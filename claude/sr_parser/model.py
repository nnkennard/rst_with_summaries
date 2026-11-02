"""SpanBERT-NoCoref shift-reduce parser (Guz and Carenini, 2020)."""

import torch
import torch.nn as nn
from transformers import AutoModel, BertTokenizerFast

from . import data as D
from .data import Node

FEAT_DIM = 10  # learnable embedding size per organizational feature


def load_tokenizer(name: str):
    # The hub config for SpanBERT does not set do_lower_case, and the default
    # (True) would lowercase input for this cased model.
    return BertTokenizerFast.from_pretrained(name, do_lower_case=False)


class MLP(nn.Sequential):
    """2 linear layers with GELU and dropout in between (Appendix B)."""

    def __init__(self, d_in, d_hidden, d_out, dropout):
        super().__init__(nn.Linear(d_in, d_hidden), nn.GELU(),
                         nn.Dropout(dropout), nn.Linear(d_hidden, d_out))


class EncodedDoc:
    """Tokenized document split into non-overlapping SpanBERT segments."""

    def __init__(self, doc: D.Doc, tokenizer, max_seg_len: int):
        ids, first, last, sent_end_tok = [], [], [], set()
        for i, edu in enumerate(doc.edus):
            toks = tokenizer(edu, add_special_tokens=False)["input_ids"]
            if not toks:  # never happens in RST-DT; keep indices valid anyway
                toks = [tokenizer.unk_token_id]
            first.append(len(ids))
            ids.extend(toks)
            last.append(len(ids) - 1)
            if i == len(doc) - 1 or doc.sent_ids[i + 1] != doc.sent_ids[i]:
                sent_end_tok.add(len(ids))
        edu_end_tok = {l + 1 for l in last}
        # Greedy segmentation, preferring to cut at sentence, then EDU ends.
        body = max_seg_len - 2
        self.segments, pos = [], 0
        while pos < len(ids):
            end = min(pos + body, len(ids))
            if end < len(ids):
                for bounds in (sent_end_tok, edu_end_tok):
                    cut = max((b for b in bounds if pos < b <= end), default=0)
                    if cut:
                        end = cut
                        break
            self.segments.append(ids[pos:end])
            pos = end
        self.first = torch.tensor(first)
        self.last = torch.tensor(last)
        self.n_tokens = len(ids)


class ShiftReduceParser(nn.Module):

    def __init__(self, bert_name="SpanBERT/spanbert-base-cased",
                 hidden=512, dropout=0.3, max_seg_len=384):
        super().__init__()
        # transformers>=5 defaults to the checkpoint's dtype, which is fp16 for
        # SpanBERT; fp16 AdamW state underflows and turns the weights into NaN.
        self.bert = AutoModel.from_pretrained(
            bert_name, add_pooling_layer=False, dtype=torch.float32)
        self.tokenizer = load_tokenizer(bert_name)
        self.max_seg_len = max_seg_len
        d = self.bert.config.hidden_size
        # One embedding table per feature; index 0 (missing) is a zero vector.
        self.feat_emb = nn.ModuleList(
            nn.Embedding(3, FEAT_DIM, padding_idx=0) for _ in range(D.N_FEATS))
        d_in = 3 * d + D.N_FEATS * FEAT_DIM
        self.act_clf = MLP(d_in, hidden, len(D.ACTIONS), dropout)
        self.rel_clf = MLP(d_in, hidden, len(D.RELATIONS), dropout)

    def encode_docs(self, docs: list) -> list:
        return [EncodedDoc(d, self.tokenizer, self.max_seg_len) for d in docs]

    # -- encoder ------------------------------------------------------------
    def edu_embeddings(self, encs: list) -> list:
        """Run all segments of all docs through SpanBERT in one batch and
        return, per doc, an (n_edus, d) tensor: mean of the first and last
        token embedding of each EDU."""
        tok = self.tokenizer
        segs = [s for e in encs for s in e.segments]
        L = max(len(s) for s in segs) + 2
        dev = self.bert.device
        input_ids = torch.full((len(segs), L), tok.pad_token_id, dtype=torch.long)
        mask = torch.zeros((len(segs), L), dtype=torch.long)
        for i, s in enumerate(segs):
            row = [tok.cls_token_id] + s + [tok.sep_token_id]
            input_ids[i, :len(row)] = torch.tensor(row)
            mask[i, :len(row)] = 1
        h = self.bert(input_ids=input_ids.to(dev),
                      attention_mask=mask.to(dev)).last_hidden_state
        out, k = [], 0
        for e in encs:
            toks = torch.cat([h[k + j, 1:1 + len(s)]
                              for j, s in enumerate(e.segments)])
            k += len(e.segments)
            out.append((toks[e.first.to(dev)] + toks[e.last.to(dev)]) / 2)
        return out

    # -- classifiers ----------------------------------------------------------
    def state_repr(self, edu_emb, heads, feats):
        """edu_emb: (n, d); heads: (B, 3) EDU indices of S2, S1, Q1's nuclear
        EDU, -1 if missing; feats: (B, N_FEATS) in {0,1,2}."""
        padded = torch.cat([edu_emb, edu_emb.new_zeros(1, edu_emb.size(1))])
        heads = torch.where(heads < 0, edu_emb.size(0), heads)
        v = padded[heads].flatten(1)  # (B, 3d)
        f = torch.cat([emb(feats[:, i]) for i, emb in enumerate(self.feat_emb)],
                      dim=1)
        return torch.cat([v, f], dim=1)

    @staticmethod
    def _state_inputs(doc, states):
        heads = [[s2.head if s2 else -1, s1.head if s1 else -1,
                  q1 if q1 is not None else -1] for s2, s1, q1 in states]
        feats = [D.org_features(doc, *s) for s in states]
        return torch.tensor(heads), torch.tensor(feats)

    # -- training -------------------------------------------------------------
    def loss(self, docs: list, encs: list):
        dev = self.bert.device
        reprs, acts, rels = [], [], []
        for doc, emb in zip(docs, self.edu_embeddings(encs)):
            steps = D.oracle_states(doc)
            heads, feats = self._state_inputs(doc, [s[:3] for s in steps])
            reprs.append(self.state_repr(emb, heads.to(dev), feats.to(dev)))
            acts += [s[3] for s in steps]
            rels += [s[4] for s in steps]
        x = torch.cat(reprs)
        acts = torch.tensor(acts, device=dev)
        rels = torch.tensor(rels, device=dev)
        red = rels >= 0
        l_act = nn.functional.cross_entropy(self.act_clf(x), acts)
        l_rel = nn.functional.cross_entropy(self.rel_clf(x[red]), rels[red])
        return l_act + l_rel, l_act.item(), l_rel.item()

    # -- inference --------------------------------------------------------------
    @torch.no_grad()
    def parse(self, docs: list, encs: list) -> list:
        """Greedy decoding with legal-action masking."""
        trees = []
        dev = self.bert.device
        for doc, emb in zip(docs, self.edu_embeddings(encs)):
            n = len(doc)
            stack, nodes, q = [], [], 0
            while not (q == n and len(stack) == 1):
                s1 = stack[-1] if stack else None
                s2 = stack[-2] if len(stack) > 1 else None
                q1 = q if q < n else None
                heads, feats = self._state_inputs(doc, [(s2, s1, q1)])
                x = self.state_repr(emb, heads.to(dev), feats.to(dev))
                logits = self.act_clf(x)[0]
                if len(stack) < 2:
                    act = D.SHIFT
                elif q == n:
                    act = int(logits[1:].argmax()) + 1
                else:
                    act = int(logits.argmax())
                if act == D.SHIFT:
                    nodes.append(Node(q, q, text=doc.edus[q]))
                else:
                    rel = D.RELATIONS[int(self.rel_clf(x)[0].argmax())]
                    nodes = nodes[:-2] + [_make_node(nodes[-2], nodes[-1],
                                                     D.ACT2NUC[act], rel)]
                stack, q = D.apply(stack, q, act)
            trees.append(nodes[0])
        return trees


def _make_node(left: Node, right: Node, nuc: str, rel: str) -> Node:
    left.nuc, right.nuc = nuc[0], nuc[1]
    left.rel = D.SPAN if nuc == "NS" else rel
    right.rel = D.SPAN if nuc == "SN" else rel
    return Node(left.start, right.end, children=[left, right])
