"""Shift-reduce RST discourse parser over gold EDUs.

Reads RST-DT .dis trees, right-binarizes them, derives static-oracle
shift/reduce action sequences, and trains an MLP to predict actions from
frozen SpanBERT representations of the top two stack spans and the next EDU
([first token; last token; mean] per span) together with sparse lexical and
structural features approximating Feng & Eisenstein (2014). Evaluates with RST-Parseval
(span / nuclearity / relation).

    python rst_shift_reduce.py --cache spanbert.pt          # train on TRAINING, eval on TEST
    python rst_shift_reduce.py --cache spanbert.pt --save model.pt
"""

import argparse
import glob
import os
import re
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "rst_discourse_treebank", "data", "RSTtrees-WSJ-main-1.0")

# Fine-grained RST-DT relations -> the 18 coarse classes of Carlson et al. (2001).
COARSE = {
    "attribution": "Attribution", "attribution-negative": "Attribution",
    "background": "Background", "circumstance": "Background",
    "cause": "Cause", "result": "Cause", "consequence": "Cause", "cause-result": "Cause",
    "comparison": "Comparison", "preference": "Comparison", "analogy": "Comparison",
    "proportion": "Comparison",
    "condition": "Condition", "hypothetical": "Condition", "contingency": "Condition",
    "otherwise": "Condition",
    "contrast": "Contrast", "concession": "Contrast", "antithesis": "Contrast",
    "example": "Elaboration", "definition": "Elaboration",
    "purpose": "Enablement", "enablement": "Enablement",
    "evaluation": "Evaluation", "interpretation": "Evaluation", "conclusion": "Evaluation",
    "comment": "Evaluation",
    "evidence": "Explanation", "explanation-argumentative": "Explanation", "reason": "Explanation",
    "list": "Joint", "disjunction": "Joint",
    "manner": "Manner-Means", "means": "Manner-Means",
    "problem-solution": "Topic-Comment", "question-answer": "Topic-Comment",
    "statement-response": "Topic-Comment", "topic-comment": "Topic-Comment",
    "comment-topic": "Topic-Comment", "rhetorical-question": "Topic-Comment",
    "summary": "Summary", "restatement": "Summary",
    "temporal-before": "Temporal", "temporal-after": "Temporal",
    "temporal-same-time": "Temporal", "sequence": "Temporal", "inverted-sequence": "Temporal",
    "topic-shift": "Topic-Change", "topic-drift": "Topic-Change",
    "textualorganization": "Textual-Organization",
    "same-unit": "Same-Unit",
}


def coarse_relation(rel: str) -> str:
    rel = rel.lower()
    # Strip embedded / multinuclear-variant suffixes such as "-e", "-s-e", "-n".
    while re.search(r"-[esn]$", rel):
        rel = rel[:-2]
    if rel.startswith("elaboration"):
        return "Elaboration"
    return COARSE.get(rel, rel)


# ---------------------------------------------------------------------------
# Trees
# ---------------------------------------------------------------------------

@dataclass
class Node:
    start: int                      # 1-based EDU index, inclusive
    end: int
    nuc: str = "N"                  # nuclearity w.r.t. parent: "N" or "S"
    rel: str = "span"               # rel2par of this node
    children: list = field(default_factory=list)
    text: str = ""                  # leaves only
    label: str = ""                 # internal nodes: "NS-Elaboration" etc.

    @property
    def is_leaf(self):
        return not self.children


_TOKEN = re.compile(r"_!.*?_!|\(|\)|[^\s()]+", re.S)


def read_dis(path: str) -> Node:
    with open(path, encoding="utf-8", errors="replace") as f:
        # A couple of files carry stray "//TT_ERR" annotator markers; drop them.
        tokens = [t for t in _TOKEN.findall(f.read()) if not t.startswith("//")]
    pos = 0

    def parse():
        nonlocal pos
        assert tokens[pos] == "("
        pos += 1
        kind = tokens[pos]
        pos += 1
        node = Node(0, 0, nuc="S" if kind == "Satellite" else "N")
        while tokens[pos] != ")":
            assert tokens[pos] == "("
            key = tokens[pos + 1]
            if key == "span":
                node.start, node.end = int(tokens[pos + 2]), int(tokens[pos + 3])
                pos += 5
            elif key == "leaf":
                node.start = node.end = int(tokens[pos + 2])
                pos += 4
            elif key == "rel2par":
                node.rel = tokens[pos + 2]
                pos += 4
            elif key == "text":
                node.text = tokens[pos + 2][2:-2]
                pos += 4
            else:
                node.children.append(parse())
        pos += 1
        return node

    return parse()


def binarize(node: Node) -> Node:
    """Right-binarize nodes with more than two children."""
    kids = [binarize(c) for c in node.children]
    while len(kids) > 2:
        right = Node(kids[-2].start, kids[-1].end, children=kids[-2:])
        # The new node inherits the role its children share (multinuclear lists),
        # otherwise it is the nucleus that the earlier satellites attach to.
        if all(k.nuc == "N" for k in kids[-2:]):
            right.rel = kids[-2].rel
        kids = kids[:-2] + [right]
    node.children = kids
    return node


def node_label(left: Node, right: Node) -> str:
    nuc = left.nuc + right.nuc
    if nuc == "NN":
        rel = left.rel
    else:
        rel = right.rel if nuc == "NS" else left.rel
    return f"{nuc}-{coarse_relation(rel)}"


def annotate_labels(node: Node) -> Node:
    if not node.is_leaf:
        for c in node.children:
            annotate_labels(c)
        node.label = node_label(*node.children)
    return node


def leaves(node: Node) -> list[Node]:
    if node.is_leaf:
        return [node]
    return [l for c in node.children for l in leaves(c)]


def labeled_spans(node: Node) -> set[tuple]:
    """(start, end, nuclearity-label, relation-label) for each child of each internal node."""
    out = set()
    if node.is_leaf:
        return out
    left, right = node.children
    nuc, rel = node.label.split("-", 1)
    for child, n in ((left, nuc[0]), (right, nuc[1])):
        r = rel if n == "S" or nuc == "NN" else "span"
        out.add((child.start, child.end, n, r))
        out |= labeled_spans(child)
    return out


def load_corpus(split_dir: str) -> list[tuple[str, Node]]:
    docs = []
    for path in sorted(glob.glob(os.path.join(split_dir, "*.dis"))):
        tree = annotate_labels(binarize(read_dis(path)))
        docs.append((os.path.basename(path), tree))
    return docs


# ---------------------------------------------------------------------------
# Transition system
# ---------------------------------------------------------------------------

SHIFT = "SHIFT"


def oracle(tree: Node) -> list[str]:
    """Post-order traversal of a binary tree = gold shift/reduce sequence."""
    if tree.is_leaf:
        return [SHIFT]
    left, right = tree.children
    return oracle(left) + oracle(right) + ["REDUCE-" + tree.label]


class State:
    def __init__(self, edus: list[Node]):
        self.edus = edus
        self.buffer = 0             # index of next EDU to shift
        self.stack: list[Node] = []

    def is_final(self):
        return self.buffer == len(self.edus) and len(self.stack) == 1

    def legal(self, action: str) -> bool:
        if action == SHIFT:
            return self.buffer < len(self.edus)
        return len(self.stack) >= 2

    def apply(self, action: str):
        if action == SHIFT:
            e = self.edus[self.buffer]
            self.stack.append(Node(e.start, e.end, text=e.text))
            self.buffer += 1
            return
        label = action[len("REDUCE-"):]
        right, left = self.stack.pop(), self.stack.pop()
        nuc, rel = label.split("-", 1)
        left.nuc, right.nuc = nuc[0], nuc[1]
        # Mirror RST-DT: satellites (and multinuclear siblings) carry the relation.
        left.rel = rel if nuc[0] == "S" or nuc == "NN" else "span"
        right.rel = rel if nuc[1] == "S" or nuc == "NN" else "span"
        self.stack.append(Node(left.start, right.end, children=[left, right], label=label))


# ---------------------------------------------------------------------------
# Sparse features (approximating Feng & Eisenstein 2014, Table 2)
# ---------------------------------------------------------------------------

def _words(text: str) -> list[str]:
    return re.findall(r"\w+|[^\w\s]", text.lower())


def _span_text(state: State, node: Node) -> str:
    return " ".join(state.edus[i - 1].text for i in range(node.start, node.end + 1))


def features(state: State) -> list[str]:
    n = len(state.edus)
    feats = ["bias", f"stack={min(len(state.stack), 3)}",
             f"buf={min(n - state.buffer, 3)}"]
    slots = {"s0": state.stack[-1] if state.stack else None,
             "s1": state.stack[-2] if len(state.stack) > 1 else None,
             "q0": state.edus[state.buffer] if state.buffer < n else None}
    for name, node in slots.items():
        if node is None:
            feats.append(f"{name}=NONE")
            continue
        first = _words(state.edus[node.start - 1].text)
        last = _words(state.edus[node.end - 1].text)
        length = node.end - node.start + 1
        feats += [
            f"{name}.w0={first[0] if first else ''}",
            f"{name}.w01={'_'.join(first[:2])}",
            f"{name}.wl={last[-1] if last else ''}",
            f"{name}.len={min(length, 10)}",
            f"{name}.sentend={bool(last) and last[-1] in {'.', '?', '!', chr(39) + chr(39)}}",
            f"{name}.start={node.start == 1}",
            f"{name}.endq={node.end == n}",
        ]
        if not node.is_leaf:
            feats.append(f"{name}.label={node.label}")
        if name != "q0":
            feats.append(f"{name}.ntok={min(len(_words(_span_text(state, node))) // 10, 10)}")
    s0, s1, q0 = slots["s0"], slots["s1"], slots["q0"]
    if s0 and s1:
        a = _words(state.edus[s1.end - 1].text)
        b = _words(state.edus[s0.start - 1].text)
        feats += [f"s1wl_s0w0={a[-1] if a else ''}_{b[0] if b else ''}",
                  f"s1len_s0len={min(s1.end - s1.start + 1, 5)}_{min(s0.end - s0.start + 1, 5)}"]
    if s0 and q0:
        a = _words(state.edus[s0.end - 1].text)
        b = _words(q0.text)
        feats.append(f"s0wl_q0w0={a[-1] if a else ''}_{b[0] if b else ''}")
    return feats



# ---------------------------------------------------------------------------
# SpanBERT features
# ---------------------------------------------------------------------------

SPANBERT = "SpanBERT/spanbert-base-cased"
TOKENIZER = "bert-base-cased"   # SpanBERT reuses BERT-base-cased's vocabulary


def default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class SpanEncoder:
    """Contextual token vectors for a whole document from frozen SpanBERT.

    Documents longer than 512 subwords are encoded in overlapping windows; each
    token keeps the vector from the window in which it is most central.
    """

    def __init__(self, device: str, window: int = 510, stride: int = 255):
        self.tok = AutoTokenizer.from_pretrained(TOKENIZER)
        self.bert = AutoModel.from_pretrained(SPANBERT).to(device).eval()
        self.device, self.window, self.stride = device, window, stride

    @torch.no_grad()
    def encode(self, edus: list[Node]) -> dict:
        ids, edu_bounds = [], []
        for e in edus:
            piece = self.tok(e.text, add_special_tokens=False)["input_ids"] or [self.tok.unk_token_id]
            edu_bounds.append((len(ids), len(ids) + len(piece) - 1))
            ids += piece
        n = len(ids)
        out = torch.zeros(n, self.bert.config.hidden_size)
        best = torch.full((n,), float("inf"))
        starts = list(range(0, max(n - self.window, 0) + 1, self.stride))
        if starts[-1] + self.window < n:
            starts.append(n - self.window)
        for s in starts:
            chunk = ids[s:s + self.window]
            inp = torch.tensor([[self.tok.cls_token_id] + chunk + [self.tok.sep_token_id]],
                               device=self.device)
            h = self.bert(input_ids=inp).last_hidden_state[0, 1:-1].float().cpu()
            pos = torch.arange(s, s + len(chunk))
            dist = (pos - (s + (len(chunk) - 1) / 2)).abs()
            take = dist < best[pos]
            out[pos[take]] = h[take]
            best[pos[take]] = dist[take]
        cums = torch.cat([torch.zeros(1, out.size(1)), out.cumsum(0)])
        return {"tokens": out, "cumsum": cums, "edu_bounds": edu_bounds}


def span_vector(enc: dict, start: int, end: int) -> torch.Tensor:
    """[first token; last token; mean] over EDUs start..end (1-based, inclusive)."""
    a = enc["edu_bounds"][start - 1][0]
    b = enc["edu_bounds"][end - 1][1]
    mean = (enc["cumsum"][b + 1] - enc["cumsum"][a]) / (b - a + 1)
    return torch.cat([enc["tokens"][a], enc["tokens"][b], mean])


def config_spans(state: State) -> list[tuple[int, int] | None]:
    """Spans of the parser configuration's feature slots: s1, s0, q0."""
    s = state.stack
    q0 = state.edus[state.buffer] if state.buffer < len(state.edus) else None
    return [(n.start, n.end) if n else None
            for n in (s[-2] if len(s) > 1 else None, s[-1] if s else None, q0)]


# ---------------------------------------------------------------------------
# Action classifier
# ---------------------------------------------------------------------------

class ActionMLP(nn.Module):
    """One hidden layer fed by both the SpanBERT slot vectors and the sparse
    indicator features (summed embeddings of feature ids = a linear layer on
    their one-hot encoding)."""

    def __init__(self, span_dim: int, n_feats: int, n_actions: int, n_slots: int = 3,
                 hidden: int = 512, dropout: float = 0.3):
        super().__init__()
        self.null = nn.Parameter(torch.zeros(n_slots, span_dim))
        self.dense = nn.Sequential(nn.Dropout(dropout), nn.Linear(n_slots * span_dim, hidden))
        self.sparse = nn.Embedding(n_feats, hidden)   # nn.EmbeddingBag has no MPS kernel
        self.out = nn.Sequential(nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, n_actions))

    def forward(self, slots, present, feat_ids, feat_rows) -> torch.Tensor:
        # slots: (B, n_slots, span_dim); present: (B, n_slots) bool;
        # feat_ids: flattened sparse feature ids; feat_rows: batch row of each id
        slots = torch.where(present.unsqueeze(-1), slots, self.null.expand_as(slots))
        h = self.dense(slots.flatten(1))
        return self.out(h.index_add(0, feat_rows, self.sparse(feat_ids)))


def pack_feats(id_lists: list[list[int]], device) -> tuple[torch.Tensor, torch.Tensor]:
    flat = torch.tensor([i for ids in id_lists for i in ids], dtype=torch.long)
    rows = torch.tensor([r for r, ids in enumerate(id_lists) for _ in ids], dtype=torch.long)
    return flat.to(device), rows.to(device)


def featurize(enc: dict, state: State, span_dim: int, vocab: dict[str, int]):
    spans = config_spans(state)
    slots = torch.stack([span_vector(enc, *sp) if sp else torch.zeros(span_dim) for sp in spans])
    present = torch.tensor([sp is not None for sp in spans])
    feat_ids = [vocab[f] for f in features(state) if f in vocab]   # unseen features are dropped
    return slots, present, feat_ids


class ShiftReduceParser:
    def __init__(self, encoder: SpanEncoder, actions: list[str] | None = None,
                 vocab: dict[str, int] | None = None, model: ActionMLP | None = None,
                 cache: dict | None = None):
        self.encoder = encoder
        self.actions = actions or []
        self.vocab = vocab or {}
        self.model = model
        self.cache = cache if cache is not None else {}
        self.span_dim = 3 * encoder.bert.config.hidden_size

    def encode(self, name: str, edus: list[Node]) -> dict:
        if name not in self.cache:
            self.cache[name] = self.encoder.encode(edus)
        return self.cache[name]

    def legal_mask(self, state: State) -> torch.Tensor:
        return torch.tensor([state.legal(a) for a in self.actions])

    def train(self, docs, epochs=10, seed=0, lr=1e-3, batch_size=64, hidden=512, dropout=0.3):
        torch.manual_seed(seed)
        self.actions = sorted({a for _, tree in docs for a in oracle(tree)})
        index = {a: i for i, a in enumerate(self.actions)}
        slots, present, feat_ids, gold, legal = [], [], [], [], []
        for name, tree in docs:
            edus = leaves(tree)
            enc = self.encode(name, edus)
            state = State(edus)
            for action in oracle(tree):     # static oracle: states follow gold
                for f in features(state):
                    self.vocab.setdefault(f, len(self.vocab))
                x, p, ids = featurize(enc, state, self.span_dim, self.vocab)
                slots.append(x)
                present.append(p)
                feat_ids.append(ids)
                gold.append(index[action])
                legal.append(self.legal_mask(state))
                state.apply(action)
        slots, present = torch.stack(slots), torch.stack(present)
        gold, legal = torch.tensor(gold), torch.stack(legal)
        print(f"{len(gold)} oracle states, {len(self.actions)} actions, "
              f"{len(self.vocab)} sparse features")

        device = self.encoder.device
        self.model = ActionMLP(self.span_dim, len(self.vocab), len(self.actions), hidden=hidden,
                               dropout=dropout).to(device)
        opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        for epoch in range(epochs):
            self.model.train()
            perm = torch.randperm(len(gold))
            correct = total_loss = 0.0
            for i in range(0, len(perm), batch_size):
                b = perm[i:i + batch_size]
                logits = self.model(slots[b].to(device), present[b].to(device),
                                    *pack_feats([feat_ids[j] for j in b], device))
                logits = logits.masked_fill(~legal[b].to(device), float("-inf"))
                loss = F.cross_entropy(logits, gold[b].to(device))
                opt.zero_grad()
                loss.backward()
                opt.step()
                total_loss += loss.item() * len(b)
                correct += (logits.argmax(-1).cpu() == gold[b]).sum().item()
            print(f"epoch {epoch + 1}: loss {total_loss / len(gold):.4f}, "
                  f"train action accuracy {correct / len(gold):.4f}")
        self.model.eval()

    @torch.no_grad()
    def parse(self, name: str, edus: list[Node]) -> Node:
        enc = self.encode(name, edus)
        device = self.encoder.device
        state = State(edus)
        while not state.is_final():
            x, p, ids = featurize(enc, state, self.span_dim, self.vocab)
            logits = self.model(x[None].to(device), p[None].to(device),
                                *pack_feats([ids], device))[0].cpu()
            logits = logits.masked_fill(~self.legal_mask(state), float("-inf"))
            state.apply(self.actions[logits.argmax().item()])
        return state.stack[0]


# ---------------------------------------------------------------------------
# Evaluation (RST-Parseval over binarized trees, gold EDUs)
# ---------------------------------------------------------------------------

def evaluate(parser: ShiftReduceParser, docs) -> dict[str, float]:
    hits = {"span": 0, "nuclearity": 0, "relation": 0, "full": 0}
    n_gold = n_pred = 0
    for name, gold_tree in docs:
        pred_tree = parser.parse(name, leaves(gold_tree))
        gold, pred = labeled_spans(gold_tree), labeled_spans(pred_tree)
        n_gold += len(gold)
        n_pred += len(pred)
        for proj, key in ((lambda s: s[:2], "span"), (lambda s: s[:3], "nuclearity"),
                          (lambda s: (s[0], s[1], s[3]), "relation"), (lambda s: s, "full")):
            hits[key] += len({proj(s) for s in gold} & {proj(s) for s in pred})
    return {k: 2 * v / (n_gold + n_pred) for k, v in hits.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", default=DATA_DIR, help="directory with TRAINING/ and TEST/")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--device", default=default_device())
    ap.add_argument("--cache", help="cache SpanBERT document encodings in this file")
    ap.add_argument("--save", help="save the trained classifier here")
    ap.add_argument("--load", help="skip training and load a saved classifier")
    args = ap.parse_args()

    cache = torch.load(args.cache) if args.cache and os.path.exists(args.cache) else {}
    encoder = SpanEncoder(args.device)
    test = load_corpus(os.path.join(args.data, "TEST"))
    if args.load:
        ckpt = torch.load(args.load, map_location=args.device)
        model = ActionMLP(3 * encoder.bert.config.hidden_size, len(ckpt["vocab"]),
                          len(ckpt["actions"]), hidden=ckpt["hidden"]).to(args.device)
        model.load_state_dict(ckpt["state_dict"])
        parser = ShiftReduceParser(encoder, ckpt["actions"], ckpt["vocab"], model.eval(), cache)
    else:
        train = load_corpus(os.path.join(args.data, "TRAINING"))
        print(f"{len(train)} training docs, {len(test)} test docs")
        parser = ShiftReduceParser(encoder, cache=cache)
        parser.train(train, epochs=args.epochs, lr=args.lr, hidden=args.hidden,
                     dropout=args.dropout)
        if args.save:
            torch.save({"actions": parser.actions, "vocab": parser.vocab, "hidden": args.hidden,
                        "state_dict": parser.model.state_dict()}, args.save)

    scores = evaluate(parser, test)
    if args.cache:
        torch.save(parser.cache, args.cache)
    for metric, score in scores.items():
        print(f"{metric:>10}: {score:.4f}")


if __name__ == "__main__":
    main()
