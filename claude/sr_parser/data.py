"""RST-DT loading for the shift-reduce parser.

Reads ``.dis`` files, maps relations to the 18 coarse classes, binarizes the
trees, derives sentence/paragraph boundaries, and produces the static oracle
(gold Shift / Reduce-XN-XR sequence) used for training.
"""

import glob
import os
import random
import re
from dataclasses import dataclass, field
from typing import Optional

# ---------------------------------------------------------------------------
# Relations
# ---------------------------------------------------------------------------

# 18 coarse classes (Carlson et al., 2001).
_COARSE = {
    "Attribution": ["attribution", "attribution-negative"],
    "Background": ["background", "circumstance"],
    "Cause": ["cause", "result", "consequence", "cause-result"],
    "Comparison": ["comparison", "preference", "analogy", "proportion"],
    "Condition": ["condition", "hypothetical", "contingency", "otherwise"],
    "Contrast": ["contrast", "concession", "antithesis"],
    "Elaboration": [
        "elaboration-additional", "elaboration-general-specific",
        "elaboration-part-whole", "elaboration-process-step",
        "elaboration-object-attribute", "elaboration-set-member", "example",
        "definition"
    ],
    "Enablement": ["purpose", "enablement"],
    "Evaluation": ["evaluation", "interpretation", "conclusion", "comment"],
    "Explanation": ["evidence", "explanation-argumentative", "reason"],
    "Joint": ["list", "disjunction"],
    "Manner-Means": ["manner", "means"],
    "Topic-Comment": [
        "problem-solution", "question-answer", "statement-response",
        "topic-comment", "comment-topic", "rhetorical-question"
    ],
    "Summary": ["summary", "restatement"],
    "Temporal": [
        "temporal-before", "temporal-after", "temporal-same-time", "sequence",
        "inverted-sequence"
    ],
    "Topic-Change": ["topic-shift", "topic-drift"],
    "Textual-Organization": ["textualorganization"],
    "Same-Unit": ["same-unit"],
}
RELATIONS = sorted(_COARSE)
REL2ID = {r: i for i, r in enumerate(RELATIONS)}
_FINE2COARSE = {f: c for c, fs in _COARSE.items() for f in fs}
SPAN = "span"


def coarsen(fine: str) -> str:
    r = fine.strip().lower()
    if r == SPAN:
        return SPAN
    if r.endswith("-e"):  # embedded variant
        r = r[:-2]
    if r not in _FINE2COARSE and r.endswith(("-n", "-s")):  # nuclearity-marked
        r = r[:-2]
    return _FINE2COARSE[r]


# Actions of the first classifier.
SHIFT, RED_NN, RED_NS, RED_SN = range(4)
ACTIONS = ["Shift", "Reduce-NN", "Reduce-NS", "Reduce-SN"]
NUC2ACT = {"NN": RED_NN, "NS": RED_NS, "SN": RED_SN}
ACT2NUC = {v: k for k, v in NUC2ACT.items()}

# ---------------------------------------------------------------------------
# Trees
# ---------------------------------------------------------------------------


@dataclass
class Node:
    start: int  # first EDU, 0-based inclusive
    end: int  # last EDU, 0-based inclusive
    nuc: str = "Root"  # this node's role w.r.t. its parent: 'N', 'S' or 'Root'
    rel: Optional[str] = None  # coarse rel2par, or 'span'
    children: list = field(default_factory=list)
    text: Optional[str] = None

    @property
    def is_leaf(self) -> bool:
        return not self.children

    @property
    def head(self) -> int:
        """Nuclear EDU (leftmost nucleus for multinuclear nodes)."""
        if self.is_leaf:
            return self.start
        for c in self.children:
            if c.nuc == "N":
                return c.head
        return self.children[0].head

    def nuc_label(self) -> str:
        return "".join(c.nuc for c in self.children)

    def relation(self) -> str:
        """Relation holding between the two children of a binary node."""
        l, r = self.children
        if l.nuc == "S":
            return l.rel
        if r.nuc == "S":
            return r.rel
        return l.rel if l.rel != SPAN else r.rel

    def __str__(self):
        if self.is_leaf:
            return str(self.start + 1)
        kids = " ".join(str(c) for c in self.children)
        return f"({self.nuc_label()}:{self.relation()} {kids})"


_TOKEN_RE = re.compile(r"_!.*?_!|\(|\)|[^\s()]+", re.S)


def _sexp(text: str) -> list:
    stack = [[]]
    for tok in _TOKEN_RE.findall(text):
        if tok == "(":
            stack.append([])
        elif tok == ")":
            done = stack.pop()
            stack[-1].append(done)
        else:
            stack[-1].append(tok)
    assert len(stack) == 1 and len(stack[0]) == 1, "unbalanced parentheses"
    return stack[0][0]


def _build(sexp: list) -> Node:
    node = Node(0, 0, nuc={"Root": "Root", "Nucleus": "N",
                           "Satellite": "S"}[sexp[0]])
    for item in sexp[1:]:
        head = item[0]
        if head == "span":
            node.start, node.end = int(item[1]) - 1, int(item[2]) - 1
        elif head == "leaf":
            node.start = node.end = int(item[1]) - 1
        elif head == "rel2par":
            node.rel = coarsen(item[1])
        elif head == "text":
            node.text = " ".join(item[1:])[2:-2]
        elif head in ("Nucleus", "Satellite"):
            node.children.append(_build(item))
    return node


def _binarize_children(kids: list) -> list:
    if len(kids) <= 2:
        return kids
    first, rest = kids[0], kids[1:]
    if any(k.nuc == "N" for k in rest):
        # Right-branching: (A B C ...) -> (A (B C ...)).
        if first.nuc == "N" and first.rel != SPAN:
            rel = first.rel  # multinuclear: new node shares the relation
        else:
            rel = SPAN
        group = Node(rest[0].start, rest[-1].end, "N", rel,
                     _binarize_children(rest))
        return [first, group]
    # N S S ...: grouping the satellites would make a node with no nucleus, so
    # attach the nucleus to the first satellite instead.
    group = Node(first.start, rest[0].end, "N", SPAN, [first, rest[0]])
    return _binarize_children([group] + rest[1:])


def binarize(node: Node) -> Node:
    for c in node.children:
        binarize(c)
    node.children = _binarize_children(node.children)
    return node


def leaves(node: Node) -> list:
    return [node] if node.is_leaf else [l for c in node.children for l in leaves(c)]


# ---------------------------------------------------------------------------
# Documents
# ---------------------------------------------------------------------------

_SENT_END_RE = re.compile(r"[.?!][\"'`)\]}]*$")


@dataclass
class Doc:
    doc_id: str
    edus: list  # EDU strings
    sent_ids: list  # sentence index of each EDU
    para_ids: list  # paragraph index of each EDU
    tree: Node  # binarized, coarse

    def __len__(self):
        return len(self.edus)


def _ids_from_ends(ends: list) -> list:
    ids, cur = [], 0
    for e in ends:
        ids.append(cur)
        cur += e
    return ids


def _raw_paragraph_ends(raw_path: str, edus: list) -> Optional[list]:
    """Paragraph ends for files whose paragraphs are indented lines in the raw
    text (file1-file5). Aligns EDUs to the raw text ignoring whitespace."""
    if not os.path.exists(raw_path):
        return None
    with open(raw_path, encoding="utf-8", errors="replace") as f:
        lines = f.read().split("\n")
    para_starts, pos = set(), 0  # offsets into the whitespace-free raw text
    for line in lines:
        if line[:1].isspace() and line.strip():
            para_starts.add(pos)
        pos += len(re.sub(r"\s", "", line))
    ends, pos = [], 0
    for e in edus:
        pos += len(re.sub(r"\s", "", e))
        ends.append(pos in para_starts)
    if sum(ends) == 0:
        return None
    return ends


def load_doc(path: str) -> Doc:
    with open(path, encoding="utf-8", errors="replace") as f:
        text = f.read().replace("//TT_ERR", "")
    tree = binarize(_build(_sexp(text)))
    lv = leaves(tree)
    assert [l.start for l in lv] == list(range(len(lv))), path
    raw = [l.text for l in lv]
    edus = [re.sub(r"\s+", " ", t.replace("<P>", " ")).strip() for t in raw]
    n = len(edus)

    para_end = [("<P>" in t) for t in raw]
    if not any(para_end[:-1]):
        raw_path = path[:-len(".out.dis")] + ".out" if path.endswith(
            ".out.dis") else path[:-len(".dis")]
        para_end = _raw_paragraph_ends(raw_path, edus) or para_end
    para_end[-1] = True
    # Sentence boundary heuristic: sentence-final punctuation (optionally
    # followed by closing quotes/brackets), or a paragraph end.
    sent_end = [para_end[i] or bool(_SENT_END_RE.search(edus[i]))
                for i in range(n)]

    doc_id = os.path.basename(path).replace(".out.dis", "").replace(".dis", "")
    return Doc(doc_id, edus, _ids_from_ends(sent_end),
               _ids_from_ends(para_end), tree)


def load_split(rst_dt_dir: str, split: str) -> list:
    """split: 'TRAINING' or 'TEST'."""
    paths = sorted(glob.glob(os.path.join(rst_dt_dir, split, "*.dis")))
    if not paths:
        raise FileNotFoundError(f"no .dis files for {split} in {rst_dt_dir}")
    return [load_doc(p) for p in paths]


def train_dev_split(docs: list, dev_frac: float = 0.1, seed: int = 0):
    """10% dev split, stratified by number of EDUs: sort by length, cut into
    consecutive buckets of 1/dev_frac docs, draw one dev doc per bucket."""
    rng = random.Random(seed)
    order = sorted(docs, key=lambda d: (len(d), d.doc_id))
    k = round(1 / dev_frac)
    train, dev = [], []
    for i in range(0, len(order), k):
        bucket = order[i:i + k]
        j = rng.randrange(len(bucket)) if len(bucket) == k else -1
        for b, d in enumerate(bucket):
            (dev if b == j else train).append(d)
    return train, dev


def cv_folds(docs: list, k: int = 10, seed: int = 0) -> list:
    """k (train, dev) pairs, stratified by number of EDUs: sort by length, cut
    into consecutive buckets of k docs, deal each bucket randomly across the
    folds. Every doc is in exactly one dev set."""
    if not 2 <= k <= len(docs):
        raise ValueError(f"cannot make {k} folds from {len(docs)} docs")
    rng = random.Random(seed)
    order = sorted(docs, key=lambda d: (len(d), d.doc_id))
    devs = [[] for _ in range(k)]
    for i in range(0, len(order), k):
        bucket = order[i:i + k]
        for d, fold in zip(bucket, rng.sample(range(k), k)):
            devs[fold].append(d)
    folds = []
    for dev in devs:
        dev_ids = {id(d) for d in dev}
        folds.append(([d for d in docs if id(d) not in dev_ids], dev))
    return folds


# ---------------------------------------------------------------------------
# Oracle and parser states
# ---------------------------------------------------------------------------


@dataclass
class Span:
    start: int
    end: int
    head: int


def oracle(tree: Node) -> list:
    """Post-order traversal -> list of (action, relation_id or -1)."""
    out = []

    def visit(n):
        if n.is_leaf:
            out.append((SHIFT, -1))
            return
        visit(n.children[0])
        visit(n.children[1])
        out.append((NUC2ACT[n.nuc_label()], REL2ID[n.relation()]))

    visit(tree)
    return out


def oracle_states(doc: Doc) -> list:
    """Replay the oracle; return (s2, s1, q1, action, rel) per step, where
    s2/s1 are Spans or None and q1 is an EDU index or None."""
    stack, q, steps = [], 0, []
    n = len(doc)
    for act, rel in oracle(doc.tree):
        s1 = stack[-1] if stack else None
        s2 = stack[-2] if len(stack) > 1 else None
        steps.append((s2, s1, q if q < n else None, act, rel))
        stack, q = apply(stack, q, act)
    assert len(stack) == 1 and q == n
    return steps


def apply(stack: list, q: int, act: int):
    """Apply an action; reduces pick the head from the predicted nuclearity."""
    if act == SHIFT:
        return stack + [Span(q, q, q)], q + 1
    s2, s1 = stack[-2], stack[-1]
    head = s1.head if act == RED_SN else s2.head
    return stack[:-2] + [Span(s2.start, s1.end, head)], q


# ---------------------------------------------------------------------------
# Organizational features (Wang et al., 2017)
# ---------------------------------------------------------------------------

# Values: 0 = missing, 1 = false, 2 = true.
PAIR_FEATS = ["same_sent", "same_para"]  # for (S2, S1) and (S1, Q1)
UNARY_FEATS = ["sent_start", "sent_end", "para_start", "para_end",
               "doc_start", "doc_end"]  # for S2, S1, Q1
N_FEATS = 2 * len(PAIR_FEATS) + 3 * len(UNARY_FEATS)


def _unary(doc: Doc, start: int, end: int) -> list:
    n, s, p = len(doc), doc.sent_ids, doc.para_ids
    return [
        start == 0 or s[start - 1] != s[start],
        end == n - 1 or s[end + 1] != s[end],
        start == 0 or p[start - 1] != p[start],
        end == n - 1 or p[end + 1] != p[end],
        start == 0,
        end == n - 1,
    ]


def _pair(doc: Doc, a: tuple, b: tuple) -> list:
    s, p = doc.sent_ids, doc.para_ids
    return [s[a[0]] == s[b[1]], p[a[0]] == p[b[1]]]


def org_features(doc: Doc, s2: Optional[Span], s1: Optional[Span],
                 q1: Optional[int]) -> list:
    spans = [(x.start, x.end) if x else None for x in (s2, s1)]
    spans.append((q1, q1) if q1 is not None else None)
    feats = []
    for a, b in ((spans[0], spans[1]), (spans[1], spans[2])):
        feats += ([1 + v for v in _pair(doc, a, b)] if a and b
                  else [0] * len(PAIR_FEATS))
    for x in spans:
        feats += ([1 + v for v in _unary(doc, *x)] if x
                  else [0] * len(UNARY_FEATS))
    return feats
