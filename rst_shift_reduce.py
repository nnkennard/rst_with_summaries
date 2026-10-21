"""Shift-reduce RST discourse parser over gold EDUs.

Reads RST-DT .dis trees, right-binarizes them, derives static-oracle
shift/reduce action sequences, trains an averaged perceptron to predict
actions, and evaluates with RST-Parseval (span / nuclearity / relation).

    python rst_shift_reduce.py                # train on TRAINING, eval on TEST
    python rst_shift_reduce.py --epochs 10 --save model.pkl
"""

import argparse
import glob
import os
import pickle
import random
import re
from collections import defaultdict
from dataclasses import dataclass, field

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
# Averaged perceptron
# ---------------------------------------------------------------------------

class Perceptron:
    def __init__(self):
        self.weights = defaultdict(lambda: defaultdict(float))
        self._totals = defaultdict(float)
        self._stamps = defaultdict(int)
        self.t = 0
        self.labels: set[str] = set()

    def score(self, feats):
        scores = defaultdict(float)
        for f in feats:
            for label, w in self.weights.get(f, {}).items():
                scores[label] += w
        return scores

    def predict(self, feats, state: State):
        scores = self.score(feats)
        legal = [a for a in self.labels if state.legal(a)]
        return max(legal, key=lambda a: (scores.get(a, 0.0), a))

    def _bump(self, f, label, delta):
        key = (f, label)
        self._totals[key] += (self.t - self._stamps[key]) * self.weights[f][label]
        self._stamps[key] = self.t
        self.weights[f][label] += delta

    def update(self, gold, guess, feats):
        self.t += 1
        if gold == guess:
            return
        for f in feats:
            self._bump(f, gold, 1.0)
            self._bump(f, guess, -1.0)

    def average(self):
        for f, row in self.weights.items():
            for label, w in row.items():
                key = (f, label)
                total = self._totals[key] + (self.t - self._stamps[key]) * w
                row[label] = total / max(self.t, 1)

    def to_plain(self):
        return {"weights": {f: dict(r) for f, r in self.weights.items()},
                "labels": sorted(self.labels)}

    @classmethod
    def from_plain(cls, data):
        model = cls()
        for f, row in data["weights"].items():
            model.weights[f].update(row)
        model.labels = set(data["labels"])
        return model


class ShiftReduceParser:
    def __init__(self, model: Perceptron | None = None):
        self.model = model or Perceptron()

    def train(self, docs, epochs=5, seed=0):
        rng = random.Random(seed)
        examples = [(leaves(tree), oracle(tree)) for _, tree in docs]
        for _, actions in examples:
            self.model.labels.update(actions)
        for epoch in range(epochs):
            rng.shuffle(examples)
            correct = total = 0
            for edus, actions in examples:
                state = State(edus)
                for gold in actions:
                    feats = features(state)
                    guess = self.model.predict(feats, state)
                    self.model.update(gold, guess, feats)
                    correct += gold == guess
                    total += 1
                    state.apply(gold)   # static oracle: always follow gold
            print(f"epoch {epoch + 1}: train action accuracy {correct / total:.4f}")
        self.model.average()

    def parse(self, edus: list[Node]) -> Node:
        state = State(edus)
        while not state.is_final():
            state.apply(self.model.predict(features(state), state))
        return state.stack[0]


# ---------------------------------------------------------------------------
# Evaluation (RST-Parseval over binarized trees, gold EDUs)
# ---------------------------------------------------------------------------

def evaluate(parser: ShiftReduceParser, docs) -> dict[str, float]:
    hits = {"span": 0, "nuclearity": 0, "relation": 0, "full": 0}
    n_gold = n_pred = 0
    for _, gold_tree in docs:
        pred_tree = parser.parse(leaves(gold_tree))
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
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--save", help="pickle the trained model here")
    ap.add_argument("--load", help="skip training and load a pickled model")
    args = ap.parse_args()

    test = load_corpus(os.path.join(args.data, "TEST"))
    if args.load:
        with open(args.load, "rb") as f:
            parser = ShiftReduceParser(Perceptron.from_plain(pickle.load(f)))
    else:
        train = load_corpus(os.path.join(args.data, "TRAINING"))
        print(f"{len(train)} training docs, {len(test)} test docs")
        parser = ShiftReduceParser()
        parser.train(train, epochs=args.epochs)
        if args.save:
            with open(args.save, "wb") as f:
                pickle.dump(parser.model.to_plain(), f)

    for metric, score in evaluate(parser, test).items():
        print(f"{metric:>10}: {score:.4f}")


if __name__ == "__main__":
    main()
