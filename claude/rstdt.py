"""
Read RST Discourse Treebank (.dis) files, convert them to Python trees,
right-binarize them, and map the fine-grained relations to the 18 coarse
classes of Carlson et al. (2001).

Usage:
    python rstdt.py path/to/RSTtrees-WSJ-main-1.0/TRAINING  [--out train.jsonl]

As a library:
    from rstdt import load_corpus
    trees = load_corpus("…/TRAINING")          # {doc_id: RSTNode}, binarized + coarse
    t = trees["wsj_0600"]
    t.to_nltk().pretty_print()                   # requires nltk
    t.to_tuple()                                 # plain nested tuples
"""

#from __future__ import annotations # not used?

import argparse
import glob
import json
import os
import re
from dataclasses import dataclass, field
from typing import Optional, cast

# ---------------------------------------------------------------------------
# Coarse relation mapping (18 classes; Carlson et al. 2001, Table 1)
# ---------------------------------------------------------------------------

_COARSE_CLASSES = {
    "Attribution": ["attribution", "attribution-negative"],
    "Background": ["background", "circumstance"],
    "Cause": ["cause", "result", "consequence", "consequence-n", "consequence-s",
              "cause-result"],
    "Comparison": ["comparison", "preference", "analogy", "proportion"],
    "Condition": ["condition", "hypothetical", "contingency", "otherwise"],
    "Contrast": ["contrast", "concession", "antithesis"],
    "Elaboration": ["elaboration-additional", "elaboration-general-specific",
                    "elaboration-part-whole", "elaboration-process-step",
                    "elaboration-object-attribute", "elaboration-set-member",
                    "example", "definition"],
    "Enablement": ["purpose", "enablement"],
    "Evaluation": ["evaluation", "evaluation-n", "evaluation-s", "interpretation",
                   "interpretation-n", "interpretation-s", "conclusion", "comment"],
    "Explanation": ["evidence", "explanation-argumentative", "reason"],
    "Joint": ["list", "disjunction"],
    "Manner-Means": ["manner", "means"],
    "Topic-Comment": ["problem-solution", "problem-solution-n", "problem-solution-s",
                      "question-answer", "question-answer-n", "question-answer-s",
                      "statement-response", "statement-response-n",
                      "statement-response-s", "topic-comment", "comment-topic",
                      "rhetorical-question"],
    "Summary": ["summary", "summary-n", "summary-s", "restatement"],
    "Temporal": ["temporal-before", "temporal-after", "temporal-same-time",
                 "sequence", "inverted-sequence"],
    "Topic-Change": ["topic-shift", "topic-drift"],
    "Textual-Organization": ["textualorganization", "textual-organization"],
    "Same-Unit": ["same-unit"],
}

FINE2COARSE = {fine: coarse for coarse, fines in _COARSE_CLASSES.items() for fine in fines}


def to_coarse(rel: Optional[str]) -> Optional[str]:
    """Map a fine RST-DT relation to its coarse class. 'span' and None pass through."""
    if rel is None:
        return None
    r = rel.strip().lower()
    if r == "span":
        return "span"
    candidates = [r]
    if r.endswith("-e"):                       # embedded variants, e.g. elaboration-additional-e
        candidates.append(r[:-2])
    for c in list(candidates):
        if c.endswith(("-n", "-s")):           # nuclearity-marked variants, e.g. consequence-n-e
            candidates.append(c[:-2])
    for c in candidates:
        if c in FINE2COARSE:
            return FINE2COARSE[c]
    raise KeyError(f"Unknown RST relation: {rel!r}")


# ---------------------------------------------------------------------------
# Tree data structure
# ---------------------------------------------------------------------------

@dataclass
class RSTNode:
    nuclearity: str                 # 'Root', 'Nucleus' or 'Satellite'
    relation: Optional[str] = None  # rel2par: relation to the parent (None for Root)
    span: tuple = (0, 0)            # (first EDU, last EDU), 1-based, inclusive
    children: list = field(default_factory=list)
    text: Optional[str] = None      # EDU text (leaves only)

    @property
    def is_leaf(self) -> bool:
        return not self.children

    # -- labels ------------------------------------------------------------
    def label(self) -> str:
        """Label of an internal binary node, e.g. 'NS-Elaboration', 'NN-Joint'."""
        if self.is_leaf:
            return "EDU"
        if len(self.children) != 2:
            nuc = "".join(c.nuclearity[0] for c in self.children)
            rels = {c.relation for c in self.children if c.relation != "span"}
            return f"{nuc}-{'/'.join(sorted(rels)) or 'span'}"
        left, right = self.children
        nuc = left.nuclearity[0] + right.nuclearity[0]
        if nuc == "NS":
            rel = right.relation
        elif nuc == "SN":
            rel = left.relation
        else:  # NN: both children carry the multinuclear relation
            rel = left.relation if left.relation != "span" else right.relation
        return f"{nuc}-{rel}"

    # -- conversions -------------------------------------------------------
    def to_tuple(self):
        """Nested tuples: (label, child, child) for internal nodes, (edu_id, text) for EDUs."""
        if self.is_leaf:
            return (self.span[0], self.text)
        return (self.label(), *[c.to_tuple() for c in self.children])

    def to_nltk(self, leaf_text: bool = True):
        """Convert to nltk.Tree. Leaves are EDU texts (or EDU ids if leaf_text=False)."""
        from nltk import Tree
        if self.is_leaf:
            return self.text if leaf_text else str(self.span[0])
        return Tree(self.label(), [c.to_nltk(leaf_text) for c in self.children])

    def to_dict(self) -> dict:
        d = {"nuclearity": self.nuclearity, "relation": self.relation, "span": list(self.span)}
        if self.is_leaf:
            d["text"] = self.text
        else:
            d["label"] = self.label()
            d["children"] = [c.to_dict() for c in self.children]
        return d

    def edus(self) -> list:
        return [self.text] if self.is_leaf else [e for c in self.children for e in c.edus()]

    def __str__(self) -> str:
        if self.is_leaf:
            return f"[{self.span[0]}]"
        return f"({self.label()} {' '.join(str(c) for c in self.children)})"


# ---------------------------------------------------------------------------
# Reading .dis files
# ---------------------------------------------------------------------------

# EDU text (_!..._!) is matched first because it may contain parentheses.
_TOKEN_RE = re.compile(r"_!.*?_!|\(|\)|[^\s()]+", re.S)


def _tokens_to_lists(tokens: list):
    stack = [[]]
    for tok in tokens:
        if tok == "(":
            stack.append([])
        elif tok == ")":
            done = stack.pop()
            stack[-1].append(done)
        else:
            stack[-1].append(tok)
    if len(stack) != 1 or len(stack[0]) != 1:
        raise ValueError("Unbalanced parentheses in .dis file")
    return stack[0][0]


def _build(sexp: list) -> RSTNode:
    node = RSTNode(nuclearity=sexp[0])
    for item in sexp[1:]:
        head = item[0]
        if head == "span":
            node.span = (int(item[1]), int(item[2]))
        elif head == "leaf":
            node.span = (int(item[1]), int(item[1]))
        elif head == "rel2par":
            node.relation = item[1]
        elif head == "text":
            node.text = " ".join(item[1:])[2:-2].strip()
        elif head in ("Nucleus", "Satellite"):
            node.children.append(_build(item))
    return node


def read_dis(path: str) -> Optional[RSTNode]:
    """Parse one .dis file into an RSTNode tree (original n-ary, fine-grained)."""
    with open(path, encoding="utf-8", errors="replace") as f:
        text = f.read()
        if "TT_ERR" in text:
            return None
    return _build(_tokens_to_lists(_TOKEN_RE.findall(text)))


# ---------------------------------------------------------------------------
# Right binarization
# ---------------------------------------------------------------------------

def _right_binarize_children(kids: list) -> list:
    if len(kids) <= 2:
        return kids
    first, rest = kids[0], kids[1:]

    # One nucleus followed only by satellites (N S S ...): a right-branching
    # group (S S) would have no nucleus, so attach the nucleus to its first
    # satellite instead, keeping every internal node well-formed.
    if first.nuclearity == "Nucleus" and all(k.nuclearity == "Satellite" for k in rest):
        group = RSTNode("Nucleus", "span", (first.span[0], rest[0].span[1]), [first, rest[0]])
        return _right_binarize_children([group] + rest[1:])

    # Multinuclear (all N): the new node keeps the shared relation (e.g. List).
    # Otherwise the new node contains the nucleus, so it relates as 'span'.
    if all(k.nuclearity == "Nucleus" for k in kids):
        rel = rest[0].relation
    else:
        rel = "span"
    group = RSTNode("Nucleus", rel, (rest[0].span[0], rest[-1].span[1]),
                    _right_binarize_children(rest))
    return [first, group]


def right_binarize(node: RSTNode) -> RSTNode:
    """Right-binarize the tree in place: (A B C D) -> (A (B (C D)))."""
    for c in node.children:
        right_binarize(c)
    node.children = _right_binarize_children(node.children)
    return node


def coarsen(node: RSTNode) -> RSTNode:
    """Replace every rel2par with its coarse class, in place."""
    node.relation = to_coarse(node.relation)
    for c in node.children:
        coarsen(c)
    return node


# ---------------------------------------------------------------------------
# Corpus loading
# ---------------------------------------------------------------------------

def load_tree(path: str, binarize: bool = True, coarse: bool = True) -> Optional[RSTNode]:
    print(path)
    tree = read_dis(path)
    if tree is None:
        return None
    if binarize:
        right_binarize(tree)
    if coarse:
        coarsen(tree)
    return tree


def load_corpus(directory: str, binarize: bool = True, coarse: bool = True) -> dict:
    """Load all .dis files in a directory (e.g. RST-DT TRAINING or TEST)."""
    paths = sorted(glob.glob(os.path.join(directory, "*.dis")))
    if not paths:
        raise FileNotFoundError(f"No .dis files found in {directory}")
    trees = {}
    for p in paths:
        doc_id = os.path.basename(p).replace(".out.dis", "").replace(".dis", "")
        maybe_tree = load_tree(p, binarize, coarse)
        if maybe_tree is not None:
            trees[doc_id] = cast(RSTNode, maybe_tree)
    return trees


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", help="a .dis file or a directory of .dis files")
    ap.add_argument("--out", help="write trees as JSON lines to this file")
    ap.add_argument("--no-binarize", action="store_true")
    ap.add_argument("--fine", action="store_true", help="keep fine-grained relations")
    args = ap.parse_args()

    kwargs = dict(binarize=not args.no_binarize, coarse=not args.fine)
    if os.path.isdir(args.path):
        trees = load_corpus(args.path, **kwargs)
    else:
        trees = {os.path.basename(args.path): load_tree(args.path, **kwargs)}

    n_edus = sum(len(t.edus()) for t in trees.values())
    print(f"Loaded {len(trees)} documents, {n_edus} EDUs")
    first_id, first = next(iter(trees.items()))
    print(f"\n{first_id}:\n{first}")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            for doc_id, t in trees.items():
                f.write(json.dumps({"doc_id": doc_id, "tree": t.to_dict()}) + "\n")
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()