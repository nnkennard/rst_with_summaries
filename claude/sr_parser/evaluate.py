"""RST-Parseval (Marcu, 2000) and original Parseval (Morey et al., 2017).

Both are micro-averaged over the corpus. Since predicted and gold binary trees
over n EDUs always have the same number of constituents, precision = recall =
F1, so a single number is reported per metric.
"""

from .data import SPAN, Node


def _child_labels(node: Node):
    """(nuc, rel) of the left and right child, derived from the parent so that
    gold and predicted trees are labeled identically."""
    nuc, rel = node.nuc_label(), node.relation()
    if nuc == "NS":
        return ("N", SPAN), ("S", rel)
    if nuc == "SN":
        return ("S", rel), ("N", SPAN)
    return ("N", rel), ("N", rel)


def rst_parseval_brackets(tree: Node) -> list:
    """Every node except the root, leaves included, labeled with its own
    nuclearity and relation to its parent."""
    out = []

    def visit(n):
        if n.is_leaf:
            return
        for c, (nuc, rel) in zip(n.children, _child_labels(n)):
            out.append((c.start, c.end, nuc, rel))
            visit(c)

    visit(tree)
    return out


def parseval_brackets(tree: Node) -> list:
    """Internal nodes except the root, labeled with the nuclearity of their
    children (NN/NS/SN) and the relation between them."""
    out = []

    def visit(n, root):
        if n.is_leaf:
            return
        if not root:
            out.append((n.start, n.end, n.nuc_label(), n.relation()))
        for c in n.children:
            visit(c, False)

    visit(tree, True)
    return out


def _score(golds: list, preds: list) -> dict:
    keys = {"span": lambda b: b[:2], "nuclearity": lambda b: b[:3],
            "relation": lambda b: (b[0], b[1], b[3]), "full": lambda b: b}
    res = {}
    for name, key in keys.items():
        match = total_g = total_p = 0
        for g, p in zip(golds, preds):
            gs = [key(b) for b in g]
            ps = set(key(b) for b in p)
            match += sum(1 for b in set(gs) if b in ps)
            total_g += len(g)
            total_p += len(p)
        prec = match / max(total_p, 1)
        rec = match / max(total_g, 1)
        res[name] = 100 * (2 * prec * rec / (prec + rec) if match else 0.0)
    return res


def evaluate(gold_trees: list, pred_trees: list) -> dict:
    return {
        "rst_parseval": _score([rst_parseval_brackets(t) for t in gold_trees],
                               [rst_parseval_brackets(t) for t in pred_trees]),
        "parseval": _score([parseval_brackets(t) for t in gold_trees],
                           [parseval_brackets(t) for t in pred_trees]),
    }


def format_scores(scores: dict) -> str:
    lines = []
    for metric, s in scores.items():
        lines.append(f"{metric:>13}: " + "  ".join(
            f"{k[:4].capitalize()} {v:5.1f}" for k, v in s.items()))
    return "\n".join(lines)
