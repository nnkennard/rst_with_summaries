import argparse
import collections
import glob

from dataclasses import dataclass, field
from typing import Optional, cast

import rst_dt_lib

TRAIN, TEST = "TRAINING TEST".split()
NUCLEUS, SATELLITE, SPAN = "Nucleus Satellite span".split()

parser = argparse.ArgumentParser(description='TODO(this)')
parser.add_argument('-r',
                    '--rst_dt_dir',
                    default='rst_discourse_treebank/',
                    type=str,
                    help='')


@dataclass
class RSTNode:

    nuclearity: str  # TODO: what is this Java-looking stuff?
    relation: Optional[str] = None
    span: tuple = (0, 0)
    children: list = field(default_factory=list)
    text: Optional[str] = None  # EDU text (leaves only)
    # TODO text or tokens? Useful to tokenize?

    @property
    def is_leaf(self) -> bool:
        return not self.children


def build_node(lis: list) -> RSTNode:
    nuclearity = lis.pop(0)
    assert rst_dt_lib.is_nuclearity(nuclearity)
    node = RSTNode(nuclearity=nuclearity)

    for item in lis:
        head = item[0]
        # TODO: I don't like this but it's short and it works :-(
        if head == "span":
            node.span = (int(item[1]), int(item[2]))
        elif head == "leaf":
            node.span = (int(item[1]), int(item[1]))
        elif head == "rel2par":
            node.relation = rst_dt_lib.coarsen(item[1])
        elif head == "text":
            node.text = " ".join(item[1:])[2:-2].strip()
        elif head in ("Nucleus", "Satellite"):
            node.children.append(build_node(item))
    return node


def _right_binarize_children(children: list) -> list:
    if len(children) <= 2:
        return children

    # More than two children.
    # Keep first child as left child, all the children to the right grouped under a new node.
    first, rest = children[0], children[1:]

    # Claude mystery
    # TODO: Delete this later.
    if first.nuclearity == NUCLEUS and all(k.nuclearity == SATELLITE
                                           for k in rest):
        # This is something that Claude had a bunch of code for but never occurred in RST-DT? Code to deal with it is in claude/rstdt.py if it comes up.
        assert False

    if all(c.nuclearity == NUCLEUS for c in children):
        # All N children: keep relation label on new node
        rel = rest[0].relation
    else:
        # There is a satellite: the new node will not be a nucleus
        # TODO: Check this
        rel = SPAN

    group_node = RSTNode(NUCLEUS, rel, (rest[0].span[0], rest[-1].span[1]),
                         _right_binarize_children(rest))
    return [first, group_node]


def right_binarize(node: RSTNode) -> RSTNode:
    for c in node.children:
        right_binarize(c)
    node.children = _right_binarize_children(node.children)
    return node


def read_tree(filename: str) -> Optional[RSTNode]:
    tree_list = rst_dt_lib.listify_dis(filename)
    if tree_list is not None:
        return right_binarize(build_node(tree_list))


def main():

    args = parser.parse_args()

    subsets = collections.defaultdict(list)

    for subset in [TRAIN, TEST]:
        for filename in glob.glob(
                f'{args.rst_dt_dir}/data/RSTtrees-WSJ-main-1.0/{subset}/*.dis'
        ):
            maybe_tree = read_tree(filename)
            # print(maybe_tree)
            if maybe_tree is not None:
                subsets[subset].append(
                    #cast(RSTNode,
                    maybe_tree
                    #    )
                )

    for k, v in subsets.items():
        print(k, len(v))


if __name__ == "__main__":
    main()
