import argparse
import collections
import glob

from dataclasses import dataclass, field
from typing import Optional, cast

import rst_dt_lib

TRAIN, TEST = "TRAINING TEST".split()

parser = argparse.ArgumentParser(description='TODO(this)')
parser.add_argument('-r', '--rst_dt_dir', default='rst_discourse_treebank/', type=str, help='')
#parser.add_argument('--mode', choices=('train', 'eval'), help='')
#parser.add_argument('-d', '--debug', action='store_true', help='')
#parser.add_argument('-s', '--year_start', default=2008, type=int, help='')

@dataclass
class RSTNode:

    nuclearity: str # TODO: what is this Java-looking stuff?
    relation: Optional[str] = None
    span: tuple = (0, 0)
    children: list = field(default_factory=list)
    text: Optional[str] = None # EDU text (leaves only)
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
            print(node.relation)
        elif head == "text":
            node.text = " ".join(item[1:])[2:-2].strip()
        elif head in ("Nucleus", "Satellite"):
            node.children.append(build_node(item))
    return node

def _right_binarize_children(children:list) -> list:
    return []

def right_binarize(node: RSTNode) -> RSTNode:
    for c in node.children:
        right_binarize(c)
    node.children = _right_binarize_children(node.children)
    return node

    

def read_tree(filename:str) -> Optional[RSTNode]:
    tree_list = rst_dt_lib.listify_dis(filename)
    if tree_list is not None:
        return build_node(tree_list)


def main():

    args = parser.parse_args()

    subsets = collections.defaultdict(list)

    for subset in [TRAIN, TEST]:
        for filename in glob.glob(f'{args.rst_dt_dir}/data/RSTtrees-WSJ-main-1.0/{subset}/*.dis'):
            maybe_tree = read_tree(filename)
            # print(maybe_tree)
            if maybe_tree is not None:
                subsets[subset].append(
                    #cast(RSTNode, 
                         maybe_tree
                     #    )
                         )

    print(subsets)


if __name__ == "__main__":
    main()