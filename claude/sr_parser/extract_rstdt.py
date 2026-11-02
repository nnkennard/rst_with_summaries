"""Copy the RST-DT files the shift-reduce parser needs into a smaller directory.

Creates TRAINING/ and TEST/, plus DEBUG/ with the files for the first
``DEBUG_DOCS`` documents of TRAINING, sorted by filename.

The parser (``data.py``) needs:

- every ``.dis`` tree;
- the raw text of ``file1``-``file5``, whose paragraph boundaries come from
  indented lines because their ``.dis`` files have no ``<P>`` markers.

The ``.edus``, ``.rst`` and ``.out`` files are not needed. ``load_doc`` does
look for the ``.out`` of WSJ documents without ``<P>`` markers, but it finds no
paragraph breaks in any of them, so a missing file gives the same result.

    # from the repo root
    python3 claude/sr_parser/extract_rstdt.py
"""

import argparse
import glob
import os
import shutil

SPLITS = ("TRAINING", "TEST")
RAW_TEXT_DOCS = tuple(f"file{i}" for i in range(1, 6))
DEBUG_SPLIT = "DEBUG"
DEBUG_SOURCE = "TRAINING"
DEBUG_DOCS = 10


def find_docs(split_dir: str) -> list:
    """Sorted .dis paths, in the order ``load_split`` reads them."""
    dis_paths = sorted(glob.glob(os.path.join(split_dir, "*.dis")))
    if not dis_paths:
        raise FileNotFoundError(f"no .dis files in {split_dir}")
    return dis_paths


def files_for_docs(dis_paths: list) -> list:
    raw_paths = [p[:-len(".dis")] for p in dis_paths
                 if os.path.basename(p)[:-len(".dis")] in RAW_TEXT_DOCS]
    for path in raw_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(path)
    return sorted(dis_paths + raw_paths)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--src",
                    default="rst_discourse_treebank/data/RSTtrees-WSJ-main-1.0",
                    help="directory containing TRAINING/ and TEST/")
    ap.add_argument("--dst", default="rstdt", help="output directory")
    args = ap.parse_args()

    # Resolve every split before copying, so a bad --src copies nothing.
    docs = {s: find_docs(os.path.join(args.src, s)) for s in SPLITS}
    docs[DEBUG_SPLIT] = docs[DEBUG_SOURCE][:DEBUG_DOCS]
    files = {s: files_for_docs(paths) for s, paths in docs.items()}
    for split, paths in files.items():
        out_dir = os.path.join(args.dst, split)
        os.makedirs(out_dir, exist_ok=True)
        for path in paths:
            shutil.copy2(path, out_dir)
        print(f"{split}: copied {len(paths)} files to {out_dir}")


if __name__ == "__main__":
    main()
