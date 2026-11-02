"""Train / evaluate the SpanBERT shift-reduce RST parser.

By default, runs 10-fold cross-validation over the training split: each fold
holds out 10% of the docs as dev set for model selection, and every fold's best
checkpoint is scored on the same test set.

    python -m claude.sr_parser.train --out runs/cv --seed 0
    python -m claude.sr_parser.train --out runs/seed0 --seed 0 --folds 1
    python -m claude.sr_parser.train --eval-only runs/cv/fold0/best.pt
    python -m claude.sr_parser.train --out runs/debug --train-split DEBUG
"""

import argparse
import gc
import json
import os
import random
import statistics
import time

import numpy as np
import torch
from transformers import get_linear_schedule_with_warmup

from . import data as D
from .evaluate import evaluate, format_scores
from .model import ShiftReduceParser


def get_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rst-dt", default="rstdt")
    ap.add_argument("--train-split", default="TRAINING",
                    choices=["TRAINING", "DEBUG"],
                    help="split to draw train and dev docs from; DEBUG is the "
                         "first 10 TRAINING docs (see extract_rstdt.py) and is "
                         "also used as the test set")
    ap.add_argument("--out", default="runs/sr_parser")
    ap.add_argument("--bert", default="SpanBERT/spanbert-base-cased")
    ap.add_argument("--max-seg-len", type=int, default=384,
                    help="SpanBERT window length (Joshi et al. 2020, base)")
    ap.add_argument("--hidden", type=int, default=512,
                    help="MLP hidden size (not given in the paper)")
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=5)
    ap.add_argument("--bert-lr", type=float, default=1e-5)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--warmup", type=float, default=0.0,
                    help="fraction of steps for linear warmup")
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--folds", type=int, default=10,
                    help="cross-validation folds; 1 trains once on a single "
                         "train/dev split")
    ap.add_argument("--device", default=None)
    ap.add_argument("--limit", type=int, default=0,
                    help="debug: use only this many train/dev/test docs")
    ap.add_argument("--eval-only", default=None, help="checkpoint to evaluate")
    args = ap.parse_args()
    if args.folds < 1:
        ap.error("--folds must be at least 1")
    return args


def pick_device(name):
    if name:
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def batches(items, size, shuffle, rng):
    idx = list(range(len(items)))
    if shuffle:
        rng.shuffle(idx)
    for i in range(0, len(idx), size):
        yield [items[j] for j in idx[i:i + size]]


def run_eval(model, docs, encs, batch_size=4):
    model.eval()
    preds = []
    for i in range(0, len(docs), batch_size):
        preds += model.parse(docs[i:i + batch_size], encs[i:i + batch_size])
    return evaluate([d.tree for d in docs], preds), preds


def empty_device_cache(device):
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def eval_checkpoint(args, device, train_all, test):
    """Dev docs come from the split.json saved next to the checkpoint, since
    the checkpoint alone does not say which fold it was trained on."""
    split_path = os.path.join(os.path.dirname(args.eval_only), "split.json")
    with open(split_path) as f:
        dev_ids = set(json.load(f)["dev"])
    dev = [d for d in train_all if d.doc_id in dev_ids]
    model = ShiftReduceParser(args.bert, args.hidden, args.dropout,
                              args.max_seg_len).to(device)
    model.load_state_dict(torch.load(args.eval_only, map_location=device))
    for name, ds in (("dev", dev), ("test", test)):
        scores, _ = run_eval(model, ds, model.encode_docs(ds))
        print(f"[{name}]\n{format_scores(scores)}")


def train_fold(args, device, train, dev, test, out):
    """Train on `train`, select the epoch on `dev`, score it on `test`, and
    write the results to `out`. Returns the (dev, test) scores."""
    set_seed(args.seed)
    print(f"docs: train {len(train)}  dev {len(dev)}  test {len(test)}  "
          f"device {device}", flush=True)

    model = ShiftReduceParser(args.bert, args.hidden, args.dropout,
                              args.max_seg_len).to(device)
    enc = {id(d): e for ds in (train, dev, test)
           for d, e in zip(ds, model.encode_docs(ds))}
    E = lambda ds: [enc[id(d)] for d in ds]

    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "split.json"), "w") as f:
        json.dump({"train": [d.doc_id for d in train],
                   "dev": [d.doc_id for d in dev]}, f, indent=2)

    bert_params = list(model.bert.named_parameters())
    head_params = [(n, p) for n, p in model.named_parameters()
                   if not n.startswith("bert.")]
    no_decay = ("bias", "LayerNorm.weight")

    def groups(params, lr):
        return [
            {"params": [p for n, p in params if not n.endswith(no_decay)],
             "lr": lr, "weight_decay": args.weight_decay},
            {"params": [p for n, p in params if n.endswith(no_decay)],
             "lr": lr, "weight_decay": 0.0},
        ]

    opt = torch.optim.AdamW(groups(bert_params, args.bert_lr) +
                            groups(head_params, args.lr))
    n_steps = args.epochs * ((len(train) + args.batch_size - 1) // args.batch_size)
    sched = get_linear_schedule_with_warmup(opt, int(args.warmup * n_steps),
                                            n_steps)

    rng = random.Random(args.seed)
    best, best_scores, log = -1.0, None, []
    ckpt = os.path.join(out, "best.pt")
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, tot, tot_a, tot_r, nb = time.time(), 0.0, 0.0, 0.0, 0
        for batch in batches(train, args.batch_size, True, rng):
            loss, la, lr_ = model.loss(batch, E(batch))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            opt.step()
            sched.step()
            tot, tot_a, tot_r, nb = tot + loss.item(), tot_a + la, tot_r + lr_, nb + 1
        scores, _ = run_eval(model, dev, E(dev))
        full = scores["rst_parseval"]["full"]
        entry = {"epoch": epoch, "loss": tot / nb, "loss_act": tot_a / nb,
                 "loss_rel": tot_r / nb, "dev": scores,
                 "time": time.time() - t0}
        log.append(entry)
        print(f"epoch {epoch:2d}  loss {tot / nb:.4f} (act {tot_a / nb:.4f} "
              f"rel {tot_r / nb:.4f})  {entry['time']:.0f}s\n"
              f"[dev]\n{format_scores(scores)}", flush=True)
        if full > best:
            best, best_scores = full, scores
            torch.save(model.state_dict(), ckpt)
            print(f"  new best dev RST-Parseval full {full:.2f}", flush=True)
        with open(os.path.join(out, "log.json"), "w") as f:
            json.dump(log, f, indent=2)

    model.load_state_dict(torch.load(ckpt, map_location=device))
    scores, preds = run_eval(model, test, E(test))
    print(f"[test, best dev checkpoint]\n{format_scores(scores)}", flush=True)
    with open(os.path.join(out, "test_scores.json"), "w") as f:
        json.dump(scores, f, indent=2)
    with open(os.path.join(out, "test_pred.txt"), "w") as f:
        for d, t in zip(test, preds):
            f.write(f"{d.doc_id}\t{t}\n")
    return best_scores, scores


def summarize(fold_scores):
    """Mean and sample standard deviation over folds of each metric."""
    out = {}
    for metric, names in fold_scores[0].items():
        out[metric] = {}
        for name in names:
            vals = [s[metric][name] for s in fold_scores]
            out[metric][name] = {"mean": statistics.mean(vals),
                                 "std": statistics.stdev(vals)}
    return out


def format_summary(summary):
    return "\n".join(
        f"{metric:>13}: " + "  ".join(
            f"{name[:4].capitalize()} {v['mean']:5.1f} \u00b1 {v['std']:.1f}"
            for name, v in s.items())
        for metric, s in summary.items())


def main():
    args = get_args()
    device = pick_device(args.device)

    train_all = D.load_split(args.rst_dt, args.train_split)
    test_split = "DEBUG" if args.train_split == "DEBUG" else "TEST"
    test = D.load_split(args.rst_dt, test_split)

    if args.eval_only:
        eval_checkpoint(args, device, train_all, test)
        return

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    if args.folds == 1:
        folds = [D.train_dev_split(train_all, seed=args.split_seed)]
    else:
        folds = D.cv_folds(train_all, args.folds, args.split_seed)

    dev_scores, test_scores = [], []
    for i, (train, dev) in enumerate(folds):
        fold_test = test
        if args.limit:
            train, dev, fold_test = (train[:args.limit], dev[:args.limit],
                                     test[:args.limit])
        out = args.out
        if args.folds > 1:
            out = os.path.join(args.out, f"fold{i}")
            print(f"=== fold {i + 1}/{args.folds} ===", flush=True)
        d, t = train_fold(args, device, train, dev, fold_test, out)
        dev_scores.append(d)
        test_scores.append(t)
        empty_device_cache(device)

    if args.folds > 1:
        summary = {"dev": summarize(dev_scores), "test": summarize(test_scores)}
        with open(os.path.join(args.out, "cv_scores.json"), "w") as f:
            json.dump({**summary, "folds": {"dev": dev_scores,
                                            "test": test_scores}}, f, indent=2)
        for name, sm in summary.items():
            print(f"[{name}, mean \u00b1 std over {args.folds} folds]\n"
                  f"{format_summary(sm)}", flush=True)


if __name__ == "__main__":
    main()
