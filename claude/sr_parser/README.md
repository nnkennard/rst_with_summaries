# SpanBERT shift-reduce RST parser

Reimplementation of the **SpanBERT-NoCoref** baseline from Guz & Carenini (2020),
*Coreference for Discourse Parsing: A Neural Approach* (CODI 2020), trained on
RST-DT. The package is self-contained: it reads only `rstdt/`, which
`extract_rstdt.py` creates from `rst_discourse_treebank/`.

```bash
# from the repo root
python3 claude/sr_parser/extract_rstdt.py                       # creates rstdt/
python3 -m claude.sr_parser.train --out runs/cv --seed 0        # 10-fold CV
python3 -m claude.sr_parser.train --out runs/seed0 --folds 1    # single split
python3 -m claude.sr_parser.train --eval-only runs/cv/fold0/best.pt  # re-evaluate
```

Outputs in `--out`: `args.json` and `cv_scores.json` (mean, std and per-fold
scores). Each `--out/fold<i>/` holds `log.json` (per-epoch dev scores), `best.pt`
(best dev checkpoint, ~440 MB, so ~4.4 GB per run), `test_scores.json`,
`test_pred.txt` (bracketed predicted trees) and `split.json` (train/dev doc ids).
With `--folds 1`, the fold files go directly in `--out` and there is no
`cv_scores.json`.

## What follows the paper

| Component | Setting |
|---|---|
| Transition system | Shift; Reduce-{NN,NS,SN} × 18 coarse relations (§3) |
| Classifiers | Two MLPs (Wang et al. 2017 style): action+nuclearity (4-way), relation (18-way, reduce steps only) (§3.1) |
| Classifier input | v(S2), v(S1), v(Q1) + organizational features (§3.1) |
| Node representation | mean of first and last token embedding of the node's **nuclear EDU** (§4) |
| Encoder | `SpanBERT/spanbert-base-cased`, fine-tuned, whole document in non-overlapping windows (§4) |
| Org. features | S2–S1 and S1–Q1 same sentence / same paragraph; S2, S1, Q1 at start/end of sentence / paragraph / document; each a learnable 10-d embedding, zeros if missing (§3.1, App. B) |
| MLPs | Linear → GELU → Dropout(0.3) → Linear (App. B) |
| Training | Cross-entropy per stack-queue step (static oracle); AdamW, lr 1e-5 SpanBERT / 2e-4 rest, batch 5 docs, 20 epochs, linear decay (App. B) |
| Decoding | Greedy (§3.2) |
| Dev split | 10% of TRAINING, stratified by #EDUs (App. B), with `--folds 1` |
| Metric | RST-Parseval micro (Marcu 2000), as in Table 1 |

## Choices the paper does not specify

- **Cross-validation** (the default): TRAINING is cut into 10 folds, stratified
  by #EDUs. Each fold is the dev set of one model, trained on the other nine,
  and each model is scored on the same TEST set. The paper instead averages 5
  runs on one split. All folds use the same `--seed`.
- **Window length**: 384 subword tokens (Joshi et al.'s best setting for base
  models). Windows are cut at sentence ends when possible, otherwise at EDU ends.
- **MLP hidden size**: 512. **Weight decay**: 0.01 (not on bias/LayerNorm).
  **Gradient clipping**: 1.0. **Warmup**: none.
- **Model selection**: the epoch with the best dev RST-Parseval *Full* score.
- **Missing stack/queue elements**: zero vectors.
- **Nuclear EDU of a multinuclear span**: the leftmost nucleus.
- **Binarization**: right-branching. In the N S S… case, the nucleus is
  attached to the first satellite (left-branching) so that every internal node
  has a nucleus.
- **Sentence boundaries**: heuristic. An EDU ends a sentence if it ends with
  `.?!` (optionally followed by closing quotes or brackets) or ends a paragraph.
  **Paragraphs** come from the `<P>` markers in the `.dis` EDU text. For
  `file1`–`file5`, which have no markers, they come from indented lines in the
  raw text. The other WSJ files without markers are single-paragraph articles.
- **`//TT_ERR`** markers (wsj_2353, wsj_2367) are stripped; both trees are
  otherwise well-formed and are kept.
- Relations are mapped to the standard 18 coarse classes, after stripping the
  `-e` and `-n`/`-s` suffixes.

## Metrics

`evaluate.py` reports two metrics:

- **RST-Parseval**: every node except the root, leaves included, labeled with
  its own nuclearity and relation. This is what the paper's Table 1 uses
  (baseline: Span 87.8, Nuclearity 75.8, Relation 63.4, averaged over 5 runs).
- **Original Parseval** (Morey et al. 2017): internal non-root nodes, labeled
  with the children's nuclearity and the relation. This metric is stricter and
  its numbers are much lower.

For binary trees, precision = recall = F1.

## Notes

- With transformers ≥ 5, `from_pretrained` keeps the checkpoint's dtype, and the
  SpanBERT checkpoint is fp16. In fp16, AdamW's second-moment state underflows
  and the weights become NaN after one step. The model is therefore loaded with
  `dtype=torch.float32`.
- The hub tokenizer config lowercases by default, so the tokenizer is loaded
  with `do_lower_case=False`.
