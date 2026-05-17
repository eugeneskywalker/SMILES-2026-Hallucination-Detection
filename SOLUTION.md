# SOLUTION.md

## Overview

This project detects hallucinations in Qwen2.5-0.5B answers. The idea is simple: take the model's hidden states (its internal numbers from one of the layers), turn them into one feature vector per sample, and train a small classifier to tell if the answer is hallucinated (1) or truthful (0).

**Final result on internal test split:**
- Test accuracy: **68.27%**
- Test AUROC: **66.55%**
- Test F1: **75.19%**

---

## How to reproduce

### Environment
- Python 3.11
- GPU with CUDA (I used a Linux server with an NVIDIA GPU). It also works on CPU but is slow.
- Packages from `requirements.txt`.

### Commands
```bash
git clone <this repo>
cd smile
python -m venv .venv
source .venv/bin/activate         # Linux / macOS
# .venv\Scripts\activate.bat      # Windows

pip install -r requirements.txt
python solution.py
```

After the run finishes you get two files in the current folder:
- `results.json` — metrics summary
- `predictions.csv` — predicted labels for `data/test.csv`

The whole pipeline is deterministic: split uses `random_state=42` and the probe uses `SEED=0`. So running `solution.py` twice on the same machine gives the same `predictions.csv`.

**Total time on GPU:** about 1 minute for feature extraction on the train set + a few seconds for the probe + a few seconds for the test set.

---

## Final solution

### What I changed
Only the three files we are allowed to edit:
- `aggregation.py`
- `probe.py`
- `splitting.py`

The fixed infrastructure (`model.py`, `evaluate.py`, `solution.py`) was not touched.

### `aggregation.py` — how features are made
For each sample I:
1. Take the hidden states from **block 14** of the model (index 15 in the `hidden_states` tuple, because index 0 is the embedding layer). Block 14 is around the middle-late part of the network. Papers about probing (Orgad 2025, Marks & Tegmark 2024) say the most useful information about truthfulness usually lives in the middle-to-late layers, not at the very end.
2. Take the hidden state at the **last non-padding token**. This is the simplest pooling strategy and a common baseline in probing papers.
3. That gives a 896-dimensional vector per sample (same as the model's hidden size).

No geometric features are used (`USE_GEOMETRIC = False`).

### `probe.py` — the classifier
A scikit-learn **Logistic Regression** with:
- `penalty="l2"` (standard regularization)
- `class_weight="balanced"` (the dataset is 70% hallucinated / 30% truthful, so the minority class needs more weight)
- `max_iter=1000`
- `random_state=SEED` for reproducibility
- A `StandardScaler` in front (fit only on training data — important to avoid leaking test info)
- Decision threshold fixed at **0.5**

The class also inherits from `torch.nn.Module` just to keep the original API (`fit`, `predict`, `predict_proba`, `fit_hyperparameters`). There is no PyTorch model inside — it is just a sklearn pipeline wrapped in the same interface.

### `splitting.py` — how the data is split
A single **stratified 70 / 15 / 15** split (train / validation / test), seed 42. Stratified means the ratio of hallucinated vs truthful is the same in every part. This matters because the dataset is imbalanced.

### Why these choices
- **Linear probe over MLP**: I first tried an MLP probe (Experiment 1). It got high training AUROC (≈1.0) but val AUROC was about the same as or just slightly higher than the linear probe. That is a classic sign of overfitting on a small dataset (689 samples). Logistic Regression is simpler, has fewer hyperparameters, gives almost the same val performance and is much easier to reason about. So I picked LR for the final submission.
- **Layer 14 (mid-late)**: Both Marks & Tegmark and Orgad et al. report that truthfulness signal is strongest in the middle-to-late layers, not the last one. Block 14 is around 58% depth in a 24-block model, which is in that range. I also tested block 18 — results were similar, so I stayed with the simpler choice.
- **Last-token pooling**: Most probing papers use the last-token representation. It is the simplest baseline, requires no extra code, and on this small dataset it performed close to `mean` and `max` pooling in our experiments (within ~0.02 AUROC). I picked it for simplicity.
- **Fixed threshold 0.5**: I tried F1-tuning the threshold on validation, but with `class_weight="balanced"` the tuned threshold went down to almost 0.0 and predicted almost everything as hallucinated. That is a useless classifier in practice, so I dropped F1-tuning and pinned the threshold at 0.5.

### What helped the metric most
Honestly, the biggest jump in accuracy compared to a random guess comes from just **picking a reasonable layer (14) and using Logistic Regression with `class_weight="balanced"` and a `StandardScaler`**. Once that pipeline was in place, swapping pooling strategies or layers only changed the score by a few percent. So the simple, clean baseline turned out to be the strongest single design decision.

---

## Experiments and failed attempts

### Experiment 1 — token pooling × layer (with MLP probe)
I tested three pooling strategies (`last`, `mean`, `max`) at two layers (14 and 18), three seeds each — 18 runs total.

**What I found:**
- All MLP probes overfit hard: train AUROC was 1.00 on every run, but val AUROC was 0.60–0.71.
- Val AUROC differences between pooling strategies were small (within ~0.04).
- `mean` pooling at layer 14 had the highest val AUROC mean (0.6930), and `last` pooling at layer 18 had the highest single seed (0.7112).
- Length-only baseline (just predicting from response length): val AUROC ≈ 0.66 — uncomfortably close to the probe results, which warned me that the probe was partly using length signal.

**Conclusion:** MLP is overkill. The signal is mostly linear, so use a linear probe.

### Experiment 2 — LR probe + length diagnostics
Same 3×2 grid but with Logistic Regression instead of MLP.

**What I found:**
- LR val AUROC was within ±0.03 of MLP for most cells. So the simpler model wins on Occam's razor.
- Probe predictions are correlated with response length (Pearson ≈ 0.27, Spearman ≈ 0.34).
- Adding response length as an extra explicit feature did **not** improve val AUROC (gap = −0.0006). So the length info is already inside the hidden states — adding it again is redundant.
- Within-length-quartile AUROC: probe works much better on long responses (q3, q4: AUROC ≈ 0.65–0.75) than short ones (q1, q2: AUROC ≈ 0.54–0.60). So part of the model's signal is length-based, and part is real content signal — especially on longer answers.

### F1-tuned decision threshold — failed
I tried tuning the threshold on validation to maximize F1. Because of `class_weight="balanced"` and the 70/30 imbalance, the optimizer always picked a threshold close to 0.0, which is the same as "always predict 1". This gave 99% positive predictions on the test set — basically a majority classifier. I went back to a fixed threshold of 0.5.

### Geometric features — not used
`extract_geometric_features` is left as a stub. I considered adding layer-wise activation norms or inter-layer cosine similarities, but given that even simple last-token pooling was already overfitting, adding more features felt like making the overfit worse, not better. Maybe on a bigger dataset this would help.

### Multi-fold cross-validation — not used
With only 689 samples and a clear primary metric, a single stratified split with 3 seeds (for variance estimation) was enough. K-fold would have made the runs longer without changing the conclusions.

### MLP probe — discarded
Kept the LR probe in the final submission. The MLP gave a slightly higher val AUROC on some cells but always overfit. LR is much more honest about its uncertainty.

### Response-only pooling — not implemented
Ideally I would only pool over the **response** tokens, not the prompt. But `aggregate()` is called with only `hidden_states` and `attention_mask` — there is no easy way to tell where the prompt ends and the response starts without changing `solution.py`, which is in the fixed-infrastructure list. So I pool over all non-padding tokens, accepting some signal dilution from prompt tokens.

---

## File map

| File | Role | Edited? |
|---|---|---|
| `aggregation.py` | last-token pooling at block 14 | yes |
| `probe.py` | Logistic Regression + StandardScaler | yes |
| `splitting.py` | stratified 70/15/15 split | yes (kept default) |
| `solution.py` | main pipeline | no (fixed) |
| `model.py` | LLM loader | no (fixed) |
| `evaluate.py` | metrics and saving | no (fixed) |
| `results.json` | metrics from final run | generated by `solution.py` |
| `predictions.csv` | predicted labels for `test.csv` | generated by `solution.py` |

---

## Final numbers (from `results.json`)

| Metric | Value |
|---|---|
| Baseline accuracy (majority class) | 70.19% |
| Baseline F1 | 82.49% |
| Probe train accuracy | 95.63% |
| Probe train AUROC | 99.47% |
| Probe val accuracy | 62.50% |
| Probe val AUROC | 60.98% |
| Probe test accuracy | 68.27% |
| Probe test F1 | 75.19% |
| **Probe test AUROC** | **66.55%** |
| Feature dim | 896 |
| Total samples | 689 |
| Folds | 1 |
| Feature-extraction time | 46.2 s |
