# SOLUTION.md

## Overview

This project detects hallucinations in Qwen2.5-0.5B answers. The pipeline is:

1. Run a forward pass on `prompt + response` and collect hidden states.
2. Pick one layer and pool over the non-padding tokens into a single vector.
3. Train a Logistic Regression probe on that vector.
4. Predict labels for `data/test.csv` and save them to `predictions.csv`.

**Final result on the internal test split (`results.json`):**
- Test accuracy: **68.27%**
- Test AUROC: **66.55%**
- Test F1: **75.19%**

---

## How to reproduce

### Environment used to generate the submitted `predictions.csv`
- **OS:** Linux server with NVIDIA GPU (CUDA)
- **Python:** 3.11 (conda env)
- **HuggingFace model revision (pinned):** `Qwen/Qwen2.5-0.5B` @ `060db6499f32faf8b98477b0a26969ef7d8b9987`
- **Package versions:** see `requirements.txt` (tested working with `transformers>=4.40`, `scikit-learn>=1.3`, `torch>=2.0`)

### Commands
```bash
git clone https://github.com/eugeneskywalker/SMILES-2026-Hallucination-Detection.git
cd SMILES-2026-Hallucination-Detection

python -m venv .venv
source .venv/bin/activate         # Linux / macOS
# .venv\Scripts\activate.bat      # Windows

pip install -r requirements.txt
python solution.py
```

After the run finishes you get two files:
- `results.json` — metrics summary
- `predictions.csv` — predicted labels for `data/test.csv`

### Verify the regenerated `predictions.csv` matches the submitted one
```bash
python -c "import hashlib; print(hashlib.sha256(open('predictions.csv','rb').read()).hexdigest())"
```
The expected SHA256 of the submitted file is:
```
f6a2eff470492f965ab36dbd1915c203ab740bc308992bd7860bce71d468f0e1
```

### Determinism notes
The pipeline uses fixed seeds (`random_state=42` in `splitting.py`, `SEED=0` in `probe.py`). With the **same package versions, same HF model revision, and same hardware (GPU model + numerical mode)** the regenerated `predictions.csv` should be bit-identical to the submitted file. If `transformers`, `torch`, or the model revision change, the hidden states change too and the predictions may differ in a handful of borderline rows; the AUROC will stay essentially the same.

**Run time on GPU:** ~46 s for feature extraction on the full train set, plus a few seconds for the probe and the test set.

---

## Final solution

### What I changed
Only the three files we are allowed to edit:
- `aggregation.py`
- `probe.py`
- `splitting.py` (kept default, see below)

The fixed infrastructure (`model.py`, `evaluate.py`, `solution.py`) was not modified — `git diff` against the organizer's base commit confirms zero changes to those three files.

### `aggregation.py` — features
For each sample I:
1. Take the hidden states from **block 14** (index 15 in the `hidden_states` tuple, since index 0 is the embedding layer). This is around the middle-late part of the network. Two probing papers I read (Marks & Tegmark 2024, Orgad et al. 2025) report that truthfulness signal is concentrated in middle-to-late layers, not at the very end.
2. Take the hidden state at the **last non-padding token**. This is the simplest pooling option and a common baseline in the probing literature.

That gives a 896-dimensional vector per sample (Qwen2.5-0.5B's hidden size). No geometric features (`USE_GEOMETRIC = False`).

### `probe.py` — classifier
A scikit-learn **Logistic Regression** with:
- `penalty="l2"` (standard regularization)
- `class_weight="balanced"` (the dataset is 70% hallucinated / 30% truthful)
- `max_iter=1000`
- `random_state=SEED`
- A `StandardScaler` fit on the training data only (no leakage into val/test)
- Decision threshold fixed at **0.5**

The class inherits from `torch.nn.Module` only to keep the original API (`fit`, `predict`, `predict_proba`, `fit_hyperparameters`). There is no PyTorch state inside — it is a sklearn pipeline behind the original interface.

### `splitting.py` — data split
A single **stratified 70 / 15 / 15** train / val / test split with `random_state=42`. Stratified means the 70/30 label ratio is preserved in each part. This matters because the dataset is imbalanced.

### Why these choices

- **Logistic Regression over an MLP probe.** I first prototyped an MLP probe. Its training AUROC was very high (≈1.0) while validation AUROC was at the same level as a linear probe — a textbook sign of overfitting on a small dataset (689 samples). Logistic Regression has far fewer hyperparameters, gives essentially the same val performance, and is easier to reason about. So I picked LR for the submission.
- **Layer 14 (middle-late).** Marks & Tegmark and Orgad et al. both place the strongest truthfulness signal in middle-to-late layers. Block 14 is around 58 % depth in a 24-block model, which fits that range.
- **Last-token pooling.** Standard baseline in probing papers, simplest possible code, and required no signature changes to `aggregate`. Mean and max pooling were also tested on the side and produced similar val AUROC (within a few percent).
- **Fixed threshold 0.5.** I tried F1-tuning the threshold on the validation set, but with `class_weight="balanced"` it kept driving the threshold close to 0.0, which collapses to "always predict hallucinated" — a useless classifier. So I dropped tuning and pinned 0.5.

### What contributed most to the metric
The single biggest contributor is the **basic pipeline itself**: layer 14 hidden state + `StandardScaler` + Logistic Regression with `class_weight="balanced"`. Once that is in place, changing pooling or layer only moves the AUROC by a few percent. The simple baseline turned out to be the strongest design decision.

---

## Experiments and failed attempts

### MLP probe — discarded
An MLP probe (256 hidden units, BCE with positive class weighting, Adam) was the original choice. On the same features it got train AUROC near 1.0 but val AUROC was within seed noise of the linear probe. Conclusion: the signal is essentially linear and an MLP is just overfitting. I replaced it with Logistic Regression and kept the same `HallucinationProbe` interface.

### Mean and max pooling — not used in the final submission
I compared `last`, `mean`, and `max` pooling at block 14. Val AUROC differences were within roughly 0.02–0.04, with no consistent winner across seeds I tried. Mean pooling is also diluted by prompt tokens (prompts are ~2× longer than responses), which weakens it for this dataset. I kept `last` for simplicity.

### Layer 18 — tested, not selected
I also tested block 18 (~75 % depth). Val AUROC was comparable to block 14 — within a couple of percent — so I kept the shallower, simpler choice that matches the literature heuristic more directly.

### F1-tuned decision threshold — failed
Tuning the threshold on validation under `class_weight="balanced"` pushed the threshold close to 0.0 in every run, producing ~99 % positive predictions on the test set. That is a majority classifier in disguise. I reverted to a fixed 0.5 threshold.

### Adding response length as an explicit feature — no gain
A logistic regression on just response length (one feature) got val AUROC ≈ 0.66 — close to the probe. So part of the probe signal is length-mediated. I then tried concatenating response length to the hidden-state features. Val AUROC barely moved (within seed noise), so the length info is already inside the hidden states and adding it again is redundant. I left length out of the final feature vector.

### Geometric / hand-crafted features — not implemented
`extract_geometric_features` is left as a stub returning a zero-length tensor. With the probe already overfitting on the 689-row training set, adding more features felt likely to make overfitting worse rather than help. On a larger dataset this would be the next thing to try.

### Response-only pooling — not implemented (infrastructure constraint)
Ideally pooling would be limited to the **response** tokens, since that is where the hallucination lives. But `aggregate(hidden_states, attention_mask)` does not receive `input_ids` or the prompt/response boundary, and `solution.py` is on the fixed-infrastructure list, so the call site cannot be extended. I pool over all non-padding tokens and accept some dilution from prompt tokens.

### K-fold cross-validation — not used
A single stratified split was enough for a clear primary metric. K-fold would have made the runs longer without changing the qualitative conclusions on this dataset size.

---

## File map

| File | Role | Edited? |
|---|---|---|
| `aggregation.py` | last-token pooling at block 14 | yes |
| `probe.py` | Logistic Regression + StandardScaler | yes |
| `splitting.py` | stratified 70/15/15 split | no (default kept) |
| `solution.py` | main pipeline | no (fixed infrastructure) |
| `model.py` | LLM loader | no (fixed infrastructure) |
| `evaluate.py` | metrics and saving | no (fixed infrastructure) |
| `results.json` | metrics from the final run | generated by `solution.py` |
| `predictions.csv` | predicted labels for `data/test.csv` | generated by `solution.py` |
| `requirements.txt` | dependency bounds | updated with upper bounds |

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
