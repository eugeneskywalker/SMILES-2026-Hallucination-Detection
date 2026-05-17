"""
probe.py — Hallucination probe classifier (student-implemented).

Implements ``HallucinationProbe``, a binary classifier that scores feature
vectors as truthful (0) or hallucinated (1).  Called from ``solution.py``
via ``evaluate.run_evaluation``.  All four public methods (``fit``,
``fit_hyperparameters``, ``predict``, ``predict_proba``) must be implemented
and their signatures must not change.

Internals follow the Experiment 2 methodology (see
``notes/experiment-2-design.md`` and ``smile/notes/experiment-2-results.md``):
a ``LogisticRegression`` probe with ``l2`` regularisation, balanced class
weighting, and a ``StandardScaler`` fit on the training set only.  The probe
inherits from ``nn.Module`` purely for API compatibility — there is no
torch-side state.
"""

from __future__ import annotations

import random

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Set by run_experiment_1.py / run_experiment_2.py before each probe-training
# run.  Threads through ``random_state`` of the underlying LR and the global
# RNG seeds so re-fits are deterministic.
SEED: int = 0


class HallucinationProbe(nn.Module):
    """Binary classifier that detects hallucinations from hidden-state features.

    Extends ``torch.nn.Module`` for backward API compatibility with the
    original MLP probe; the internal classifier is now a scikit-learn
    ``LogisticRegression`` with ``StandardScaler`` pre-processing.  See
    ``notes/experiment-2-design.md`` for the methodology that selected this
    configuration over the MLP baseline.
    """

    def __init__(self) -> None:
        super().__init__()
        self._scaler = StandardScaler()
        self._lr = LogisticRegression(
            penalty="l2",
            max_iter=1000,
            class_weight="balanced",
            random_state=SEED,
        )
        # Fixed decision threshold — see ``fit_hyperparameters`` for rationale.
        self._threshold: float = 0.5

    def fit(self, X: np.ndarray, y: np.ndarray) -> "HallucinationProbe":
        """Train the probe on labelled feature vectors.

        Fits the ``StandardScaler`` and ``LogisticRegression`` on ``X``, ``y``.
        Seeds python/numpy/torch RNGs for determinism (the LR itself takes
        ``random_state=SEED`` at construction time).

        Args:
            X: Feature matrix of shape ``(n_samples, feature_dim)``.
            y: Integer label vector of shape ``(n_samples,)``; 0 = truthful,
               1 = hallucinated.

        Returns:
            ``self`` (for method chaining).
        """
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        np.random.seed(SEED)
        random.seed(SEED)

        X_scaled = self._scaler.fit_transform(X)
        self._lr.fit(X_scaled, y)
        return self

    def fit_hyperparameters(
        self, X_val: np.ndarray, y_val: np.ndarray
    ) -> "HallucinationProbe":
        """Set the decision threshold.

        F1-tuning on the validation set was found to be degenerate under
        ``class_weight='balanced'`` LR with the 689-row imbalanced training
        set: every per-seed run picked a threshold ≈ 0.0, producing a
        majority classifier that flagged ~99% of test rows as positive
        (see ``smile/notes/experiment-2-results.md`` §Predictions).  We
        therefore pin the threshold at the principled fallback of 0.5.

        Args:
            X_val: Validation feature matrix (unused — kept for API parity).
            y_val: Validation label vector (unused — kept for API parity).

        Returns:
            ``self`` (for method chaining).
        """
        self._threshold = 0.5
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict binary labels for feature vectors.

        Uses the decision threshold in ``self._threshold`` (fixed at ``0.5``;
        see ``fit_hyperparameters`` for rationale).

        Args:
            X: Feature matrix of shape ``(n_samples, feature_dim)``.

        Returns:
            Integer array of shape ``(n_samples,)`` with values in ``{0, 1}``.
        """
        return (self.predict_proba(X)[:, 1] >= self._threshold).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return class probability estimates.

        Args:
            X: Feature matrix of shape ``(n_samples, feature_dim)``.

        Returns:
            Array of shape ``(n_samples, 2)`` where column 1 contains the
            estimated probability of the hallucinated class (label 1).
            Used to compute AUROC.
        """
        X_scaled = self._scaler.transform(X)
        return self._lr.predict_proba(X_scaled)
