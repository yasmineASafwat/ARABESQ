
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from sklearn.isotonic import IsotonicRegression
import joblib

@dataclass
class IsotonicCalibrator:
    """Per-label isotonic regression calibrator (for 3 marginal probabilities)."""
    iso_mito: IsotonicRegression
    iso_plastid: IsotonicRegression
    iso_other: IsotonicRegression

    def predict(self, probs_marginal3: np.ndarray) -> np.ndarray:
        p = probs_marginal3.copy()
        p[:, 0] = self.iso_mito.predict(p[:, 0])
        p[:, 1] = self.iso_plastid.predict(p[:, 1])
        p[:, 2] = self.iso_other.predict(p[:, 2])
        return np.clip(p, 0.0, 1.0)

def fit_isotonic(probs_val: np.ndarray, y_val: np.ndarray) -> IsotonicCalibrator:
    """Fit isotonic regression per marginal probability (mito, plastid, other)."""
    iso0 = IsotonicRegression(out_of_bounds="clip")
    iso1 = IsotonicRegression(out_of_bounds="clip")
    iso2 = IsotonicRegression(out_of_bounds="clip")

    iso0.fit(probs_val[:, 0], y_val[:, 0])
    iso1.fit(probs_val[:, 1], y_val[:, 1])
    iso2.fit(probs_val[:, 2], y_val[:, 2])
    return IsotonicCalibrator(iso0, iso1, iso2)

def save_calibrator(cal: IsotonicCalibrator, path: str) -> None:
    joblib.dump(cal, path)

def load_calibrator(path: str) -> IsotonicCalibrator:
    return joblib.load(path)
