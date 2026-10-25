"""Run the exported EEG workload model on an uploaded STEW-format recording.

Preprocessing and features come straight from clearmind_pipeline.py, so they
match the paper exactly.
"""
import json
import os
import tempfile
from pathlib import Path

import numpy as np

import clearmind_pipeline as cp

MODEL_PATH = Path(__file__).parent / "models" / "eeg_model.json"


class EEGModel:
    def __init__(self, path=MODEL_PATH):
        self.meta = json.loads(path.read_text())
        self.features = self.meta["feature_names"]
        self.mean = np.array(self.meta["scaler_mean"])
        self.scale = np.array(self.meta["scaler_scale"])
        self.coef = np.array(self.meta["coef"])
        self.intercept = self.meta["intercept"]

    def predict_file(self, raw_bytes, suffix=".txt"):
        """Returns per-epoch probabilities and the average feature contributions,
        or raises ValueError with a readable message."""
        fd, tmp = tempfile.mkstemp(suffix=suffix)
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(raw_bytes)
            try:
                data = cp.load_recording(tmp)
            except (IOError, ValueError) as e:
                raise ValueError(str(e).replace(tmp, "this file")) from None
        finally:
            os.remove(tmp)

        n_samples = data.shape[1]
        epochs_all = cp.epoch(cp.preprocess(data))
        epochs, n_rejected = cp.reject(epochs_all)
        if len(epochs) == 0:
            raise ValueError(
                f"All {len(epochs_all)} epochs exceeded the {cp.REJECT_UV:.0f} µV artifact "
                "threshold, so there's nothing clean to score.")

        z = (cp.extract_features(epochs) - self.mean) / self.scale
        contrib = z * self.coef                      # exact per-feature log-odds terms
        logit = contrib.sum(axis=1) + self.intercept
        prob = 1 / (1 + np.exp(-logit))
        mean_contrib = contrib.mean(axis=0)
        order = np.argsort(np.abs(mean_contrib))[::-1][:10]
        return {
            "duration_s": n_samples / cp.SFREQ,
            "n_epochs": int(len(epochs_all)),
            "n_rejected": int(n_rejected),
            "epoch_probs": prob.tolist(),
            "mean_prob": float(prob.mean()),
            "share_high": float((prob >= 0.5).mean()),
            "top_contributions": [(self.features[i], float(mean_contrib[i])) for i in order],
        }
