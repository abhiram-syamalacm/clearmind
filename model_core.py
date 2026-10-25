"""Shared model code: load the exported XGBoost model, map 1-10 answers onto
dataset scales, predict risk, and compute exact per-feature SHAP values."""
import json
from pathlib import Path

import numpy as np
import xgboost as xgb

MODEL_DIR = Path(__file__).parent / "models"

# Each question the student answers on a 1-10 scale, mapped linearly onto the
# range the feature has in the Kaggle "Student Stress Factors" dataset.
# invert=True means a 10 from the student is the LOW end of the dataset scale.
QUESTIONS = {
    "sleep_quality": {
        "label": "Sleep quality",
        "prompt": "How well have you been sleeping lately?",
        "low": "Very poorly", "high": "Very well",
        "range": (0, 5),
    },
    "anxiety_level": {
        "label": "Anxiety",
        "prompt": "Over the last two weeks, how anxious or on-edge have you felt?",
        "low": "Not at all", "high": "Nearly all the time",
        "range": (0, 21),  # GAD-7 total score in the dataset
    },
    "study_load": {
        "label": "Study load",
        "prompt": "How heavy is your study workload right now?",
        "low": "Very light", "high": "Overwhelming",
        "range": (0, 5),
    },
    "extracurricular_activities": {
        "label": "Extracurricular load",
        "prompt": "How much of your time and energy go to extracurriculars?",
        "low": "Almost none", "high": "A huge amount",
        "range": (0, 5),
    },
    "academic_performance": {
        "label": "Academic performance",
        "prompt": "How well do you feel you're doing in your classes?",
        "low": "Struggling a lot", "high": "Doing great",
        "range": (0, 5),
    },
}
# mental_health_history is binary in the dataset, so it's asked as yes / no /
# prefer not to say (NaN, which XGBoost handles natively as "missing").
HISTORY_FEATURE = "mental_health_history"


def scale_answer(feature, answer_1_to_10):
    lo, hi = QUESTIONS[feature]["range"]
    return lo + (answer_1_to_10 - 1) / 9 * (hi - lo)


def band(risk):
    if risk < 0.4:
        return "low"
    if risk < 0.7:
        return "medium"
    return "high"


class RiskModel:
    def __init__(self, model_dir=MODEL_DIR):
        self.meta = json.loads((model_dir / "model_meta.json").read_text())
        self.booster = xgb.Booster()
        self.booster.load_model(str(model_dir / "model.json"))
        self.features = self.meta["feature_names"]
        scaler = self.meta.get("scaler")
        self.mean = np.array(scaler["mean"]) if scaler else None
        self.scale = np.array(scaler["scale"]) if scaler else None

    def _matrix(self, raw):
        x = np.array([[raw[f] for f in self.features]], dtype=float)
        if self.mean is not None:
            x = (x - self.mean) / self.scale  # NaN stays NaN -> "missing"
        return xgb.DMatrix(x, feature_names=self.features)

    def predict(self, raw):
        """raw: dict of feature -> value on the dataset's own scale.
        Returns risk probability, band, and SHAP contributions (log-odds)."""
        dm = self._matrix(raw)
        risk = float(self.booster.predict(dm)[0])
        # pred_contribs gives exact TreeSHAP values; last column is the bias.
        contribs = self.booster.predict(dm, pred_contribs=True)[0]
        shap = dict(zip(self.features, contribs[:-1].tolist()))
        return {
            "risk": risk,
            "band": band(risk),
            "shap": shap,
            "base_logit": float(contribs[-1]),
            "logit": float(contribs.sum()),
        }


def feature_label(feature):
    if feature == HISTORY_FEATURE:
        return "Mental-health history"
    return QUESTIONS[feature]["label"]
