from __future__ import annotations
import hashlib, json
from pathlib import Path
import joblib
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from .features import FEATURE_COLUMNS
from .types import DecisionAction, ModelDecision

class TradeQualityModel:
    def __init__(self, model_id: str, pipeline, safety_margin_bps: float):
        self.model_id = model_id
        self.pipeline = pipeline
        self.safety_margin_bps = float(safety_margin_bps)

    def probability(self, features: dict[str, float]) -> float:
        frame = pd.DataFrame([[features[name] for name in FEATURE_COLUMNS]], columns=FEATURE_COLUMNS)
        return float(self.pipeline.predict_proba(frame)[0, 1])

    def decide(self, candidate, costs: dict) -> ModelDecision:
        probability = self.probability(candidate.features)
        gain = candidate.target_price / candidate.entry_price - 1
        loss = 1 - candidate.stop_price / candidate.entry_price
        total_cost = (2 * (float(costs["fee_bps_per_side"]) + float(costs["spread_bps_per_side"]) + float(costs["slippage_bps_per_side"])) + float(costs.get("latency_bps_round_trip", 0))) / 10000
        expected_net = probability * gain - (1 - probability) * loss - total_cost
        margin = self.safety_margin_bps / 10000
        action = DecisionAction.ACCEPT_FULL_SIZE if expected_net > margin else DecisionAction.REJECT
        return ModelDecision(candidate.candidate_id, self.model_id, probability, expected_net, margin, action)

    def save(self, path: str | Path, manifest: dict) -> None:
        destination = Path(path)
        destination.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.pipeline, destination / "model.joblib")
        (destination / "feature_schema.json").write_text(json.dumps(FEATURE_COLUMNS, indent=2), encoding="utf-8")
        (destination / "model_manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
        digest = hashlib.sha256((destination / "model.joblib").read_bytes()).hexdigest()
        (destination / "checksums.sha256").write_text(f"{digest}  model.joblib\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "TradeQualityModel":
        source = Path(path)
        manifest = json.loads((source / "model_manifest.json").read_text(encoding="utf-8"))
        return cls(manifest["model_id"], joblib.load(source / "model.joblib"), manifest["safety_margin_bps"])

def train_logistic(frame: pd.DataFrame, model_id: str, safety_margin_bps: float = 5.0) -> tuple[TradeQualityModel, dict]:
    data = frame.dropna(subset=FEATURE_COLUMNS + ["target_hit_first"])
    x = data[FEATURE_COLUMNS]
    y = data["target_hit_first"].astype(int)
    if y.nunique() < 2:
        raise ValueError("Training data must contain target and stop outcomes")
    base = Pipeline([("scale", StandardScaler()), ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0))])
    calibrated = CalibratedClassifierCV(base, method="sigmoid", cv=3)
    calibrated.fit(x, y)
    probabilities = calibrated.predict_proba(x)[:, 1]
    metrics = {"rows": len(data), "positive_rate": float(y.mean()), "brier_score": float(brier_score_loss(y, probabilities)), "log_loss": float(log_loss(y, probabilities))}
    return TradeQualityModel(model_id, calibrated, safety_margin_bps), metrics
