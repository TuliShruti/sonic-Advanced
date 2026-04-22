"""Outlier analysis wrappers for sonic log QC results."""

from __future__ import annotations

import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, IsolationForest
from sklearn.preprocessing import StandardScaler

from sonic_dashboard.processing.qc_engine import (
    _build_flags,
    _coerce_logs,
    _engineer_features,
    _filled_features,
    _rolling_median,
)


MODEL_DIR = "models"
MODEL_PATH_DTCO = os.path.join(MODEL_DIR, "gbr_dtco.joblib")
MODEL_PATH_DTSM = os.path.join(MODEL_DIR, "gbr_dtsm.joblib")


def load_model(path):
    import os
    import joblib

    if not os.path.exists(path):
        return None

    try:
        return joblib.load(path)
    except Exception:
        # corrupted / incompatible model -> delete and retrain
        os.remove(path)
        return None


def save_model(model, path):
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, path)


def _fit_predict_persisted_corrections(
    logs: pd.DataFrame,
    features: pd.DataFrame,
    good_mask: pd.Series,
    target_col: str,
) -> tuple[np.ndarray, GradientBoostingRegressor | None]:
    corrected = logs[target_col].to_numpy(dtype=float).copy()
    target_good = good_mask & logs[target_col].notna()
    bad_target = ~target_good

    if target_col == "DTCO":
        model_path = MODEL_PATH_DTCO
    else:
        model_path = MODEL_PATH_DTSM

    model_features = _filled_features(features)
    model = load_model(model_path)

    if model is None:
        if int(target_good.sum()) < 20:
            return corrected, None

        model = GradientBoostingRegressor(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            random_state=42,
        )
        model.fit(model_features.loc[target_good], logs.loc[target_good, target_col])
        save_model(model, model_path)

    if bool(bad_target.any()):
        try:
            predicted = pd.Series(model.predict(model_features), index=logs.index)
        except ValueError:
            if int(target_good.sum()) < 20:
                return corrected, model

            model = GradientBoostingRegressor(
                n_estimators=300,
                max_depth=4,
                learning_rate=0.05,
                random_state=42,
            )
            model.fit(model_features.loc[target_good], logs.loc[target_good, target_col])
            save_model(model, model_path)
            predicted = pd.Series(model.predict(model_features), index=logs.index)

        smoothed = _rolling_median(predicted, window=5).to_numpy(dtype=float)
        corrected[bad_target.to_numpy()] = smoothed[bad_target.to_numpy()]

    return corrected, model


def run_outlier_detection(df: pd.DataFrame) -> dict:
    """Run QC-backed outlier detection and expose UI-friendly flag names."""
    logs = _coerce_logs(df)
    features = _engineer_features(logs)
    model_features = _filled_features(features)

    iso_labels = np.ones(len(logs), dtype=int)
    iso_scores = np.zeros(len(logs), dtype=float)

    if len(logs) >= 20:
        scaled_features = StandardScaler().fit_transform(model_features)
        iso_model = IsolationForest(
            n_estimators=200,
            contamination="auto",
            random_state=42,
        )
        iso_labels = iso_model.fit_predict(scaled_features)
        iso_scores = iso_model.decision_function(scaled_features)

    flags = _build_flags(logs, features, iso_labels).copy()
    good_mask = ~flags["bad_zone"] & logs["DTCO"].notna() & logs["DTSM"].notna()

    dtco_corr, model_dtco = _fit_predict_persisted_corrections(logs, features, good_mask, "DTCO")
    dtsm_corr, model_dtsm = _fit_predict_persisted_corrections(logs, features, good_mask, "DTSM")

    if "cycle_skip_co" not in flags:
        flags["cycle_skip_co"] = flags.get("cycle_skipping", False)
    if "cycle_skip_sm" not in flags:
        flags["cycle_skip_sm"] = flags.get("cycle_skipping", False)
    if "bad_dtco" not in flags:
        flags["bad_dtco"] = flags.get("bad_zone", False)
    if "bad_dtsm" not in flags:
        flags["bad_dtsm"] = flags.get("bad_zone", False)

    return {
        "dtco_corr": dtco_corr,
        "dtsm_corr": dtsm_corr,
        "flags": flags,
        "iso_scores": iso_scores,
        "models": {
            "dtco": model_dtco,
            "dtsm": model_dtsm,
        },
    }


def detect_outliers(data):
    """Backward-compatible alias for outlier detection."""
    return run_outlier_detection(data)
