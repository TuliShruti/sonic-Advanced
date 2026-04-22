"""Sonic log quality control and ML-based outlier correction."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, IsolationForest
from sklearn.preprocessing import StandardScaler


NULL_VALUE = -999.25
EXPECTED_COLUMNS = [
    "DTCO",
    "DTSM",
    "RHOB",
    "NPHI",
    "GR",
    "CALI",
    "RT",
    "TENS",
    "SNR2",
    "DCI2",
    "DCI4",
    "VPVS",
    "DEPTH_M",
]


def _coerce_logs(df: pd.DataFrame) -> pd.DataFrame:
    logs = df.copy()

    for column in EXPECTED_COLUMNS:
        if column not in logs:
            logs[column] = np.nan

    logs = logs[EXPECTED_COLUMNS].apply(pd.to_numeric, errors="coerce")
    return logs.replace(NULL_VALUE, np.nan)


def _rolling_std(series: pd.Series, window: int = 7) -> pd.Series:
    return series.rolling(window=window, center=True, min_periods=2).std()


def _rolling_median(series: pd.Series, window: int = 5) -> pd.Series:
    return series.rolling(window=window, center=True, min_periods=1).median()


def _engineer_features(logs: pd.DataFrame) -> pd.DataFrame:
    features = pd.DataFrame(index=logs.index)

    base_columns = [
        "RHOB",
        "NPHI",
        "GR",
        "CALI",
        "RT",
        "TENS",
        "SNR2",
        "DCI2",
        "DCI4",
        "VPVS",
        "DEPTH_M",
    ]
    for column in base_columns:
        features[column] = logs[column]

    cali_baseline = _rolling_median(logs["CALI"], window=31)
    features["washout"] = logs["CALI"] - cali_baseline
    features["dtco_diff"] = logs["DTCO"].diff().abs()
    features["dtsm_diff"] = logs["DTSM"].diff().abs()
    features["dtco_roll_std"] = _rolling_std(logs["DTCO"])
    features["dtsm_roll_std"] = _rolling_std(logs["DTSM"])
    features["gr_roll_std"] = _rolling_std(logs["GR"])
    features["cali_roll_std"] = _rolling_std(logs["CALI"])
    features["vpvs_from_dt"] = logs["DTSM"] / logs["DTCO"]
    features["rt_log10"] = np.log10(logs["RT"].where(logs["RT"] > 0))

    return features.replace([np.inf, -np.inf], np.nan)


def _filled_features(features: pd.DataFrame) -> pd.DataFrame:
    filled = features.copy()
    medians = filled.median(numeric_only=True)
    filled = filled.fillna(medians)
    return filled.fillna(0.0)


def _robust_jump_flag(series: pd.Series, minimum_threshold: float) -> pd.Series:
    diff = series.diff().abs()
    finite_diff = diff[np.isfinite(diff)]
    if finite_diff.empty:
        threshold = minimum_threshold
    else:
        mad = np.nanmedian(np.abs(finite_diff - np.nanmedian(finite_diff)))
        threshold = max(minimum_threshold, float(np.nanmedian(finite_diff) + 8.0 * mad))

    return diff > threshold


def _build_flags(logs: pd.DataFrame, features: pd.DataFrame, iso_labels: np.ndarray) -> pd.DataFrame:
    flags = pd.DataFrame(index=logs.index)

    cali = logs["CALI"]
    cali_baseline = _rolling_median(cali, window=31)
    flags["washout"] = (cali - cali_baseline > 1.5) | (cali > 20.0)

    flags["cycle_skipping"] = (
        _robust_jump_flag(logs["DTCO"], minimum_threshold=12.0)
        | _robust_jump_flag(logs["DTSM"], minimum_threshold=25.0)
        | (features["dtco_roll_std"] > 10.0)
        | (features["dtsm_roll_std"] > 20.0)
    )

    flags["out_of_range"] = (
        logs["DTCO"].notna() & ~logs["DTCO"].between(40.0, 240.0)
    ) | (
        logs["DTSM"].notna() & ~logs["DTSM"].between(80.0, 500.0)
    ) | (
        logs["RHOB"].notna() & ~logs["RHOB"].between(1.5, 3.2)
    ) | (
        logs["NPHI"].notna() & ~logs["NPHI"].between(-0.15, 0.80)
    ) | (
        logs["GR"].notna() & ~logs["GR"].between(0.0, 300.0)
    ) | (
        logs["CALI"].notna() & ~logs["CALI"].between(4.0, 20.0)
    ) | (
        logs["RT"].notna() & ~logs["RT"].between(0.1, 100000.0)
    ) | (
        logs["VPVS"].notna() & ~logs["VPVS"].between(1.0, 4.5)
    )

    flags["missing_sonic"] = logs["DTCO"].isna() | logs["DTSM"].isna()
    flags["iso_anomaly"] = iso_labels == -1
    flags["bad_zone"] = flags.any(axis=1)

    return flags.fillna(False)


def _fit_predict_corrections(
    logs: pd.DataFrame,
    features: pd.DataFrame,
    good_mask: pd.Series,
    target: str,
) -> np.ndarray:
    corrected = logs[target].to_numpy(dtype=float).copy()
    target_good = good_mask & logs[target].notna()
    bad_target = ~target_good

    if int(target_good.sum()) < 20 or not bool(bad_target.any()):
        return corrected

    model_features = _filled_features(features)
    model = GradientBoostingRegressor(random_state=42)
    model.fit(model_features.loc[target_good], logs.loc[target_good, target])

    predicted = pd.Series(model.predict(model_features), index=logs.index)
    smoothed = _rolling_median(predicted, window=5).to_numpy(dtype=float)
    corrected[bad_target.to_numpy()] = smoothed[bad_target.to_numpy()]

    return corrected


def run_qc_and_correction(df: pd.DataFrame) -> dict:
    """
    Run sonic log QC and ML-based correction.

    Input:
        df: pandas DataFrame with logs:
            ['DTCO','DTSM','RHOB','NPHI','GR','CALI','RT','TENS','SNR2',
             'DCI2','DCI4','VPVS','DEPTH_M']

    Output:
        {
            "flags": DataFrame,
            "dtco_corrected": array,
            "dtsm_corrected": array,
            "iso_scores": array
        }
    """
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

    flags = _build_flags(logs, features, iso_labels)
    good_mask = ~flags["bad_zone"] & logs["DTCO"].notna() & logs["DTSM"].notna()

    dtco_corrected = _fit_predict_corrections(logs, features, good_mask, "DTCO")
    dtsm_corrected = _fit_predict_corrections(logs, features, good_mask, "DTSM")

    return {
        "flags": flags,
        "dtco_corrected": dtco_corrected,
        "dtsm_corrected": dtsm_corrected,
        "iso_scores": iso_scores,
    }
