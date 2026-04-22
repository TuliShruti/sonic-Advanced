"""Outlier detection and compatibility imports for the sonic dashboard."""

import os

import joblib
import numpy as np
import pandas as pd
from scipy.stats import skew as sp_skew
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import PowerTransformer, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_validate

from sonic_dashboard.processing.outlier_detection import (
    MODEL_PATH_DTCO,
    MODEL_PATH_DTSM,
    detect_outliers,
    run_outlier_detection,
)

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_PATH_DT = os.path.join(MODEL_DIR, "rf_dt_model.joblib")


def save_model(model, path):
    try:
        joblib.dump(model, path)
    except Exception as e:
        print(f"Model save failed: {e}")


def load_model(path):
    if os.path.exists(path):
        try:
            return joblib.load(path)
        except Exception as e:
            print(f"Model load failed: {e}")
    return None


def train_and_fill(df, dt_ch, predictor_channels,
                   n_estimators=100, random_state=42, use_saved_model=True):

    # -- Select available predictors ---------------------------------------
    available_predictors = [c for c in predictor_channels if c in df.columns]
    feature_cols = available_predictors + ["DEPTH_M"]

    # -- MUTE bad zones -----------------------------------------------------
    df_work = df.copy()
    df_work.loc[df_work["FLAG_BAD"], dt_ch] = np.nan

    # -- Split data ---------------------------------------------------------
    df_good = df_work[~df_work["FLAG_BAD"]].dropna(subset=[dt_ch] + feature_cols).copy()
    df_bad = df_work[df_work["FLAG_BAD"]].dropna(subset=feature_cols).copy()

    if len(df_bad) == 0:
        raise RuntimeError("No bad samples to fill.")

    # -- Detect skew + preprocessing ---------------------------------------
    pos_skew, yeo_skew, depth_cols = [], [], []

    for col in feature_cols:
        if col == "DEPTH_M":
            depth_cols.append(col)
            continue
        s = sp_skew(df_good[col].dropna())
        mn = df_good[col].min()
        if abs(s) > 0.75:
            (pos_skew if mn > 0 else yeo_skew).append(col)

    transformers = []
    if pos_skew:
        transformers.append((PowerTransformer(method='box-cox'), pos_skew))
    if yeo_skew:
        transformers.append((PowerTransformer(method='yeo-johnson'), yeo_skew))
    if depth_cols:
        transformers.append((MinMaxScaler(), depth_cols))

    col_trans = make_column_transformer(*transformers, remainder='passthrough')

    X_train = col_trans.fit_transform(df_good[feature_cols])
    y_train = df_good[dt_ch].values
    X_fill = col_trans.transform(df_bad[feature_cols])

    # -- Train model --------------------------------------------------------
    loaded_bundle = load_model(MODEL_PATH_DT) if use_saved_model else None
    loaded_model = None
    if isinstance(loaded_bundle, dict):
        bundle_dt = loaded_bundle.get("dt_ch")
        bundle_features = loaded_bundle.get("feature_cols")
        if bundle_dt == dt_ch and bundle_features == feature_cols:
            loaded_model = loaded_bundle.get("model")
    elif loaded_bundle is not None:
        loaded_model = loaded_bundle

    if loaded_model is not None:
        rf = loaded_model
        metrics = {
            "train_mae": np.nan,
            "val_mae": np.nan,
        }
    else:
        rf = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=20,
            min_samples_split=4,
            random_state=random_state,
            n_jobs=-1
        )

        cv = cross_validate(rf, X_train, y_train,
                            cv=5,
                            scoring='neg_mean_absolute_error',
                            return_train_score=True)

        rf.fit(X_train, y_train)
        save_model(
            {
                "model": rf,
                "dt_ch": dt_ch,
                "feature_cols": feature_cols,
            },
            MODEL_PATH_DT,
        )

        metrics = {
            "train_mae": -cv['train_score'].mean(),
            "val_mae": -cv['test_score'].mean()
        }

    # -- Predict ------------------------------------------------------------
    dt_predicted = rf.predict(X_fill)

    # -- Merge results ------------------------------------------------------
    df_merged = df.copy()
    df_merged["DT_FILLED"] = df_merged[dt_ch].copy()
    df_merged["DT_SOURCE"] = "original"

    df_merged.loc[df_bad.index, "DT_FILLED"] = dt_predicted
    df_merged.loc[df_bad.index, "DT_SOURCE"] = "ml_predicted"

    return df_merged, metrics


__all__ = [
    "MODEL_DIR",
    "MODEL_PATH_DT",
    "MODEL_PATH_DTCO",
    "MODEL_PATH_DTSM",
    "detect_outliers",
    "load_model",
    "run_outlier_detection",
    "save_model",
    "train_and_fill",
]
