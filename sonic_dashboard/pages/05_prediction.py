"""ML-based sonic log prediction page."""

from __future__ import annotations

import io

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from sonic_dashboard.utils.session_state import initialize_session_state


DROP_COLS = [
    "TENS",
    "DRHO",
    "XPHI",
    "SPHI",
    "DPHI",
    "NDSN",
    "FDSN",
    "ITTT",
    "MSFL",
    "LLS",
    "AHVT",
    "GRTO",
    "GRTH",
    "GRKT",
    "GKUT",
    "DXDT",
    "DYDT",
    "BHVT",
]
FEATURE_COLS = ["URAN", "THOR", "LLD", "GR", "RHOB", "PE", "NPHI", "DEPT"]
TARGET_COL = "MDT"


def load_las_dataframe(file_bytes: bytes) -> pd.DataFrame:
    """Convert an uploaded LAS file to a curve dataframe."""
    try:
        import lasio
    except ImportError as exc:
        raise ImportError("Install lasio to upload LAS files for prediction.") from exc

    try:
        las = lasio.read(io.BytesIO(file_bytes))
    except Exception:
        text = file_bytes.decode("utf-8", errors="ignore")
        las = lasio.read(io.StringIO(text))

    df = las.df().reset_index()
    df = df.replace([-999.25, -999.0, -9999.0, -99999.0], np.nan)
    for col in df.select_dtypes(include=[np.number]).columns:
        df.loc[df[col] < -900, col] = np.nan
    if "MNDT" in df.columns and TARGET_COL not in df.columns:
        df = df.rename(columns={"MNDT": TARGET_COL})
    return df


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the notebook preprocessing: numeric conversion and missing-row removal."""
    df = df.copy()
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()
    return df


def train_model(X, y, model_type: str, params: dict):
    if model_type == "Linear Regression":
        from sklearn.linear_model import LinearRegression

        model = LinearRegression()
    elif model_type == "Random Forest":
        from sklearn.ensemble import RandomForestRegressor

        model = RandomForestRegressor(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            min_samples_split=params["min_samples_split"],
            random_state=42,
            n_jobs=-1,
        )
    elif model_type == "Gradient Boosting":
        from sklearn.ensemble import GradientBoostingRegressor

        model = GradientBoostingRegressor(
            n_estimators=params["n_estimators"],
            learning_rate=params["learning_rate"],
            max_depth=params["max_depth"],
        )
    elif model_type == "Neural Network":
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras.layers import Dense
        from tensorflow.keras.models import Sequential

        tf.random.set_seed(42)
        model = Sequential(
            [
                Dense(
                    params["nodes"],
                    activation="relu",
                    input_dim=X.shape[1],
                    kernel_initializer=tf.keras.initializers.RandomNormal(seed=42),
                ),
                Dense(params["nodes"], activation="relu"),
                Dense(1, activation="linear"),
            ]
        )
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=params["learning_rate"]),
            loss="mean_absolute_error",
            metrics=["mean_absolute_error"],
        )
        model.fit(
            X,
            y,
            batch_size=params["batch_size"],
            epochs=params["epochs"],
            verbose=0,
        )
        return model
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    model.fit(X, y)
    return model


def predict_model(model, X) -> np.ndarray:
    try:
        predicted = model.predict(X, verbose=0)
    except TypeError:
        predicted = model.predict(X)
    return np.asarray(predicted, dtype=float).reshape(-1)


def _model_feature_count(model) -> int | None:
    if hasattr(model, "n_features_in_"):
        return int(model.n_features_in_)

    input_shape = getattr(model, "input_shape", None)
    if input_shape and len(input_shape) >= 2 and input_shape[-1] is not None:
        return int(input_shape[-1])

    return None


def _clear_prediction_model_state() -> None:
    for key in [
        "prediction_model",
        "model",
        "prediction_training_metrics",
        "prediction_validation_metrics",
        "prediction_training_measured",
        "prediction_training_pred",
        "prediction_validation_measured",
        "prediction_validation_pred",
        "prediction_model_type",
    ]:
        st.session_state[key] = None


def _stored_model_is_compatible(model, features: list[str]) -> bool:
    if model is None:
        return True

    if st.session_state.get("prediction_features") != features:
        return False

    feature_count = _model_feature_count(model)
    return feature_count in (None, len(features))


def evaluate(measured, predicted) -> dict:
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    measured, predicted = _paired_numeric_arrays(measured, predicted)
    if len(measured) == 0:
        return {"mae": np.nan, "mse": np.nan, "r2": np.nan}

    return {
        "mae": mean_absolute_error(measured, predicted),
        "mse": mean_squared_error(measured, predicted),
        "r2": r2_score(measured, predicted),
    }


def _numeric_columns(df: pd.DataFrame) -> list[str]:
    return [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]


def _curve_summary(df: pd.DataFrame) -> pd.DataFrame:
    summary = []
    for col in df.columns:
        numeric_col = pd.to_numeric(df[col], errors="coerce")
        summary.append(
            {
                "Curve": col,
                "Missing %": df[col].isna().mean() * 100,
                "Min": numeric_col.min(),
                "Max": numeric_col.max(),
                "Mean": numeric_col.mean(),
            }
        )

    return pd.DataFrame(summary)


def _select_distribution_logs(df: pd.DataFrame, target: str | None = None) -> list[str]:
    candidate_logs = [
        "GR",
        "RHOB",
        "NPHI",
        "LLD",
        "MSFL",
        "POTA",
        "URAN",
        "THOR",
        "CALI",
    ]
    return [
        log
        for log in candidate_logs
        if log in df.columns
        and log != target
        and "_grad" not in log
        and "_roll" not in log
    ][:6]


def _default_target(columns: list[str]) -> str:
    for candidate in (TARGET_COL, "MNDT", "DT", "DTCO", "DTSM"):
        if candidate in columns:
            return candidate
    return columns[0]


def _default_features(columns: list[str], target: str) -> list[str]:
    depth_names = {"DEPT", "DEPTH", "DEPTH_M", "TDEP"}
    return [
        col
        for col in columns
        if col != target
        and not col.startswith(f"{target}_")
        and col.upper() not in depth_names
    ]


def _depth_column(df: pd.DataFrame) -> str | None:
    for col in ("DEPT", "DEPTH", "DEPTH_M", "TDEP"):
        if col in df.columns:
            return col
    return None


def _format_metric(value: float) -> str:
    if pd.isna(value):
        return "N/A"
    return f"{value:.4f}"


def _paired_numeric_arrays(measured, predicted) -> tuple[np.ndarray, np.ndarray]:
    measured_array = np.asarray(measured, dtype=float).reshape(-1)
    predicted_array = np.asarray(predicted, dtype=float).reshape(-1)

    min_len = min(len(measured_array), len(predicted_array))
    measured_array = measured_array[:min_len]
    predicted_array = predicted_array[:min_len]

    valid = np.isfinite(measured_array) & np.isfinite(predicted_array)
    return measured_array[valid], predicted_array[valid]


def _plot_measured_vs_predicted(measured, predicted, target: str) -> plt.Figure:
    measured, predicted = _paired_numeric_arrays(measured, predicted)

    fig, ax = plt.subplots(figsize=(7, 5))
    if len(measured) == 0:
        ax.text(0.5, 0.5, "No paired training values to plot.", ha="center", va="center")
    else:
        ax.scatter(measured, predicted, s=12, alpha=0.7, color="#2563eb")

        min_value = min(float(np.min(measured)), float(np.min(predicted)))
        max_value = max(float(np.max(measured)), float(np.max(predicted)))
        ax.plot([min_value, max_value], [min_value, max_value], color="#111827", linewidth=1.2)

    ax.set_xlabel(f"Measured {target}")
    ax.set_ylabel(f"Predicted {target}")
    ax.set_title("Measured vs Predicted")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    fig.tight_layout()
    return fig


def _plot_residuals(measured, predicted) -> plt.Figure:
    measured, predicted = _paired_numeric_arrays(measured, predicted)
    residuals = measured - predicted

    fig, ax = plt.subplots(figsize=(7, 4))
    if len(predicted) == 0:
        ax.text(0.5, 0.5, "No paired residual values to plot.", ha="center", va="center")
    else:
        ax.scatter(predicted, residuals, s=12, alpha=0.7, color="#7c3aed")

    ax.axhline(0.0, color="#111827", linewidth=1.2)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Residual")
    ax.set_title("Residuals")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    fig.tight_layout()
    return fig


def _plot_actual_predicted_log_track(prediction_df, y_pred, model_type, measured):
    if "DEPT" in prediction_df.columns:
        depth = prediction_df["DEPT"].values
    else:
        depth = np.arange(len(y_pred))

    y_pred = np.array(y_pred)
    measured = np.asarray(measured, dtype=float).reshape(-1)

    min_len = min(len(depth), len(y_pred), len(measured))
    depth = depth[:min_len]
    y_pred = y_pred[:min_len]
    measured = measured[:min_len]

    order = np.argsort(depth)
    d = depth[order]
    act = measured[order]
    pr = y_pred[order]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=act,
            y=d,
            mode="lines",
            name=f"Actual {TARGET_COL}",
            line=dict(color="#1f77b4", width=1.5),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=pr,
            y=d,
            mode="lines",
            name=f"Predicted ({model_type})",
            line=dict(color="#d62728", width=1.5, dash="dot"),
        )
    )

    metrics = evaluate(measured, y_pred)

    fig.update_layout(
        title=(
            f"{TARGET_COL} Log Track - {model_type}"
            f"<br><sup>MAE: {_format_metric(metrics['mae'])} | R2: {_format_metric(metrics['r2'])}</sup>"
        ),
        title_font=dict(color="#111827", size=18),
        xaxis_title=f"{TARGET_COL} (us/ft)",
        yaxis_title="Depth (ft)",
        yaxis_autorange="reversed",
        height=850,
        width=520,
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(color="#111827", size=12),
        legend=dict(
            x=0.55,
            y=0.02,
            font=dict(color="#111827", size=12),
            bgcolor="rgba(255,255,255,0.96)",
            bordercolor="#9ca3af",
            borderwidth=1,
        ),
    )

    fig.update_xaxes(
        showgrid=True,
        gridcolor="#e5e7eb",
        title_font=dict(color="#111827", size=13),
        tickfont=dict(color="#374151", size=11),
        linecolor="#9ca3af",
        zerolinecolor="#d1d5db",
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor="#e5e7eb",
        title_font=dict(color="#111827", size=13),
        tickfont=dict(color="#374151", size=11),
        linecolor="#9ca3af",
        zerolinecolor="#d1d5db",
    )

    return fig


def _plot_log_distributions(df: pd.DataFrame, selected_logs: list[str]) -> plt.Figure:
    n_logs = len(selected_logs)
    cols = 3
    rows = int(np.ceil(n_logs / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(12, 8))
    axes = np.asarray(axes).flatten()

    for i, log in enumerate(selected_logs):
        data = pd.to_numeric(df[log], errors="coerce").dropna()

        lower = data.quantile(0.01)
        upper = data.quantile(0.99)
        data = data[(data >= lower) & (data <= upper)]

        axes[i].hist(data, bins=50, color="#2563eb", edgecolor="white", alpha=0.85)
        axes[i].set_title(log)
        axes[i].grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.5)

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    fig.tight_layout()
    return fig


def _render_metrics(metrics: dict) -> None:
    col1, col2, col3 = st.columns(3)
    col1.metric("MAE", _format_metric(metrics["mae"]))
    col2.metric("MSE", _format_metric(metrics["mse"]))
    col3.metric("R2", _format_metric(metrics["r2"]))


def _model_params(model_type: str) -> dict:
    if model_type == "Linear Regression":
        return {}

    if model_type == "Random Forest":
        return {
            "n_estimators": st.slider("n_estimators", 50, 500, 300, 10),
            "max_depth": st.slider("max_depth", 3, 30, 20),
            "min_samples_split": st.slider("min_samples_split", 2, 12, 4),
        }

    if model_type == "Gradient Boosting":
        return {
            "n_estimators": st.slider("n_estimators", 50, 600, 400, 10),
            "learning_rate": st.slider("learning_rate", 0.01, 0.2, 0.05, 0.01),
            "max_depth": st.slider("max_depth", 2, 10, 5),
        }

    return {
        "epochs": st.slider("epochs", 10, 300, 100, 10),
        "batch_size": st.slider("batch_size", 8, 128, 32, 8),
        "nodes": st.slider("nodes", 16, 256, 100, 4),
        "learning_rate": st.slider("learning_rate", 0.0001, 0.01, 0.001, 0.0001, format="%.4f"),
    }


def _iqr_bounds(df: pd.DataFrame) -> dict[str, tuple[float, float]]:
    details = df.describe()
    bounds = {}
    for col in details.columns:
        iqr = details.loc["75%", col] - details.loc["25%", col]
        bounds[col] = (
            details.loc["25%", col] - 1.5 * iqr,
            details.loc["75%", col] + 1.5 * iqr,
        )
    return bounds


def _drop_outliers(df: pd.DataFrame, bounds: dict[str, tuple[float, float]], mode: str = "all") -> pd.DataFrame:
    if mode == "cali" and "CALI" in df.columns:
        lower, upper = bounds["CALI"]
        return df[(df["CALI"] >= lower) & (df["CALI"] <= upper)]

    filtered = df
    for col, (lower, upper) in bounds.items():
        filtered = filtered[(filtered[col] >= lower) & (filtered[col] <= upper)]
    return filtered


def _feature_transformer():
    from sklearn.compose import make_column_transformer
    from sklearn.preprocessing import MinMaxScaler, PowerTransformer

    return make_column_transformer(
        (PowerTransformer(), ["URAN", "THOR", "LLD", "GR", "RHOB", "PE"]),
        (PowerTransformer(method="yeo-johnson"), ["NPHI"]),
        (MinMaxScaler(), ["DEPT"]),
    )


def _modeling_frame(df_raw: pd.DataFrame, require_target: bool) -> tuple[pd.DataFrame | None, list[str]]:
    df = preprocess_data(df_raw).drop(columns=DROP_COLS, errors="ignore")
    required = FEATURE_COLS + ([TARGET_COL] if require_target else [])
    missing = [col for col in required if col not in df.columns]
    if missing:
        return None, missing

    return df.copy(), []


def _split_features(df: pd.DataFrame, include_target: bool = True):
    X = df[FEATURE_COLS].copy()
    if include_target:
        return X, df[TARGET_COL].copy()
    return X, None


def _prepare_training_data(df_raw: pd.DataFrame):
    from sklearn.model_selection import train_test_split

    train, missing = _modeling_frame(df_raw, require_target=True)
    if train is None:
        return None, missing
    if train.empty:
        return None, ["No rows remain after missing-value removal"]

    iqr_dict = _iqr_bounds(train)
    train_no_outliers = _drop_outliers(train, iqr_dict, "all")
    if train_no_outliers.empty:
        return None, ["No rows remain after IQR outlier removal"]

    X, y = _split_features(train_no_outliers)
    transformer = _feature_transformer()
    X_transformed = transformer.fit_transform(X)
    X_train, X_val, y_train, y_val = train_test_split(
        X_transformed,
        y,
        test_size=0.2,
        random_state=42,
    )

    return {
        "frame": train_no_outliers,
        "X_train": X_train,
        "X_val": X_val,
        "y_train": y_train,
        "y_val": y_val,
        "transformer": transformer,
        "features": FEATURE_COLS,
        "row_counts": {
            "Raw usable rows": len(train),
            "IQR-clean rows": len(train_no_outliers),
        },
    }, []


def _prepare_test_data(df_raw: pd.DataFrame, transformer):
    test, missing = _modeling_frame(df_raw, require_target=False)
    if test is None:
        return None, None, None, missing
    if test.empty:
        return None, None, None, ["No rows remain after missing-value removal"]

    test_has_target = TARGET_COL in preprocess_data(df_raw).columns
    if test_has_target:
        test, missing = _modeling_frame(df_raw, require_target=True)
        if test is None:
            return None, None, None, missing
        if test.empty:
            return None, None, None, ["No rows remain after missing-value removal"]

    test_iqr = _iqr_bounds(test)
    test_clean = _drop_outliers(test, test_iqr, "all")
    if test_clean.empty:
        return None, None, None, ["No rows remain after IQR outlier removal"]

    X_test, y_test = _split_features(test_clean, include_target=test_has_target)
    X_test_transformed = transformer.transform(X_test)
    return test_clean, X_test_transformed, y_test, []


def _candidate_features(columns: list[str], target: str) -> list[str]:
    return [
        col
        for col in columns
        if col != target and not col.startswith(f"{target}_")
    ]


initialize_session_state()

st.title("Sonic Log Prediction")
st.caption("Train an ML model on LAS logs and predict sonic response on a second LAS file.")

for key, value in {
    "prediction_model": None,
    "prediction_transformer": None,
    "prediction_features": None,
    "prediction_target": None,
    "prediction_model_type": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = value

st.subheader("1. Training")
training_file = st.file_uploader("Upload training LAS file", type=["las"], key="prediction_training_las")

if training_file is None:
    st.info("Upload a training LAS file to begin.")
else:
    previous_training_file = st.session_state.get("prediction_training_file_name")
    if previous_training_file is not None and previous_training_file != training_file.name:
        st.session_state["prediction_ran"] = False
    st.session_state["prediction_training_file_name"] = training_file.name

    if st.button("Process Data", type="primary"):
        st.session_state["prediction_ran"] = True

    if not st.session_state.get("prediction_ran", False):
        st.info("Upload data and click 'Process Data' to start.")
        st.stop()

    try:
        df_raw = load_las_dataframe(training_file.getvalue())
    except Exception as exc:
        st.error(f"Could not load training LAS file: {exc}")
        st.stop()

    st.subheader("Well Log Overview")
    summary_df = _curve_summary(df_raw)
    st.dataframe(summary_df, use_container_width=True)

    raw_numeric_columns = _numeric_columns(df_raw)
    overview_target = _default_target(raw_numeric_columns) if raw_numeric_columns else None

    st.subheader("Log Variable Distributions")
    selected_logs = _select_distribution_logs(df_raw, overview_target)
    distribution_options = [
        col
        for col in df_raw.columns
        if col != overview_target
        and "_grad" not in col
        and "_roll" not in col
        and pd.to_numeric(df_raw[col], errors="coerce").notna().any()
    ]
    user_logs = st.multiselect(
        "Select logs to visualize",
        distribution_options,
        default=selected_logs,
    )[:6]

    if user_logs:
        st.pyplot(_plot_log_distributions(df_raw, user_logs), use_container_width=True)
    else:
        st.info("No base log curves are available for distribution plotting.")

    with st.expander("Data preview"):
        st.dataframe(df_raw.head(50), use_container_width=True)

    st.divider()

    cleaned_preview = preprocess_data(df_raw).drop(columns=DROP_COLS, errors="ignore")
    numeric_columns = _numeric_columns(cleaned_preview)

    missing_required = [col for col in FEATURE_COLS + [TARGET_COL] if col not in numeric_columns]
    if missing_required:
        st.warning(
            "The training LAS file is missing required notebook curves: "
            + ", ".join(missing_required)
        )
        st.stop()

    target = TARGET_COL

    training_data, missing_train = _prepare_training_data(df_raw)
    if missing_train:
        st.warning(f"Missing training columns after preprocessing: {', '.join(missing_train)}")
        st.stop()

    st.session_state["features"] = training_data["features"]
    st.session_state["target"] = target
    st.session_state["prediction_transformer"] = training_data["transformer"]

    if not _stored_model_is_compatible(
        st.session_state.get("prediction_model"),
        training_data["features"],
    ):
        _clear_prediction_model_state()
        st.info("The previous model used an older feature pipeline. Please train the model again.")

    with st.expander("pipeline row counts"):
        st.dataframe(
            pd.DataFrame(
                training_data["row_counts"].items(),
                columns=["Step", "Rows"],
            ),
            use_container_width=True,
            hide_index=True,
        )

    st.divider()
    st.subheader("2. Model + Hyperparameters")
    model_type = st.selectbox(
        "Model",
        ["Random Forest", "Gradient Boosting", "Linear Regression", "Neural Network"],
    )
    params = _model_params(model_type)

    st.divider()
    st.subheader("3. Train Model")
    if st.button("Train model", type="primary"):
        with st.spinner("Training model..."):
            try:
                model = train_model(
                    training_data["X_train"],
                    training_data["y_train"],
                    model_type,
                    params,
                )
            except ImportError as exc:
                st.error(f"Could not train {model_type}: {exc}")
                st.stop()
            train_pred = predict_model(model, training_data["X_train"])
            val_pred = predict_model(model, training_data["X_val"])
            train_metrics = evaluate(training_data["y_train"], train_pred)
            val_metrics = evaluate(training_data["y_val"], val_pred)

        st.session_state["prediction_model"] = model
        st.session_state["model"] = model
        st.session_state["features"] = training_data["features"]
        st.session_state["target"] = target
        st.session_state["prediction_transformer"] = training_data["transformer"]
        st.session_state["prediction_features"] = training_data["features"]
        st.session_state["prediction_target"] = target
        st.session_state["prediction_model_type"] = model_type
        st.session_state["prediction_training_metrics"] = train_metrics
        st.session_state["prediction_validation_metrics"] = val_metrics
        st.session_state["prediction_training_measured"] = training_data["y_train"].to_numpy(dtype=float)
        st.session_state["prediction_training_pred"] = train_pred
        st.session_state["prediction_validation_measured"] = training_data["y_val"].to_numpy(dtype=float)
        st.session_state["prediction_validation_pred"] = val_pred
        st.success("Model trained.")

    model = st.session_state.get("prediction_model")
    trained_target = st.session_state.get("target", target)
    train_metrics = st.session_state.get("prediction_training_metrics")
    val_metrics = st.session_state.get("prediction_validation_metrics")
    train_measured = st.session_state.get("prediction_training_measured")
    train_pred = st.session_state.get("prediction_training_pred")
    trained_model_type = st.session_state.get("prediction_model_type") or model_type

    if model is not None and train_metrics is not None:
        st.divider()
        st.subheader("4. Training Output")
        st.markdown("**Train Metrics**")
        _render_metrics(train_metrics)
        if val_metrics is not None:
            st.markdown("**Validation Metrics**")
            _render_metrics(val_metrics)

        col1, col2 = st.columns(2)
        with col1:
            st.pyplot(_plot_measured_vs_predicted(train_measured, train_pred, trained_target), use_container_width=True)
        with col2:
            st.pyplot(_plot_residuals(train_measured, train_pred), use_container_width=True)

        st.divider()
        st.subheader("5. Test Data")
        test_file = st.file_uploader("Upload test LAS file", type=["las"], key="prediction_test_las")

        if test_file is not None:
            try:
                df_test_raw = load_las_dataframe(test_file.getvalue())
            except Exception as exc:
                st.error(f"Could not load test LAS file: {exc}")
                st.stop()

            prediction_df, X_test, y_test, missing_test = _prepare_test_data(
                df_test_raw,
                st.session_state["prediction_transformer"],
            )

            if missing_test:
                st.warning(f"Prediction skipped. Test data issue: {', '.join(missing_test)}")
                st.stop()

            if prediction_df is None or X_test is None:
                st.warning("Prediction skipped. No valid rows remain after preprocessing.")
            else:
                st.divider()
                st.subheader("6. Prediction")
                if not _stored_model_is_compatible(model, training_data["features"]):
                    _clear_prediction_model_state()
                    st.warning("The stored model does not match the current notebook feature pipeline. Train the model again.")
                    st.stop()

                try:
                    y_pred = predict_model(model, X_test)
                except ValueError as exc:
                    _clear_prediction_model_state()
                    st.warning(f"Prediction skipped because the stored model is incompatible. Train the model again. Details: {exc}")
                    st.stop()

                if y_test is not None:
                    test_metrics = evaluate(y_test, y_pred)
                    st.markdown("**Test Metrics**")
                    _render_metrics(test_metrics)
                    st.pyplot(
                        _plot_measured_vs_predicted(y_test, y_pred, trained_target),
                        use_container_width=True,
                    )
                else:
                    st.info("The test LAS has no MDT/MNDT curve, so test metrics cannot be calculated.")

                st.divider()
                st.subheader("7. Final Visualization")
                if y_test is None:
                    st.info("The notebook-style actual vs predicted plot requires MDT/MNDT in the test LAS.")
                else:
                    fig = _plot_actual_predicted_log_track(
                        prediction_df,
                        y_pred,
                        trained_model_type,
                        y_test,
                    )
                    _, plot_col, _ = st.columns([1, 2, 1])
                    with plot_col:
                        st.plotly_chart(fig, use_container_width=False)
