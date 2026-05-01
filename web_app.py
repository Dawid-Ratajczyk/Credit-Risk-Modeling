"""
Interactive credit-risk modeling: upload data, pick a model, tune parameters,
and compare accuracy / precision / recall / F1 across probability thresholds.
Run: streamlit run web_app.py
"""

from __future__ import annotations

import io
import itertools
from collections.abc import Callable
from typing import Any, Literal

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from locale_strings import MESSAGES, tr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier

ZERO_DIV = 0
MAX_CSV_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MiB cap for uploaded CSV


def section_title(title: str) -> None:
    """Simple section title without inline theory popovers."""
    st.subheader(title)


def render_theory_page(lang: str) -> None:
    """Standalone page with explanations previously behind ℹ️ buttons."""
    st.title(tr(lang, "theory_title"))
    st.caption(tr(lang, "theory_intro"))

    section_title(tr(lang, "header_model"))
    st.markdown(f"### {tr(lang, 'model_logistic')}")
    st.markdown(tr(lang, "theory_model_logistic"))
    st.markdown(
        "\n".join(
            [
                f"- `{tr(lang, 'max_iter')}`: {tr(lang, 'max_iter_help')}",
                f"- `{tr(lang, 'lr_c')}`: {tr(lang, 'lr_c_help')}",
                f"- `{tr(lang, 'penalty')}`: {tr(lang, 'penalty_help')}",
                f"- `{tr(lang, 'fit_intercept')}`: {tr(lang, 'fit_intercept_help')}",
                f"- `{tr(lang, 'tol')}`: {tr(lang, 'tol_help')}",
            ]
        )
    )

    st.markdown(f"### {tr(lang, 'model_tree')}")
    st.markdown(tr(lang, "theory_model_tree"))
    st.markdown(
        "\n".join(
            [
                f"- `{tr(lang, 'tree_max_depth')}`: {tr(lang, 'tree_max_depth_help')}",
                f"- `{tr(lang, 'min_samples_leaf')}`: {tr(lang, 'min_samples_leaf_help')}",
                f"- `{tr(lang, 'min_samples_split')}`: {tr(lang, 'min_samples_split_help')}",
                f"- `{tr(lang, 'criterion')}`: {tr(lang, 'criterion_help')}",
                f"- `{tr(lang, 'max_features')}`: {tr(lang, 'max_features_help')}",
                f"- `{tr(lang, 'max_leaf_nodes')}`: {tr(lang, 'max_leaf_nodes_help')}",
                f"- `{tr(lang, 'min_impurity_decrease')}`: {tr(lang, 'min_impurity_decrease_help')}",
            ]
        )
    )

    st.markdown(f"### {tr(lang, 'model_random_forest')}")
    st.markdown(tr(lang, "theory_model_random_forest"))
    st.markdown(
        "\n".join(
            [
                f"- `{tr(lang, 'rf_n_estimators')}`: {tr(lang, 'rf_n_estimators_help')}",
                f"- `{tr(lang, 'rf_max_depth')}`: {tr(lang, 'rf_max_depth_help')}",
                f"- `{tr(lang, 'min_samples_leaf')}`: {tr(lang, 'min_samples_leaf_help')}",
                f"- `{tr(lang, 'min_samples_split')}`: {tr(lang, 'min_samples_split_help')}",
                f"- `{tr(lang, 'max_features')}`: {tr(lang, 'max_features_help')}",
                f"- `{tr(lang, 'max_leaf_nodes')}`: {tr(lang, 'max_leaf_nodes_help')}",
                f"- `{tr(lang, 'min_impurity_decrease')}`: {tr(lang, 'min_impurity_decrease_help')}",
                f"- `{tr(lang, 'rf_bootstrap')}`: {tr(lang, 'rf_bootstrap_help')}",
            ]
        )
    )

    st.markdown(f"### {tr(lang, 'model_xgboost')}")
    st.markdown(tr(lang, "theory_model_xgboost"))
    st.markdown(
        "\n".join(
            [
                f"- `{tr(lang, 'n_estimators')}`: {tr(lang, 'n_estimators_help')}",
                f"- `{tr(lang, 'xgb_max_depth')}`: {tr(lang, 'xgb_max_depth_help')}",
                f"- `{tr(lang, 'learning_rate')}`: {tr(lang, 'learning_rate_help')}",
                f"- `{tr(lang, 'subsample')}`: {tr(lang, 'subsample_help')}",
                f"- `{tr(lang, 'colsample_bytree')}`: {tr(lang, 'colsample_bytree_help')}",
                f"- `{tr(lang, 'min_child_weight')}`: {tr(lang, 'min_child_weight_help')}",
                f"- `{tr(lang, 'gamma')}`: {tr(lang, 'gamma_help')}",
                f"- `{tr(lang, 'reg_alpha')}`: {tr(lang, 'reg_alpha_help')}",
                f"- `{tr(lang, 'reg_lambda')}`: {tr(lang, 'reg_lambda_help')}",
                f"- `scale_pos_weight`: {tr(lang, 'spw_popover')}",
            ]
        )
    )

    st.markdown(f"### {tr(lang, 'model_catboost')}")
    st.markdown(tr(lang, "theory_model_catboost"))
    st.markdown(
        "\n".join(
            [
                f"- `{tr(lang, 'cat_iterations')}`: {tr(lang, 'cat_iterations_help')}",
                f"- `{tr(lang, 'cat_depth')}`: {tr(lang, 'cat_depth_help')}",
                f"- `{tr(lang, 'cat_learning_rate')}`: {tr(lang, 'cat_learning_rate_help')}",
                f"- `{tr(lang, 'cat_l2_leaf_reg')}`: {tr(lang, 'cat_l2_leaf_reg_help')}",
            ]
        )
    )

    st.markdown(f"### {tr(lang, 'class_weights')}")
    st.markdown(tr(lang, "class_weights_popover"))

    section_title(tr(lang, "header_threshold_sweep"))
    st.markdown(f"- {tr(lang, 't_min_help')}")
    st.markdown(f"- {tr(lang, 't_max_help')}")
    st.markdown(f"- {tr(lang, 't_step_help')}")
    st.markdown(f"- {tr(lang, 'slider_threshold_help')}")

    section_title(tr(lang, "sec_metrics_vs_t"))
    st.markdown(tr(lang, "sec_metrics_vs_t_info"))

    section_title(tr(lang, "sec_pick_threshold"))
    st.markdown(f"- {tr(lang, 'sec_pick_threshold_info')}")
    st.markdown(f"- {tr(lang, 'cm_popover')}")
    st.markdown(f"- {tr(lang, 'cr_popover')}")

    section_title(tr(lang, "sec_numeric"))
    st.markdown(tr(lang, "sec_numeric_info"))


def preprocess_features(
    df: pd.DataFrame,
    target_col: str,
    drop_cols: list[str],
    err: Callable[..., str],
) -> tuple[pd.DataFrame, pd.Series]:
    df = df.copy()
    drop_cols = [c for c in drop_cols if c in df.columns and c != target_col]
    df = df.drop(columns=drop_cols, errors="ignore")
    if target_col not in df.columns:
        raise ValueError(err("err_target_not_found", col=target_col))
    y = df[target_col]
    X = df.drop(columns=[target_col])
    obj_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    if obj_cols:
        X = pd.get_dummies(X, columns=obj_cols, drop_first=True)
    num = X.select_dtypes(include=[np.number])
    if num.shape[1] == 0:
        raise ValueError(err("err_no_numeric_features"))
    X = num.fillna(num.median(numeric_only=True))
    y = pd.to_numeric(y, errors="coerce")
    mask = y.notna()
    X, y = X.loc[mask], y.loc[mask]
    y = y.astype(int)
    return X, y


def build_class_weight(w0: float, w1: float) -> dict[int, float]:
    return {0: float(w0), 1: float(w1)}


def _logistic_solver_for_penalty(penalty: str) -> str:
    """Pick a solver compatible with the penalty (sklearn 1.2+)."""
    if penalty == "l1":
        return "saga"
    # l2 and none work with lbfgs
    return "lbfgs"


def _tree_max_features_ui(val: str) -> Any:
    if val == "all":
        return None
    return val


def make_model(
    kind: Literal["logistic", "tree", "random_forest", "xgboost", "catboost"],
    random_state: int,
    lr_max_iter: int,
    lr_C: float,
    lr_penalty: str,
    lr_fit_intercept: bool,
    lr_tol: float,
    cw0: float,
    cw1: float,
    tree_max_depth: int | None,
    tree_min_samples_leaf: int,
    tree_min_samples_split: int,
    tree_criterion: str,
    tree_max_features_ui: str,
    tree_max_leaf_nodes: int,
    tree_min_impurity_decrease: float,
    rf_n_estimators: int,
    rf_max_depth: int | None,
    rf_min_samples_leaf: int,
    rf_min_samples_split: int,
    rf_max_features_ui: str,
    rf_max_leaf_nodes: int,
    rf_min_impurity_decrease: float,
    rf_bootstrap: bool,
    xgb_n_estimators: int,
    xgb_max_depth: int,
    xgb_learning_rate: float,
    xgb_scale_pos_weight: float,
    xgb_subsample: float,
    xgb_colsample_bytree: float,
    xgb_min_child_weight: float,
    xgb_gamma: float,
    xgb_reg_alpha: float,
    xgb_reg_lambda: float,
    cat_iterations: int,
    cat_depth: int,
    cat_learning_rate: float,
    cat_l2_leaf_reg: float,
) -> Any:
    cw = build_class_weight(cw0, cw1)
    if kind == "logistic":
        pen = str(lr_penalty).lower()
        pen_kw: str = "none" if pen == "none" else pen
        solver_pen = "l2" if pen_kw == "none" else pen_kw
        return LogisticRegression(
            max_iter=int(lr_max_iter),
            C=float(lr_C),
            penalty=pen_kw,
            solver=_logistic_solver_for_penalty(solver_pen),
            class_weight=cw,
            random_state=random_state,
            fit_intercept=bool(lr_fit_intercept),
            tol=float(lr_tol),
        )
    if kind == "tree":
        mlf = (
            None
            if int(tree_max_leaf_nodes) <= 0
            else int(tree_max_leaf_nodes)
        )
        crit = str(tree_criterion).lower()
        return DecisionTreeClassifier(
            class_weight=cw,
            random_state=random_state,
            max_depth=None if tree_max_depth == 0 else int(tree_max_depth),
            min_samples_leaf=int(tree_min_samples_leaf),
            min_samples_split=int(tree_min_samples_split),
            criterion=crit,
            max_features=_tree_max_features_ui(tree_max_features_ui),
            max_leaf_nodes=mlf,
            min_impurity_decrease=float(tree_min_impurity_decrease),
        )
    if kind == "random_forest":
        rf_mlf = (
            None
            if int(rf_max_leaf_nodes) <= 0
            else int(rf_max_leaf_nodes)
        )
        return RandomForestClassifier(
            n_estimators=int(rf_n_estimators),
            class_weight=cw,
            random_state=random_state,
            max_depth=None if rf_max_depth == 0 else int(rf_max_depth),
            min_samples_leaf=int(rf_min_samples_leaf),
            min_samples_split=int(rf_min_samples_split),
            max_features=_tree_max_features_ui(rf_max_features_ui),
            max_leaf_nodes=rf_mlf,
            min_impurity_decrease=float(rf_min_impurity_decrease),
            bootstrap=bool(rf_bootstrap),
            n_jobs=-1,
        )
    if kind == "catboost":
        return CatBoostClassifier(
            random_seed=random_state,
            iterations=int(cat_iterations),
            depth=int(cat_depth),
            learning_rate=float(cat_learning_rate),
            l2_leaf_reg=float(cat_l2_leaf_reg),
            loss_function="Logloss",
            eval_metric="Logloss",
            class_weights=[float(cw0), float(cw1)],
            verbose=False,
        )
    return XGBClassifier(
        random_state=random_state,
        n_estimators=int(xgb_n_estimators),
        max_depth=int(xgb_max_depth),
        learning_rate=float(xgb_learning_rate),
        scale_pos_weight=float(xgb_scale_pos_weight),
        subsample=float(xgb_subsample),
        colsample_bytree=float(xgb_colsample_bytree),
        min_child_weight=float(xgb_min_child_weight),
        gamma=float(xgb_gamma),
        reg_alpha=float(xgb_reg_alpha),
        reg_lambda=float(xgb_reg_lambda),
        eval_metric="logloss",
        verbosity=0,
    )


def threshold_curve(
    model: Any,
    X: pd.DataFrame,
    y_true: np.ndarray,
    thresholds: np.ndarray,
    eval_msg: str,
) -> pd.DataFrame:
    if len(np.unique(y_true)) < 2:
        raise ValueError(eval_msg)
    proba = model.predict_proba(X)[:, 1]
    rows = []
    for t in thresholds:
        y_pred = (proba > float(t)).astype(int)
        rows.append(
            {
                "threshold": float(t),
                "accuracy": accuracy_score(y_true, y_pred),
                "precision": precision_score(
                    y_true, y_pred, zero_division=ZERO_DIV
                ),
                "recall": recall_score(y_true, y_pred, zero_division=ZERO_DIV),
                "f1": f1_score(y_true, y_pred, zero_division=ZERO_DIV),
            }
        )
    return pd.DataFrame(rows)


def metrics_vs_threshold_chart(curve: pd.DataFrame, lang: str) -> alt.Chart:
    """Altair chart: F1 as a thicker orange line with large markers on top."""
    df_long = curve.melt(
        id_vars=["threshold"],
        value_vars=["accuracy", "precision", "recall", "f1"],
        var_name="metric",
        value_name="value",
    )
    label_map = {
        "accuracy": tr(lang, "col_accuracy"),
        "precision": tr(lang, "col_precision"),
        "recall": tr(lang, "col_recall"),
        "f1": tr(lang, "col_f1"),
    }
    df_long["series"] = df_long["metric"].map(label_map)
    order = [label_map[m] for m in ("accuracy", "precision", "recall", "f1")]
    colors = ["#95a5a6", "#3498db", "#8e7cc3", "#e65100"]
    f1_only = df_long[df_long["metric"] == "f1"]

    base_enc = dict(
        x=alt.X("threshold:Q", title=tr(lang, "col_threshold")),
        y=alt.Y("value:Q", title="", scale=alt.Scale(domain=[0, 1])),
    )
    lines = (
        alt.Chart(df_long)
        .transform_calculate(
            line_w="datum.metric == 'f1' ? 4.5 : 1.6",
        )
        .mark_line(interpolate="monotone", strokeCap="round")
        .encode(
            **base_enc,
            color=alt.Color(
                "series:N",
                title=tr(lang, "chart_legend_series"),
                sort=order,
                scale=alt.Scale(domain=order, range=colors),
            ),
            strokeWidth=alt.StrokeWidth("line_w:Q", legend=None),
            tooltip=[
                alt.Tooltip("threshold:Q", format=".4f", title=tr(lang, "col_threshold")),
                alt.Tooltip("series:N", title=tr(lang, "chart_legend_series")),
                alt.Tooltip("value:Q", format=".4f", title=tr(lang, "chart_tooltip_value")),
            ],
        )
    )
    f1_markers = (
        alt.Chart(f1_only)
        .mark_circle(
            size=120,
            fill="#ff6f00",
            stroke="#ffffff",
            strokeWidth=2,
            opacity=0.95,
        )
        .encode(
            **base_enc,
            tooltip=[
                alt.Tooltip("threshold:Q", format=".4f", title=tr(lang, "col_threshold")),
                alt.Tooltip("series:N", title=tr(lang, "chart_legend_series")),
                alt.Tooltip("value:Q", format=".4f", title=tr(lang, "chart_tooltip_value")),
            ],
        )
    )
    return (
        (lines + f1_markers)
        .properties(height=420)
        .configure_view(strokeWidth=0)
        .configure_axis(grid=True)
    )


def confusion_matrix_chart(cm: np.ndarray, lang: str) -> alt.Chart | None:
    """2×2 heatmap with TN/FP/FN/TP labels and counts; None if shape is not 2×2."""
    if cm.shape != (2, 2):
        return None
    tags: dict[tuple[int, int], str] = {
        (0, 0): tr(lang, "cm_tag_tn"),
        (0, 1): tr(lang, "cm_tag_fp"),
        (1, 0): tr(lang, "cm_tag_fn"),
        (1, 1): tr(lang, "cm_tag_tp"),
    }
    rows: list[dict[str, Any]] = []
    for i in range(2):
        for j in range(2):
            rows.append(
                {
                    "yt": str(i),
                    "yp": str(j),
                    "n": int(cm[i, j]),
                    "tag": tags[(i, j)],
                }
            )
    df = pd.DataFrame(rows)
    base = alt.Chart(df).encode(
        x=alt.X(
            "yp:N",
            title=tr(lang, "cm_axis_pred"),
            scale=alt.Scale(domain=["0", "1"]),
            axis=alt.Axis(tickMinStep=1, labelFontSize=13, titlePadding=8),
        ),
        y=alt.Y(
            "yt:N",
            title=tr(lang, "cm_axis_true"),
            sort=["1", "0"],
            scale=alt.Scale(domain=["1", "0"]),
            axis=alt.Axis(tickMinStep=1, labelFontSize=13, titlePadding=8),
        ),
    )
    heat = base.mark_rect(stroke="#ffffff", strokeWidth=2).encode(
        color=alt.Color(
            "n:Q",
            title=tr(lang, "cm_legend_count"),
            scale=alt.Scale(scheme="tealblues", reverse=False),
            legend=alt.Legend(orient="right", tickMinStep=1),
        ),
        tooltip=[
            alt.Tooltip("tag:N", title=tr(lang, "cm_tooltip_type")),
            alt.Tooltip("yt:N", title=tr(lang, "cm_axis_true")),
            alt.Tooltip("yp:N", title=tr(lang, "cm_axis_pred")),
            alt.Tooltip("n:Q", format="d", title=tr(lang, "chart_tooltip_value")),
        ],
    )
    lab = (
        alt.Chart(df)
        .transform_calculate(
            cell_label="datum.tag + ' · ' + datum.n",
        )
        .mark_text(
            baseline="middle",
            align="center",
            fontSize=15,
            fontWeight="bold",
            lineHeight=18,
        )
        .encode(
            x=alt.X("yp:N", scale=alt.Scale(domain=["0", "1"])),
            y=alt.Y(
                "yt:N",
                sort=["1", "0"],
                scale=alt.Scale(domain=["1", "0"]),
            ),
            text=alt.Text("cell_label:N"),
            color=alt.condition(
                alt.datum.n > 0,
                alt.value("#0d3b4a"),
                alt.value("#94a3b8"),
            ),
        )
    )
    return (
        (heat + lab)
        .properties(
            width=320,
            height=260,
            padding={"top": 12, "right": 12, "bottom": 12, "left": 12},
        )
        .configure_view(strokeWidth=0)
        .configure_axis(grid=False)
    )


def best_threshold_on_curve(
    curve: pd.DataFrame, metric_col: str
) -> tuple[float, float]:
    """Threshold and metric value where metric is highest (first max if tied)."""
    if curve.empty or metric_col not in curve.columns:
        return float("nan"), float("nan")
    idx = int(curve[metric_col].to_numpy().argmax())
    return float(curve["threshold"].iloc[idx]), float(curve[metric_col].iloc[idx])


def _threshold_grid(t_min: float, t_max: float, t_step: float) -> np.ndarray:
    """Same construction as the main app; fallback if min >= max."""
    if t_min >= t_max:
        arr = np.arange(0.05, 1.0, 0.05)
    else:
        arr = np.arange(float(t_min), float(t_max) + float(t_step) / 2, float(t_step))
    arr = np.clip(np.round(arr, 4), 0.0, 1.0)
    u = np.unique(arr)
    if len(u) == 0:
        return np.array([0.5], dtype=float)
    return u


def _max_f1_on_test_grid(
    model: Any,
    model_kind: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    thresholds: np.ndarray,
    eval_msg: str,
) -> float:
    try:
        if model_kind in ("xgboost", "catboost"):
            model.fit(X_train, y_train)
        else:
            model.fit(X_train, np.ravel(y_train.values))
        curve = threshold_curve(
            model,
            X_test,
            np.ravel(y_test.values),
            thresholds,
            eval_msg,
        )
        return float(curve["f1"].max())
    except Exception:
        return -1.0


def search_best_f1_hyperparams(
    model_kind: Literal["logistic", "tree", "random_forest", "xgboost", "catboost"],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    thresholds: np.ndarray,
    random_state: int,
    cw0: float,
    cw1: float,
    base_spw: float,
    eval_msg: str,
    *,
    lr_max_iter: int,
    lr_fit_intercept: bool,
    lr_tol: float,
    tree_criterion: str,
    tree_max_features_ui: str,
    tree_max_leaf_nodes: int,
    tree_min_impurity_decrease: float,
    rf_max_features_ui: str,
    rf_max_leaf_nodes: int,
    rf_min_impurity_decrease: float,
    rf_bootstrap: bool,
    xgb_subsample: float,
    xgb_colsample_bytree: float,
    xgb_min_child_weight: float,
    xgb_gamma: float,
    xgb_reg_alpha: float,
    xgb_reg_lambda: float,
    cat_l2_leaf_reg: float,
    progress_callback: Callable[[int, int], None] | None = None,
) -> tuple[float, dict[str, Any]]:
    """
    Grid search over a small hyperparameter grid; maximize best F1 on the
    threshold sweep (test set). Returns (best_f1, session_state key updates).
    """
    best_f1 = -1.0
    best_updates: dict[str, Any] = {}

    base_kwargs: dict[str, Any] = {
        "random_state": int(random_state),
        "lr_max_iter": int(lr_max_iter),
        "lr_C": 1.0,
        "lr_penalty": "l2",
        "lr_fit_intercept": bool(lr_fit_intercept),
        "lr_tol": float(lr_tol),
        "cw0": float(cw0),
        "cw1": float(cw1),
        "tree_max_depth": 5,
        "tree_min_samples_leaf": 10,
        "tree_min_samples_split": 2,
        "tree_criterion": str(tree_criterion),
        "tree_max_features_ui": str(tree_max_features_ui),
        "tree_max_leaf_nodes": int(tree_max_leaf_nodes),
        "tree_min_impurity_decrease": float(tree_min_impurity_decrease),
        "rf_n_estimators": 200,
        "rf_max_depth": 8,
        "rf_min_samples_leaf": 1,
        "rf_min_samples_split": 2,
        "rf_max_features_ui": str(rf_max_features_ui),
        "rf_max_leaf_nodes": int(rf_max_leaf_nodes),
        "rf_min_impurity_decrease": float(rf_min_impurity_decrease),
        "rf_bootstrap": bool(rf_bootstrap),
        "xgb_n_estimators": 100,
        "xgb_max_depth": 6,
        "xgb_learning_rate": 0.1,
        "xgb_scale_pos_weight": float(base_spw),
        "xgb_subsample": float(xgb_subsample),
        "xgb_colsample_bytree": float(xgb_colsample_bytree),
        "xgb_min_child_weight": float(xgb_min_child_weight),
        "xgb_gamma": float(xgb_gamma),
        "xgb_reg_alpha": float(xgb_reg_alpha),
        "xgb_reg_lambda": float(xgb_reg_lambda),
        "cat_iterations": 400,
        "cat_depth": 6,
        "cat_learning_rate": 0.1,
        "cat_l2_leaf_reg": float(cat_l2_leaf_reg),
    }

    if model_kind == "logistic":
        grid_c = [0.01, 0.07, 0.3, 1.0, 3.0, 10.0]
        grid_pen = ["l2", "l1"]
        combos = list(itertools.product(grid_c, grid_pen))
        total = len(combos)
        for i, (C, pen) in enumerate(combos):
            m = make_model(kind="logistic", **base_kwargs, lr_C=float(C), lr_penalty=str(pen))
            f1m = _max_f1_on_test_grid(
                m, "logistic", X_train, y_train, X_test, y_test, thresholds, eval_msg
            )
            if f1m > best_f1:
                best_f1 = f1m
                best_updates = {"p_lr_C": float(C), "p_lr_penalty": str(pen)}
            if progress_callback is not None:
                progress_callback(i + 1, total)
    elif model_kind == "tree":
        grid_depth = [0, 4, 8, 12, 20]
        grid_leaf = [1, 5, 10, 20, 40]
        grid_split = [2, 6, 14]
        combos = list(itertools.product(grid_depth, grid_leaf, grid_split))
        total = len(combos)
        for i, (td, lf, sp) in enumerate(combos):
            m = make_model(
                kind="tree",
                **base_kwargs,
                tree_max_depth=int(td),
                tree_min_samples_leaf=int(lf),
                tree_min_samples_split=int(sp),
            )
            f1m = _max_f1_on_test_grid(
                m, "tree", X_train, y_train, X_test, y_test, thresholds, eval_msg
            )
            if f1m > best_f1:
                best_f1 = f1m
                best_updates = {
                    "p_tree_max_depth": int(td),
                    "p_tree_min_samples_leaf": int(lf),
                    "p_tree_min_samples_split": int(sp),
                }
            if progress_callback is not None:
                progress_callback(i + 1, total)
    elif model_kind == "random_forest":
        grid_ne = [100, 200, 400]
        grid_depth = [0, 6, 12]
        grid_leaf = [1, 5, 10]
        combos = list(itertools.product(grid_ne, grid_depth, grid_leaf))
        total = len(combos)
        for i, (ne, md, lf) in enumerate(combos):
            m = make_model(
                kind="random_forest",
                **base_kwargs,
                rf_n_estimators=int(ne),
                rf_max_depth=int(md),
                rf_min_samples_leaf=int(lf),
                rf_min_samples_split=2,
            )
            f1m = _max_f1_on_test_grid(
                m, "random_forest", X_train, y_train, X_test, y_test, thresholds, eval_msg
            )
            if f1m > best_f1:
                best_f1 = f1m
                best_updates = {
                    "p_rf_n_estimators": int(ne),
                    "p_rf_max_depth": int(md),
                    "p_rf_min_samples_leaf": int(lf),
                }
            if progress_callback is not None:
                progress_callback(i + 1, total)
    else:
        if model_kind == "xgboost":
            grid_ne = [50, 100, 200]
            grid_md = [3, 6, 9]
            grid_lr = [0.05, 0.1, 0.2]
            combos = list(itertools.product(grid_ne, grid_md, grid_lr))
            total = len(combos)
            for i, (ne, md, lr) in enumerate(combos):
                m = make_model(
                    kind="xgboost",
                    **base_kwargs,
                    xgb_n_estimators=int(ne),
                    xgb_max_depth=int(md),
                    xgb_learning_rate=float(lr),
                    xgb_scale_pos_weight=float(base_spw),
                )
                f1m = _max_f1_on_test_grid(
                    m, "xgboost", X_train, y_train, X_test, y_test, thresholds, eval_msg
                )
                if f1m > best_f1:
                    best_f1 = f1m
                    best_updates = {
                        "p_xgb_n_estimators": int(ne),
                        "p_xgb_max_depth": int(md),
                        "p_xgb_learning_rate": float(lr),
                    }
                if progress_callback is not None:
                    progress_callback(i + 1, total)
        else:
            grid_it = [200, 400, 600]
            grid_depth = [4, 6, 8]
            grid_lr = [0.03, 0.1, 0.2]
            combos = list(itertools.product(grid_it, grid_depth, grid_lr))
            total = len(combos)
            for i, (it, depth, lr) in enumerate(combos):
                m = make_model(
                    kind="catboost",
                    **base_kwargs,
                    cat_iterations=int(it),
                    cat_depth=int(depth),
                    cat_learning_rate=float(lr),
                    cat_l2_leaf_reg=float(cat_l2_leaf_reg),
                )
                f1m = _max_f1_on_test_grid(
                    m, "catboost", X_train, y_train, X_test, y_test, thresholds, eval_msg
                )
                if f1m > best_f1:
                    best_f1 = f1m
                    best_updates = {
                        "p_cat_iterations": int(it),
                        "p_cat_depth": int(depth),
                        "p_cat_learning_rate": float(lr),
                    }
                if progress_callback is not None:
                    progress_callback(i + 1, total)

    return best_f1, best_updates


def main() -> None:
    st.session_state.setdefault("app_lang", "en")
    st.set_page_config(
        page_title=tr(st.session_state.app_lang, "page_title"),
        layout="wide",
    )

    # Apply F1 grid-search results before any widget with keys p_* is created (same-run writes
    # to those keys after instantiation raise StreamlitAPIException).
    if "_f1_pending_param_patch" in st.session_state:
        _patch = st.session_state.pop("_f1_pending_param_patch")
        for _pk, _pv in _patch.items():
            st.session_state[_pk] = _pv

    with st.sidebar:
        st.selectbox(
            "Language / Język",
            ["en", "pl"],
            index=["en", "pl"].index(st.session_state.app_lang)
            if st.session_state.app_lang in ("en", "pl")
            else 0,
            format_func=lambda c: MESSAGES[c]["lang_en"] if c == "en" else MESSAGES[c]["lang_pl"],
            key="app_lang",
        )
        lang = st.session_state.app_lang
        page = st.radio(
            tr(lang, "nav_page"),
            ["workbench", "theory"],
            format_func=lambda x: {
                "workbench": tr(lang, "nav_workbench"),
                "theory": tr(lang, "nav_theory"),
            }[x],
            key="app_page",
        )

        st.caption(tr(lang, "ux_sidebar_hint"))
        st.divider()

        st.header(tr(lang, "header_data"))
        uploaded = st.file_uploader(
            tr(lang, "csv_file"),
            type=["csv"],
            help=tr(lang, "csv_help"),
        )
        use_sample = st.checkbox(
            tr(lang, "use_sample"),
            value=uploaded is None,
            help=tr(lang, "use_sample_help"),
        )
        target_col = st.text_input(
            tr(lang, "target_col"),
            value="default",
            help=tr(lang, "target_help"),
        )
        id_drop = st.text_input(
            tr(lang, "id_drop"),
            value="client_id",
            help=tr(lang, "id_drop_help"),
        )
        test_size = st.slider(
            tr(lang, "test_size"),
            0.1,
            0.5,
            0.3,
            0.05,
            help=tr(lang, "test_size_help"),
        )
        random_state = st.number_input(
            tr(lang, "random_seed"),
            value=123,
            step=1,
            help=tr(lang, "random_seed_help"),
        )

        st.divider()
        st.header(tr(lang, "header_model"))
        model_kind = st.selectbox(
            tr(lang, "model"),
            ["logistic", "tree", "random_forest", "xgboost", "catboost"],
            format_func=lambda x: {
                "logistic": tr(lang, "model_logistic"),
                "tree": tr(lang, "model_tree"),
                "random_forest": tr(lang, "model_random_forest"),
                "xgboost": tr(lang, "model_xgboost"),
                "catboost": tr(lang, "model_catboost"),
            }[x],
            help=tr(lang, "model_help"),
            key="p_model_kind",
        )

        st.subheader(tr(lang, "class_weights"))
        cw0 = st.number_input(
            tr(lang, "weight_0"),
            min_value=0.01,
            value=1.0,
            step=0.1,
            help=tr(lang, "weight_0_help"),
        )
        cw1 = st.number_input(
            tr(lang, "weight_1"),
            min_value=0.01,
            value=4.0,
            step=0.1,
            help=tr(lang, "weight_1_help"),
        )

        lr_max_iter = 1000
        lr_C = 1.0
        lr_penalty = "l2"
        lr_fit_intercept = True
        lr_tol = 1e-4
        tree_max_depth = 5
        tree_min_samples_leaf = 10
        tree_min_samples_split = 2
        tree_criterion = "gini"
        tree_max_features_ui = "all"
        tree_max_leaf_nodes = 0
        tree_min_impurity_decrease = 0.0
        rf_n_estimators = 200
        rf_max_depth = 8
        rf_min_samples_leaf = 1
        rf_min_samples_split = 2
        rf_max_features_ui = "sqrt"
        rf_max_leaf_nodes = 0
        rf_min_impurity_decrease = 0.0
        rf_bootstrap = True
        xgb_n_estimators = 100
        xgb_max_depth = 6
        xgb_learning_rate = 0.1
        xgb_subsample = 1.0
        xgb_colsample_bytree = 1.0
        xgb_min_child_weight = 1.0
        xgb_gamma = 0.0
        xgb_reg_alpha = 0.0
        xgb_reg_lambda = 1.0
        cat_iterations = 400
        cat_depth = 6
        cat_learning_rate = 0.1
        cat_l2_leaf_reg = 3.0

        if model_kind == "logistic":
            lr_max_iter = st.number_input(
                tr(lang, "max_iter"),
                100,
                5000,
                1000,
                100,
                help=tr(lang, "max_iter_help"),
                key="p_lr_max_iter",
            )
            lr_C = st.number_input(
                tr(lang, "lr_c"),
                0.001,
                100.0,
                1.0,
                0.1,
                help=tr(lang, "lr_c_help"),
                key="p_lr_C",
            )
            lr_penalty = st.selectbox(
                tr(lang, "penalty"),
                ["l2", "l1", "none"],
                index=0,
                help=tr(lang, "penalty_help"),
                key="p_lr_penalty",
            )
            with st.expander(tr(lang, "ux_advanced"), expanded=False):
                lr_fit_intercept = st.checkbox(
                    tr(lang, "fit_intercept"),
                    value=True,
                    help=tr(lang, "fit_intercept_help"),
                    key="p_lr_fit_intercept",
                )
                lr_tol = st.number_input(
                    tr(lang, "tol"),
                    value=0.0001,
                    format="%.6f",
                    min_value=1e-9,
                    max_value=1.0,
                    step=1e-5,
                    help=tr(lang, "tol_help"),
                    key="p_lr_tol",
                )
        elif model_kind == "tree":
            tree_max_depth = st.number_input(
                tr(lang, "tree_max_depth"),
                0,
                50,
                5,
                1,
                help=tr(lang, "tree_max_depth_help"),
                key="p_tree_max_depth",
            )
            tree_min_samples_leaf = st.number_input(
                tr(lang, "min_samples_leaf"),
                1,
                200,
                10,
                1,
                help=tr(lang, "min_samples_leaf_help"),
                key="p_tree_min_samples_leaf",
            )
            tree_min_samples_split = st.number_input(
                tr(lang, "min_samples_split"),
                2,
                500,
                2,
                1,
                help=tr(lang, "min_samples_split_help"),
                key="p_tree_min_samples_split",
            )
            with st.expander(tr(lang, "ux_advanced"), expanded=False):
                tree_criterion = st.selectbox(
                    tr(lang, "criterion"),
                    ["gini", "entropy", "log_loss"],
                    index=0,
                    help=tr(lang, "criterion_help"),
                    key="p_tree_criterion",
                )
                tree_max_features_ui = st.selectbox(
                    tr(lang, "max_features"),
                    ["all", "sqrt", "log2"],
                    index=0,
                    help=tr(lang, "max_features_help"),
                    key="p_tree_max_features_ui",
                )
                tree_max_leaf_nodes = st.number_input(
                    tr(lang, "max_leaf_nodes"),
                    0,
                    500,
                    0,
                    1,
                    help=tr(lang, "max_leaf_nodes_help"),
                    key="p_tree_max_leaf_nodes",
                )
                tree_min_impurity_decrease = st.number_input(
                    tr(lang, "min_impurity_decrease"),
                    0.0,
                    1.0,
                    0.0,
                    0.001,
                    format="%.4f",
                    help=tr(lang, "min_impurity_decrease_help"),
                    key="p_tree_min_impurity_decrease",
                )
        elif model_kind == "xgboost":
            xgb_n_estimators = st.number_input(
                tr(lang, "n_estimators"),
                10,
                500,
                100,
                10,
                help=tr(lang, "n_estimators_help"),
                key="p_xgb_n_estimators",
            )
            xgb_max_depth = st.number_input(
                tr(lang, "xgb_max_depth"),
                1,
                20,
                6,
                1,
                help=tr(lang, "xgb_max_depth_help"),
                key="p_xgb_max_depth",
            )
            xgb_learning_rate = st.number_input(
                tr(lang, "learning_rate"),
                0.01,
                0.5,
                0.1,
                0.01,
                help=tr(lang, "learning_rate_help"),
                key="p_xgb_learning_rate",
            )
            with st.expander(tr(lang, "ux_advanced"), expanded=False):
                xgb_subsample = st.slider(
                    tr(lang, "subsample"),
                    0.1,
                    1.0,
                    1.0,
                    0.05,
                    help=tr(lang, "subsample_help"),
                    key="p_xgb_subsample",
                )
                xgb_colsample_bytree = st.slider(
                    tr(lang, "colsample_bytree"),
                    0.1,
                    1.0,
                    1.0,
                    0.05,
                    help=tr(lang, "colsample_bytree_help"),
                    key="p_xgb_colsample_bytree",
                )
                xgb_min_child_weight = st.number_input(
                    tr(lang, "min_child_weight"),
                    0.0,
                    30.0,
                    1.0,
                    0.5,
                    help=tr(lang, "min_child_weight_help"),
                    key="p_xgb_min_child_weight",
                )
                xgb_gamma = st.number_input(
                    tr(lang, "gamma"),
                    0.0,
                    10.0,
                    0.0,
                    0.1,
                    help=tr(lang, "gamma_help"),
                    key="p_xgb_gamma",
                )
                xgb_reg_alpha = st.number_input(
                    tr(lang, "reg_alpha"),
                    0.0,
                    10.0,
                    0.0,
                    0.1,
                    help=tr(lang, "reg_alpha_help"),
                    key="p_xgb_reg_alpha",
                )
                xgb_reg_lambda = st.number_input(
                    tr(lang, "reg_lambda"),
                    0.0,
                    20.0,
                    1.0,
                    0.5,
                    help=tr(lang, "reg_lambda_help"),
                    key="p_xgb_reg_lambda",
                )
                spw_c1, spw_c2 = st.columns([0.78, 0.22], gap="small")
                with spw_c1:
                    st.caption(tr(lang, "spw_caption"))
        elif model_kind == "random_forest":
            rf_n_estimators = st.number_input(
                tr(lang, "rf_n_estimators"),
                50,
                1000,
                200,
                50,
                help=tr(lang, "rf_n_estimators_help"),
                key="p_rf_n_estimators",
            )
            rf_max_depth = st.number_input(
                tr(lang, "rf_max_depth"),
                0,
                50,
                8,
                1,
                help=tr(lang, "rf_max_depth_help"),
                key="p_rf_max_depth",
            )
            rf_min_samples_leaf = st.number_input(
                tr(lang, "min_samples_leaf"),
                1,
                200,
                1,
                1,
                help=tr(lang, "min_samples_leaf_help"),
                key="p_rf_min_samples_leaf",
            )
            with st.expander(tr(lang, "ux_advanced"), expanded=False):
                rf_min_samples_split = st.number_input(
                    tr(lang, "min_samples_split"),
                    2,
                    500,
                    2,
                    1,
                    help=tr(lang, "min_samples_split_help"),
                    key="p_rf_min_samples_split",
                )
                rf_max_features_ui = st.selectbox(
                    tr(lang, "max_features"),
                    ["all", "sqrt", "log2"],
                    index=1,
                    help=tr(lang, "max_features_help"),
                    key="p_rf_max_features_ui",
                )
                rf_max_leaf_nodes = st.number_input(
                    tr(lang, "max_leaf_nodes"),
                    0,
                    500,
                    0,
                    1,
                    help=tr(lang, "max_leaf_nodes_help"),
                    key="p_rf_max_leaf_nodes",
                )
                rf_min_impurity_decrease = st.number_input(
                    tr(lang, "min_impurity_decrease"),
                    0.0,
                    1.0,
                    0.0,
                    0.001,
                    format="%.4f",
                    help=tr(lang, "min_impurity_decrease_help"),
                    key="p_rf_min_impurity_decrease",
                )
                rf_bootstrap = st.checkbox(
                    tr(lang, "rf_bootstrap"),
                    value=True,
                    help=tr(lang, "rf_bootstrap_help"),
                    key="p_rf_bootstrap",
                )
        else:
            cat_iterations = st.number_input(
                tr(lang, "cat_iterations"),
                50,
                2000,
                400,
                50,
                help=tr(lang, "cat_iterations_help"),
                key="p_cat_iterations",
            )
            cat_depth = st.number_input(
                tr(lang, "cat_depth"),
                2,
                12,
                6,
                1,
                help=tr(lang, "cat_depth_help"),
                key="p_cat_depth",
            )
            cat_learning_rate = st.number_input(
                tr(lang, "cat_learning_rate"),
                0.01,
                0.5,
                0.1,
                0.01,
                help=tr(lang, "cat_learning_rate_help"),
                key="p_cat_learning_rate",
            )
            with st.expander(tr(lang, "ux_advanced"), expanded=False):
                cat_l2_leaf_reg = st.number_input(
                    tr(lang, "cat_l2_leaf_reg"),
                    0.1,
                    50.0,
                    3.0,
                    0.1,
                    help=tr(lang, "cat_l2_leaf_reg_help"),
                    key="p_cat_l2_leaf_reg",
                )

        st.divider()
        st.subheader(tr(lang, "ux_f1_search_section"))
        st.caption(tr(lang, "f1_search_help"))
        if st.button(
            tr(lang, "f1_search_btn"),
            key="btn_f1_search",
            type="primary",
            use_container_width=True,
        ):
            st.session_state["_f1_search_requested"] = True
            st.rerun()

        st.divider()
        st.header(tr(lang, "header_threshold_sweep"))
        t_min = st.slider(
            tr(lang, "t_min"),
            0.0,
            0.95,
            0.05,
            0.05,
            help=tr(lang, "t_min_help"),
        )
        t_max = st.slider(
            tr(lang, "t_max"),
            0.05,
            1.0,
            0.95,
            0.05,
            help=tr(lang, "t_max_help"),
        )
        t_step = st.selectbox(
            tr(lang, "t_step"),
            [0.01, 0.02, 0.05],
            index=2,
            help=tr(lang, "t_step_help"),
        )

    lang = st.session_state.get("app_lang", "en")
    if st.session_state.get("app_page", "workbench") == "theory":
        render_theory_page(lang)
        return

    def err_msg(key: str, **kwargs: Any) -> str:
        return tr(lang, key, **kwargs)

    st.title(tr(lang, "title"))
    st.caption(tr(lang, "caption"))
    if "_f1_search_notice" in st.session_state:
        _nf1 = st.session_state.pop("_f1_search_notice")
        st.success(tr(lang, "f1_search_done", f1=float(_nf1)))
        st.toast(tr(lang, "ux_toast_f1", f1=float(_nf1)), icon="✅")

    drop_list = [c.strip() for c in id_drop.split(",") if c.strip()]

    try:
        if use_sample and uploaded is None:
            df = pd.read_csv("credit_data.csv")
        elif uploaded is not None:
            raw_csv = uploaded.getvalue()
            if len(raw_csv) > MAX_CSV_UPLOAD_BYTES:
                st.error(
                    tr(
                        lang,
                        "err_csv_too_large",
                        max_mb=MAX_CSV_UPLOAD_BYTES // (1024 * 1024),
                    )
                )
                return
            df = pd.read_csv(io.BytesIO(raw_csv))
        else:
            st.info(tr(lang, "info_upload"))
            return
    except FileNotFoundError:
        st.error(tr(lang, "err_no_credit_file"))
        return
    except Exception as e:
        st.error(f"{tr(lang, 'err_load_data')} {e}")
        return

    try:
        X, y = preprocess_features(df, target_col, drop_list, err_msg)
    except Exception as e:
        st.error(str(e))
        return

    _mk_label = {
        "logistic": tr(lang, "model_logistic"),
        "tree": tr(lang, "model_tree"),
        "random_forest": tr(lang, "model_random_forest"),
        "xgboost": tr(lang, "model_xgboost"),
        "catboost": tr(lang, "model_catboost"),
    }[model_kind]
    c_m1, c_m2, c_m3, c_m4 = st.columns(4)
    with c_m1:
        st.metric(
            tr(lang, "metric_rows"),
            len(X),
            help=tr(lang, "metric_rows_help"),
        )
    with c_m2:
        st.metric(
            tr(lang, "metric_features"),
            X.shape[1],
            help=tr(lang, "metric_features_help"),
        )
    with c_m3:
        st.metric(
            tr(lang, "metric_pos_rate"),
            f"{y.mean():.3f}",
            help=tr(lang, "metric_pos_rate_help"),
        )
    with c_m4:
        st.metric(
            tr(lang, "ux_model_active"),
            _mk_label,
            help=tr(lang, "ux_model_active_help"),
        )

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=float(test_size),
            random_state=int(random_state),
            stratify=y,
        )
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=float(test_size),
            random_state=int(random_state),
            stratify=None,
        )
        st.warning(tr(lang, "stratify_warning"))

    y_tr = np.ravel(y_train.values)
    n_neg = int((y_tr == 0).sum())
    n_pos = int((y_tr == 1).sum())
    base_spw = (n_neg / max(n_pos, 1)) * (float(cw1) / float(cw0))

    _thr_for_search = _threshold_grid(float(t_min), float(t_max), float(t_step))
    if st.session_state.pop("_f1_search_requested", False):
        em = tr(lang, "err_two_classes_eval")
        _prog = st.progress(0)

        def _on_prog(cur: int, tot: int) -> None:
            _prog.progress(min(cur / max(tot, 1), 1.0))

        with st.spinner(tr(lang, "f1_search_spinner")):
            bf1, upd = search_best_f1_hyperparams(
                model_kind,
                X_train,
                y_train,
                X_test,
                y_test,
                _thr_for_search,
                int(random_state),
                float(cw0),
                float(cw1),
                float(base_spw),
                em,
                lr_max_iter=int(lr_max_iter),
                lr_fit_intercept=bool(lr_fit_intercept),
                lr_tol=float(lr_tol),
                tree_criterion=str(tree_criterion),
                tree_max_features_ui=str(tree_max_features_ui),
                tree_max_leaf_nodes=int(tree_max_leaf_nodes),
                tree_min_impurity_decrease=float(tree_min_impurity_decrease),
                rf_max_features_ui=str(rf_max_features_ui),
                rf_max_leaf_nodes=int(rf_max_leaf_nodes),
                rf_min_impurity_decrease=float(rf_min_impurity_decrease),
                rf_bootstrap=bool(rf_bootstrap),
                xgb_subsample=float(xgb_subsample),
                xgb_colsample_bytree=float(xgb_colsample_bytree),
                xgb_min_child_weight=float(xgb_min_child_weight),
                xgb_gamma=float(xgb_gamma),
                xgb_reg_alpha=float(xgb_reg_alpha),
                xgb_reg_lambda=float(xgb_reg_lambda),
                cat_l2_leaf_reg=float(cat_l2_leaf_reg),
                progress_callback=_on_prog,
            )
        _prog.empty()
        if bf1 < 0.0 or not upd:
            st.error(tr(lang, "f1_search_fail"))
        else:
            st.session_state["_f1_pending_param_patch"] = upd
            st.session_state["_f1_search_notice"] = float(bf1)
            st.rerun()

    model = make_model(
        kind=model_kind,
        random_state=int(random_state),
        lr_max_iter=int(lr_max_iter),
        lr_C=float(lr_C),
        lr_penalty=str(lr_penalty),
        lr_fit_intercept=bool(lr_fit_intercept),
        lr_tol=float(lr_tol),
        cw0=float(cw0),
        cw1=float(cw1),
        tree_max_depth=tree_max_depth,
        tree_min_samples_leaf=int(tree_min_samples_leaf),
        tree_min_samples_split=int(tree_min_samples_split),
        tree_criterion=str(tree_criterion),
        tree_max_features_ui=str(tree_max_features_ui),
        tree_max_leaf_nodes=int(tree_max_leaf_nodes),
        tree_min_impurity_decrease=float(tree_min_impurity_decrease),
        rf_n_estimators=int(rf_n_estimators),
        rf_max_depth=rf_max_depth,
        rf_min_samples_leaf=int(rf_min_samples_leaf),
        rf_min_samples_split=int(rf_min_samples_split),
        rf_max_features_ui=str(rf_max_features_ui),
        rf_max_leaf_nodes=int(rf_max_leaf_nodes),
        rf_min_impurity_decrease=float(rf_min_impurity_decrease),
        rf_bootstrap=bool(rf_bootstrap),
        xgb_n_estimators=int(xgb_n_estimators),
        xgb_max_depth=int(xgb_max_depth),
        xgb_learning_rate=float(xgb_learning_rate),
        xgb_scale_pos_weight=float(base_spw if model_kind == "xgboost" else 1.0),
        xgb_subsample=float(xgb_subsample),
        xgb_colsample_bytree=float(xgb_colsample_bytree),
        xgb_min_child_weight=float(xgb_min_child_weight),
        xgb_gamma=float(xgb_gamma),
        xgb_reg_alpha=float(xgb_reg_alpha),
        xgb_reg_lambda=float(xgb_reg_lambda),
        cat_iterations=int(cat_iterations),
        cat_depth=int(cat_depth),
        cat_learning_rate=float(cat_learning_rate),
        cat_l2_leaf_reg=float(cat_l2_leaf_reg),
    )

    with st.spinner(tr(lang, "spinner_fit")):
        if model_kind in ("xgboost", "catboost"):
            model.fit(X_train, y_train)
        else:
            model.fit(X_train, np.ravel(y_train))

    if t_min >= t_max:
        st.error(tr(lang, "err_t_order"))
        return

    thresholds = np.arange(float(t_min), float(t_max) + t_step / 2, float(t_step))
    thresholds = np.clip(np.round(thresholds, 4), 0.0, 1.0)
    thresholds = np.unique(thresholds)

    try:
        curve = threshold_curve(
            model,
            X_test,
            np.ravel(y_test.values),
            thresholds,
            tr(lang, "err_two_classes_eval"),
        )
    except Exception as e:
        st.error(str(e))
        return

    y_te = np.ravel(y_test.values)
    proba_test = model.predict_proba(X_test)[:, 1]
    try:
        roc_auc = roc_auc_score(y_te, proba_test)
    except ValueError:
        roc_auc = float("nan")
    with st.container(border=True):
        st.metric(
            tr(lang, "metric_roc"),
            f"{roc_auc:.4f}" if np.isfinite(roc_auc) else tr(lang, "na"),
            help=tr(lang, "metric_roc_help"),
        )

        col_rename = {
            c: tr(lang, f"col_{c}")
            for c in ["threshold", "accuracy", "precision", "recall", "f1"]
        }
        section_title(tr(lang, "sec_metrics_vs_t"))
        st.altair_chart(
            metrics_vs_threshold_chart(curve, lang), use_container_width=True
        )

        t_acc, v_acc = best_threshold_on_curve(curve, "accuracy")
        t_pre, v_pre = best_threshold_on_curve(curve, "precision")
        t_rec, v_rec = best_threshold_on_curve(curve, "recall")
        t_f1, v_f1 = best_threshold_on_curve(curve, "f1")

        st.subheader(tr(lang, "best_grid_subheader"))
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric(
                tr(lang, "best_acc"),
                f"t = {t_acc:.4f}",
                delta=f"{tr(lang, 'delta_accuracy')} {v_acc:.4f}",
                help=tr(lang, "best_acc_help"),
            )
        with m2:
            st.metric(
                tr(lang, "best_precision"),
                f"t = {t_pre:.4f}",
                delta=f"{tr(lang, 'delta_precision')} {v_pre:.4f}",
                help=tr(lang, "best_precision_help"),
            )
        with m3:
            st.metric(
                tr(lang, "best_recall"),
                f"t = {t_rec:.4f}",
                delta=f"{tr(lang, 'delta_recall')} {v_rec:.4f}",
                help=tr(lang, "best_recall_help"),
            )
        with m4:
            st.metric(
                tr(lang, "best_f1"),
                f"t = {t_f1:.4f}",
                delta=f"{tr(lang, 'delta_f1')} {v_f1:.4f}",
                help=tr(lang, "best_f1_help"),
            )

        section_title(tr(lang, "sec_pick_threshold"))
        st.caption(tr(lang, "ux_threshold_snaps"))
        sb1, sb2, sb3, sb4 = st.columns(4)
        with sb1:
            if st.button(
                tr(lang, "ux_snap_f1"),
                key="snap_t_f1",
                use_container_width=True,
            ):
                st.session_state.inspect_threshold = float(t_f1)
                st.rerun()
        with sb2:
            if st.button(
                tr(lang, "ux_snap_recall"),
                key="snap_t_rec",
                use_container_width=True,
            ):
                st.session_state.inspect_threshold = float(t_rec)
                st.rerun()
        with sb3:
            if st.button(
                tr(lang, "ux_snap_precision"),
                key="snap_t_pre",
                use_container_width=True,
            ):
                st.session_state.inspect_threshold = float(t_pre)
                st.rerun()
        with sb4:
            if st.button(
                tr(lang, "ux_snap_mid"),
                key="snap_t_mid",
                use_container_width=True,
            ):
                st.session_state.inspect_threshold = float(
                    thresholds[len(thresholds) // 2]
                )
                st.rerun()

        _sl_min = float(thresholds.min())
        _sl_max = float(thresholds.max())
        _sl_mid = float(thresholds[len(thresholds) // 2])
        if "inspect_threshold" not in st.session_state:
            st.session_state.inspect_threshold = _sl_mid
        st.session_state.inspect_threshold = float(
            min(max(float(st.session_state.inspect_threshold), _sl_min), _sl_max)
        )
        st.slider(
            tr(lang, "slider_threshold"),
            _sl_min,
            _sl_max,
            key="inspect_threshold",
            step=float(t_step),
            help=tr(lang, "slider_threshold_help"),
        )
        sel = float(st.session_state.inspect_threshold)
        y_hat = (proba_test > sel).astype(int)

        c1, c2 = st.columns(2)
        with c1:
            st.text(tr(lang, "cm_label"))
            _cm = confusion_matrix(y_te, y_hat)
            _cm_chart = confusion_matrix_chart(_cm, lang)
            if _cm_chart is not None:
                st.altair_chart(_cm_chart, use_container_width=False)
                st.caption(tr(lang, "cm_visual_caption"))
            else:
                st.dataframe(
                    pd.DataFrame(_cm),
                    use_container_width=True,
                    hide_index=False,
                )
        with c2:
            st.text(tr(lang, "cr_label"))
            st.text(classification_report(y_te, y_hat, zero_division=ZERO_DIV))

        section_title(tr(lang, "sec_numeric"))
        st.dataframe(
            curve.round(4).rename(columns=col_rename),
            use_container_width=True,
            hide_index=True,
        )


if __name__ == "__main__":
    main()
