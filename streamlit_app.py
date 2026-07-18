from __future__ import annotations

import pandas as pd
import streamlit as st

from regime_signal_lab.backtest import run_backtest
from regime_signal_lab.features import build_features, feature_columns
from regime_signal_lab.model import candidate_models, choose_best, walk_forward_validate
from regime_signal_lab.regime_detection import detect_regimes
from regime_signal_lab.simulate import simulate_market


@st.cache_data(show_spinner=False)
def run_research(n_days: int, seed: int, entry_threshold: float, transaction_cost_bps: float) -> dict:
    raw = simulate_market(n_days=n_days, seed=seed)
    frame = build_features(raw)
    features = feature_columns(frame)
    results = [
        walk_forward_validate(frame, features, name, model)
        for name, model in candidate_models().items()
    ]
    best = choose_best(results)
    backtest, metrics = run_backtest(
        best.predictions,
        entry_threshold=entry_threshold,
        transaction_cost=transaction_cost_bps / 10_000,
    )
    # Whole-sample HMM fit for the regime-detection diagnostic tab. This uses
    # smoothed (non-causal) posteriors -- fine for illustrating detection
    # quality against ground truth, but NOT how regime features are built for
    # the trading model (see regime_detection.walk_forward_regime_posteriors,
    # which is causal and refit per walk-forward fold to avoid leakage).
    detection = detect_regimes(raw["return"].to_numpy(), raw["regime"].to_numpy(), n_states=3, seed=0)
    return {
        "raw": raw,
        "features": frame,
        "results": results,
        "best": best,
        "backtest": backtest,
        "metrics": metrics,
        "detection": detection,
    }


st.set_page_config(page_title="Market Regime Signal Lab", layout="wide")

st.title("Market Regime Signal Lab")
st.caption(
    "Noisy time-series simulation, feature engineering, walk-forward validation "
    "and cost-aware strategy evaluation."
)

with st.sidebar:
    st.header("Experiment")
    n_days = st.slider("Simulated trading days", 1200, 5000, 2500, step=100)
    seed = st.number_input("Random seed", min_value=1, max_value=9999, value=7, step=1)
    entry_threshold = st.slider("Entry probability threshold", 0.50, 0.65, 0.54, step=0.01)
    transaction_cost_bps = st.slider("Transaction cost, basis points", 0.0, 10.0, 2.0, step=0.5)

research = run_research(n_days, seed, entry_threshold, transaction_cost_bps)
raw = research["raw"]
results = research["results"]
best = research["best"]
backtest = research["backtest"]
metrics = research["metrics"]

metric_cols = st.columns(5)
metric_cols[0].metric("Selected model", best.name.replace("_", " ").title())
metric_cols[1].metric("AUC", f"{best.auc:.3f}")
metric_cols[2].metric("Accuracy", f"{best.accuracy:.3f}")
metric_cols[3].metric("Total return", f"{metrics['total_return']:.1%}")
metric_cols[4].metric("Sharpe", f"{metrics['annualised_sharpe']:.2f}")

tab_names = ["Market Data", "Regime Detection", "Model Validation", "Strategy", "Sample Data"]
tab_price, tab_regimes, tab_models, tab_strategy, tab_data = st.tabs(tab_names)

with tab_price:
    st.subheader("Simulated Market With Hidden Regimes")
    st.line_chart(raw.set_index("date")[["price"]])
    st.bar_chart(raw["regime"].value_counts().sort_index())

with tab_regimes:
    st.subheader("Gaussian HMM Regime Detection vs Ground Truth")
    detection = research["detection"]
    det_cols = st.columns(2)
    det_cols[0].metric(
        "Detection accuracy", f"{detection.accuracy:.1%}", help="Chance level for 3 states = 33.3%"
    )
    det_cols[1].metric("Mean detection lag", f"{detection.mean_detection_lag:.1f} trading days")

    st.write("True regime vs HMM-inferred regime (state labels aligned via Hungarian matching)")
    regime_compare = pd.DataFrame(
        {
            "true_regime": raw["regime"].to_numpy(),
            "inferred_regime": detection.aligned_states,
        },
        index=raw["date"],
    )
    st.line_chart(regime_compare)

    st.write("Filtered/smoothed posterior probability per regime")
    posterior_cols = [f"P(regime {i})" for i in range(detection.posterior.shape[1])]
    posterior_df = pd.DataFrame(detection.posterior, columns=posterior_cols, index=raw["date"])
    st.area_chart(posterior_df)

    st.write("Confusion matrix (rows = true regime, columns = inferred regime)")
    confusion_df = pd.DataFrame(
        detection.confusion,
        index=[f"true_{i}" for i in range(detection.confusion.shape[0])],
        columns=[f"pred_{i}" for i in range(detection.confusion.shape[1])],
    )
    st.dataframe(confusion_df, use_container_width=True)

with tab_models:
    st.subheader("Walk-Forward Model Comparison")
    model_rows = [
        {"model": result.name, "auc": result.auc, "accuracy": result.accuracy}
        for result in results
    ]
    st.dataframe(model_rows, use_container_width=True, hide_index=True)
    st.write("Predicted probability distribution for the selected model")
    st.bar_chart(best.predictions["probability"].round(2).value_counts().sort_index())

with tab_strategy:
    st.subheader("Cost-Aware Strategy Evaluation")
    st.line_chart(backtest.set_index("date")[["equity"]])
    st.area_chart(backtest.set_index("date")[["position"]])
    st.dataframe(
        {
            "metric": list(metrics.keys()),
            "value": [round(value, 4) for value in metrics.values()],
        },
        use_container_width=True,
        hide_index=True,
    )

with tab_data:
    st.subheader("Engineered Feature Sample")
    st.dataframe(research["features"].tail(100), use_container_width=True, hide_index=True)
