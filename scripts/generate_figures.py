"""Generate the headline figures referenced in the README and research
report. Static PNGs (not interactive charts) since they're embedded in
GitHub-rendered markdown. Palette is the colorblind-safe Okabe-Ito set;
the cost-sensitivity heatmap uses a diverging colormap centered at zero
(Sharpe has a genuine sign / polarity), everything else uses a fixed
categorical color per series (never cycled, never color-as-rank).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

from regime_signal_lab.backtest import run_backtest
from regime_signal_lab.data import load_real
from regime_signal_lab.evaluation import buy_and_hold_baseline, random_signal_baseline
from regime_signal_lab.features import build_features, feature_columns
from regime_signal_lab.model import candidate_models, choose_best, walk_forward_validate
from regime_signal_lab.portfolio import (
    marchenko_pastur_bounds,
    pca_eigendecomposition,
    simulate_factor_returns,
)
from regime_signal_lab.regime_detection import detect_regimes
from regime_signal_lab.simulate import simulate_market

FIG_DIR = Path(__file__).resolve().parents[1] / "assets" / "figures"

# Okabe-Ito colorblind-safe palette, fixed categorical order.
BLUE = "#0072B2"
ORANGE = "#E69F00"
VERMILLION = "#D55E00"
GREEN = "#009E73"
GRAY = "#7F7F7F"
INK = "#222222"

plt.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#B0B0B0",
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "axes.grid": True,
        "grid.color": "#E5E5E5",
        "grid.linewidth": 0.7,
        "axes.axisbelow": True,
    }
)


def figure_equity_curves() -> None:
    raw = simulate_market(n_days=4000, seed=7)
    frame = build_features(raw)
    features = feature_columns(frame)
    results = [
        walk_forward_validate(frame, features, name, model, min_train_size=800, test_size=125)
        for name, model in candidate_models().items()
    ]
    best = choose_best(results)
    strategy_frame, strategy_metrics = run_backtest(best.predictions)
    bh_frame, bh_metrics = buy_and_hold_baseline(best.predictions)
    rand_frame, rand_metrics = random_signal_baseline(best.predictions, seed=0)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(
        strategy_frame["date"], strategy_frame["equity"],
        color=BLUE, linewidth=2.0,
        label=f"Strategy (Sharpe {strategy_metrics['annualised_sharpe']:.2f})",
    )
    ax.plot(
        bh_frame["date"], bh_frame["equity"],
        color=ORANGE, linewidth=1.5, linestyle="--",
        label=f"Buy & hold (Sharpe {bh_metrics['annualised_sharpe']:.2f})",
    )
    ax.plot(
        rand_frame["date"], rand_frame["equity"],
        color=GRAY, linewidth=1.3, linestyle=":",
        label=f"Random signal (Sharpe {rand_metrics['annualised_sharpe']:.2f})",
    )
    ax.set_title("Strategy equity vs. baselines (synthetic data, out-of-sample)")
    ax.set_ylabel("Growth of $1")
    ax.set_xlabel("Date")
    ax.legend(loc="upper left", frameon=False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "equity_curves.png")
    plt.close(fig)


def figure_regime_detection() -> None:
    raw = simulate_market(n_days=4000, seed=7)
    detection = detect_regimes(raw["return"].to_numpy(), raw["regime"].to_numpy(), n_states=3, seed=0)

    fig, (ax_price, ax_post) = plt.subplots(
        2, 1, figsize=(9, 6), sharex=True, gridspec_kw={"height_ratios": [1.3, 1]}
    )

    regime_colors = {0: "#DCE6F1", 1: "#FCE8CC", 2: "#F6D6D6"}  # light calm / trend / stress bands
    regimes = raw["regime"].to_numpy()
    dates = raw["date"].to_numpy()
    change_points = np.where(np.diff(regimes) != 0)[0] + 1
    boundaries = [0, *change_points.tolist(), len(regimes)]
    for start_idx, end_idx in zip(boundaries[:-1], boundaries[1:]):
        span_color = regime_colors[regimes[start_idx]]
        ax_price.axvspan(dates[start_idx], dates[min(end_idx, len(dates) - 1)], color=span_color, zorder=0)

    ax_price.plot(dates, raw["price"], color=INK, linewidth=1.2)
    ax_price.set_ylabel("Price")
    fig.suptitle("True regimes (shaded) vs HMM-inferred posterior", x=0.02, ha="left", fontsize=12)
    ax_price.set_title(
        f"Detection accuracy {detection.accuracy:.0%} (chance = 33%)", fontsize=10, loc="left", color=GRAY
    )

    post_colors = [BLUE, ORANGE, VERMILLION]
    ax_post.stackplot(
        dates,
        detection.posterior.T,
        colors=post_colors,
        labels=[f"P(regime {i})" for i in range(detection.posterior.shape[1])],
        alpha=0.85,
    )
    ax_post.set_ylabel("Posterior probability")
    ax_post.set_xlabel("Date")
    ax_post.set_ylim(0, 1)
    ax_post.legend(loc="upper left", frameon=False, ncol=3, fontsize=8)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "regime_detection.png")
    plt.close(fig)


def figure_cost_sensitivity_heatmap() -> None:
    raw = load_real("SPY", start="2010-01-01")
    frame = build_features(raw)
    features = feature_columns(frame)
    model = candidate_models()["logistic_regression"]
    result = walk_forward_validate(frame, features, "SPY", model, min_train_size=800, test_size=125)

    thresholds = [0.50, 0.52, 0.55, 0.58, 0.60]
    costs_bps = [0.0, 2.0, 5.0, 10.0]
    sharpe_grid = np.zeros((len(thresholds), len(costs_bps)))
    for i, threshold in enumerate(thresholds):
        for j, cost_bps in enumerate(costs_bps):
            _, metrics = run_backtest(
                result.predictions, entry_threshold=threshold, transaction_cost=cost_bps / 10_000
            )
            sharpe_grid[i, j] = metrics["annualised_sharpe"]

    diverging = LinearSegmentedColormap.from_list("okabe_diverging", [VERMILLION, "#F5F5F5", BLUE])
    norm = TwoSlopeNorm(vcenter=0.0, vmin=sharpe_grid.min(), vmax=max(sharpe_grid.max(), 0.01))

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    im = ax.imshow(sharpe_grid, cmap=diverging, norm=norm, aspect="auto")
    ax.set_xticks(range(len(costs_bps)), [f"{c:.0f}" for c in costs_bps])
    ax.set_yticks(range(len(thresholds)), [f"{t:.2f}" for t in thresholds])
    ax.set_xlabel("Transaction cost (bps)")
    ax.set_ylabel("Entry probability threshold")
    ax.set_title("SPY: annualized Sharpe by cost & selectivity (real data)")
    ax.grid(False)

    for i in range(len(thresholds)):
        for j in range(len(costs_bps)):
            value = sharpe_grid[i, j]
            text_color = "white" if abs(value) > 0.25 else INK
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", color=text_color, fontsize=9)

    fig.colorbar(im, ax=ax, label="Annualized Sharpe")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "cost_sensitivity_heatmap.png")
    plt.close(fig)


def figure_pca_scree() -> None:
    n_assets, n_days = 40, 150
    _, _, true_cov = simulate_factor_returns(n_assets, n_days, seed=7)
    returns, _, _ = simulate_factor_returns(n_assets, n_days, seed=7)
    sample_cov = np.cov(returns, rowvar=False)
    eigenvalues, _ = pca_eigendecomposition(sample_cov)

    std = np.sqrt(np.diag(sample_cov))
    avg_variance = float(np.mean(std**2))
    _, lambda_plus_corr = marchenko_pastur_bounds(n_assets, n_days, variance=1.0)
    noise_threshold = lambda_plus_corr * avg_variance

    colors = [BLUE if val > noise_threshold else GRAY for val in eigenvalues]

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.bar(range(1, len(eigenvalues) + 1), eigenvalues, color=colors, width=0.7)
    ax.axhline(
        noise_threshold, color=VERMILLION, linewidth=1.5, linestyle="--",
        label="Marchenko-Pastur noise edge",
    )
    ax.set_xlabel("Principal component")
    ax.set_ylabel("Eigenvalue (variance)")
    ax.set_title(f"Eigenvalue spectrum: {n_assets} simulated assets, {n_days}-day sample")
    ax.legend(loc="upper right", frameon=False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "pca_scree.png")
    plt.close(fig)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    figure_equity_curves()
    print("wrote", FIG_DIR / "equity_curves.png")
    figure_regime_detection()
    print("wrote", FIG_DIR / "regime_detection.png")
    figure_cost_sensitivity_heatmap()
    print("wrote", FIG_DIR / "cost_sensitivity_heatmap.png")
    figure_pca_scree()
    print("wrote", FIG_DIR / "pca_scree.png")


if __name__ == "__main__":
    main()
