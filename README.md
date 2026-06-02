# Market Regime Signal Lab

A compact quantitative research project for modelling noisy financial time-series data. The project simulates market data with hidden regimes, engineers predictive features, trains models with walk-forward validation, and evaluates a simple long/short strategy after transaction costs.

The goal is not to claim a profitable trading strategy. The goal is to show a research workflow: define a hypothesis, generate data, build features, validate honestly on time-ordered splits, and measure whether a signal survives costs and regime shifts.

## Why This Project Exists

Jane Street's Quantitative Research material emphasises noisy data, time series, feature engineering, model building, experiment design and careful communication. This project is designed to demonstrate those skills in a finance-relevant setting.

## Features

- Synthetic price and return generation with hidden volatility/trend regimes.
- Feature engineering from lagged returns, rolling volatility, momentum and drawdown.
- Walk-forward validation to avoid leakage from future data.
- Model comparison using logistic regression and gradient boosting.
- Transaction-cost-aware backtest with position sizing and risk metrics.
- Reproducible command-line research run.

## Project Structure

```text
market-regime-signal-lab/
  README.md
  requirements.txt
  scripts/
    run_research.py
  src/
    regime_signal_lab/
      __init__.py
      backtest.py
      features.py
      model.py
      simulate.py
```

## Quick Start

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python scripts/run_research.py
```

On macOS/Linux, use `source .venv/bin/activate` instead of the Windows activation command.

## Example Output

```text
Model comparison
logistic_regression auc=0.55 accuracy=0.53
gradient_boosting  auc=0.58 accuracy=0.55

Backtest
total_return=0.14 sharpe=1.10 max_drawdown=-0.08 turnover=0.42
```

Exact values vary by random seed and simulation parameters.

## Research Notes

- A high in-sample score is not enough. The script reports walk-forward validation and a simple out-of-sample backtest.
- Costs matter. A weak signal can look attractive before costs and disappear after spread/fee assumptions.
- Regimes matter. The data generating process changes through time, so the model has to work with instability rather than a clean stationary dataset.
