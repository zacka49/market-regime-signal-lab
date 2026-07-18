# Research Report: Market Regime Signal Lab

## Hypothesis

Daily equity returns are not iid: regime-dependent drift/volatility and
short-horizon autocorrelation (momentum or mean reversion) create weak but
potentially exploitable structure. A model that (a) engineers leak-free
features from lagged returns and rolling statistics, (b) infers the hidden
regime driving those dynamics, and (c) validates strictly out-of-sample
should be able to detect that structure where it exists, and — just as
importantly — should honestly report when it doesn't.

## Method

1. **Simulation** (`simulate.py`, `sde.py`): a discrete-time regime-switching
   process with AR(1) noise (baseline), plus a proper continuous-time SDE
   engine (GBM, Ornstein-Uhlenbeck, Heston, Merton jump-diffusion) used to
   validate discretization schemes against closed-form solutions.
2. **Regime detection** (`regime_detection.py`, `filtering.py`): a Gaussian
   HMM fit on returns, scored directly against the simulator's ground-truth
   regime labels — accuracy, confusion matrix, detection lag — rather than
   assumed to be sensible.
3. **Feature/alpha engineering** (`features.py`, `alphas.py`): lagged
   returns, rolling momentum/volatility/drawdown, plus a small alpha zoo
   (mean reversion, momentum, volatility-managed, regime-transition, and a
   placebo) each evaluated through a standard IC-based harness
   (`alpha_eval.py`).
4. **Validation** (`model.py`): expanding-window walk-forward with an
   optional purge/embargo gap, never training on future data
   (`tests/test_no_lookahead.py`).
5. **Backtest** (`backtest.py`, `sizing.py`): cost-aware, with binary ±1,
   Kelly-derived, and volatility-targeted position sizing.
6. **Statistical rigor** (`evaluation.py`): every headline number is
   reported next to a buy-and-hold and random-signal baseline, with
   stationary block-bootstrap confidence intervals (not iid — returns are
   serially dependent) and a deflated Sharpe ratio that accounts for every
   configuration actually tried, not a hypothetical one.
7. **Portfolio construction** (`portfolio.py`): a simulated multi-asset
   factor panel with a known true covariance, used to score PCA, Marchenko-
   Pastur denoising, and Ledoit-Wolf shrinkage against ground truth, and to
   demonstrate naive mean-variance "error maximization" directly.
8. **Real data** (`data.py`): the exact same pipeline, unchanged, run on
   five real ETFs.

Full derivations for every method above are in
[`mathematical_notes.md`](mathematical_notes.md), cross-linked to the code
that implements them.

## Results

### Regime detection recovers real structure

![Regime detection vs ground truth](../assets/figures/regime_detection.png)

A 3-state Gaussian HMM fit on returns alone recovers the simulator's true
regime with **68% accuracy** (chance = 33%) and a mean detection lag of
~11 trading days. Feeding these (leak-free, walk-forward-refit) regime
probabilities into the trading model gives a small AUC improvement over the
regime-agnostic baseline (0.524 → ~0.53); a separate classifier per
inferred regime gives a similar-sized improvement by a different route.
Neither is dramatic — the regime label is a noisy, lagged proxy for
information the model can partly infer already from realized volatility —
but the improvement is real and, unlike the underlying trading edge below,
survives the honest evaluation in the next section reasonably well.

### The synthetic strategy beats its baselines, but the effect is not statistically distinguishable from zero

![Strategy equity vs baselines](../assets/figures/equity_curves.png)

On synthetic data (which has genuine AR(1) and regime structure by
construction), the walk-forward strategy clears both a buy-and-hold and a
random-signal baseline (Sharpe 0.47 vs 0.20 vs -0.71). But its **90%
stationary block-bootstrap confidence interval for Sharpe is
[-0.04, +0.94]** — it includes zero. The deflated Sharpe ratio, accounting
for the 6 configurations actually compared during model selection, is
**0.59**: modestly more likely than not to reflect genuine skill, a long
way from the false confidence a single point-estimate Sharpe would imply.

### Real data: the signal does not survive contact with reality

![Cost sensitivity heatmap](../assets/figures/cost_sensitivity_heatmap.png)

Run unchanged on five real ETFs, the same pipeline's AUC sits at
**0.484–0.502** — statistically indistinguishable from a coin flip, and
below the synthetic result. Some configurations still show a positive
Sharpe (up to +0.86 pre-cost on SPY at the most permissive threshold), but
the cost/threshold sweep above shows why that's not a real edge: at the
loosest threshold the model is long almost every day and is mostly
re-deriving SPY's own positive drift; as the threshold tightens (trading
only on the model's most "confident" calls) Sharpe collapses to -0.37. A
model with genuine skill should get *more* confident on its best calls, not
less — this pattern is a symptom of noise, not signal. A momentum-signal
parameter-stability sweep (lookback × cost) tells the same story: negative
Sharpe almost everywhere, with the one positive cell vanishing under any
realistic transaction cost.

### Regularized covariance estimation measurably reduces portfolio risk

![PCA eigenvalue spectrum vs Marchenko-Pastur](../assets/figures/pca_scree.png)

On a simulated multi-asset panel with a known true covariance, PCA recovers
the true market-factor loadings almost exactly (cosine similarity > 0.99).
Marchenko-Pastur denoising and Ledoit-Wolf shrinkage both reduce
out-of-sample minimum-variance portfolio risk relative to the raw sample
covariance (averaged over 15 seeds). Most strikingly, feeding a
**deliberately pure-noise** expected-return estimate (a short-sample mean
of a process with true mean exactly zero) into naive mean-variance
optimization shows the raw covariance taking ~431x gross leverage and
realizing 0.76 true portfolio variance, versus ~244x leverage and 0.27
realized variance for the denoised covariance — a direct demonstration of
Michaud's (1989) "error maximization": an unregularized optimizer doesn't
filter out estimation noise, it actively seeks out whichever inputs are
most overestimated.

## What this project does and does not claim

- **Does not claim** a profitable trading strategy. The honest headline
  result is that the one place a real, cost-surviving directional edge was
  tested for (real ETF data) it was not found, and the synthetic result
  that came closest cannot be statistically distinguished from luck at 90%
  confidence.
- **Does claim**: a correctly-built research *process* — leak-free
  validation, ground-truth-scored regime detection, an alpha-evaluation
  harness that correctly identifies its own placebo signal as uninformative
  (and, along the way, exposed and fixed a real statistical pitfall in
  naive rolling-window t-stats — Section 3.2 of the mathematical notes),
  honest multiple-testing accounting, and portfolio-construction techniques
  validated against known ground truth.

## Limitations

- Daily bars only; no intraday microstructure, no borrow costs, no market
  impact beyond a flat per-turnover cost.
- Real-data results use five liquid, large-cap-adjacent US ETFs over one
  historical window; results would likely differ (and should be re-run,
  not assumed) for other assets, periods, or frequencies.
- The alpha zoo (Section 3) is intentionally small and simple; production
  research groups maintain much larger, more specialized signal libraries.
- Deflated Sharpe / block-bootstrap parameters (block length, number of
  trials counted) are reasonable choices, not the only defensible ones —
  see the corresponding docstrings for what was assumed and why.
- No portfolio-level cross-sectional trading strategy is backtested
  end-to-end on real multi-asset data; Section 6's portfolio-construction
  results use simulated assets specifically because they provide a known
  ground truth to validate estimators against.

## Next steps

- Extend the alpha zoo and re-run the full IC/decile/orthogonalization
  harness on real multi-asset data.
- Build a genuine cross-sectional long/short backtest on real ETFs/sectors,
  using the portfolio-construction tools from Section 6.
- Explore whether the modest regime-feature improvement (Section 3) is
  itself statistically significant once put through the same
  block-bootstrap treatment as the headline Sharpe.
