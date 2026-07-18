# Expansion Plan — Market Regime Signal Lab

> **Status: Phases 1–7 implemented.** See [`research_report.md`](research_report.md)
> for results and [`mathematical_notes.md`](mathematical_notes.md) for the
> derivations. Phase 9 (stretch) was intentionally left undone. This document
> is kept as the original roadmap / rationale, not a live TODO list.

Goal: turn this project from a clean demo into something that reads like real
quant research work, suitable for an entry-level Quantitative Researcher CV.
Each phase is independently shippable — the project should look finished after
every phase, not only at the end.

Guiding principles:

1. Hiring managers care less about whether the strategy "works" and more about
   whether the research process is honest — hypothesis, validation without
   leakage, statistical significance, cost sensitivity, clear communication.
2. Mathematical depth (stochastic calculus, linear algebra, financial maths)
   must appear as **working code plus written derivations you can defend at a
   whiteboard** — a `docs/mathematical_notes.md` with the derivations, and a
   module that implements each result. Anything you write down, you should be
   able to re-derive in an interview.

---

## Phase 1 — Engineering credibility (1–2 days)

Cheap wins that signal professional habits. Interviewers *will* open the repo.

- [ ] Convert to a proper package: `pyproject.toml` (replaces `requirements.txt`
      + the `sys.path` hack in `scripts/run_research.py`), installable with
      `pip install -e .`.
- [ ] Add `pytest` suite: unit tests for feature leakage (assert no feature at
      time *t* uses data from *t* or later), regime transition matrix rows sum
      to 1, backtest turnover/cost accounting, walk-forward split boundaries.
      Leakage tests are a talking point in interviews — name the test file
      `test_no_lookahead.py`.
- [ ] GitHub Actions CI: run tests + `ruff` lint + `mypy` on push. Badge in README.
- [ ] Pin a `seed` everywhere and add a smoke test that `run_research.py`
      reproduces the same metrics end-to-end (reproducibility is a core
      research value).

## Phase 2 — Stochastic-calculus simulator upgrade (3–5 days)

The current simulator is a discrete Markov chain over drift/vol with an AR(1)
noise term. Rebuild it on continuous-time foundations so the project
demonstrates real stochastic calculus, discretized properly.

- [ ] `sde.py`: a small SDE simulation engine with Euler–Maruyama and Milstein
      schemes. Implement and document each process:
      - Geometric Brownian Motion — baseline; derive the exact log-price
        solution via Itô's lemma and use it to unit-test the discretization
        (weak/strong convergence check as step size shrinks).
      - Ornstein–Uhlenbeck on log-price or spread — the canonical
        mean-reversion model and the theoretical basis for your reversal alpha.
      - Heston stochastic volatility — CIR variance process with correlated
        Brownian motions (Cholesky the 2×2 correlation); note the Feller
        condition and use full truncation to keep variance non-negative.
      - Merton jump-diffusion — compound Poisson jumps for crash regimes.
      - Regime-switching drift/vol (what you have now) reframed as a
        continuous-time Markov-modulated process.
- [ ] `estimation.py`: closed-form MLE for OU parameters (it's an AR(1) in
      disguise — derive the mapping), method-of-moments for jump intensity,
      and realized-variance estimation. Recover the true parameters from your
      own simulations and report estimator bias/variance vs sample size —
      a genuinely researchy experiment.
- [ ] Connect regime inference to filtering theory: implement the Hamilton
      filter for the regime-switching model (forward algorithm on Gaussian
      emissions) and write up its relationship to the HMM you fit in Phase 3.
- [ ] `docs/mathematical_notes.md` section: Itô's lemma statement + GBM
      derivation, OU solution and its stationary distribution, Euler–Maruyama
      vs Milstein error orders, Feller condition. Keep each derivation short
      enough to reproduce on a whiteboard.

## Phase 3 — Regime detection (3–5 days)

Right now regimes exist in the simulator but nothing *detects* them. The
simulation gives you ground truth to validate against — a genuinely nice
experimental design that almost no CV project can offer.

- [ ] Add `regime_detection.py`: fit a Gaussian HMM (`hmmlearn`) and/or a
      rolling Gaussian mixture on returns/volatility to infer the hidden state.
      Compare against your own Hamilton filter from Phase 2.
- [ ] Score the detector against ground truth: confusion matrix, regime
      classification accuracy, detection lag (how many days after a true
      regime switch does the model catch it?). Lead with this result.
- [ ] Regime-conditioned strategy: compare (a) the current regime-agnostic
      model, (b) a model with inferred regime probability as a feature,
      (c) separate models per inferred regime. Report which wins and why.
- [ ] Add the regime posterior probabilities to the Streamlit dashboard,
      shaded over the price path against the true regimes.

## Phase 4 — Alpha signal research framework (4–6 days)

Reframe "features" as **candidate alphas**, each a hypothesis with an economic
rationale, evaluated in a standardized harness. This is the day job of a QR —
make the repo look like that day job.

- [ ] `alphas.py`: a common signal interface (`signal_t = f(data up to t)`,
      output standardized to z-scores). Implement a small zoo, each with a
      one-line economic hypothesis in its docstring:
      - short-horizon mean reversion (justified by the OU maths in Phase 2)
      - time-series momentum (12–1 style, scaled down to daily horizon)
      - volatility-managed exposure (low-vol → lever up; cite vol clustering)
      - regime-transition signal (posterior regime-change probability from
        Phase 3)
      - a deliberately weak/placebo signal, to show the harness correctly
        rejects it.
- [ ] `alpha_eval.py`: the standard alpha report per signal —
      information coefficient (rank correlation with forward return) and its
      time series, IC t-stat, IC decay across horizons (1/5/10/20 days),
      signal autocorrelation → implied turnover, quantile/decile forward
      returns, and after-cost Sharpe at realistic cost assumptions.
- [ ] Alpha combination: equal-weight z-scores vs IC-weighted vs ridge
      stacking. Orthogonalize signals first (regress each on the others and
      keep residuals — Gram–Schmidt in finance clothing) and show how
      combination changes when signals are correlated.
- [ ] Report a signal-correlation matrix and discuss redundancy — knowing two
      alphas are the same trade is a core practitioner insight.
- [ ] `docs/mathematical_notes.md` section: IC → IR → expected Sharpe link
      (Grinold's fundamental law of active management, with its assumptions
      and why they break), and the algebra of combining correlated signals.

## Phase 5 — Statistical rigor (3–4 days)

This is what separates "quant researcher" from "person who ran a backtest."

- [ ] Baselines: buy-and-hold, always-long, and a random-signal Monte Carlo.
      Every reported metric should sit next to a baseline.
- [ ] Uncertainty on everything: stationary block bootstrap confidence
      intervals for Sharpe and total return; report `sharpe = 1.1 [0.4, 1.8]`,
      not a point estimate.
- [ ] Significance: test whether the strategy return distribution beats zero /
      beats baseline (bootstrap p-value). Be explicit when it doesn't.
- [ ] Multiple-testing honesty: you compared models, alphas and thresholds —
      apply a Deflated Sharpe Ratio and discuss selection bias in the README.
      Naming this problem unprompted is a strong interview signal.
- [ ] Upgrade validation: add an embargo gap between train and test in the
      walk-forward split (purged walk-forward); document why (serial
      correlation + overlapping labels).

## Phase 6 — Linear algebra depth: factor structure & portfolio construction (3–5 days)

Extend from one asset to a small cross-section (simulate correlated assets via
a factor model; later, real ETFs) so the linear algebra has something real to
chew on.

- [ ] Simulate N correlated assets from a factor model
      `r = B f + ε` (market + 1–2 style factors); you now control the true
      covariance, so every estimator below can be scored against truth.
- [ ] PCA on the return covariance: eigendecomposition, scree plot, interpret
      PC1 as the market factor; verify recovered loadings against the true `B`.
- [ ] Random matrix theory denoising: compare the empirical eigenvalue
      spectrum to the Marchenko–Pastur distribution, clip noise eigenvalues,
      and show the denoised covariance predicts out-of-sample portfolio risk
      better than the raw sample covariance.
- [ ] Shrinkage: implement Ledoit–Wolf (or use sklearn's and verify against
      your own implementation) and benchmark against sample + RMT-denoised
      covariance.
- [ ] Portfolio construction with the combined alpha from Phase 4:
      closed-form minimum-variance and mean-variance weights
      (`w ∝ Σ⁻¹ μ` — derive it with a Lagrangian), long/short
      dollar-neutral quantile portfolios, and show how covariance estimation
      error destroys naive mean-variance ("error maximization").
- [ ] `docs/mathematical_notes.md` section: eigendecomposition of covariance
      matrices and why they're PSD, the mean-variance Lagrangian derivation,
      Marchenko–Pastur statement, and the intuition for shrinkage as a
      bias-variance trade-off.

## Phase 7 — Real data + robustness (3–5 days)

Synthetic data is fine for methodology, but pair it with reality.

- [ ] Data module with a clean interface: `load_synthetic()` and
      `load_real(tickers)` (e.g. daily SPY/QQQ/sector ETFs via `yfinance`),
      cached to `data/` (gitignored). Everything downstream stays unchanged —
      that's the point of the abstraction.
- [ ] Run the full alpha harness and portfolio construction on 5–10 real
      ETFs. Expect signals to weaken or die — **that is the result**; write it
      up honestly ("the mean-reversion structure the model exploits in
      simulation is largely absent in daily index returns after costs").
- [ ] Cost sensitivity sweep: Sharpe as a function of transaction cost
      (0–10 bps) and entry threshold — a heatmap. Show the break-even cost.
- [ ] Position sizing as financial maths: derive the Kelly criterion for the
      strategy's return distribution, explain why full Kelly is reckless
      (estimation error, fat tails), and implement fractional-Kelly /
      vol-targeted sizing vs the current binary ±1. Compare drawdowns.
- [ ] Parameter stability: heatmap of Sharpe across (threshold × lookback)
      to show you're not sitting on an overfit spike.

## Phase 8 — Communication (2–3 days, do continuously)

Quant research is a writing job. This phase is what gets read in screening.

- [ ] `docs/mathematical_notes.md` — assembled from the sections above:
      Itô/GBM/OU derivations, Euler–Maruyama error orders, Hamilton filter,
      Grinold's law, mean-variance Lagrangian, Marchenko–Pastur, Kelly.
      Every equation must have a corresponding implementation in `src/` —
      cross-link both ways. This document *is* your interview prep.
- [ ] `docs/research_report.md` (or a rendered notebook): hypothesis → method
      → results → limitations → next steps, ~2 pages, with figures. Include
      negative results prominently.
- [ ] README rewrite around findings, not features: lead with 2–3 figures
      (equity curve vs baseline with CI band, regime detection vs truth,
      IC decay / cost-sensitivity heatmap) and one honest headline result.
- [ ] A limitations section: daily bars only, no borrow costs, no market
      impact, small cross-section, survivorship-free by construction, etc.
      Pre-empting the interviewer's objections is the best interview prep.

## Phase 9 — Stretch (optional, pick at most one)

- [ ] Experiment tracking: a simple `experiments/` log (config YAML + metrics
      JSON per run) — shows you think about research infrastructure without
      dragging in MLflow.
- [ ] Option-pricing tie-in: Black–Scholes via risk-neutral pricing on your
      GBM simulator, verified against Monte Carlo; implied vs realized vol as
      an extra alpha.
- [ ] A simple execution/microstructure touch: model slippage as a function of
      turnover rather than a flat cost.

---

## Suggested CV bullets once complete

- Built an end-to-end alpha research pipeline in Python: continuous-time
  market simulation (Heston, Merton jump-diffusion, Markov-modulated regimes
  via Euler–Maruyama), HMM/Hamilton-filter regime inference validated against
  ground truth, and purged walk-forward validation.
- Designed a standardized alpha evaluation harness (information coefficients,
  IC decay, turnover, after-cost Sharpe) over a library of economically
  motivated signals; combined signals via orthogonalization and ridge stacking.
- Applied covariance denoising (Marchenko–Pastur eigenvalue clipping,
  Ledoit–Wolf shrinkage) and closed-form mean-variance optimization to a
  simulated factor cross-section; quantified robustness with block-bootstrap
  confidence intervals and deflated Sharpe ratios, documenting negative
  results on real ETF data.

## Ordering rationale

Phases 1–5 make the project interview-ready (rigor > breadth); Phase 6 adds
the linear-algebra/portfolio story and Phase 7 the "does it survive reality"
story. If time is short, the minimum credible cut is: Phase 1, the OU + GBM
slice of Phase 2, regime detection (Phase 3), the alpha harness with 2–3
signals (Phase 4), baselines + bootstrap from Phase 5, and the two docs from
Phase 8. The mathematical notes are non-negotiable — they convert project work
into interview answers.
