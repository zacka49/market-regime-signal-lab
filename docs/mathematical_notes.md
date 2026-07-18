# Mathematical Notes

Every derivation below has a corresponding implementation in `src/` and a
test in `tests/` that checks the code against the theory (not just against
itself). The rule for this document: nothing goes in here that can't be
reproduced on a whiteboard.

---

## 1. Stochastic calculus and the simulator (`sde.py`, `filtering.py`)

### 1.1 Ito's lemma and Geometric Brownian Motion

GBM is defined by the SDE

```
dS_t = mu S_t dt + sigma S_t dW_t
```

Let `f(S) = log S`. Ito's lemma states that for `f` twice differentiable,

```
df(S_t) = f'(S_t) dS_t + (1/2) f''(S_t) (dS_t)^2
```

using the Ito multiplication rules `dt*dt = 0`, `dt*dW_t = 0`, `dW_t*dW_t = dt`.
With `f'(S) = 1/S`, `f''(S) = -1/S^2`:

```
d(log S_t) = (1/S_t)(mu S_t dt + sigma S_t dW_t) - (1/2)(1/S_t^2)(sigma S_t)^2 dt
           = (mu - sigma^2/2) dt + sigma dW_t
```

This is now a driftless-in-the-Ito-correction SDE for `log S_t`, which
integrates directly (it's just Brownian motion with drift):

```
log S_t = log S_0 + (mu - sigma^2/2) t + sigma W_t
S_t = S_0 exp( (mu - sigma^2/2) t + sigma W_t )
```

Implemented as `gbm_exact()`. The `-sigma^2/2` term is the Ito correction --
forgetting it (i.e. treating `d(log S)` as if it obeyed ordinary calculus) is
the single most common stochastic-calculus mistake, and is exactly what
distinguishes the exact solution from a naive guess.

### 1.2 Discretization schemes and their error orders

For a general SDE `dX_t = a(X_t) dt + b(X_t) dW_t`:

**Euler-Maruyama:** `X_{n+1} = X_n + a(X_n) dt + b(X_n) dW_n`. Strong order
0.5, weak order 1.0 (strong error, i.e. `E|X_T^discrete - X_T^exact|`, shrinks
like `sqrt(dt)`).

**Milstein:** adds the first correction term from the Ito-Taylor expansion,

```
X_{n+1} = X_n + a(X_n) dt + b(X_n) dW_n + (1/2) b(X_n) b'(X_n) (dW_n^2 - dt)
```

Strong order 1.0 -- error shrinks like `dt`, a full order better than Euler.
For GBM, `b(S) = sigma S`, `b'(S) = sigma`, so the correction term is
`(1/2) sigma^2 S_n (dW_n^2 - dt)`.

For OU (Section 1.3), `b(x) = sigma` is constant, so `b'(x) = 0` and Milstein
collapses to Euler -- there is no Milstein correction for additive noise.

`tests/test_sde.py::test_milstein_has_smaller_strong_error_than_euler_for_gbm`
verifies both orders empirically: it discretizes a *fixed* Brownian path
(built by summing fine increments into coarser ones, so every scheme sees the
same noise) at shrinking step sizes and confirms Milstein's pathwise error is
smaller than Euler's at matched step counts.

### 1.3 Ornstein-Uhlenbeck: exact solution and stationary distribution

```
dX_t = kappa (theta - X_t) dt + sigma dW_t
```

This is a linear SDE; multiplying by the integrating factor `e^{kappa t}` and
integrating gives the exact solution

```
X_t = theta + (X_0 - theta) e^{-kappa t} + sigma * Integral_0^t e^{-kappa(t-s)} dW_s
```

The stochastic integral is a Gaussian (Wiener integral of a deterministic
integrand), so conditional on `X_s`, `X_{s+dt}` is exactly Gaussian:

```
X_{s+dt} | X_s ~ Normal( theta + (X_s - theta) e^{-kappa dt},
                          sigma^2 (1 - e^{-2 kappa dt}) / (2 kappa) )
```

This is `simulate_ou(..., scheme="exact")` -- because the transition kernel
is known in closed form, this scheme has **zero discretization bias at any
step size**, unlike Euler-Maruyama, which only recovers the right one-step
variance in the `dt -> 0` limit (`tests/test_sde.py::test_ou_euler_scheme_is_biased_at_large_dt`
demonstrates the gap directly).

Sending `t -> infinity` in the exact solution gives the **stationary
distribution**:

```
X_infinity ~ Normal( theta, sigma^2 / (2 kappa) )
```

implemented as `ou_stationary_variance()`.

### 1.4 OU parameter estimation is AR(1) MLE in disguise

The exact transition kernel above is a Gaussian AR(1) model:

```
X_{t+dt} = a + b X_t + eps_t,   b = e^{-kappa dt},  a = theta (1 - b)
Var(eps_t) = sigma^2 (1 - b^2) / (2 kappa)
```

So the conditional-Gaussian MLE for `(kappa, theta, sigma)` is *exactly* OLS
on `X_{t+dt}` regressed on `X_t`, followed by inverting the map above:

```
kappa_hat  = -ln(b_hat) / dt
theta_hat  = a_hat / (1 - b_hat)
sigma_hat  = sqrt( 2 kappa_hat * Var(residuals) / (1 - b_hat^2) )
```

implemented in `estimate_ou_mle()`. A practical subtlety this uncovers
(verified in `test_ou_mle_recovers_known_parameters`): `kappa`'s sampling
variance scales like `2 kappa / (n dt)`, i.e. what matters for identifying
the mean-reversion speed is the **total observed time span** `n*dt`, not
the sample count `n` alone -- collecting more high-frequency data with the
same total span barely helps, because `b = e^{-kappa dt}` is pinned close to
1 and tiny errors in `b` blow up under `-ln(b)/dt`.

### 1.5 Heston stochastic volatility

```
dS_t = mu S_t dt + sqrt(v_t) S_t dW_t^S
dv_t = kappa (theta - v_t) dt + xi sqrt(v_t) dW_t^v,   corr(dW^S, dW^v) = rho
```

`v_t` follows a CIR (square-root) process. The **Feller condition**

```
2 kappa theta > xi^2
```

is necessary and sufficient for `v_t > 0` almost surely in continuous time
(`feller_condition()`). Discretized paths can still go negative even when
Feller holds, because Euler-type schemes don't respect the process's exact
transition law. `simulate_heston` uses **full truncation** (Lord, Koekkoek &
van Dijk, 2010): replace `v_t` by `v_t^+ = max(v_t, 0)` inside the diffusion
coefficient and the mean-reversion term,

```
v_{n+1} = v_n + kappa (theta - v_n^+) dt + xi sqrt(v_n^+) dW_n^v
```

which keeps the recursion well-defined (no `sqrt` of a negative number) while
still pulling `v` back toward `theta` from below. `test_heston_variance_stays_nonnegative_under_full_truncation`
deliberately violates Feller and checks the path never goes negative.

Correlated Brownian increments are built via the Cholesky factor of the 2x2
correlation matrix `[[1, rho], [rho, 1]]`: draw independent
`Z1, Z2 ~ N(0,1)`, set `dW^v = sqrt(dt) Z1`, `dW^S = sqrt(dt) (rho Z1 + sqrt(1-rho^2) Z2)`.

### 1.6 Merton jump-diffusion

```
dS_t / S_t = (mu - lambda k) dt + sigma dW_t + dJ_t
```

`J_t` is a compound Poisson process with intensity `lambda` and log-jump
sizes `Y ~ Normal(mu_J, sigma_J^2)`. The compensator

```
k = E[e^Y] - 1 = exp(mu_J + sigma_J^2 / 2) - 1
```

is subtracted from the drift so that `E[dS_t/S_t] = mu dt` regardless of the
jump parameters -- jumps redistribute risk between the diffusion and jump
components without changing the expected return. Over one step,

```
log S_{t+dt} = log S_t + (mu - sigma^2/2 - lambda k) dt + sigma sqrt(dt) Z
               + sum_{i=1}^{N_dt} Y_i,   N_dt ~ Poisson(lambda dt)
```

A defining statistical signature of jumps is **excess kurtosis** relative to
pure GBM at the same diffusion volatility -- `test_merton_jumps_increase_return_kurtosis_versus_pure_diffusion`
checks this directly rather than just checking the code runs.

### 1.7 From a discrete regime chain to a continuous-time Markov chain

`simulate.py` drives regimes with a *daily* transition matrix `P`. A
continuous-time Markov chain (CTMC) with generator `Q` observed at spacing
`dt_day` satisfies `P = exp(Q * dt_day)`; to first order in `dt_day`,
`Q ~ (P - I) / dt_day` (`daily_transition_to_generator()`). Given `Q`, the
chain is simulated *exactly* (no discretization at all) via the Gillespie
algorithm: from state `i`, wait an `Exponential(-Q_ii)` holding time, then
jump to state `j != i` with probability `-Q_ij / Q_ii` (`simulate_ctmc_path()`).
Both constructions share the same stationary distribution `pi` (solving
`pi P = pi` is equivalent to `pi Q = 0`), which
`test_ctmc_path_visits_stationary_distribution_over_long_run` checks by
comparing long-run time-in-state fractions to `pi`.

### 1.8 The Hamilton filter

For the regime model `r_t | s_t = i ~ Normal(mu_i, sigma_i)` with hidden
Markov state `s_t`, the Hamilton (1989) filter computes
`P(s_t = i | r_1, ..., r_t)` recursively via a predict/update pair:

```
Predict:  xi_{t|t-1} = P' xi_{t-1|t-1}
Update:   xi_{t|t}[i] = xi_{t|t-1}[i] * f(r_t | s_t=i)  /  sum_j( xi_{t|t-1}[j] * f(r_t | s_t=j) )
```

where `f(.|s_t=i)` is the Gaussian density for regime `i`. This is the
forward pass of the forward-backward algorithm applied to a Markov-switching
(rather than a hidden i.i.d.-emission) model; `filtered_probs.sum(axis=1) == 1`
at every `t` is the basic sanity check the implementation satisfies. The
initial state `xi_{0|0}` is set to the chain's stationary distribution,
found by solving `pi P = pi`, `sum(pi) = 1` as a linear system
(`stationary_distribution()`). Because this filter is handed the *true*
`P`, `mu_i`, `sigma_i`, it is a ceiling on what an estimated model (Section 3,
`regime_detection.py`) can do with the same information set -- a useful
baseline to quote.

### 1.9 Aligning unlabeled HMM states to ground truth

A fitted `GaussianHMM`'s state indices are arbitrary -- state 0 on one fit
might be state 2 on the next. Before any label-based score (accuracy,
confusion matrix) is meaningful, states must be matched to true regimes.
This is a linear assignment problem: build the confusion matrix `C` between
inferred and true labels, then find the permutation maximizing
`sum_i C[i, pi(i)]`, solved exactly and efficiently by the Hungarian
algorithm (`scipy.optimize.linear_sum_assignment` on `-C`). Implemented in
`regime_detection._align_states`.

---

## 2. Regime detection results (`regime_detection.py`)

With the true regime-generating process known (Section 1.7-1.8), regime
detection quality can be scored directly rather than assumed. On a 4,000-day
simulated path (`scripts/run_regime_research.py`), a 3-state Gaussian HMM
fit on returns alone recovers the true regime with **~68% accuracy**
(chance = 33%) and a **mean detection lag of ~11 trading days** after a true
regime change. The confusion matrix shows most of the error is the HMM
conflating the two higher-volatility regimes (1 and 2) with each other, not
with the calm regime -- consistent with those two regimes differing more in
drift than in the volatility the HMM's Gaussian emissions weight most
heavily.

Feeding leak-free regime-probability features (walk-forward refit HMM +
causal Hamilton filter, Section 1.8) into the downstream classifier gives a
small AUC improvement over the regime-agnostic baseline; a separate
classifier per inferred regime gives a similar-sized improvement by a
different route. Neither is a dramatic effect -- an honest reading is that
the regime *label* is a noisy, lagged proxy for the drift/vol state the
downstream model can already partially infer from realized volatility
features, so it helps a little rather than transforming the result. See
`docs/research_report.md` for the full writeup.

---

## 3. Alpha research (`alphas.py`, `alpha_eval.py`)

### 3.1 The Information Coefficient and Grinold's Fundamental Law

The **information coefficient (IC)** of a signal is the (rank) correlation
between the signal and the forward return it targets -- `information_coefficient()`.
Grinold's *Fundamental Law of Active Management* (Grinold, 1989) links IC to
achievable performance via the **information ratio**:

```
IR ~= IC * sqrt(BR)
```

where `BR` ("breadth") is the number of *independent* forecasts made per
year. The intuition: a weak but genuine edge (small IC), applied often and
independently, compounds into a much better risk-adjusted return than the
same IC applied rarely. This is precisely why the same tiny IC (~0.01) that
looks economically negligible per-trade can still matter at daily
rebalancing over years -- and precisely why the "independent" qualifier on
`BR` is doing enormous work, which motivates Section 3.2.

### 3.2 A statistical pitfall: overlapping windows inflate significance

The natural way to check whether a signal's IC is *stable* is to compute it
on rolling windows (`rolling_ic()`) and look at `mean(IC) / (std(IC)/sqrt(n))`.
This project's own `alpha_eval.ic_t_stat()` does exactly that -- and it is
**wrong** as a significance test, on purpose, to make a point that's worth
documenting. Consecutive 60-day rolling windows share 59 days of data, so
the `n` windowed IC values are nowhere close to independent; treating them
as iid in the standard-error formula badly understates the true standard
error.

`tests/test_alpha_eval.py::test_rolling_ic_t_stat_is_anti_conservative_under_noise`
quantifies the damage directly: feeding the naive rolling t-stat two
*independent* white-noise series (zero true IC, by construction) reports
`|t| > 2` -- nominally "95% significant" -- far more than the nominal 5% of
the time. `non_overlapping_ic_t_stat()` fixes this by computing IC on
**non-overlapping** blocks, so each block-IC is a much more nearly
independent draw (at the cost of a far smaller effective sample size and
therefore lower power). Running `scripts/run_alpha_research.py` on the
simulator shows the gap concretely: the naive rolling t-stat for the
momentum signal is -58.7 (absurd), while the non-overlapping version is a
much more believable -7.9 for the same data.

Neither number is the final word -- the rigorous fix is the block-bootstrap
significance test in Phase 5, which resamples *contiguous blocks* of the
original series (preserving serial dependence) rather than trusting any
closed-form standard-error formula. The lesson generalizes: **any time-series
statistic computed on overlapping windows needs either a block-based
resampling test or a Newey-West-style autocorrelation correction** -- a
naive iid formula will lie in the same direction (overstating significance)
essentially every time.

### 3.3 Orthogonalization as sequential linear regression

Two signals with high correlation are largely the same bet -- combining them
naively double-counts that bet. `orthogonalize()` removes this by, for each
signal in turn, regressing it on all signals already processed and keeping
only the residual:

```
signal_k_orthogonal = signal_k - X_{<k} (X_{<k}'X_{<k})^{-1} X_{<k}' signal_k
```

solved via `numpy.linalg.lstsq` rather than an explicit matrix inverse (more
numerically stable when signals are near-collinear). This is exactly
Gram-Schmidt orthogonalization applied to the columns of the signal matrix,
which is why the result depends on column order: whichever signal is placed
first keeps 100% of its own variance, and each subsequent signal is stripped
of everything linearly explained by the ones before it. Placing the most
economically fundamental signal first is a deliberate choice, not an
arbitrary one -- see `scripts/run_alpha_research.py`.

### 3.4 Combining correlated signals

`combine_ic_weighted()` weights each (z-scored) signal by its own IC, which
implicitly *under*-weights redundant signal pairs relative to what their
combined predictive power could support (their shared component gets
counted twice in the weights but not in the outcome). `combine_ridge()`
instead regresses the forward return on all signals jointly with an L2
penalty:

```
beta_ridge = argmin_beta  ||y - X beta||^2 + alpha * ||beta||^2
           = (X'X + alpha*I)^{-1} X'y
```

Ridge (rather than OLS) is necessary specifically *because* the signals are
correlated: `X'X` is then close to singular, and small changes in the sample
would otherwise produce wildly different OLS coefficients (classical
multicollinearity). The `alpha * I` term regularizes `X'X` away from
singularity at the cost of some bias -- a direct, small-scale instance of the
bias-variance trade-off that reappears in Section 6 (covariance shrinkage
for portfolio construction).

---

## 4. Statistical rigor (`evaluation.py`)

### 4.1 The stationary block bootstrap

Daily strategy returns are serially dependent (positions persist across
days by construction, and the underlying return process itself has AR(1)
structure -- Section 1.6). An ordinary iid bootstrap resamples single days
independently and would therefore *understate* the true uncertainty of any
statistic computed on the return series -- the same failure mode as the
overlapping-window IC t-stat in Section 3.2, just one level up (whole
trading strategies, not individual signals).

The fix is the same family of tool: the **stationary bootstrap** (Politis &
Romano, 1994) resamples contiguous *blocks* whose length is drawn from a
Geometric distribution with a chosen mean, strung together (with wraparound)
to build a resampled series of the original length. Using a random block
length rather than a fixed one avoids the resampled series having artificial
periodicity at exactly the block boundary. `block_bootstrap_ci()` applies
this to compute confidence intervals for arbitrary statistics (annualized
Sharpe, total return); `bootstrap_p_value()` uses the same resampling,
applied to a mean-centered series (imposing the null of zero true mean
return while preserving the series' higher moments and dependence
structure), to get a p-value that doesn't assume independence either.

Running `scripts/run_statistical_rigor.py` on the selected strategy shows
exactly why this matters: the point-estimate annualized Sharpe is +0.47, but
its 90% block-bootstrap CI is **[-0.04, +0.94]** -- it includes zero. The
naive iid-normal-theory standard error would report a much tighter (and
misleadingly reassuring) interval.

### 4.2 The Deflated Sharpe Ratio

Section 3.1's Fundamental Law already showed a small IC can still be
valuable. The flip side: if you evaluate `N` candidate strategies and report
the best one, the *best-of-N* Sharpe ratio is biased upward even when none
of the `N` have any true skill, purely from selection -- the same logic as
p-hacking. The **Deflated Sharpe Ratio (DSR)** (Bailey & Lopez de Prado,
2014) corrects for this by testing the observed Sharpe not against a null of
0, but against the *expected maximum* Sharpe achievable by `N` independent
skill-less strategies:

```
SR* = sigma_SR * [ (1 - gamma_E) * Phi^-1(1 - 1/N) + gamma_E * Phi^-1(1 - 1/(N*e)) ]
```

(`gamma_E` = Euler-Mascheroni constant ~0.5772, `sigma_SR` = the standard
deviation of the Sharpe ratios actually observed across the `N` trials,
`Phi^-1` = inverse normal CDF) -- `expected_max_sharpe_under_null()`. The
observed Sharpe is then converted to a probability of genuine skill via

```
DSR = Phi( (SR_hat - SR*) / se(SR_hat) )
```

where `se(SR_hat)` accounts for the return distribution's skewness `gamma_3`
and kurtosis `gamma_4` (Mertens, 2002 / Lo, 2002):

```
se(SR_hat) = sqrt( (1 - gamma_3*SR_hat + (gamma_4 - 1)/4 * SR_hat^2) / (T - 1) )
```

which reduces to the familiar `sqrt((1 + SR^2/2) / T)` under normality
(`gamma_3 = 0`, `gamma_4 = 3`). `deflated_sharpe_ratio()` takes the actual
per-period Sharpe ratios observed across every configuration considered
during this project's own model/strategy selection (`trial_sharpes` in
`scripts/run_statistical_rigor.py`: both candidate classifiers plus the
individual alpha signals from Section 3), not a hypothetical count -- so the
multiple-testing penalty reflects the real search that was done. With 6
configurations considered, the selected strategy's DSR is **0.59**: modestly
more likely than not to reflect genuine skill, but a long way from the false
confidence a single reported Sharpe ratio would suggest.

### 4.3 Purging and embargo

`model.walk_forward_validate(..., embargo=k)` drops the most recent `k` rows
of each fold's training window, leaving a gap before the test window starts
(Lopez de Prado, *Advances in Financial Machine Learning*, ch. 7). The
motivating failure mode: when labels are constructed from overlapping
multi-day forward windows, or features leak information across the
train/test boundary via a rolling window that spans it, samples adjacent to
the boundary are effectively "the same observation" split across train and
test, and the resulting validation score is optimistic. In this specific
pipeline, labels are single-day-ahead (no overlapping horizons) and features
are strictly backward-looking (`tests/test_no_lookahead.py`), so leakage is
already avoided by construction and the embargo is not load-bearing here --
`scripts/run_statistical_rigor.py` confirms AUC is essentially flat across
`embargo in {0, 10, 30}`. It is included anyway both as standard extra
margin of safety and because a pipeline with overlapping labels (e.g. a
5-day-forward-return target) would need it for correctness, not just safety.

---

## 5. Linear algebra and portfolio construction (`portfolio.py`)

### 5.1 Eigendecomposition of a covariance matrix

A covariance matrix `Sigma` is symmetric and positive semi-definite, so it
admits the spectral decomposition `Sigma = V diag(lambda) V'` with real,
non-negative eigenvalues `lambda_i` and orthonormal eigenvectors (the
columns of `V`). `pca_eigendecomposition()` uses `numpy.linalg.eigh`
specifically (not the general `eig`) because it assumes symmetry, which is
both faster and numerically safer (guaranteed real output, no complex
round-off). Each eigenvector is a portfolio of assets with **uncorrelated**
returns (the principal components); its eigenvalue is that portfolio's
variance. `explained_variance_ratio()` normalizes eigenvalues by their sum
(the trace, i.e. total variance) to say what fraction of total risk each
component explains. On the simulated factor panel, PC1 recovers the true
market-factor loadings almost exactly (cosine similarity > 0.99,
`test_pca_recovers_the_market_factor_direction`) precisely because that
factor dominates the variance.

### 5.2 The Marchenko-Pastur law and eigenvalue denoising

If a `(T x N)` matrix has iid entries of variance `sigma^2`, the eigenvalues
of its `N x N` sample correlation matrix converge (as `T, N -> infinity`
with `q = N/T` fixed) to the **Marchenko-Pastur distribution**, supported on
`[sigma^2 (1-sqrt(q))^2, sigma^2 (1+sqrt(q))^2]` (`marchenko_pastur_bounds()`).
This is a precise statement of "how much eigenvalue spread finite-sample
noise alone produces" -- any empirical eigenvalue that falls *inside* this
band is statistically indistinguishable from noise, even if the true
covariance were exactly the identity (uncorrelated assets).

`denoise_covariance_mp()` acts on this directly: eigenvalues of the sample
correlation matrix at or below the MP upper edge are replaced by their
average (a noise-only subspace should carry no directional information, so
spreading its total variance evenly across those directions is the
information-theoretically neutral choice), while eigenvalues above the edge
(judged to reflect genuine factor structure) are kept as estimated. The
`test_pca_recovers...` and `test_denoising_preserves_a_genuine_factor...`
tests confirm the dominant (market) eigenvalue survives denoising
essentially unchanged.

**Important subtlety, found empirically while testing this module**: MP
denoising does *not* reliably reduce Frobenius-norm distance to the true
covariance matrix -- in several parameter regimes tried here it made that
distance slightly *worse*. What it reliably improves is **out-of-sample
minimum-variance portfolio risk** (`test_denoising_reduces_out_of_sample_minimum_variance_portfolio_risk`,
averaged over 15 seeds to avoid a single noisy draw). The reason is
mechanical: `minimum_variance_weights()` computes `Sigma^-1 * 1`, and matrix
inversion *amplifies* whichever eigen-directions have the smallest
eigenvalues -- exactly the directions most corrupted by estimation noise in
the raw sample covariance. Fixing the small/noisy eigenvalues (denoising)
therefore helps the specific quantity (`w' Sigma_true w`) that portfolio
construction actually cares about, even when it doesn't help the covariance
matrix's overall matrix-norm accuracy. This is a good general lesson: pick
the evaluation metric that matches how an estimate will actually be used.

### 5.3 Ledoit-Wolf shrinkage as a bias-variance trade-off

`ledoit_wolf_shrinkage()` implements Ledoit & Wolf (2004): shrink the sample
covariance `S` toward a scaled-identity target `F = mu*I` (`mu` = the
average sample variance),

```
S_shrunk = delta * F + (1 - delta) * S
```

with the shrinkage intensity `delta` chosen (via estimators `phi_hat` for
the sampling variance of `S`'s entries and `gamma_hat` for how much `F`
misspecifies `S`) to minimize expected Frobenius loss against the unknown
true covariance. This project's implementation was checked against
`sklearn.covariance.LedoitWolf` on identical data and matches to float
precision (`test_ledoit_wolf_shrinkage_matches_sklearn`) -- both the
shrinkage intensity and the resulting matrix agree to 1e-6. `S` (unbiased,
high variance when `N` is comparable to `T`) and `F` (biased -- real assets
aren't actually uncorrelated with equal variance -- but zero estimation
variance) sit at opposite ends of the bias-variance trade-off; `delta`
interpolates between them based on how much each source of error actually
matters for the data at hand.

### 5.4 Portfolio weights via Lagrangian optimization

**Minimum variance** (`minimum_variance_weights()`): minimize `w'Sigma w`
subject to `1'w = 1`. Lagrangian `L = w'Sigma w - lambda(1'w - 1)`; the
first-order condition `dL/dw = 2 Sigma w - lambda 1 = 0` gives
`w ~ Sigma^-1 1`, normalized so the weights sum to 1. For a diagonal
covariance this reduces to the well-known inverse-variance weighting
(verified directly in `test_minimum_variance_weights_are_inverse_variance_for_diagonal_covariance`).

**Mean-variance** (`mean_variance_weights()`): maximize
`w'mu - (risk_aversion/2) w'Sigma w`. FOC: `mu = risk_aversion * Sigma * w`,
so `w = (1/risk_aversion) Sigma^-1 mu` -- the direction classical Markowitz
optimization points in (unnormalized for any budget constraint).

### 5.5 Error maximization

Feeding a **pure-noise** expected-return estimate (a 150-day sample mean of
a process whose true mean is exactly 0 everywhere) into
`mean_variance_weights()` alongside each covariance estimate
(`scripts/run_portfolio_research.py`, Section 3) makes the failure mode
concrete: the optimizer built on the **raw sample covariance** takes ~431x
gross leverage and realizes 0.76 true portfolio variance betting on a
signal that carries, by construction, exactly zero true expected return.
With the **MP-denoised covariance**, the same noisy input produces ~244x
leverage and 0.27 realized variance -- roughly a third of the raw case's
risk, on inputs that were identical apart from the covariance estimate.
This is Michaud's (1989) "error maximization": naive mean-variance
optimization doesn't just fail to filter out estimation error, it actively
*seeks out* whichever assets have the most overestimated return and most
underestimated risk, because that is precisely what looks best to an
optimizer that takes its inputs at face value. Regularizing the covariance
estimate (denoising or shrinkage) is a partial but real mitigation, visible
directly in the leverage and realized-risk numbers above.

---

## 6. Real data and robustness (`data.py`, `sizing.py`)

### 6.1 The Kelly criterion, and why full Kelly is reckless

For a bet sized as fraction `f` of capital with per-period return `r`,
wealth compounds as `W_T = W_0 * prod(1 + f*r_t)`, so long-run growth rate
is `E[log(1 + f*r)]`. For small `r` (a reasonable approximation for daily
equity returns), a second-order Taylor expansion gives

```
E[log(1 + f*r)] ~= f*mu - (f^2 * sigma^2) / 2
```

Maximizing over `f` (`d/df = mu - f*sigma^2 = 0`) gives the **Kelly
fraction**:

```
f* = mu / sigma^2
```

`sizing.kelly_fraction()` implements this directly. Running it on the SPY
strategy's realized daily returns (`scripts/run_real_data_research.py`,
Section 4) gives **f\* = 5.03** -- over 5x leverage. This number is exactly
why full Kelly is considered reckless in practice, not a strategy
recommendation: `f*` is *linear* in `mu`, and `mu` (a daily mean return) is
by far the hardest of the two moments to estimate reliably -- its standard
error shrinks only as `1/sqrt(T)`, while `sigma` is comparatively stable.
Since Section 1's own findings show this strategy's AUC on real data is
essentially at chance (~0.49-0.50, Section 6.2), the `mu` feeding this Kelly
calculation is itself mostly noise -- levering 5x into a noisy mean
estimate is a textbook way to blow up an account. `fractional_kelly_position()`
(e.g. half-Kelly) is the standard practical mitigation: it trades away some
theoretical growth rate for a large reduction in sensitivity to `mu`
estimation error (variance of terminal wealth under fractional Kelly `c`
scales roughly with `c^2`, while growth rate scales closer to linearly in
`c` for `c` below 1 -- a favorable trade near `c=1` and a clearly favorable
one well below it).

**Volatility targeting** (`vol_target_position_scale()`,
`run_vol_targeted_backtest()`) sidesteps the problem rather than mitigating
it: position size = `target_vol / realized_vol`, which needs only a
*variance* estimate, not a *mean* estimate. This is precisely why
volatility targeting -- not literal Kelly sizing -- is what's actually used
in practice. On SPY (`scripts/run_real_data_research.py` Section 4),
vol-targeting produced a slightly *lower* Sharpe than plain binary +-1
sizing (0.333 vs 0.401) in this specific run -- reported as-is rather than
cherry-picked, since with an underlying signal this weak (Section 6.2),
there is no strong prior that a smarter sizing rule should improve a
strategy whose direction calls are barely better than chance.

### 6.2 Real data: an honest negative(-ish) result

Running the exact same pipeline validated on synthetic data (features,
walk-forward logistic regression, cost-aware backtest -- zero code changes)
against five real ETFs (`scripts/run_real_data_research.py` Section 1)
gives AUC in **0.484-0.502** -- statistically indistinguishable from the
0.500 coin-flip baseline, and *below* the synthetic simulator's own
0.52-0.55. This is the expected result stated up front in the script's
docstring: the synthetic simulator has genuine AR(1)-driven short-horizon
structure and persistent regimes *by construction*; real daily ETF returns
are close to a random walk once that structure is absent.

The more interesting, easy-to-miss finding is in Section 2 (the cost x
threshold sensitivity sweep on SPY): several configurations show a
*positive* Sharpe (up to +0.86 pre-cost at the most permissive threshold)
despite AUC essentially at chance. The explanation is visible in the
threshold sweep itself: at threshold 0.50 the model is long almost every
day (predicting "up" is right about half the time anyway, but SPY drifts up
~55% of days), so the backtest is mostly re-deriving **buy-and-hold's own
positive drift**, not detecting anything. As the threshold rises (the model
becomes more selective, trading only on its most confident calls) Sharpe
collapses toward -0.37 -- exactly what you'd expect if the "confidence"
being filtered on carries no real information. Section 3's momentum-signal
parameter-stability sweep tells the same story from a different angle: Sharpe
is negative across nearly every (lookback, cost) combination, with the one
positive cell (lookback=20, zero cost, Sharpe=+0.13) vanishing the moment
any realistic transaction cost is applied. Both findings independently
support the same conclusion reached via AUC alone: **this signal does not
survive contact with real cost-aware, selectivity-aware evaluation** -- a
result that would be easy to miss by looking at total-return or Sharpe
alone without the AUC and cost-sensitivity cross-checks. See
`docs/research_report.md` for the full writeup and its implications for
what this project does and doesn't claim.

---

*(This is the final section for now; see the Research Notes / Future Work
sections of the README and research report for possible extensions.)*
