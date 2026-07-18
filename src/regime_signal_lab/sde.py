"""Continuous-time stochastic process simulation.

Discretization schemes (Euler-Maruyama, Milstein, full-truncation) and their
error orders are derived in docs/mathematical_notes.md. Every process here
has a closed-form or near-closed-form target used to validate the scheme in
tests/test_sde.py -- the point of this module is not just to generate paths,
but to demonstrate that the discretization is correct.
"""

from __future__ import annotations

import numpy as np


def gbm_exact(mu: float, sigma: float, x0: float, t: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Exact solution of dS = mu*S dt + sigma*S dW via Ito's lemma on log(S)."""
    return x0 * np.exp((mu - 0.5 * sigma**2) * t + sigma * w)


def simulate_gbm(
    mu: float,
    sigma: float,
    x0: float,
    T: float,
    n_steps: int,
    seed: int | None = None,
    scheme: str = "euler",
    increments: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate one GBM path. Returns (t, x) of length n_steps + 1.

    scheme: "euler" (strong order 0.5) or "milstein" (strong order 1.0).
    Pass `increments` (shape (n_steps,), Brownian increments N(0, dt)) to
    reuse the same driving noise across schemes/step counts for a strong
    (pathwise) convergence comparison.
    """
    dt = T / n_steps
    if increments is None:
        rng = np.random.default_rng(seed)
        increments = rng.normal(0.0, np.sqrt(dt), size=n_steps)

    t = np.linspace(0.0, T, n_steps + 1)
    x = np.empty(n_steps + 1)
    x[0] = x0
    for n in range(n_steps):
        dw = increments[n]
        if scheme == "euler":
            x[n + 1] = x[n] + mu * x[n] * dt + sigma * x[n] * dw
        elif scheme == "milstein":
            x[n + 1] = (
                x[n]
                + mu * x[n] * dt
                + sigma * x[n] * dw
                + 0.5 * sigma**2 * x[n] * (dw**2 - dt)
            )
        else:
            raise ValueError(f"unknown scheme: {scheme}")
    return t, x


def ou_stationary_variance(kappa: float, sigma: float) -> float:
    """Var(X_infinity) for dX = kappa*(theta - X) dt + sigma dW."""
    return sigma**2 / (2.0 * kappa)


def simulate_ou(
    kappa: float,
    theta: float,
    sigma: float,
    x0: float,
    T: float,
    n_steps: int,
    seed: int | None = None,
    scheme: str = "exact",
    n_paths: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate Ornstein-Uhlenbeck paths.

    scheme="exact" uses the known Gaussian transition kernel
        X_{n+1} | X_n ~ Normal(theta + (X_n - theta) e^{-kappa dt},
                                sigma^2 (1 - e^{-2 kappa dt}) / (2 kappa))
    and has zero discretization bias at any step size. scheme="euler" uses
    the naive Euler-Maruyama update and is biased for large kappa*dt (the
    diffusion term for OU is additive, so Milstein collapses to Euler here).

    Returns (t, x) where x has shape (n_paths, n_steps + 1) if n_paths > 1
    else shape (n_steps + 1,).
    """
    dt = T / n_steps
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, T, n_steps + 1)

    x = np.empty((n_paths, n_steps + 1))
    x[:, 0] = x0

    if scheme == "exact":
        phi = np.exp(-kappa * dt)
        cond_std = np.sqrt(sigma**2 * (1.0 - phi**2) / (2.0 * kappa))
        z = rng.standard_normal((n_paths, n_steps))
        for n in range(n_steps):
            x[:, n + 1] = theta + (x[:, n] - theta) * phi + cond_std * z[:, n]
    elif scheme == "euler":
        dw = rng.normal(0.0, np.sqrt(dt), size=(n_paths, n_steps))
        for n in range(n_steps):
            x[:, n + 1] = x[:, n] + kappa * (theta - x[:, n]) * dt + sigma * dw[:, n]
    else:
        raise ValueError(f"unknown scheme: {scheme}")

    return t, (x[0] if n_paths == 1 else x)


def estimate_ou_mle(series: np.ndarray, dt: float) -> tuple[float, float, float]:
    """Closed-form MLE for (kappa, theta, sigma) from an observed OU path.

    The exact transition kernel is a Gaussian AR(1): X_{t+dt} = a + b*X_t + eps,
    with b = e^{-kappa dt}, a = theta*(1-b), Var(eps) = sigma^2 (1-b^2)/(2 kappa).
    OLS recovers (a, b) and the residual variance; kappa/theta/sigma follow by
    inverting that map. See docs/mathematical_notes.md for the derivation.
    """
    x_t = series[:-1]
    x_next = series[1:]

    n = len(x_t)
    x_mean, y_mean = x_t.mean(), x_next.mean()
    b = np.sum((x_t - x_mean) * (x_next - y_mean)) / np.sum((x_t - x_mean) ** 2)
    a = y_mean - b * x_mean

    residuals = x_next - (a + b * x_t)
    residual_var = np.sum(residuals**2) / (n - 2)

    b = np.clip(b, 1e-8, 1 - 1e-8)
    kappa = -np.log(b) / dt
    theta = a / (1.0 - b)
    sigma = np.sqrt(2.0 * kappa * residual_var / (1.0 - b**2))
    return float(kappa), float(theta), float(sigma)


def simulate_heston(
    mu: float,
    kappa: float,
    theta: float,
    xi: float,
    rho: float,
    s0: float,
    v0: float,
    T: float,
    n_steps: int,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simulate Heston stochastic-volatility paths via full truncation Euler.

    dS = mu*S dt + sqrt(v)*S dW^S,  dv = kappa*(theta - v) dt + xi*sqrt(v) dW^v,
    corr(dW^S, dW^v) = rho. Feller condition 2*kappa*theta > xi^2 keeps v > 0
    in continuous time; full truncation clips v to 0 inside sqrt()/drift terms
    (Lord, Koekkoek & Van Dijk, 2010) so the discretized variance stays finite
    without ad-hoc reflection.
    """
    dt = T / n_steps
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, T, n_steps + 1)

    z1 = rng.standard_normal(n_steps)
    z2 = rng.standard_normal(n_steps)
    dw_v = np.sqrt(dt) * z1
    dw_s = np.sqrt(dt) * (rho * z1 + np.sqrt(1.0 - rho**2) * z2)

    log_s = np.empty(n_steps + 1)
    v = np.empty(n_steps + 1)
    log_s[0] = np.log(s0)
    v[0] = v0

    for n in range(n_steps):
        v_pos = max(v[n], 0.0)
        sqrt_v = np.sqrt(v_pos)
        log_s[n + 1] = log_s[n] + (mu - 0.5 * v_pos) * dt + sqrt_v * dw_s[n]
        v[n + 1] = v[n] + kappa * (theta - v_pos) * dt + xi * sqrt_v * dw_v[n]

    return t, np.exp(log_s), np.clip(v, 0.0, None)


def feller_condition(kappa: float, theta: float, xi: float) -> bool:
    """True if 2*kappa*theta > xi**2, i.e. the CIR variance process cannot hit 0."""
    return 2.0 * kappa * theta > xi**2


def simulate_merton_jump_diffusion(
    mu: float,
    sigma: float,
    jump_intensity: float,
    jump_mean: float,
    jump_std: float,
    s0: float,
    T: float,
    n_steps: int,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simulate Merton's jump-diffusion model.

    dS/S = (mu - lambda*k) dt + sigma dW + dJ, J a compound Poisson process
    with intensity lambda (jump_intensity) and log-jump sizes Y ~ N(jump_mean,
    jump_std^2). k = E[e^Y] - 1 compensates the drift so E[dS/S] = mu*dt
    regardless of the jump parameters.

    Returns (t, price, jump_counts).
    """
    dt = T / n_steps
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, T, n_steps + 1)

    k = np.exp(jump_mean + 0.5 * jump_std**2) - 1.0
    diffusion_drift = (mu - 0.5 * sigma**2 - jump_intensity * k) * dt

    jump_counts = rng.poisson(jump_intensity * dt, size=n_steps)
    jump_sizes = np.zeros(n_steps)
    for n in range(n_steps):
        if jump_counts[n] > 0:
            jump_sizes[n] = rng.normal(jump_mean, jump_std, size=jump_counts[n]).sum()

    diffusion = sigma * rng.normal(0.0, np.sqrt(dt), size=n_steps)
    log_returns = diffusion_drift + diffusion + jump_sizes
    log_price = np.log(s0) + np.concatenate([[0.0], np.cumsum(log_returns)])

    return t, np.exp(log_price), np.concatenate([[0], jump_counts])


def daily_transition_to_generator(daily_transition: np.ndarray, dt_day: float = 1.0) -> np.ndarray:
    """Approximate a continuous-time generator Q from a discrete daily transition
    matrix P via Q = (P - I) / dt_day. Exact for a genuinely continuous-time chain
    observed at spacing dt_day; here it is a first-order approximation that lets
    the discrete regime model in simulate.py be re-simulated at arbitrary
    (sub-daily) resolution as a proper continuous-time Markov chain (CTMC).
    """
    n = daily_transition.shape[0]
    return (daily_transition - np.eye(n)) / dt_day


def simulate_ctmc_path(
    generator: np.ndarray, T: float, x0: int, seed: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Exact CTMC simulation (Gillespie / stochastic simulation algorithm).

    From state i, the holding time is Exponential(-Q[i, i]) and the chain
    jumps to state j != i with probability -Q[i, j] / Q[i, i] (the embedded
    jump chain). Returns (jump_times, states) with jump_times[0] == 0.0 and
    the path defined by holding `states[k]` on [jump_times[k], jump_times[k+1]).
    """
    rng = np.random.default_rng(seed)
    n_states = generator.shape[0]
    jump_probs = -generator / np.diag(generator)[:, None]
    np.fill_diagonal(jump_probs, 0.0)

    times = [0.0]
    states = [x0]
    t = 0.0
    state = x0
    while t < T:
        rate = -generator[state, state]
        if rate <= 0:
            break
        t += rng.exponential(1.0 / rate)
        if t >= T:
            break
        state = rng.choice(n_states, p=jump_probs[state])
        times.append(t)
        states.append(state)

    return np.array(times), np.array(states)
