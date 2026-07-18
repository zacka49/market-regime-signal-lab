from __future__ import annotations

import numpy as np
import pytest

from regime_signal_lab.filtering import stationary_distribution
from regime_signal_lab.sde import (
    daily_transition_to_generator,
    estimate_ou_mle,
    feller_condition,
    gbm_exact,
    ou_stationary_variance,
    simulate_ctmc_path,
    simulate_gbm,
    simulate_heston,
    simulate_merton_jump_diffusion,
    simulate_ou,
)
from regime_signal_lab.simulate import REGIME_TRANSITION_MATRIX


def test_milstein_has_smaller_strong_error_than_euler_for_gbm():
    """Pathwise (strong) convergence: discretize the SAME Brownian path at
    successively finer step counts and compare each scheme's path to the
    exact Ito solution. Milstein (strong order 1.0) must eventually beat
    Euler-Maruyama (strong order 0.5) on the same path."""
    T, mu, sigma, x0 = 1.0, 0.05, 0.3, 100.0
    rng = np.random.default_rng(0)
    finest_steps = 2**10
    dt_fine = T / finest_steps
    fine_increments = rng.normal(0.0, np.sqrt(dt_fine), size=finest_steps)

    step_counts = [2**k for k in range(4, 11)]
    errors_euler, errors_milstein = [], []

    for n_steps in step_counts:
        factor = finest_steps // n_steps
        coarse_increments = fine_increments.reshape(n_steps, factor).sum(axis=1)
        t = np.linspace(0.0, T, n_steps + 1)
        w = np.concatenate([[0.0], np.cumsum(coarse_increments)])
        exact = gbm_exact(mu, sigma, x0, t, w)

        _, euler = simulate_gbm(mu, sigma, x0, T, n_steps, scheme="euler", increments=coarse_increments)
        _, milstein = simulate_gbm(mu, sigma, x0, T, n_steps, scheme="milstein", increments=coarse_increments)

        errors_euler.append(np.max(np.abs(euler - exact)))
        errors_milstein.append(np.max(np.abs(milstein - exact)))

    errors_euler = np.array(errors_euler)
    errors_milstein = np.array(errors_milstein)

    assert errors_euler[-1] < errors_euler[0]
    assert errors_milstein[-1] < errors_milstein[0]
    assert errors_milstein[-1] < errors_euler[-1]


def test_ou_exact_scheme_matches_stationary_variance_at_large_dt():
    """The exact OU transition kernel has zero discretization bias at ANY
    step size, unlike Euler-Maruyama. Simulate many one-step transitions
    from stationarity with a deliberately large dt and check the empirical
    variance matches the theoretical stationary variance."""
    kappa, theta, sigma = 1.0, 0.0, 1.0
    true_var = ou_stationary_variance(kappa, sigma)

    rng = np.random.default_rng(1)
    x0_samples = rng.normal(theta, np.sqrt(true_var), size=20_000)

    large_dt = 3.0
    finals = np.empty_like(x0_samples)
    for i, x0 in enumerate(x0_samples):
        _, path = simulate_ou(kappa, theta, sigma, x0, T=large_dt, n_steps=1, seed=1000 + i, scheme="exact")
        finals[i] = path[-1]

    assert finals.var() == pytest.approx(true_var, rel=0.05)


def test_ou_euler_scheme_is_biased_at_large_dt():
    """Euler-Maruyama's one-step variance is sigma^2*dt regardless of kappa,
    which overstates the true mean-reverting variance sigma^2*(1-e^-2k dt)/2k
    once kappa*dt is not small -- i.e. Euler is a biased discretization here."""
    kappa, theta, sigma = 1.0, 0.0, 1.0
    true_var = ou_stationary_variance(kappa, sigma)
    large_dt = 3.0

    _, path = simulate_ou(
        kappa, theta, sigma, x0=0.0, T=large_dt, n_steps=1, seed=7, scheme="euler", n_paths=20_000
    )
    euler_var = path[:, -1].var()

    # naive Euler one-step variance is sigma^2 * dt = 3.0, ~6x the true
    # stationary variance of 0.5 -- well outside a 5% tolerance band.
    assert abs(euler_var - true_var) / true_var > 0.5


def test_ou_mle_recovers_known_parameters():
    # kappa*dt must not be tiny, or the AR(1) coefficient b = e^{-kappa dt} sits
    # so close to 1 that sampling noise in b swamps -ln(b)/dt (kappa's implied
    # variance scales as ~2*kappa/(n*dt), i.e. total span n*dt is what matters).
    kappa, theta, sigma = 0.8, 0.02, 0.15
    dt = 1.0
    _, path = simulate_ou(kappa, theta, sigma, x0=theta, T=dt * 5000, n_steps=5000, seed=3, scheme="exact")

    kappa_hat, theta_hat, sigma_hat = estimate_ou_mle(path, dt)

    assert kappa_hat == pytest.approx(kappa, rel=0.25)
    assert theta_hat == pytest.approx(theta, abs=0.01)
    assert sigma_hat == pytest.approx(sigma, rel=0.1)


def test_heston_variance_stays_nonnegative_under_full_truncation():
    # Deliberately violate the Feller condition (2*kappa*theta <= xi^2) so
    # the discretized variance would go negative without full truncation.
    kappa, theta, xi = 1.0, 0.02, 0.6
    assert not feller_condition(kappa, theta, xi)

    _, _, v = simulate_heston(
        mu=0.05, kappa=kappa, theta=theta, xi=xi, rho=-0.7,
        s0=100.0, v0=theta, T=2.0, n_steps=2000, seed=5,
    )
    assert np.all(v >= 0.0)


def test_heston_variance_reverts_towards_theta():
    kappa, theta, xi = 3.0, 0.04, 0.3
    assert feller_condition(kappa, theta, xi)

    _, _, v = simulate_heston(
        mu=0.05, kappa=kappa, theta=theta, xi=xi, rho=-0.5,
        s0=100.0, v0=0.20, T=5.0, n_steps=5000, seed=6,
    )
    burn_in = len(v) // 2
    assert v[burn_in:].mean() == pytest.approx(theta, rel=0.3)


def test_merton_jumps_increase_return_kurtosis_versus_pure_diffusion():
    """A defining feature of jump-diffusion vs. pure GBM: fat tails. Compare
    excess kurtosis of daily log-returns across many simulated paths with and
    without jumps, holding the diffusion volatility fixed."""
    from scipy.stats import kurtosis

    n_paths, n_steps, T = 300, 250, 1.0
    diffusion_only_returns = []
    jump_returns = []

    for i in range(n_paths):
        _, price_no_jump, _ = simulate_merton_jump_diffusion(
            mu=0.05, sigma=0.2, jump_intensity=0.0, jump_mean=0.0, jump_std=0.0,
            s0=100.0, T=T, n_steps=n_steps, seed=1000 + i,
        )
        diffusion_only_returns.append(np.diff(np.log(price_no_jump)))

        _, price_jump, _ = simulate_merton_jump_diffusion(
            mu=0.05, sigma=0.2, jump_intensity=5.0, jump_mean=0.0, jump_std=0.05,
            s0=100.0, T=T, n_steps=n_steps, seed=2000 + i,
        )
        jump_returns.append(np.diff(np.log(price_jump)))

    diffusion_only_returns = np.concatenate(diffusion_only_returns)
    jump_returns = np.concatenate(jump_returns)

    assert kurtosis(jump_returns, fisher=True) > kurtosis(diffusion_only_returns, fisher=True)


def test_generator_rows_sum_to_zero():
    generator = daily_transition_to_generator(REGIME_TRANSITION_MATRIX, dt_day=1.0)
    np.testing.assert_allclose(generator.sum(axis=1), np.zeros(3), atol=1e-10)


def test_ctmc_path_visits_stationary_distribution_over_long_run():
    generator = daily_transition_to_generator(REGIME_TRANSITION_MATRIX, dt_day=1.0)
    pi = stationary_distribution(REGIME_TRANSITION_MATRIX)

    times, states = simulate_ctmc_path(generator, T=20_000.0, x0=0, seed=42)
    durations = np.diff(np.append(times, 20_000.0))

    time_in_state = np.zeros(3)
    for state, duration in zip(states, durations):
        time_in_state[state] += duration
    empirical = time_in_state / time_in_state.sum()

    np.testing.assert_allclose(empirical, pi, atol=0.03)
