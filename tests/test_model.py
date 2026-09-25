import numpy as np
import pytest

from kvwap.data import default_true_params, simulate_log_volume, u_shape_phi
from kvwap.model import (
    KalmanParams,
    KalmanVolumeModel,
    VolumeState,
    advance,
    fit_em,
    forecast_log_volume,
    kalman_filter,
    rts_smoother,
    soft_threshold,
)


def small_params(I=4):
    return KalmanParams(
        pi1=np.array([0.3, -0.1]),
        sigma1=np.array([[0.2, 0.03], [0.03, 0.1]]),
        a_eta=0.9,
        a_mu=0.6,
        sigma_eta2=0.05,
        sigma_mu2=0.08,
        r=0.04,
        phi=np.linspace(-0.5, 0.5, I),
    )


def transition(p, tau, I):
    """A_tau, Q_tau for the move tau -> tau+1 (0-based tau)."""
    boundary = (tau + 1) % I == 0
    A = np.diag([p.a_eta if boundary else 1.0, p.a_mu])
    Q = np.diag([p.sigma_eta2 if boundary else 0.0, p.sigma_mu2])
    return A, Q


def reference_filter(y, p):
    """Textbook matrix Kalman filter (Algorithm 1)."""
    I = p.bins_per_day
    C = np.array([[1.0, 1.0]])
    x, P = p.pi1.copy(), p.sigma1.copy()
    xf, Pf = [], []
    for t, yt in enumerate(y):
        S = (C @ P @ C.T).item() + p.r
        K = P @ C.T / S
        x = x + (K * (yt - p.phi[t % I] - (C @ x).item())).ravel()
        P = P - K @ C @ P
        xf.append(x.copy())
        Pf.append(P.copy())
        A, Q = transition(p, t, I)
        x, P = A @ x, A @ P @ A.T + Q
    return np.array(xf), np.array(Pf)


def batch_posterior(y, p):
    """Exact Gaussian posterior of the stacked states given all observations."""
    I = p.bins_per_day
    N = len(y)
    # X = mean + M u,  u = [x1 - pi1, w_1, ..., w_{N-1}]
    M = np.zeros((2 * N, 2 * N))
    D = np.zeros((2 * N, 2 * N))
    D[:2, :2] = p.sigma1
    for t in range(1, N):
        _, Q = transition(p, t - 1, I)
        D[2 * t:2 * t + 2, 2 * t:2 * t + 2] = Q
    for t in range(N):
        for s in range(t + 1):
            # effect of u_s (block s) on x_t
            G = np.eye(2)
            for k in range(s, t):
                A, _ = transition(p, k, I)
                G = A @ G
            M[2 * t:2 * t + 2, 2 * s:2 * s + 2] = G
    mean = M[:, :2] @ p.pi1
    covX = M @ D @ M.T
    Cb = np.kron(np.eye(N), np.array([[1.0, 1.0]]))
    ymean = Cb @ mean + np.tile(p.phi, N // I)
    covY = Cb @ covX @ Cb.T + p.r * np.eye(N)
    gain = covX @ Cb.T @ np.linalg.inv(covY)
    post_mean = mean + gain @ (y - ymean)
    post_cov = covX - gain @ Cb @ covX
    return post_mean.reshape(N, 2), post_cov


def test_filter_matches_reference():
    p = small_params()
    rng = np.random.default_rng(0)
    y = rng.normal(0, 1, 20)
    fr = kalman_filter(y, p)
    xf, Pf = reference_filter(y, p)
    np.testing.assert_allclose(fr.x_filt, xf, atol=1e-12)
    np.testing.assert_allclose(fr.P_filt[:, 0], Pf[:, 0, 0], atol=1e-12)
    np.testing.assert_allclose(fr.P_filt[:, 1], Pf[:, 0, 1], atol=1e-12)
    np.testing.assert_allclose(fr.P_filt[:, 2], Pf[:, 1, 1], atol=1e-12)


def test_smoother_and_cross_covariance_match_exact_posterior():
    p = small_params(I=4)
    rng = np.random.default_rng(1)
    y = rng.normal(0, 1, 12)  # 3 days x 4 bins
    fr = kalman_filter(y, p)
    sm = rts_smoother(fr, p)
    post_mean, post_cov = batch_posterior(y, p)
    np.testing.assert_allclose(sm.x, post_mean, atol=1e-10)
    for t in range(12):
        blk = post_cov[2 * t:2 * t + 2, 2 * t:2 * t + 2]
        np.testing.assert_allclose(sm.P[t], [blk[0, 0], blk[0, 1], blk[1, 1]], atol=1e-10)
    for t in range(1, 12):
        cross = post_cov[2 * t:2 * t + 2, 2 * (t - 1):2 * t]  # Cov(x_t, x_{t-1} | all)
        np.testing.assert_allclose(sm.P_cross[t], cross, atol=1e-10)


def test_missing_observations_skip_correction():
    p = small_params()
    y = np.array([0.1, np.nan, 0.3, 0.2])
    fr = kalman_filter(y, p)
    # with the observation missing, the filtered state equals the prediction
    np.testing.assert_allclose(fr.x_filt[1], fr.x_pred[1])
    np.testing.assert_allclose(fr.P_filt[1], fr.P_pred[1])


def test_em_loglik_is_monotone_and_recovers_parameters():
    true = default_true_params(26)
    clean, _ = simulate_log_volume(true, 150, seed=11)
    fit = fit_em(clean, max_iter=200)
    ll = np.array(fit.loglik_path)
    assert np.all(np.diff(ll) > -1e-6), "EM must not decrease the likelihood"
    q = fit.params
    assert abs(q.a_mu - true.a_mu) < 0.05
    assert abs(q.a_eta - true.a_eta) < 0.08
    assert q.r == pytest.approx(true.r, rel=0.25)
    assert q.sigma_mu2 == pytest.approx(true.sigma_mu2, rel=0.25)
    assert q.sigma_eta2 == pytest.approx(true.sigma_eta2, rel=0.6)
    # seasonal shape is recovered up to the level shared with eta
    shape = q.phi - q.phi.mean()
    np.testing.assert_allclose(shape, true.phi - true.phi.mean(), atol=0.08)


def test_em_is_robust_to_initialisation():
    true = default_true_params(26)
    clean, _ = simulate_log_volume(true, 120, seed=3)
    fits = []
    for a_eta, a_mu in [(0.5, 0.2), (0.99, 0.95)]:
        init = fit_em(clean, max_iter=1).params  # data-driven phi / variances
        init.a_eta, init.a_mu = a_eta, a_mu
        fits.append(fit_em(clean, init=init, max_iter=300).params)
    assert abs(fits[0].a_mu - fits[1].a_mu) < 0.02
    assert abs(fits[0].r - fits[1].r) < 0.005


def test_soft_threshold():
    assert soft_threshold(3.0, 1.0) == 2.0
    assert soft_threshold(-3.0, 1.0) == -2.0
    assert soft_threshold(0.5, 1.0) == 0.0


def test_robust_filter_thresholds():
    p = small_params()
    y = np.array([0.0, 5.0, 0.0, 0.0])
    std = kalman_filter(y, p)
    huge = kalman_filter(y, p, lam=1e9)
    np.testing.assert_allclose(std.x_filt, huge.x_filt)
    rob = kalman_filter(y, p, lam=2.0)
    S = rob.P_pred[1, 0] + 2 * rob.P_pred[1, 1] + rob.P_pred[1, 2] + p.r
    e = y[1] - p.phi[1] - rob.x_pred[1].sum()
    assert rob.z[1] == pytest.approx(e - 2.0 * np.sqrt(S))
    paper = kalman_filter(y, p, lam=2.0, lam_mode="paper")
    assert paper.z[1] == pytest.approx(e - 0.5 * 2.0 * S)  # lam / (2W), W = 1/S
    assert abs(rob.x_filt[1].sum()) < abs(std.x_filt[1].sum())


def test_robust_em_ignores_outliers():
    true = default_true_params(26)
    _, dirty = simulate_log_volume(true, 120, seed=5, outlier_frac=0.1, outlier_scale=3.0)
    std = fit_em(dirty).params
    rob = fit_em(dirty, lam=2.0, max_iter=150).params
    # outliers inflate the standard model's noise and wreck a_mu; the robust one stays close
    assert abs(rob.a_mu - true.a_mu) < abs(std.a_mu - true.a_mu)
    assert abs(rob.a_mu - true.a_mu) < 0.15
    assert rob.r < std.r


def test_forecast_and_advance_consistency():
    true = default_true_params(26)
    clean, _ = simulate_log_volume(true, 40, seed=2)
    m = KalmanVolumeModel().fit(clean[:-1])
    st = m.next_day_state()
    assert st.bin == 0
    static = m.forecast_next_day()
    assert static.shape == (26,)
    # advancing with all-NaN observations equals multi-step prediction
    st5 = advance(st, m.params, np.full(5, np.nan))
    mean5, _ = forecast_log_volume(st5, m.params, 21)
    np.testing.assert_allclose(mean5, static[5:])
    # filtering real observations then forecasting matches a one-shot filter
    st_obs = advance(st, m.params, clean[-1, :10])
    fr = kalman_filter(np.r_[clean[:-1].ravel(), clean[-1, :10]], m.params)
    assert st_obs.bin == fr.next_bin == 10
    np.testing.assert_allclose(st_obs.mean, fr.next_mean, atol=1e-10)


def test_forecast_crosses_day_boundary():
    p = small_params(I=4)
    st = VolumeState(mean=(1.0, 1.0), cov=(0.1, 0.0, 0.1), bin=3)
    mean, var = forecast_log_volume(st, p, 2)
    assert mean[0] == pytest.approx(2.0 + p.phi[3])
    assert mean[1] == pytest.approx(p.a_eta * 1.0 + p.a_mu * 1.0 + p.phi[0])
    assert var[1] > var[0]


def test_params_roundtrip():
    p = small_params()
    q = KalmanParams.from_dict(p.to_dict())
    np.testing.assert_allclose(q.phi, p.phi)
    assert q.a_mu == p.a_mu


def test_u_shape():
    phi = u_shape_phi(26)
    assert phi[0] > phi[13] and phi[-1] > phi[13]
