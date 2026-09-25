"""Kalman-filter model for intraday log-volume (Chen, Feng & Palomar).

State-space model (paper Eqs. 4-5), with tau = I*(t-1) + i:

    x_{tau+1} = A_tau x_tau + w_tau,        x = [eta, mu]'
    y_tau     = C x_tau + phi_tau + v_tau (+ z_tau for the robust model)

* eta   -- log daily-average component. Piecewise constant within a day: it
           only moves (a_eta, sigma_eta^2) on the transition into a new day.
* mu    -- log intraday dynamic component, AR(1) bin to bin (a_mu, sigma_mu^2).
* phi_i -- log intraday periodic (seasonal) component, one value per bin.
* v     -- Gaussian observation noise with variance r.
* z     -- sparse outlier term handled by Lasso-regularised correction (Sec. 3).

Everything is two-dimensional, so the filter/smoother are written with
unrolled scalar arithmetic: that is ~50x faster than 2x2 numpy calls in a
Python loop and keeps the package dependency-light (numpy only).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Optional

import numpy as np

_LOG2PI = math.log(2.0 * math.pi)
_VAR_FLOOR = 1e-10


@dataclass
class KalmanParams:
    """theta = (pi_1, Sigma_1, a_eta, a_mu, sigma_eta^2, sigma_mu^2, r, phi) -- Eq. 6."""

    pi1: np.ndarray  # (2,)
    sigma1: np.ndarray  # (2, 2)
    a_eta: float
    a_mu: float
    sigma_eta2: float
    sigma_mu2: float
    r: float
    phi: np.ndarray  # (I,)

    @property
    def bins_per_day(self) -> int:
        return len(self.phi)

    def copy(self) -> "KalmanParams":
        return replace(self, pi1=self.pi1.copy(), sigma1=self.sigma1.copy(), phi=self.phi.copy())

    def to_dict(self) -> dict:
        return {
            "pi1": self.pi1.tolist(),
            "sigma1": self.sigma1.tolist(),
            "a_eta": self.a_eta,
            "a_mu": self.a_mu,
            "sigma_eta2": self.sigma_eta2,
            "sigma_mu2": self.sigma_mu2,
            "r": self.r,
            "phi": self.phi.tolist(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "KalmanParams":
        return cls(
            pi1=np.asarray(d["pi1"], float),
            sigma1=np.asarray(d["sigma1"], float),
            a_eta=float(d["a_eta"]),
            a_mu=float(d["a_mu"]),
            sigma_eta2=float(d["sigma_eta2"]),
            sigma_mu2=float(d["sigma_mu2"]),
            r=float(d["r"]),
            phi=np.asarray(d["phi"], float),
        )


def initial_params(y: np.ndarray) -> KalmanParams:
    """Data-driven starting point for EM from a (T, I) log-volume matrix.

    The seasonal shape phi absorbs the overall level; eta starts as the first
    day's deviation from it. EM is insensitive to these choices (paper Fig. 4),
    they just shorten the number of iterations.
    """
    y = np.asarray(y, float)
    T, I = y.shape
    phi = np.nanmean(y, axis=0)
    resid = y - phi
    daily = np.nanmean(resid, axis=1)
    intraday = resid - daily[:, None]
    v_day = float(np.nanvar(np.diff(daily))) if T > 2 else 0.05
    v_in = float(np.nanvar(intraday))
    return KalmanParams(
        pi1=np.array([daily[0], 0.0]),
        sigma1=np.diag([max(v_day, 1e-3), max(v_in, 1e-3)]),
        a_eta=0.9,
        a_mu=0.5,
        sigma_eta2=max(0.5 * v_day, 1e-4),
        sigma_mu2=max(0.5 * v_in, 1e-4),
        r=max(0.5 * v_in, 1e-4),
        phi=phi,
    )


# --------------------------------------------------------------------------- #
# Filtering (Algorithm 1, with the robust correction of Eqs. 31-34)
# --------------------------------------------------------------------------- #
@dataclass
class FilterResult:
    x_pred: np.ndarray  # (N, 2)  x_{tau|tau-1}
    P_pred: np.ndarray  # (N, 3)  Sigma_{tau|tau-1} as (p11, p12, p22)
    x_filt: np.ndarray  # (N, 2)  x_{tau|tau}
    P_filt: np.ndarray  # (N, 3)
    z: np.ndarray  # (N,)    robust outlier estimates (0 for the standard filter)
    loglik: float  # Gaussian innovation log-likelihood
    next_mean: tuple  # x_{N+1|N}
    next_cov: tuple  # Sigma_{N+1|N}
    next_bin: int  # bin index (0-based) of observation N+1


def soft_threshold(e: float, thr: float) -> float:
    """Solution of min_z W(e - z)^2 + lam|z| with thr = lam / (2W) -- Eq. 33."""
    if e > thr:
        return e - thr
    if e < -thr:
        return e + thr
    return 0.0


def kalman_filter(
    y: np.ndarray,
    params: KalmanParams,
    lam: Optional[float] = None,
    lam_mode: str = "sigma",
    first_bin: int = 0,
    mean0: Optional[tuple] = None,
    cov0: Optional[tuple] = None,
) -> FilterResult:
    """Run the (robust) Kalman filter over a flat log-volume series.

    ``y`` may contain NaN for unobserved bins: the correction step is skipped
    and the prediction steps run back to back (paper Sec. 2.2).
    ``lam`` enables the Lasso-robust correction; ``None`` gives the standard filter.
    The prior defaults to (pi_1, Sigma_1) for the first observation.

    ``lam_mode`` picks how ``lam`` maps to the soft threshold on the innovation e:

    * ``"paper"`` -- Eq. 34 verbatim: threshold = lam / (2W) = lam * S / 2, where
      S = C Sigma C' + r is the innovation variance.
    * ``"sigma"`` (default) -- threshold = lam * sqrt(S), i.e. the same Lasso
      problem (Eq. 30) with lam_tau = 2*lam/sqrt(S_tau), so ``lam`` is "how many
      predictive standard deviations count as an outlier". With a fixed paper-
      style lam the threshold shrinks faster (~S) than the noise (~sqrt(S)); inside
      EM that feedback can collapse r -> 0 with every bin flagged as an outlier.
      The sigma form keeps the threshold scale-free and EM stable.
    """
    if lam_mode not in ("sigma", "paper"):
        raise ValueError("lam_mode must be 'sigma' or 'paper'")
    sigma_mode = lam_mode == "sigma"
    y = np.asarray(y, float).ravel()
    N = y.size
    I = params.bins_per_day
    phi = params.phi.tolist()
    a_eta, a_mu = params.a_eta, params.a_mu
    q_eta, q_mu, r = params.sigma_eta2, params.sigma_mu2, params.r
    a_eta2, a_mu2, a_em = a_eta * a_eta, a_mu * a_mu, a_eta * a_mu

    if mean0 is None:
        m1, m2 = float(params.pi1[0]), float(params.pi1[1])
    else:
        m1, m2 = mean0
    if cov0 is None:
        p11, p12, p22 = float(params.sigma1[0, 0]), float(params.sigma1[0, 1]), float(params.sigma1[1, 1])
    else:
        p11, p12, p22 = cov0

    xp, Pp, xf, Pf = [None] * N, [None] * N, [None] * N, [None] * N
    z = [0.0] * N
    ll = 0.0
    b = first_bin % I
    yl = y.tolist()
    for t in range(N):
        xp[t] = (m1, m2)
        Pp[t] = (p11, p12, p22)
        yt = yl[t]
        if yt == yt:  # observed (not NaN)
            S = p11 + 2.0 * p12 + p22 + r
            k1 = (p11 + p12) / S
            k2 = (p12 + p22) / S
            e = yt - phi[b] - m1 - m2
            ll -= 0.5 * (_LOG2PI + math.log(S) + e * e / S)
            if lam is not None:
                # W = 1/S; paper threshold lam/(2W) = lam*S/2 (Eq. 34)
                zt = soft_threshold(e, lam * math.sqrt(S) if sigma_mode else 0.5 * lam * S)
                z[t] = zt
                e -= zt
            m1 += k1 * e
            m2 += k2 * e
            p11 -= S * k1 * k1
            p12 -= S * k1 * k2
            p22 -= S * k2 * k2
        xf[t] = (m1, m2)
        Pf[t] = (p11, p12, p22)
        # predict tau -> tau+1
        b += 1
        if b == I:  # day boundary: eta moves
            b = 0
            m1 *= a_eta
            p11 = a_eta2 * p11 + q_eta
            p12 *= a_em
        else:
            p12 *= a_mu
        m2 *= a_mu
        p22 = a_mu2 * p22 + q_mu

    return FilterResult(
        x_pred=np.array(xp).reshape(N, 2),
        P_pred=np.array(Pp).reshape(N, 3),
        x_filt=np.array(xf).reshape(N, 2),
        P_filt=np.array(Pf).reshape(N, 3),
        z=np.array(z),
        loglik=ll,
        next_mean=(m1, m2),
        next_cov=(p11, p12, p22),
        next_bin=b,
    )


# --------------------------------------------------------------------------- #
# Smoothing (Algorithm 2) + lag-one cross covariance
# --------------------------------------------------------------------------- #
@dataclass
class SmootherResult:
    x: np.ndarray  # (N, 2)     x_{tau|N}
    P: np.ndarray  # (N, 3)     Sigma_{tau|N}
    P_cross: np.ndarray  # (N, 2, 2)  Sigma_{tau,tau-1|N}; row 0 unused


def rts_smoother(fr: FilterResult, params: KalmanParams, first_bin: int = 0) -> SmootherResult:
    """Rauch-Tung-Striebel smoother.

    The lag-one covariance uses the identity Sigma_{tau+1,tau|N} = Sigma_{tau+1|N} L_tau',
    which is algebraically equivalent to the recursion in Eqs. A.20-A.21.
    """
    N = fr.x_filt.shape[0]
    I = params.bins_per_day
    a_eta, a_mu = params.a_eta, params.a_mu
    xf, Pf, xp, Pp = fr.x_filt.tolist(), fr.P_filt.tolist(), fr.x_pred.tolist(), fr.P_pred.tolist()
    xs = [None] * N
    Ps = [None] * N
    Pc = [(0.0, 0.0, 0.0, 0.0)] * N
    xs[N - 1] = tuple(xf[N - 1])
    Ps[N - 1] = tuple(Pf[N - 1])
    for t in range(N - 2, -1, -1):
        a1 = a_eta if (first_bin + t + 1) % I == 0 else 1.0
        f11, f12, f22 = Pf[t]
        q11, q12, q22 = Pp[t + 1]
        det = q11 * q22 - q12 * q12
        i11, i12, i22 = q22 / det, -q12 / det, q11 / det
        # B = Sigma_{t|t} A'
        b11, b12, b21, b22 = f11 * a1, f12 * a_mu, f12 * a1, f22 * a_mu
        L11 = b11 * i11 + b12 * i12
        L12 = b11 * i12 + b12 * i22
        L21 = b21 * i11 + b22 * i12
        L22 = b21 * i12 + b22 * i22
        s1, s2 = xs[t + 1]
        d1, d2 = s1 - xp[t + 1][0], s2 - xp[t + 1][1]
        xs[t] = (xf[t][0] + L11 * d1 + L12 * d2, xf[t][1] + L21 * d1 + L22 * d2)
        S11, S12, S22 = Ps[t + 1]
        D11, D12, D22 = S11 - q11, S12 - q12, S22 - q22
        LD11 = L11 * D11 + L12 * D12
        LD12 = L11 * D12 + L12 * D22
        LD21 = L21 * D11 + L22 * D12
        LD22 = L21 * D12 + L22 * D22
        Ps[t] = (
            f11 + LD11 * L11 + LD12 * L12,
            f12 + LD11 * L21 + LD12 * L22,
            f22 + LD21 * L21 + LD22 * L22,
        )
        Pc[t + 1] = (
            S11 * L11 + S12 * L12,
            S11 * L21 + S12 * L22,
            S12 * L11 + S22 * L12,
            S12 * L21 + S22 * L22,
        )
    return SmootherResult(
        x=np.array(xs).reshape(N, 2),
        P=np.array(Ps).reshape(N, 3),
        P_cross=np.array(Pc).reshape(N, 2, 2),
    )


# --------------------------------------------------------------------------- #
# EM calibration (Algorithm 3; robust variant Eqs. 35-36)
# --------------------------------------------------------------------------- #
def m_step(y: np.ndarray, sm: SmootherResult, z: np.ndarray, I: int) -> KalmanParams:
    """Closed-form M-step, Eqs. 17-24 (Eqs. 35-36 when z != 0)."""
    y = np.asarray(y, float).ravel()
    N = y.size
    T = N // I
    xs, Ps, Pc = sm.x, sm.P, sm.P_cross

    P11 = Ps[:, 0] + xs[:, 0] ** 2
    P22 = Ps[:, 2] + xs[:, 1] ** 2
    Pc11 = Pc[:, 0, 0] + np.r_[0.0, xs[1:, 0] * xs[:-1, 0]]
    Pc22 = Pc[:, 1, 1] + np.r_[0.0, xs[1:, 1] * xs[:-1, 1]]

    # eta: only the T-1 day-boundary transitions tau = kI+1 carry information
    bnd = np.arange(I, N, I)
    a_eta = float(Pc11[bnd].sum() / P11[bnd - 1].sum())
    sigma_eta2 = float(
        np.mean(P11[bnd] + a_eta**2 * P11[bnd - 1] - 2.0 * a_eta * Pc11[bnd])
    )
    a_mu = float(Pc22[1:].sum() / P22[:-1].sum())
    sigma_mu2 = float(np.mean(P22[1:] + a_mu**2 * P22[:-1] - 2.0 * a_mu * Pc22[1:]))

    cx = xs[:, 0] + xs[:, 1]
    resid = (y - cx - z).reshape(T, I)
    phi = np.nanmean(resid, axis=0)
    obs = ~np.isnan(y)
    e = y - np.tile(phi, T) - cx - z
    cpc = Ps[:, 0] + 2.0 * Ps[:, 1] + Ps[:, 2]
    r = float(np.mean(e[obs] ** 2 + cpc[obs]))

    sigma1 = np.array([[Ps[0, 0], Ps[0, 1]], [Ps[0, 1], Ps[0, 2]]])
    sigma1 = sigma1 + np.eye(2) * _VAR_FLOOR
    return KalmanParams(
        pi1=xs[0].copy(),
        sigma1=sigma1,
        a_eta=a_eta,
        a_mu=a_mu,
        sigma_eta2=max(sigma_eta2, _VAR_FLOOR),
        sigma_mu2=max(sigma_mu2, _VAR_FLOOR),
        r=max(r, _VAR_FLOOR),
        phi=phi,
    )


@dataclass
class FitResult:
    params: KalmanParams
    filter: FilterResult
    loglik_path: list = field(default_factory=list)
    n_iter: int = 0
    converged: bool = False


def fit_em(
    y: np.ndarray,
    lam: Optional[float] = None,
    init: Optional[KalmanParams] = None,
    max_iter: int = 100,
    tol: float = 1e-6,
    lam_mode: str = "sigma",
) -> FitResult:
    """Calibrate theta by EM on a (T, I) log-volume matrix.

    Stops when the innovation log-likelihood improves by less than ``tol`` per
    observation. (The split of the overall level between eta and phi is only
    weakly identified when a_eta ~ 1 and drifts slowly along a flat ridge, so a
    parameter-change criterion would keep iterating without changing forecasts.)
    """
    y = np.asarray(y, float)
    if y.ndim != 2:
        raise ValueError("y must be a (days, bins) matrix")
    T, I = y.shape
    if T < 3:
        raise ValueError("need at least 3 days of history to calibrate")
    flat = y.ravel()
    params = init.copy() if init is not None else initial_params(y)
    if len(params.phi) != I:
        raise ValueError("init params have a different number of bins per day")
    n_obs = max(int(np.sum(~np.isnan(flat))), 1)
    lls = []
    converged = False
    it = 0
    for it in range(1, max_iter + 1):
        fr = kalman_filter(flat, params, lam=lam, lam_mode=lam_mode)
        if lls and abs(fr.loglik - lls[-1]) < tol * n_obs:
            converged = True
            break
        lls.append(fr.loglik)
        sm = rts_smoother(fr, params)
        params = m_step(flat, sm, fr.z, I)
    fr = kalman_filter(flat, params, lam=lam, lam_mode=lam_mode)
    return FitResult(params=params, filter=fr, loglik_path=lls, n_iter=it, converged=converged)


# --------------------------------------------------------------------------- #
# Online use: state tracking + static / dynamic forecasts (Eq. 9)
# --------------------------------------------------------------------------- #
@dataclass
class VolumeState:
    """Predicted state x_{tau|tau-1} for the upcoming bin ``bin`` (0-based)."""

    mean: tuple
    cov: tuple
    bin: int


def forecast_log_volume(state: VolumeState, params: KalmanParams, horizon: int):
    """h-step-ahead log-volume mean and variance, starting at ``state.bin``.

    Returns (mean[h], var[h]); var includes the observation noise r.
    """
    I = params.bins_per_day
    m1, m2 = state.mean
    p11, p12, p22 = state.cov
    b = state.bin
    means = np.empty(horizon)
    vars_ = np.empty(horizon)
    for h in range(horizon):
        means[h] = m1 + m2 + params.phi[b]
        vars_[h] = p11 + 2.0 * p12 + p22 + params.r
        b += 1
        if b == I:
            b = 0
            m1 *= params.a_eta
            p11 = params.a_eta**2 * p11 + params.sigma_eta2
            p12 *= params.a_eta * params.a_mu
        else:
            p12 *= params.a_mu
        m2 *= params.a_mu
        p22 = params.a_mu**2 * p22 + params.sigma_mu2
    return means, vars_


def advance(
    state: VolumeState,
    params: KalmanParams,
    y_obs,
    lam: Optional[float] = None,
    lam_mode: str = "sigma",
) -> VolumeState:
    """Consume observations (NaN = missing) for bins starting at ``state.bin``."""
    y_obs = np.atleast_1d(np.asarray(y_obs, float))
    if y_obs.size == 0:
        return state
    fr = kalman_filter(
        y_obs, params, lam=lam, lam_mode=lam_mode, first_bin=state.bin, mean0=state.mean, cov0=state.cov
    )
    return VolumeState(mean=fr.next_mean, cov=fr.next_cov, bin=fr.next_bin)


class KalmanVolumeModel:
    """Convenience wrapper: calibrate on history, then forecast the next day.

    Parameters
    ----------
    lam : Lasso regularisation for the robust filter (None = standard filter).
    lam_mode : "sigma" (default) or "paper"; see ``kalman_filter``.
    """

    def __init__(
        self,
        lam: Optional[float] = None,
        max_iter: int = 100,
        tol: float = 1e-6,
        lam_mode: str = "sigma",
    ):
        self.lam = lam
        self.lam_mode = lam_mode
        self.max_iter = max_iter
        self.tol = tol
        self.fit_: Optional[FitResult] = None

    @property
    def params(self) -> KalmanParams:
        if self.fit_ is None:
            raise RuntimeError("model is not calibrated; call fit() first")
        return self.fit_.params

    def fit(self, log_volume: np.ndarray, init: Optional[KalmanParams] = None) -> "KalmanVolumeModel":
        self.fit_ = fit_em(
            log_volume, lam=self.lam, init=init, max_iter=self.max_iter, tol=self.tol, lam_mode=self.lam_mode
        )
        return self

    def next_day_state(self) -> VolumeState:
        """Predicted state for bin 0 of the day after the training window."""
        fr = self.fit_.filter
        return VolumeState(mean=fr.next_mean, cov=fr.next_cov, bin=fr.next_bin)

    def forecast_next_day(self) -> np.ndarray:
        """Static prediction: log-volume for all I bins of the next day."""
        return forecast_log_volume(self.next_day_state(), self.params, self.params.bins_per_day)[0]
