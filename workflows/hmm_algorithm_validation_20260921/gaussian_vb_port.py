"""Order-0/full-covariance equations ported from HMM-MAR commit 023905b.
Scope: batch updates only; not a reproduction of PNAS stochastic training.
Source mapping: obsinit, updateW, updateOmega, obslike, hsupdate.
Original authors Diego Vidaurre et al.; see vendored HMM-MAR license.
"""

import numpy as np
from scipy.special import digamma
from scipy.linalg import solve


def priors(x, k):
    d = x.shape[1]
    span = np.ptp(x, axis=0)
    if np.any(span <= 0):
        raise ValueError("Constant channel")
    return dict(
        mean_precision=4 / span**2,
        rate=np.diag(span),
        nu=d - 0.9,
        transition=np.ones((k, k)) + 9 * np.eye(k),
        initial=np.ones(k),
    )


def update(x, g, xi, starts, old, p):
    k = g.shape[1]
    d = x.shape[1]
    n = g.sum(0)
    if np.any(n <= 1e-10):
        raise ValueError("Empty state")
    means = []
    scov = []
    rates = []
    for j in range(k):
        precision = old["nu"][j] * np.linalg.inv(old["rate"][j])
        gram = n[j] * precision
        s = np.linalg.inv(np.diag(p["mean_precision"]) + gram)
        m = s @ precision @ (g[:, j] @ x)
        e = x - m
        rate = p["rate"] + (e * g[:, j, None]).T @ e + n[j] * s
        means.append(m)
        scov.append(s)
        rates.append((rate + rate.T) / 2)
    ad = p["transition"] + xi
    a = np.exp(digamma(ad) - digamma(ad.sum(1))[:, None])
    a /= a.sum(1)[:, None]
    pd = p["initial"] + g[starts].sum(0)
    pi = np.exp(digamma(pd) - digamma(pd.sum()))
    pi /= pi.sum()
    return dict(
        mean=np.array(means),
        mean_cov=np.array(scov),
        rate=np.array(rates),
        nu=p["nu"] + n,
        A=a,
        pi=pi,
        adir=ad,
        pidir=pd,
    )


def log_emission(x, m):
    n, d = x.shape
    k = len(m["mean"])
    out = np.empty((n, k))
    for j in range(k):
        rate = m["rate"][j]
        nu = m["nu"][j]
        prec = nu * np.linalg.inv(rate)
        e = x - m["mean"][j]
        out[:, j] = (
            -d / 2 * np.log(2 * np.pi)
            - 0.5 * np.linalg.slogdet(rate)[1]
            + 0.5 * digamma((nu + 1 - np.arange(1, d + 1)) / 2).sum()
            - 0.5 * np.sum((e @ prec) * e, axis=1)
            - 0.5 * np.sum(prec * m["mean_cov"][j])
        )
    # MATLAB omits d/2*log(2), a state-independent constant; preserved here.
    return out


def infer(loge, A, pi, length):
    """Scaled full-sequence forward/backward; scans remain independent."""
    k = len(pi)
    e = loge.reshape(-1, length, k)
    shift = e.max(2)
    b = np.exp(e - shift[:, :, None])
    ns = len(e)
    alpha = np.empty_like(e)
    scale = np.empty((ns, length))
    alpha[:, 0] = pi * b[:, 0]
    scale[:, 0] = alpha[:, 0].sum(1)
    alpha[:, 0] /= scale[:, 0, None]
    for t in range(1, length):
        alpha[:, t] = (alpha[:, t - 1] @ A) * b[:, t]
        scale[:, t] = alpha[:, t].sum(1)
        alpha[:, t] /= scale[:, t, None]
    beta = np.ones((ns, k))
    g = np.empty_like(e)
    g[:, -1] = alpha[:, -1]
    xi = np.zeros((k, k))
    for t in range(length - 2, -1, -1):
        term = b[:, t + 1] * beta / scale[:, t + 1, None]
        xi += A * (alpha[:, t].T @ term)
        beta = term @ A.T
        g[:, t] = alpha[:, t] * beta
        g[:, t] /= g[:, t].sum(1)[:, None]
    return g.reshape(-1, k), xi, (np.log(scale) + shift).sum()
