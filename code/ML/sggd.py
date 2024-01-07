import numpy as np
import scipy


def sggd(params: np.ndarray, x: np.ndarray):
    beta = params[0]
    mean = params[1]
    alpha = params[2]

    exponent_term = np.exp(-((np.abs(x - mean) / alpha) ** beta))
    scalar = beta / (2 * alpha * scipy.special.gamma(1 / beta))
    ggd = scalar * exponent_term
    return ggd


def sggd_cost(params: np.ndarray, data: np.ndarray):
    return -np.log(sggd(params, data)).sum()


def compute_sggd_variance(beta: float, alpha: float) -> float:
    return ((alpha ** 2) * scipy.special.gamma(3 / beta)) / scipy.special.gamma(1 / beta)



