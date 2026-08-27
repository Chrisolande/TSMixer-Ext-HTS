import numpy as np
from scipy import stats

from hier_forecast.models.distribution import NegativeBinomial


def pinball_loss(y_true: np.ndarray, y_pred_q: np.ndarray, q: float) -> float:
    """Pinball / quantile loss for a specific quantile level q in (0, 1)."""
    y_t = np.asarray(y_true, dtype=np.float64)
    y_p = np.asarray(y_pred_q, dtype=np.float64)
    diff = y_t - y_p
    loss = np.maximum(q * diff, (q - 1.0) * diff)
    return float(np.mean(loss))


def weighted_interval_score(
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    alpha: float = 0.2,
) -> float:
    """Interval score for central (1 - alpha) prediction interval [lower, upper]."""
    y_t = np.asarray(y_true, dtype=np.float64)
    lower_arr = np.asarray(lower, dtype=np.float64)
    upper_arr = np.asarray(upper, dtype=np.float64)

    width = upper_arr - lower_arr
    under = np.maximum(0.0, lower_arr - y_t) * (2.0 / alpha)
    over = np.maximum(0.0, y_t - upper_arr) * (2.0 / alpha)

    score = width + under + over
    return float(np.mean(score))


def empirical_coverage(y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    """Empirical coverage proportion of interval [lower, upper]."""
    y_t = np.asarray(y_true, dtype=np.float64)
    lower_arr = np.asarray(lower, dtype=np.float64)
    upper_arr = np.asarray(upper, dtype=np.float64)
    inside = (y_t >= lower_arr) & (y_t <= upper_arr)
    return float(np.mean(inside))


def sharpness(lower: np.ndarray, upper: np.ndarray) -> float:
    """Mean interval width (sharpness)."""
    lower_arr = np.asarray(lower, dtype=np.float64)
    upper_arr = np.asarray(upper, dtype=np.float64)
    return float(np.mean(upper_arr - lower_arr))


def discrete_crps_nb(
    y_true: np.ndarray,
    mu: np.ndarray,
    alpha: np.ndarray,
    max_k: int | None = None,
) -> float:
    """Exact discrete CRPS for Negative Binomial distribution.
    CRPS(F, y) = sum_{k=0}^infty (F(k) - 1(y <= k))^2
    """
    y_t = np.asarray(y_true, dtype=np.float64)
    mu_np = np.asarray(mu, dtype=np.float64)
    alpha_np = np.asarray(alpha, dtype=np.float64)

    dist = NegativeBinomial(mu=mu_np, alpha=alpha_np)
    r, p = dist._to_numpy_params()

    std = np.sqrt(mu_np + alpha_np * (mu_np**2))
    if max_k is None:
        k_upper = int(np.nanmax(y_t) + 4 * np.nanmax(std) + 10)
        k_upper = min(max(k_upper, 30), 2000)
    else:
        k_upper = max_k

    crps_sum = 0.0
    for k in range(k_upper + 1):
        f_k = stats.nbinom.cdf(k, n=r, p=p)
        ind_k = (y_t <= k).astype(np.float64)
        crps_sum += (f_k - ind_k) ** 2

    return float(np.mean(crps_sum))
