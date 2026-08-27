import numpy as np
from scipy import stats

from hier_forecast.models.distribution import NegativeBinomial


def pit_calibration_test(
    y_true: np.ndarray,
    mu: np.ndarray,
    alpha: np.ndarray,
    num_bins: int = 10,
    randomized: bool = True,
) -> dict:
    """Evaluate Probability Integral Transform (PIT) uniformity for probabilistic calibration."""
    y_t = np.asarray(y_true, dtype=np.float64)
    dist = NegativeBinomial(mu=mu, alpha=alpha)
    pit_values = dist.pit(y_t, randomized=randomized)

    ks_stat, p_value = stats.kstest(pit_values.flatten(), "uniform")
    hist, bin_edges = np.histogram(pit_values.flatten(), bins=num_bins, range=(0.0, 1.0))

    return {
        "uniform_ks_stat": float(ks_stat),
        "uniform_ks_pvalue": float(p_value),
        "bin_counts": hist.tolist(),
        "bin_edges": bin_edges.tolist(),
    }
