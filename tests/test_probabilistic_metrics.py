import numpy as np

from hier_forecast.evaluation.calibration import pit_calibration_test
from hier_forecast.evaluation.probabilistic import (
    discrete_crps_nb,
    empirical_coverage,
    pinball_loss,
    sharpness,
    weighted_interval_score,
)
from hier_forecast.models.distribution import NegativeBinomial


def test_pinball_loss():
    y_true = np.array([2.0, 5.0, 10.0])
    # Underprediction for q=0.9 should penalize with weight q
    y_pred_low = np.array([1.0, 4.0, 8.0])
    loss_09 = pinball_loss(y_true, y_pred_low, q=0.9)
    assert np.isclose(loss_09, np.mean(0.9 * (y_true - y_pred_low)))

    # Overprediction for q=0.1 should penalize with weight (1 - q)
    y_pred_high = np.array([4.0, 7.0, 12.0])
    loss_01 = pinball_loss(y_true, y_pred_high, q=0.1)
    assert np.isclose(loss_01, np.mean(0.9 * (y_pred_high - y_true)))


def test_weighted_interval_score_and_coverage():
    y_true = np.array([3.0, 5.0, 12.0])
    lower = np.array([2.0, 4.0, 6.0])
    upper = np.array([5.0, 7.0, 10.0])

    cov = empirical_coverage(y_true, lower, upper)
    # y=3 in [2,5] (yes), y=5 in [4,7] (yes), y=12 in [6,10] (no) -> 2/3 coverage
    assert np.isclose(cov, 2.0 / 3.0)

    shp = sharpness(lower, upper)
    # widths: 3, 3, 4 -> mean = 10/3
    assert np.isclose(shp, 10.0 / 3.0)

    # WIS for alpha = 0.2 (80% interval)
    wis = weighted_interval_score(y_true, lower, upper, alpha=0.2)
    assert wis > 0.0


def test_discrete_crps_nb():
    y_true = np.array([2, 5, 0])
    mu = np.array([2.2, 4.8, 0.5])
    alpha = np.array([0.3, 0.2, 0.5])

    crps_val = discrete_crps_nb(y_true, mu, alpha)
    assert crps_val >= 0.0
    assert not np.isnan(crps_val)

    # Perfect forecast should yield minimal CRPS
    crps_exact = discrete_crps_nb(np.array([2]), np.array([2.0]), np.array([0.01]))
    crps_bad = discrete_crps_nb(np.array([20]), np.array([2.0]), np.array([0.01]))
    assert crps_bad > crps_exact


def test_pit_calibration_test():
    np.random.seed(42)
    mu = np.array([5.0] * 500)
    alpha = np.array([0.2] * 500)
    dist = NegativeBinomial(mu=mu, alpha=alpha)

    # Sample true counts from the distribution
    r, p = dist._to_numpy_params()
    from scipy.stats import nbinom
    y_sample = nbinom.rvs(n=r, p=p)

    pit_result = pit_calibration_test(y_sample, mu, alpha, num_bins=10)
    assert "uniform_ks_pvalue" in pit_result
    assert "bin_counts" in pit_result
    assert len(pit_result["bin_counts"]) == 10
    # Randomized PIT on true samples should follow uniform distribution (p > 0.01)
    assert pit_result["uniform_ks_pvalue"] > 0.01
