import numpy as np
import torch
from scipy import special, stats


class NegativeBinomial:
    """Negative Binomial distribution wrapper for discrete count forecasts and probabilistic evaluation.

    Parameterization:
      mu: expected value (mu > 0)
      alpha: overdispersion parameter (alpha > 0)
      r = 1.0 / alpha (number of failures)
      p = 1.0 / (1.0 + alpha * mu) (probability of success)
      Var(Y) = mu + alpha * mu^2
    """

    def __init__(self, mu, alpha, eps: float = 1e-6):
        self.eps = eps
        if isinstance(mu, torch.Tensor):
            self.mu = torch.clamp(mu, min=eps)
            self.alpha = torch.clamp(alpha, min=eps)
            self.r = 1.0 / self.alpha
            self.p = 1.0 / (1.0 + self.alpha * self.mu)
            self.variance = self.mu + self.alpha * (self.mu**2)
            self._is_torch = True
        else:
            self.mu = np.clip(np.asarray(mu, dtype=np.float64), a_min=eps, a_max=None)
            self.alpha = np.clip(np.asarray(alpha, dtype=np.float64), a_min=eps, a_max=None)
            self.r = 1.0 / self.alpha
            self.p = 1.0 / (1.0 + self.alpha * self.mu)
            self.variance = self.mu + self.alpha * (self.mu**2)
            self._is_torch = False

    @property
    def mean(self):
        return self.mu

    def log_prob(self, y):
        """Negative Binomial log-likelihood for observed count y."""
        if self._is_torch:
            y_t = torch.as_tensor(y, dtype=self.mu.dtype, device=self.mu.device)
            r = self.r
            p = self.p
            return (
                torch.lgamma(y_t + r)
                - torch.lgamma(y_t + 1.0)
                - torch.lgamma(r)
                + r * torch.log(p + self.eps)
                + y_t * torch.log(1.0 - p + self.eps)
            )
        else:
            y_np = np.asarray(y, dtype=np.float64)
            r = self.r
            p = self.p
            return (
                special.gammaln(y_np + r)
                - special.gammaln(y_np + 1.0)
                - special.gammaln(r)
                + r * np.log(p + self.eps)
                + y_np * np.log(1.0 - p + self.eps)
            )

    def _to_numpy_params(self):
        if self._is_torch:
            r = self.r.detach().cpu().numpy().astype(np.float64)
            p = self.p.detach().cpu().numpy().astype(np.float64)
        else:
            r = self.r
            p = self.p
        return r, p

    def cdf(self, k):
        """Cumulative distribution function P(Y <= k)."""
        r, p = self._to_numpy_params()
        k_np = np.asarray(k, dtype=np.float64)
        return stats.nbinom.cdf(k_np, n=r, p=p)

    def ppf(self, q):
        """Percent point function (inverse CDF / quantile). Smallest integer k >= 0 with F(k) >= q."""
        r, p = self._to_numpy_params()
        q_np = np.asarray(q, dtype=np.float64)
        quantiles = stats.nbinom.ppf(q_np, n=r, p=p)
        quantiles = np.nan_to_num(quantiles, nan=0.0)
        return np.maximum(0, quantiles).astype(int)

    @property
    def median(self):
        """Exact discrete median (q = 0.5)."""
        return self.ppf(0.5)

    def pit(self, y, randomized: bool = True):
        """Probability Integral Transform (PIT) value in [0, 1]."""
        r, p = self._to_numpy_params()
        y_np = np.asarray(y, dtype=np.float64)
        upper = stats.nbinom.cdf(y_np, n=r, p=p)
        if not randomized:
            return upper
        lower = stats.nbinom.cdf(y_np - 1.0, n=r, p=p)
        lower = np.where(y_np <= 0, 0.0, lower)
        u = np.random.uniform(0.0, 1.0, size=y_np.shape)
        return lower + u * (upper - lower)
