import torch.nn as nn

from hier_forecast.models.distribution import NegativeBinomial


class NegativeBinomialLoss(nn.Module):
    """Continuous Negative Binomial Log-Likelihood Loss: NLL = -log P(y | mu, alpha)."""

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, mu, alpha, y):
        # mu: (B, T, C), alpha: (B, T, C), y: (B, T, C)
        dist = NegativeBinomial(mu=mu, alpha=alpha, eps=self.eps)
        return -dist.log_prob(y).mean()
