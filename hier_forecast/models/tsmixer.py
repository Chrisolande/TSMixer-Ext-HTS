import torch
import torch.nn as nn

from hier_forecast.models.layers import MixerLayer, RevIN, TemporalProjection


class TSMixer(nn.Module):
    """Vanilla TSMixer architecture: RevIN -> Stacked MixerLayers -> TemporalProjection -> RevIN Denorm."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        num_features: int,
        num_blocks: int = 2,
        hidden_size: int | None = None,
        dropout: float = 0.0,
        norm_type: str = "layer",
        pre_norm: bool = False,
        use_revin: bool = False,
    ):
        super().__init__()
        self.use_revin = use_revin
        if self.use_revin:
            self.revin = RevIN(num_features)

        self.blocks = nn.ModuleList(
            [MixerLayer(seq_len, num_features, hidden_size, dropout, norm_type, pre_norm) for _ in range(num_blocks)]
        )
        self.tp = TemporalProjection(seq_len, pred_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, L_in, C) -> (B, L_out, C)
        if self.use_revin:
            x = self.revin(x, "norm")

        for block in self.blocks:
            x = block(x)

        out = self.tp(x)

        if self.use_revin:
            out = self.revin(out, "denorm")
        return out
