import torch
import torch.nn as nn
import torch.nn.functional as F

from hier_forecast.models.layers import (
    ConditionalFeatureMixing,
    ConditionalMixerLayer,
    MeanScaling,
    StaticEmbeddingBlock,
    TemporalProjection,
)


class TSMixerExt(nn.Module):
    """Extended TSMixer with exogenous features, static metadata conditioning, and probabilistic heads."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        num_features: int = 1,
        hist_exog_dim: int = 10,
        futr_exog_dim: int = 10,
        static_cont_dim: int = 1,
        cat_cardinalities: list[int] | None = None,
        cat_emb_dims: list[int] | None = None,
        num_blocks: int = 2,
        hidden_size: int = 64,
        dropout: float = 0.0,
        norm_type: str = "layer",
        pre_norm: bool = False,
        probabilistic: bool = False,
        use_mean_scaling: bool = False,
    ):
        super().__init__()
        self.probabilistic = probabilistic
        self.use_mean_scaling = use_mean_scaling

        if self.use_mean_scaling:
            self.mean_scaler = MeanScaling()

        self.has_embeddings = cat_cardinalities is not None and cat_emb_dims is not None
        if self.has_embeddings:
            self.static_embedder = StaticEmbeddingBlock(
                cat_cardinalities=cat_cardinalities,
                cat_emb_dims=cat_emb_dims,
                continuous_dim=static_cont_dim,
                out_dim=hidden_size,
            )
            static_dim_for_mix = hidden_size
        else:
            self.static_embedder = nn.Linear(static_cont_dim, hidden_size)
            static_dim_for_mix = hidden_size

        self.tp_past = TemporalProjection(seq_len, pred_len)
        self.cfm_past = ConditionalFeatureMixing(
            in_features=num_features + hist_exog_dim,
            out_features=hidden_size,
            seq_len=pred_len,
            static_dim=static_dim_for_mix,
            hidden_size=hidden_size,
            dropout=dropout,
            norm_type=norm_type,
            pre_norm=pre_norm,
        )

        self.cfm_futr = ConditionalFeatureMixing(
            in_features=futr_exog_dim,
            out_features=hidden_size,
            seq_len=pred_len,
            static_dim=static_dim_for_mix,
            hidden_size=hidden_size,
            dropout=dropout,
            norm_type=norm_type,
            pre_norm=pre_norm,
        )

        self.block1 = ConditionalMixerLayer(
            in_features=2 * hidden_size,
            out_features=hidden_size,
            seq_len=pred_len,
            static_dim=static_dim_for_mix,
            hidden_size=hidden_size,
            dropout=dropout,
            norm_type=norm_type,
            pre_norm=pre_norm,
        )

        self.blocks = nn.ModuleList(
            [
                ConditionalMixerLayer(
                    in_features=hidden_size,
                    out_features=hidden_size,
                    seq_len=pred_len,
                    static_dim=static_dim_for_mix,
                    hidden_size=hidden_size,
                    dropout=dropout,
                    norm_type=norm_type,
                    pre_norm=pre_norm,
                )
                for _ in range(num_blocks - 1)
            ]
        )

        if self.probabilistic:
            self.mu_head = nn.Linear(hidden_size, num_features)
            self.alpha_head = nn.Linear(hidden_size, num_features)
        else:
            self.fc_head = nn.Linear(hidden_size, num_features)

    def forward(
        self,
        x: torch.Tensor,
        x_hist_exog: torch.Tensor,
        z_futr: torch.Tensor,
        s_cat: torch.Tensor | None = None,
        s_cont: torch.Tensor | None = None,
    ):
        # x: (B, L, C), x_hist_exog: (B, L, D_hist), z_futr: (B, T, D_futr), s_cat: (B, K), s_cont: (B, S)
        mean_scale = None
        if self.use_mean_scaling:
            x, mean_scale = self.mean_scaler(x, "norm")

        if self.has_embeddings:
            assert s_cat is not None and s_cont is not None, "Metadata category indices required"
            v_static = self.static_embedder(s_cat, s_cont)
        else:
            assert s_cont is not None, "Continuous static features required"
            v_static = self.static_embedder(s_cont)

        past_concat = torch.cat([x, x_hist_exog], dim=-1)
        past_proj = self.tp_past(past_concat)
        x_aligned = self.cfm_past(past_proj, v_static)

        z_aligned = self.cfm_futr(z_futr, v_static)

        h = torch.cat([x_aligned, z_aligned], dim=-1)
        h = self.block1(h, v_static)

        for block in self.blocks:
            h = block(h, v_static)

        if self.probabilistic:
            # Softplus activation ensures strictly positive parameters mu > 0, alpha > 0
            mu = F.softplus(self.mu_head(h)) + 1e-4
            alpha = F.softplus(self.alpha_head(h)) + 1e-4
            if self.use_mean_scaling:
                mu = self.mean_scaler(mu, "denorm", mean_scale)
            return mu, alpha
        else:
            out = self.fc_head(h)
            if self.use_mean_scaling:
                out = self.mean_scaler(out, "denorm", mean_scale)
            return out
