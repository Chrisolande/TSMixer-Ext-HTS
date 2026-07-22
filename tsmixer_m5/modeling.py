import torch
import torch.nn as nn
import torch.nn.functional as F


class BatchNorm2D(nn.Module):
    """Reshape 3D tensor (B, L, C) to apply 1D BatchNorm over combined spatial/temporal dims."""

    def __init__(self, num_time_steps, num_features):
        super().__init__()
        self.bn = nn.BatchNorm1d(num_time_steps * num_features)
        self.L = num_time_steps
        self.C = num_features

    def forward(self, x):
        # x shape: (B, L, C)
        B, L, C = x.shape
        x_flat = x.reshape(B, L * C)
        x_norm = self.bn(x_flat)
        return x_norm.reshape(B, L, C)


class RevIN(nn.Module):
    """Reversible Instance Normalization: x_norm = ((x - mu) / sigma) * gamma + beta."""

    def __init__(self, num_features, eps=1e-5, affine=True):
        super().__init__()
        self.eps = eps
        self.affine = affine
        if self.affine:
            self.affine_weight = nn.Parameter(torch.ones(num_features))
            self.affine_bias = nn.Parameter(torch.zeros(num_features))
        self.mean = None
        self.stdev = None

    def forward(self, x, mode):
        # x shape: (B, L, C)
        if mode == "norm":
            self.get_statistics(x)
            x = self.normalize(x)
        elif mode == "denorm":
            x = self.denormalize(x)
        return x

    def get_statistics(self, x):
        # mu = E[x], sigma = sqrt(Var[x] + eps) along temporal dim L
        self.mean = torch.mean(x, dim=1, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + self.eps).detach()

    def normalize(self, x):
        # x_norm = ((x - mu) / sigma) * weight + bias
        x = (x - self.mean) / self.stdev
        if self.affine:
            x = x * self.affine_weight + self.affine_bias
        return x

    def denormalize(self, x):
        # x_denorm = ((x_norm - bias) / weight) * sigma + mu
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps)
        return x * self.stdev + self.mean


class MeanScaling(nn.Module):
    """Scale normalization: scale = 1.0 + mean(x) along temporal dimension L."""

    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.mean = None

    def forward(self, x, mode, mean=None):
        # x shape: (B, L, C)
        if mode == "norm":
            self.mean = 1.0 + torch.mean(x, dim=1, keepdim=True).detach()
            return x / self.mean, self.mean
        elif mode == "denorm":
            assert mean is not None
            return x * mean


class TemporalProjection(nn.Module):
    """Project time dimensions via transposed linear mapping: (B, L_in, C) -> (B, L_out, C)."""

    def __init__(self, in_len, out_len):
        super().__init__()
        self.linear = nn.Linear(in_len, out_len)

    def forward(self, x):
        # x shape: (B, L_in, C) -> transpose to (B, C, L_in) -> Linear(L_in, L_out) -> transpose to (B, L_out, C)
        B, L, C = x.shape
        x_t = x.transpose(1, 2)
        out_t = self.linear(x_t)
        return out_t.transpose(1, 2)


class TimeMixing(nn.Module):
    """Time-mixing MLP block: Y = Norm(X + Dropout(ReLU(W_t * X^T))^T)."""

    def __init__(self, seq_len, num_features, dropout=0.0, norm_type="layer", pre_norm=False):
        super().__init__()
        self.pre_norm = pre_norm
        self.tp = nn.Linear(seq_len, seq_len)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout)

        if norm_type == "layer":
            self.norm = nn.LayerNorm([seq_len, num_features])
        elif norm_type == "batch":
            self.norm = BatchNorm2D(seq_len, num_features)
        else:
            self.norm = nn.Identity()

    def forward(self, x):
        # x shape: (B, L, C)
        if self.pre_norm:
            x_norm = self.norm(x)
            x_norm_t = x_norm.transpose(1, 2)
            out_t = self.tp(x_norm_t)
            out = out_t.transpose(1, 2)
            return x + self.drop(self.act(out))
        else:
            x_t = x.transpose(1, 2)
            out_t = self.tp(x_t)
            out = out_t.transpose(1, 2)
            return self.norm(x + self.drop(self.act(out)))


class FeatureMixing(nn.Module):
    """Feature-mixing MLP block: Y = Norm(X_proj + W2 * Dropout(ReLU(W1 * X)))."""

    def __init__(
        self,
        in_features,
        out_features,
        seq_len,
        hidden_size=None,
        dropout=0.0,
        norm_type="layer",
        pre_norm=False,
    ):
        super().__init__()
        self.pre_norm = pre_norm
        hidden_size = hidden_size or out_features

        self.linear1 = nn.Linear(in_features, hidden_size)
        self.act = nn.ReLU()
        self.drop1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(hidden_size, out_features)
        self.drop2 = nn.Dropout(dropout)

        if in_features != out_features:
            self.shortcut_proj = nn.Linear(in_features, out_features)
        else:
            self.shortcut_proj = nn.Identity()

        if norm_type == "layer":
            if self.pre_norm:
                self.norm = nn.LayerNorm([seq_len, in_features])
            else:
                self.norm = nn.LayerNorm([seq_len, out_features])
        elif norm_type == "batch":
            if self.pre_norm:
                self.norm = BatchNorm2D(seq_len, in_features)
            else:
                self.norm = BatchNorm2D(seq_len, out_features)
        else:
            self.norm = nn.Identity()

    def forward(self, x):
        # x shape: (B, L, C_in) -> (B, L, C_out)
        if self.pre_norm:
            x_norm = self.norm(x)
            u = self.drop1(self.act(self.linear1(x_norm)))
            out = self.drop2(self.linear2(u))
            return self.shortcut_proj(x) + out
        else:
            u = self.drop1(self.act(self.linear1(x)))
            out = self.drop2(self.linear2(u))
            return self.norm(self.shortcut_proj(x) + out)


class MixerLayer(nn.Module):
    """Sequential combination of TimeMixing followed by FeatureMixing."""

    def __init__(
        self,
        seq_len,
        num_features,
        hidden_size=None,
        dropout=0.0,
        norm_type="layer",
        pre_norm=False,
    ):
        super().__init__()
        self.tm = TimeMixing(seq_len, num_features, dropout, norm_type, pre_norm)
        self.fm = FeatureMixing(
            num_features,
            num_features,
            seq_len,
            hidden_size,
            dropout,
            norm_type,
            pre_norm,
        )

    def forward(self, x):
        # x shape: (B, L, C) -> TimeMixing -> FeatureMixing -> (B, L, C)
        return self.fm(self.tm(x))


class TSMixer(nn.Module):
    """Vanilla TSMixer architecture: RevIN -> Stacked MixerLayers -> TemporalProjection -> RevIN Denorm."""

    def __init__(
        self,
        seq_len,
        pred_len,
        num_features,
        num_blocks=2,
        hidden_size=None,
        dropout=0.0,
        norm_type="layer",
        pre_norm=False,
        use_revin=False,
    ):
        super().__init__()
        self.use_revin = use_revin
        if self.use_revin:
            self.revin = RevIN(num_features)

        self.blocks = nn.ModuleList(
            [MixerLayer(seq_len, num_features, hidden_size, dropout, norm_type, pre_norm) for _ in range(num_blocks)]
        )
        self.tp = TemporalProjection(seq_len, pred_len)

    def forward(self, x):
        # x shape: (B, L_in, C) -> (B, L_out, C)
        if self.use_revin:
            x = self.revin(x, "norm")

        for block in self.blocks:
            x = block(x)

        out = self.tp(x)

        if self.use_revin:
            out = self.revin(out, "denorm")
        return out


class StaticEmbeddingBlock(nn.Module):
    """Static metadata embedder: Concat(Embeddings(cat_cols), cont_cols) -> Linear Projection."""

    def __init__(self, cat_cardinalities, cat_emb_dims, continuous_dim, out_dim):
        super().__init__()
        self.embeddings = nn.ModuleList(
            [nn.Embedding(num_embeddings=card, embedding_dim=dim) for card, dim in zip(cat_cardinalities, cat_emb_dims)]
        )
        total_dim = sum(cat_emb_dims) + continuous_dim
        self.proj = nn.Linear(total_dim, out_dim)

    def forward(self, s_cat, s_cont):
        # s_cat shape: (B, num_cats), s_cont shape: (B, continuous_dim) -> Output: (B, out_dim)
        embedded = []
        for i, emb_layer in enumerate(self.embeddings):
            col_ids = s_cat[:, i].long()
            embedded.append(emb_layer(col_ids))

        s_all = torch.cat(embedded + [s_cont], dim=-1)
        return self.proj(s_all)


class ConditionalFeatureMixing(nn.Module):
    """Feature mixing conditioned on static metadata: Concat(X, Expand(Linear(v_static))) -> FeatureMixing."""

    def __init__(
        self,
        in_features,
        out_features,
        seq_len,
        static_dim,
        hidden_size=None,
        dropout=0.0,
        norm_type="layer",
        pre_norm=False,
    ):
        super().__init__()
        self.fr = nn.Linear(static_dim, out_features)
        self.fm = FeatureMixing(
            in_features + out_features,
            out_features,
            seq_len,
            hidden_size,
            dropout,
            norm_type,
            pre_norm,
        )

    def forward(self, x, v_static):
        # x shape: (B, T, C_in), v_static shape: (B, static_dim) -> Output: (B, T, C_out)
        B, T, _ = x.shape
        v_proj = self.fr(v_static)
        v_expanded = v_proj.unsqueeze(1).expand(-1, T, -1)
        x_concat = torch.cat([x, v_expanded], dim=-1)
        return self.fm(x_concat)


class ConditionalMixerLayer(nn.Module):
    """Mixer layer conditioned on static metadata: TimeMixing -> ConditionalFeatureMixing."""

    def __init__(
        self,
        in_features,
        out_features,
        seq_len,
        static_dim,
        hidden_size=None,
        dropout=0.0,
        norm_type="layer",
        pre_norm=False,
    ):
        super().__init__()
        self.tm = TimeMixing(seq_len, in_features, dropout, norm_type, pre_norm)
        self.cfm = ConditionalFeatureMixing(
            in_features,
            out_features,
            seq_len,
            static_dim,
            hidden_size,
            dropout,
            norm_type,
            pre_norm,
        )

    def forward(self, x, v_static):
        # x shape: (B, T, C_in), v_static shape: (B, static_dim) -> Output: (B, T, C_out)
        return self.cfm(self.tm(x), v_static)


class TSMixerExt(nn.Module):
    """Extended TSMixer with exogenous features, static metadata conditioning, and probabilistic heads."""

    def __init__(
        self,
        seq_len,
        pred_len,
        num_features,
        hist_exog_dim,
        futr_exog_dim,
        static_cont_dim,
        cat_cardinalities=None,
        cat_emb_dims=None,
        num_blocks=2,
        hidden_size=64,
        dropout=0.0,
        norm_type="layer",
        pre_norm=False,
        probabilistic=False,
        use_mean_scaling=False,
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

    def forward(self, x, x_hist_exog, z_futr, s_cat=None, s_cont=None):
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


class NegativeBinomialLoss(nn.Module):
    """Continuous Negative Binomial Log-Likelihood Loss: NLL = -log P(y | mu, alpha)."""

    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, mu, alpha, y):
        # mu: (B, T, C), alpha: (B, T, C), y: (B, T, C)
        mu = torch.clamp(mu, min=self.eps)
        alpha = torch.clamp(alpha, min=self.eps)
        r = 1.0 / alpha
        p = 1.0 / (1.0 + alpha * mu)
        log_prob = (
            torch.lgamma(y + r)
            - torch.lgamma(y + 1.0)
            - torch.lgamma(r)
            + r * torch.log(p + self.eps)
            + y * torch.log(1.0 - p + self.eps)
        )
        return -log_prob.mean()
