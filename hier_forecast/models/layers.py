import torch
import torch.nn as nn


class BatchNorm1D(nn.Module):
    """Applies native 1D BatchNorm over feature channels for (B, L, C) sequence tensors."""

    def __init__(self, num_features: int):
        super().__init__()
        self.bn = nn.BatchNorm1d(num_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, L, C) -> transpose to (B, C, L) -> bn -> transpose back to (B, L, C)
        return self.bn(x.transpose(1, 2)).transpose(1, 2)


class RevIN(nn.Module):
    """Reversible Instance Normalization: x_norm = ((x - mu) / sigma) * gamma + beta."""

    def __init__(self, num_features: int, eps: float = 1e-5, affine: bool = True):
        super().__init__()
        self.eps = eps
        self.affine = affine
        if self.affine:
            self.affine_weight = nn.Parameter(torch.ones(num_features))
            self.affine_bias = nn.Parameter(torch.zeros(num_features))
        self.mean = None
        self.stdev = None

    def forward(self, x: torch.Tensor, mode: str) -> torch.Tensor:
        # x shape: (B, L, C)
        if mode == "norm":
            self.get_statistics(x)
            x = self.normalize(x)
        elif mode == "denorm":
            x = self.denormalize(x)
        return x

    def get_statistics(self, x: torch.Tensor):
        # mu = E[x], sigma = sqrt(Var[x] + eps) along temporal dim L
        self.mean = torch.mean(x, dim=1, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + self.eps).detach()

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        # x_norm = ((x - mu) / sigma) * weight + bias
        x = (x - self.mean) / self.stdev
        if self.affine:
            x = x * self.affine_weight + self.affine_bias
        return x

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        # x_denorm = ((x_norm - bias) / weight) * sigma + mu
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps)
        return x * self.stdev + self.mean


class MeanScaling(nn.Module):
    """Scale normalization: scale = 1.0 + mean(x) along temporal dimension L."""

    def __init__(self, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.mean = None

    def forward(self, x: torch.Tensor, mode: str, mean: torch.Tensor | None = None):
        # x shape: (B, L, C)
        if mode == "norm":
            self.mean = 1.0 + torch.mean(x, dim=1, keepdim=True).detach()
            return x / self.mean, self.mean
        elif mode == "denorm":
            assert mean is not None, "Mean scale required for denormalization"
            return x * mean


class TemporalProjection(nn.Module):
    """Project time dimensions via transposed linear mapping: (B, L_in, C) -> (B, L_out, C)."""

    def __init__(self, in_len: int, out_len: int):
        super().__init__()
        self.linear = nn.Linear(in_len, out_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, L_in, C) -> transpose to (B, C, L_in) -> Linear(L_in, L_out) -> transpose to (B, L_out, C)
        x_t = x.transpose(1, 2)
        out_t = self.linear(x_t)
        return out_t.transpose(1, 2)


class TimeMixing(nn.Module):
    """Time-mixing MLP block: Y = Norm(X + Dropout(ReLU(W_t * X^T))^T)."""

    def __init__(
        self,
        seq_len: int,
        num_features: int,
        dropout: float = 0.0,
        norm_type: str = "layer",
        pre_norm: bool = False,
    ):
        super().__init__()
        self.pre_norm = pre_norm
        self.tp = nn.Linear(seq_len, seq_len)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout)

        if norm_type == "layer":
            self.norm = nn.LayerNorm([seq_len, num_features])
        elif norm_type == "batch":
            self.norm = BatchNorm1D(num_features)
        else:
            self.norm = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
        in_features: int,
        out_features: int,
        seq_len: int,
        hidden_size: int | None = None,
        dropout: float = 0.0,
        norm_type: str = "layer",
        pre_norm: bool = False,
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
                self.norm = BatchNorm1D(in_features)
            else:
                self.norm = BatchNorm1D(out_features)
        else:
            self.norm = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
        seq_len: int,
        num_features: int,
        hidden_size: int | None = None,
        dropout: float = 0.0,
        norm_type: str = "layer",
        pre_norm: bool = False,
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, L, C) -> TimeMixing -> FeatureMixing -> (B, L, C)
        return self.fm(self.tm(x))


class StaticEmbeddingBlock(nn.Module):
    """Static metadata embedder: Concat(Embeddings(cat_cols), cont_cols) -> Linear Projection."""

    def __init__(
        self,
        cat_cardinalities: list[int],
        cat_emb_dims: list[int],
        continuous_dim: int,
        out_dim: int,
    ):
        super().__init__()
        self.embeddings = nn.ModuleList(
            [nn.Embedding(num_embeddings=card, embedding_dim=dim) for card, dim in zip(cat_cardinalities, cat_emb_dims)]
        )
        total_dim = sum(cat_emb_dims) + continuous_dim
        self.proj = nn.Linear(total_dim, out_dim)

    def forward(self, s_cat: torch.Tensor, s_cont: torch.Tensor) -> torch.Tensor:
        # s_cat shape: (B, num_cats), s_cont shape: (B, continuous_dim) -> Output: (B, out_dim)
        embedded = []
        for i, emb_layer in enumerate(self.embeddings):
            col_ids = s_cat[:, i].long()
            if emb_layer.num_embeddings > 0:
                col_ids = torch.clamp(col_ids, 0, emb_layer.num_embeddings - 1)
            embedded.append(emb_layer(col_ids))

        s_all = torch.cat(embedded + [s_cont], dim=-1)
        return self.proj(s_all)


class ConditionalFeatureMixing(nn.Module):
    """Feature mixing conditioned on static metadata: Concat(X, Expand(Linear(v_static))) -> FeatureMixing."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        seq_len: int,
        static_dim: int,
        hidden_size: int | None = None,
        dropout: float = 0.0,
        norm_type: str = "layer",
        pre_norm: bool = False,
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

    def forward(self, x: torch.Tensor, v_static: torch.Tensor) -> torch.Tensor:
        # x shape: (B, T, C_in), v_static shape: (B, static_dim) -> Output: (B, T, C_out)
        _, T, _ = x.shape
        v_proj = self.fr(v_static)
        v_expanded = v_proj.unsqueeze(1).expand(-1, T, -1)
        x_concat = torch.cat([x, v_expanded], dim=-1)
        return self.fm(x_concat)


class ConditionalMixerLayer(nn.Module):
    """Mixer layer conditioned on static metadata: TimeMixing -> ConditionalFeatureMixing."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        seq_len: int,
        static_dim: int,
        hidden_size: int | None = None,
        dropout: float = 0.0,
        norm_type: str = "layer",
        pre_norm: bool = False,
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

    def forward(self, x: torch.Tensor, v_static: torch.Tensor) -> torch.Tensor:
        # x shape: (B, T, C_in), v_static shape: (B, static_dim) -> Output: (B, T, C_out)
        return self.cfm(self.tm(x), v_static)
