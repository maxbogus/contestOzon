"""T3.19/T3.20: sequence encoders (GRU / LSTM / Transformer / Mamba / RWKV / xLSTM) + attention pooling.

T3.20 (2026-08-30): new-architecture probes — Mamba (S6 selective SSM), RWKV
(token-shift recurrent mixing), xLSTM (mLSTM matrix memory). All pure PyTorch,
self-implemented (R1: no external deps, BSD-compatible), same outer interface:
embedding -> encoder -> attention pooling -> MLP head, d_in=16 (the neural-sweep
config that produced the LB winner)."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding (Transformer)."""

    def __init__(self, d_model: int, max_len: int = 512) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[: x.size(1)]


class SequenceModel(nn.Module):
    """Embedding -> encoder (gru/lstm/transformer) -> attention pooling -> MLP head."""

    def __init__(
        self, d_in: int = 16, hidden: int = 64, layers: int = 2, kind: str = "gru"
    ) -> None:
        super().__init__()
        self.kind = kind
        self.emb = nn.Linear(d_in, hidden)
        if kind == "gru":
            self.encoder = nn.GRU(hidden, hidden, num_layers=layers, batch_first=True)
        elif kind == "lstm":
            self.encoder = nn.LSTM(hidden, hidden, num_layers=layers, batch_first=True)
        elif kind == "transformer":
            layer = nn.TransformerEncoderLayer(
                d_model=hidden, nhead=4, dim_feedforward=hidden * 2, batch_first=True, dropout=0.1
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
            self.pos = PositionalEncoding(hidden)
        else:
            raise ValueError(kind)
        self.query = nn.Parameter(torch.randn(hidden) * 0.05)
        self.head = nn.Sequential(nn.Linear(hidden, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.emb(x))  # [B, L, H]
        if self.kind == "transformer":
            h = self.pos(h)
            out = self.encoder(h)  # [B, L, H]
        elif self.kind == "gru":
            out, _ = self.encoder(h)
        else:
            out, _ = self.encoder(h)
        w = F.softmax(torch.einsum("blh,h->bl", out, self.query), dim=1)
        pooled = torch.einsum("bl,blh->bh", w, out)
        return self.head(pooled).squeeze(-1)


class SequenceModelStatic(nn.Module):
    """Two-input: sequence encoder (GRU/LSTM) + static features -> concat -> MLP."""

    def __init__(
        self,
        d_in: int = 16,
        hidden: int = 128,
        layers: int = 2,
        kind: str = "gru",
        static_dim: int = 102,
    ) -> None:
        super().__init__()
        self.kind = kind
        self.emb = nn.Linear(d_in, hidden)
        if kind == "gru":
            self.encoder = nn.GRU(hidden, hidden, num_layers=layers, batch_first=True)
        elif kind == "lstm":
            self.encoder = nn.LSTM(hidden, hidden, num_layers=layers, batch_first=True)
        else:
            raise ValueError(kind)
        self.query = nn.Parameter(torch.randn(hidden) * 0.05)
        self.head = nn.Sequential(nn.Linear(hidden + static_dim, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x: torch.Tensor, static: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.emb(x))
        out, _ = self.encoder(h)
        w = F.softmax(torch.einsum("blh,h->bl", out, self.query), dim=1)
        pooled = torch.einsum("bl,blh->bh", w, out)
        z = torch.cat([pooled, static], dim=-1)
        return self.head(z).squeeze(-1)


class SequenceModelPatched(nn.Module):
    """Patch-tokenized Transformer (iTransformer/Timer-inspired), CLS pooling.

    T3.19/Phase 2.1 "unusual" variant: instead of one token per day (linear
    emb -> GRU/Transformer), group consecutive days into non-overlapping patches
    of `patch` calendar days, project each patch [P*D] -> H, add sinusoidal
    positional encoding, prepend a CLS token and run a TransformerEncoder; the
    CLS output feeds the MLP head.

    Rationale: compresses L=409 days into ~59 weekly tokens (cheap attention),
    captures weekly/periodic trajectory, filters day-level noise. All input days
    are historical (<= ref_date), so full attention is leak-free (same contract
    as the bidirectional Transformer in SequenceModel).

    R1: pure PyTorch (BSD), self-implemented, no external deps.
    """

    def __init__(
        self,
        d_in: int = 16,
        hidden: int = 128,
        layers: int = 2,
        patch: int = 7,
        nhead: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.patch = patch
        self.emb = nn.Linear(patch * d_in, hidden)
        self.pos = PositionalEncoding(hidden, max_len=512)
        self.cls = nn.Parameter(torch.randn(hidden) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=nhead,
            dim_feedforward=hidden * 2,
            batch_first=True,
            dropout=dropout,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.head = nn.Sequential(nn.Linear(hidden, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        P = self.patch
        n_p = (L + P - 1) // P
        pad = n_p * P - L
        if pad:
            x = F.pad(x, (0, 0, 0, pad))  # zero-pad trailing days
        xp = x.reshape(B, n_p, P * D)  # [B, n_p, P*D]
        h = torch.relu(self.emb(xp))  # [B, n_p, H]
        h = self.pos(h)
        cls = self.cls.view(1, 1, -1).expand(B, 1, -1)
        out = self.encoder(torch.cat([cls, h], dim=1))  # [B, n_p+1, H]
        return self.head(out[:, 0]).squeeze(-1)


# ---------------------------------------------------------------------------
# T3.20: new-architecture probes (Mamba / RWKV / xLSTM) — pure PyTorch, R1
# ---------------------------------------------------------------------------


class MambaBlock(nn.Module):
    """Selective SSM (S6) block — pure PyTorch, no CUDA kernels (R1/R4/R6).

    Faithful to the Mamba paper (Gu & Dao 2023): causal depthwise conv over the
    input, input-dependent dt/B/C projections, Euler-discretized recurrence
    (B_bar = dt*B), and a chunked-parallel selective scan (cumprod/cumsum closed
    form with boundary state carry; dt and A clamped so the cumprod never
    underflows in fp32). Fully parallel GPU ops -> fast on the 8GB card.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 8,
        d_conv: int = 4,
        expand: int = 2,
        d_rank: int | None = None,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = int(expand * d_model)
        self.d_rank = d_rank if d_rank is not None else d_state
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv = nn.Conv1d(
            self.d_inner,
            self.d_inner,
            d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            bias=False,
        )
        self.x_proj = nn.Linear(self.d_inner, self.d_rank + 2 * self.d_state, bias=False)
        self.dt_proj = nn.Linear(self.d_rank, self.d_inner, bias=True)
        # structured A (log-space, negative): -(1, 2, ..., d_state) per channel
        self.A_log = nn.Parameter(
            torch.log(torch.arange(1, self.d_state + 1))
            .view(1, self.d_state)
            .repeat(self.d_inner, 1)
        )
        self.D = nn.Parameter(torch.ones(self.d_inner))  # SSM skip connection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, _ = x.shape
        xz = self.in_proj(x)  # (B, L, 2*d_inner)
        x0, z = xz.chunk(2, dim=-1)
        x0 = F.silu(x0)
        x0 = self.conv(x0.transpose(1, 2))[..., :L].transpose(1, 2)  # causal conv
        x0 = F.silu(x0)
        dt_bc = self.x_proj(x0)  # (B, L, d_rank + 2*d_state)
        dt = F.softplus(self.dt_proj(dt_bc[..., : self.d_rank]))  # (B, L, d_inner)
        Bs = dt_bc[..., self.d_rank : self.d_rank + self.d_state]  # (B, L, d_state)
        Cs = dt_bc[..., self.d_rank + self.d_state :]  # (B, L, d_state)
        A = -torch.exp(self.A_log)  # (d_inner, d_state)
        y = self._scan(x0, dt, A, Bs, Cs)  # (B, L, d_inner)
        return self.out_proj(y * F.silu(z))

    def _scan(
        self,
        x: torch.Tensor,
        delta: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        chunk: int = 32,
    ) -> torch.Tensor:
        """Chunked Hillis-Steele parallel selective scan — division-free.

        Within a chunk the linear recurrence h_t = a_t*h_{t-1} + b_t (a_t =
        exp(dt_t*A) > 0, b_t = dt_t*B_t*x_t) is solved with the Hillis-Steele
        parallel prefix scan over the (a, b) combine: (a2,b2) after (a1,b1) ->
        (a2*a1, a2*b1 + b2). The boundary state is carried across chunks via
        cp = cumprod(a). There is NO division anywhere, so both forward and
        backward stay bounded: the cumprod/cumsum form's backward term
        1/cp^2 ~ 1e27 blew up training (mamba smoke 2026-08-30: fold-B 1.9982).
        Product underflow for long ranges just truncates negligible memory.
        """
        Bn, L, D = x.shape
        N = A.shape[1]
        dA = torch.exp(delta.unsqueeze(-1) * A)  # (B, L, D, N) in (0, 1]
        xb = delta.unsqueeze(-1) * B.unsqueeze(2) * x.unsqueeze(-1)  # (B, L, D, N)
        h = torch.zeros(Bn, D, N, device=x.device)
        ys = []
        for s in range(0, L, chunk):
            C_len = min(chunk, L - s)
            ac = dA[:, s : s + C_len]  # (B, C, D, N)
            bc = xb[:, s : s + C_len]
            hc = self._hillis_steele(ac, bc)  # (B, C, D, N) scan from zero init
            cp = torch.cumprod(ac, dim=1)  # running products (carry decay)
            h_full = hc + cp * h.unsqueeze(1)  # (B, C, D, N)
            h = h_full[:, -1]  # boundary state for the next chunk
            y_c = torch.einsum("bcdn,bcn->bcd", h_full, C[:, s : s + C_len])  # (B, C, D)
            ys.append(y_c)
        return torch.cat(ys, dim=1) + self.D * x

    def _hillis_steele(
        self, a: torch.Tensor, b: torch.Tensor
    ) -> torch.Tensor:
        """Hillis-Steele parallel prefix scan, h_t = a_t*h_{t-1} + b_t (zero init).

        a, b: (B, C, D, N). Returns h: (B, C, D, N). O(C log C) work, O(log C)
        depth; the combine uses products and sums only (division-free, so the
        backward cannot blow up through a 1/x^2 term).
        """
        Bn, C, D, N = a.shape
        C2 = 1 << (C - 1).bit_length()
        if C2 > C:
            pad = C2 - C
            a = torch.cat([a, a.new_ones(Bn, pad, D, N)], dim=1)  # identity a=1
            b = torch.cat([b, b.new_zeros(Bn, pad, D, N)], dim=1)  # identity b=0
        sa, sb = a, b
        step = 1
        while step < C2:
            src = torch.arange(C2, device=a.device) - step
            mask = (src >= 0).view(1, C2, 1, 1)
            sa_src = torch.where(mask, sa[:, src.clamp(min=0)], a.new_ones(Bn, 1, D, N))
            sb_src = torch.where(mask, sb[:, src.clamp(min=0)], a.new_zeros(Bn, 1, D, N))
            sa_new = sa * sa_src
            sb_new = sa * sb_src + sb  # uses OLD sa (before reassignment)
            sa, sb = sa_new, sb_new
            step *= 2
        return sb[:, :C]  # (B, C, D, N) = h_t from zero init


class SequenceModelMamba(nn.Module):
    """Mamba (S6) encoder: emb -> [MambaBlock x layers] -> attention pool -> MLP."""

    def __init__(
        self,
        d_in: int = 16,
        hidden: int = 128,
        layers: int = 2,
        d_state: int = 8,
        expand: int = 2,
    ) -> None:
        super().__init__()
        self.emb = nn.Linear(d_in, hidden)
        self.blocks = nn.ModuleList(
            [MambaBlock(hidden, d_state=d_state, expand=expand) for _ in range(layers)]
        )
        self.query = nn.Parameter(torch.randn(hidden) * 0.05)
        self.head = nn.Sequential(nn.Linear(hidden, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.emb(x))  # [B, L, H]
        for blk in self.blocks:
            h = blk(h)
        w = F.softmax(torch.einsum("blh,h->bl", h, self.query), dim=1)
        pooled = torch.einsum("bl,blh->bh", w, h)
        return self.head(pooled).squeeze(-1)


class RWKVBlock(nn.Module):
    """RWKV-style block: token-shift time-mixing + channel-mixing (pure torch).

    Self-implemented from the RWKV paper (Apache-2.0). Time-mixing: token-shift
    (mix with the previous timestep), per-channel key/value/receptance, and a
    per-channel-decay recurrent state computed in closed form with a parallel
    cumsum (out = sigmoid(r) * state); then a residual channel-mixing FFN with
    squared ReLU gated by receptance. No sequential L-loop -> fast training.
    """

    def __init__(self, d: int, layer_id: int, n_layers: int) -> None:
        super().__init__()
        # RWKV time-mix schedule: deeper layers shift less (keep more history).
        t = 1.0 / (2.0 ** (layer_id / (n_layers + 1)))
        self.mix_r = nn.Parameter(torch.full((1, 1, d), 1.0))
        self.mix_k = nn.Parameter(torch.full((1, 1, d), t))
        self.mix_v = nn.Parameter(torch.full((1, 1, d), t))
        self.ln = nn.LayerNorm(d)
        self.key = nn.Linear(d, d, bias=False)
        self.value = nn.Linear(d, d, bias=False)
        self.receptance = nn.Linear(d, d, bias=False)
        self.output = nn.Linear(d, d, bias=False)
        self.w = nn.Parameter(torch.full((1, 1, d), -3.0))  # per-channel decay logit
        # channel mixing
        self.mix_ck = nn.Parameter(torch.full((1, 1, d), t))
        self.mix_cr = nn.Parameter(torch.full((1, 1, d), 1.0))
        self.ln_c = nn.LayerNorm(d)
        self.ck = nn.Linear(d, d, bias=False)
        self.cr = nn.Linear(d, d, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, d = x.shape
        x_shift = torch.cat([x.new_zeros(B, 1, d), x[:, :-1]], dim=1)  # x_{t-1}
        xn = self.ln(x)
        k = self.key(xn * self.mix_k + x_shift * (1 - self.mix_k))
        v = self.value(xn * self.mix_v + x_shift * (1 - self.mix_v))
        r = torch.sigmoid(self.receptance(xn * self.mix_r + x_shift * (1 - self.mix_r)))
        # parallel WKV: state_t = decay^t * cumsum_{i<=t}(k_i*v_i / decay^i)
        # (exact closed form of state = decay*state + k*v; decay floor 0.9 keeps
        #  the 1/decay^i terms finite in fp32 at L<=409)
        decay = torch.exp(-torch.exp(self.w)).view(1, 1, d).clamp(min=0.9)  # (1, 1, d)
        powers = decay ** torch.arange(1, L + 1, device=x.device).view(1, L, 1)  # (1, L, d)
        kv = k * v  # (B, L, d)
        state = powers * torch.cumsum(kv / powers, dim=1)  # (B, L, d)
        x2 = x + self.output(r * state)
        # channel mixing (residual)
        x_shift2 = torch.cat([x2.new_zeros(B, 1, d), x2[:, :-1]], dim=1)
        xc = self.ln_c(x2)
        kk = self.ck(xc * self.mix_ck + x_shift2 * (1 - self.mix_ck))
        rr = torch.sigmoid(self.cr(xc * self.mix_cr + x_shift2 * (1 - self.mix_cr)))
        return x2 + rr * torch.square(torch.relu(kk))


class SequenceModelRWKV(nn.Module):
    """RWKV encoder: emb -> [RWKVBlock x layers] -> attention pool -> MLP."""

    def __init__(self, d_in: int = 16, hidden: int = 128, layers: int = 2) -> None:
        super().__init__()
        self.emb = nn.Linear(d_in, hidden)
        self.blocks = nn.ModuleList(
            [RWKVBlock(hidden, layer_id=i, n_layers=layers) for i in range(layers)]
        )
        self.query = nn.Parameter(torch.randn(hidden) * 0.05)
        self.head = nn.Sequential(nn.Linear(hidden, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.emb(x))  # [B, L, H]
        for blk in self.blocks:
            h = blk(h)
        w = F.softmax(torch.einsum("blh,h->bl", h, self.query), dim=1)
        pooled = torch.einsum("bl,blh->bh", w, h)
        return self.head(pooled).squeeze(-1)


class mLSTMBlock(nn.Module):
    """xLSTM mLSTM cell (Beck et al. 2024): exponential input gate + matrix memory.

    Pure PyTorch, self-implemented. Per head: matrix memory C (hh x hh) and
    normalizer n (hh); update C = f*C + outer(v, i*k), n = f*n + i*k; output
    h = o * (C^T q) / max(|n^T q|, 1). The input gate is max-stabilized over
    heads (i = exp(logits - max)), so exp never overflows. Projections are
    batched over the whole sequence; the memory recurrence is solved in closed
    form per chunk (chunked-parallel cumprod/cumsum, forget floor 0.3 keeps the
    cumprod positive). Memory O(L*B*H*hh*hh/|chunk|) -> batch capped (128).
    """

    def __init__(self, d: int, n_heads: int = 4) -> None:
        super().__init__()
        assert d % n_heads == 0
        self.d = d
        self.n_heads = n_heads
        self.hh = d // n_heads
        self.ln = nn.LayerNorm(d)
        self.w_q = nn.Linear(d, d, bias=False)
        self.w_k = nn.Linear(d, d, bias=False)
        self.w_v = nn.Linear(d, d, bias=False)
        self.w_i = nn.Linear(d, n_heads, bias=False)  # input gate logits (per head)
        self.w_f = nn.Linear(d, n_heads)  # forget gate logits (bias=0 -> f=0.5 init)
        self.w_o = nn.Linear(d, d, bias=False)  # output gate
        self.out_ln = nn.LayerNorm(d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, _ = x.shape
        H, hh = self.n_heads, self.hh
        # batch all per-step projections (LayerNorm + linear) as full-sequence ops
        xn = F.layer_norm(x, (self.d,), self.ln.weight, self.ln.bias)  # (B, L, d)
        q = self.w_q(xn).view(B, L, H, hh).transpose(1, 2)  # (B, H, L, hh)
        k = self.w_k(xn).view(B, L, H, hh).transpose(1, 2)
        v = self.w_v(xn).view(B, L, H, hh).transpose(1, 2)
        o = torch.sigmoid(self.w_o(xn)).view(B, L, H, hh).transpose(1, 2)
        i_log = self.w_i(xn)  # (B, L, H) input-gate logits
        i = (
            torch.exp(i_log - i_log.max(dim=-1, keepdim=True).values)
            .transpose(1, 2)
            .unsqueeze(-1)  # (B, H, L, 1), in (0, 1]
        )
        # forget floor [0.9, 1): the chunked-parallel cumsum divides by
        # cp = cumprod(f); with f >= 0.9 the within-chunk cp stays >= 0.034 so
        # the backward 1/cp^2 term stays ~O(1e3) — the f>=0.3 floor allowed
        # cp ~ 1.9e-17 -> 1/cp^2 ~ 1e33, the same explosion that broke the
        # Mamba cumsum scan (2026-08-30 smoke, fold-B 1.9982). Long-only memory
        # is fine for cumulative GMV behaviour (the normalizer keeps the output
        # scale bounded).
        f = (0.9 + 0.1 * torch.sigmoid(self.w_f(xn))).transpose(1, 2).unsqueeze(-1)  # (B, H, L, 1)
        ik = i * k  # (B, H, L, hh)
        # chunked-parallel matrix memory: within a chunk the linear recurrence
        # C_t = f_t*C_{t-1} + outer(v, ik) is solved with cumprod/cumsum
        # (C_t = cp_t * (C_0 + sum_{i<=t} outer(v_i, ik_i)/cp_i)); boundary
        # state carried across chunks. f>=0.3 and chunk<=32 keep cp > 0.
        chunk = 32
        outs = []
        C = torch.zeros(B, H, hh, hh, device=x.device)
        n = torch.zeros(B, H, hh, device=x.device)
        for s in range(0, L, chunk):
            C_len = min(chunk, L - s)
            f_c = f[:, :, s : s + C_len]  # (B, H, C, 1)
            ik_c = ik[:, :, s : s + C_len]  # (B, H, C, hh)
            v_c = v[:, :, s : s + C_len]  # (B, H, C, hh)
            q_c = q[:, :, s : s + C_len]  # (B, H, C, hh)
            cp = torch.cumprod(f_c, dim=2)  # (B, H, C, 1)
            iks = ik_c / cp  # (B, H, C, hh)
            C_zero = cp.unsqueeze(-1) * torch.cumsum(
                torch.einsum("bhci,bhcj->bhcij", v_c, iks), dim=2
            )  # (B, H, C, hh, hh)
            n_zero = cp * torch.cumsum(iks, dim=2)  # (B, H, C, hh)
            C_full = C_zero + cp.unsqueeze(-1) * C.unsqueeze(2)  # (B, H, C, hh, hh)
            n_full = n_zero + cp * n.unsqueeze(2)  # (B, H, C, hh)
            C = C_full[:, :, -1]
            n = n_full[:, :, -1]
            qn = torch.einsum("bhci,bhci->bhc", q_c, n_full).abs().clamp_min(1.0)  # (B, H, C)
            hq = torch.einsum("bhci,bhcij->bhcj", q_c, C_full) / qn.unsqueeze(-1)  # (B, H, C, hh)
            outs.append((hq * o[:, :, s : s + C_len]).transpose(1, 2).reshape(B, C_len, self.d))
        return self.out_ln(torch.cat(outs, dim=1))


class SequenceModelXLSTM(nn.Module):
    """xLSTM (mLSTM) encoder: emb -> [mLSTMBlock x layers] -> attention pool -> MLP."""

    def __init__(
        self, d_in: int = 16, hidden: int = 128, layers: int = 2, n_heads: int = 4
    ) -> None:
        super().__init__()
        self.emb = nn.Linear(d_in, hidden)
        self.blocks = nn.ModuleList([mLSTMBlock(hidden, n_heads=n_heads) for _ in range(layers)])
        self.query = nn.Parameter(torch.randn(hidden) * 0.05)
        self.head = nn.Sequential(nn.Linear(hidden, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.emb(x))  # [B, L, H]
        for blk in self.blocks:
            h = blk(h)
        w = F.softmax(torch.einsum("blh,h->bl", h, self.query), dim=1)
        pooled = torch.einsum("bl,blh->bh", w, h)
        return self.head(pooled).squeeze(-1)
