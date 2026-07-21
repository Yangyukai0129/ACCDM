# import math
# import torch
# import torch.nn as nn
# import torch.nn.functional as F


# # =========================================================
# # <<< 時間嵌入（與 UNet 完全相同）>>>
# # =========================================================
# def sinusoidal_embedding(t, dim, device):
#     half_dim = dim // 2
#     emb = math.log(10000) / (half_dim - 1)
#     emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
#     emb = t[:, None].float() * emb[None, :]
#     emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
#     return emb


# # =========================================================
# # <<< 時間 Transformer >>>
# # =========================================================
# class TemporalTransformerLayer(nn.Module):
#     """
#     時間 Transformer：讓 C 個預報時次（channel）互相 attend。

#     設計邏輯：
#         輸入 (B, C, H, W)
#         → reshape 成 (B, C, H*W)
#         → C 個時步是 seq，H*W 是 token 維度
#         → Self-Attention 讓不同時次互相溝通
#         → reshape 回 (B, C, H, W)

#     spatial_size : H * W，用於初始化 LayerNorm 和 Linear
#     """
#     def __init__(self, channels, spatial_size, num_heads=4, dropout=0.1):
#         super().__init__()
#         self.norm = nn.LayerNorm(spatial_size)
#         self.attn = nn.MultiheadAttention(spatial_size, num_heads,
#                                           dropout=dropout, batch_first=True)
#         self.ff   = nn.Sequential(
#             nn.LayerNorm(spatial_size),
#             nn.Linear(spatial_size, spatial_size * 2),
#             nn.GELU(),
#             nn.Linear(spatial_size * 2, spatial_size),
#         )

#     def forward(self, x):
#         B, C, H, W = x.shape
#         x_t      = x.reshape(B, C, H * W)        # (B, C, H*W)
#         normed   = self.norm(x_t)
#         attn_out, _ = self.attn(normed, normed, normed)  # (B, C, H*W)
#         x_t      = x_t + attn_out                 # residual
#         x_t      = x_t + self.ff(x_t)
#         return x_t.reshape(B, C, H, W)


# # =========================================================
# # <<< 空間 Transformer >>>
# # =========================================================
# class SpatialTransformerLayer(nn.Module):
#     """
#     空間 Transformer：讓 H*W 個空間位置互相 attend。

#     設計邏輯：
#         輸入 (B, C, H, W)
#         → permute + reshape 成 (B, H*W, C)
#         → H*W 個空間位置是 seq，C 是 token 維度
#         → Self-Attention 讓不同位置互相溝通
#         → reshape 回 (B, C, H, W)
#     """
#     def __init__(self, channels, num_heads=4, dropout=0.1):
#         super().__init__()
#         self.norm = nn.LayerNorm(channels)
#         self.attn = nn.MultiheadAttention(channels, num_heads,
#                                           dropout=dropout, batch_first=True)
#         self.ff   = nn.Sequential(
#             nn.LayerNorm(channels),
#             nn.Linear(channels, channels * 4),
#             nn.GELU(),
#             nn.Linear(channels * 4, channels),
#         )

#     def forward(self, x):
#         B, C, H, W = x.shape
#         x_s      = x.permute(0, 2, 3, 1).reshape(B, H * W, C)  # (B, H*W, C)
#         normed   = self.norm(x_s)
#         attn_out, _ = self.attn(normed, normed, normed)          # (B, H*W, C)
#         x_s      = x_s + attn_out                                # residual
#         x_s      = x_s + self.ff(x_s)
#         return x_s.reshape(B, H, W, C).permute(0, 3, 1, 2)      # (B, C, H, W)


# # =========================================================
# # <<< CSDI Block >>>
# # =========================================================
# class CSDIBlock(nn.Module):
#     """
#     一個 CSDI Block = 時間 Transformer + 空間 Transformer。
#     先讓不同時步互相溝通，再讓不同空間位置互相溝通。
#     """
#     def __init__(self, channels, spatial_size, num_heads=4, dropout=0.1):
#         super().__init__()
#         self.temporal = TemporalTransformerLayer(channels, spatial_size, num_heads, dropout)
#         self.spatial  = SpatialTransformerLayer(channels, num_heads, dropout)

#     def forward(self, x):
#         x = self.temporal(x)
#         x = self.spatial(x)
#         return x


# # =========================================================
# # <<< CSDITransformer >>>
# # =========================================================
# class CSDITransformer(nn.Module):
#     """
#     CSDI-B Transformer 模型。

#     介面與 UNet 完全相同：
#         model(x_t, cond, t, beta) -> pred_noise

#     內部架構：
#         1. channel concat：noisy target + cond -> init_conv -> (B, d_model, H, W)
#         2. 時間嵌入注入（與 UNet 相同）
#         3. N 個 CSDIBlock（時間 Transformer + 空間 Transformer）
#         4. 輸出卷積 -> pred_noise (B, out_ch, H, W)

#     config 需要包含：
#         in_steps      : int   — 輸入時間步數（cond channel = in_steps * 2）
#         target_steps  : int   — 預測時間步數（out_ch）
#         spatial_size  : int   — H * W（例如格網 32x64 則填 2048）
#         d_model       : int   — Transformer hidden dim（建議 64~256，預設 128）
#         num_layers    : int   — CSDIBlock 數量（建議 4~8，預設 6）
#         num_heads     : int   — Attention head 數（建議 4~8，預設 4）
#         dropout       : float — 預設 0.1
#         time_dim      : int   — 預設 32
#     """

#     def __init__(self, config):
#         super().__init__()
#         self.config = config

#         cond_ch      = config['in_steps'] * 2
#         out_ch       = config['target_steps']
#         spatial_size = config['spatial_size']        # H * W
#         d_model      = config.get('d_model',    128)
#         n_layers     = config.get('num_layers',   6)
#         n_heads      = config.get('num_heads',    4)
#         dropout      = config.get('dropout',    0.1)
#         time_dim     = config.get('time_dim',    32)

#         self.out_ch      = out_ch
#         self.cond_ch     = cond_ch
#         self.time_dim    = time_dim

#         # 時間嵌入（與 UNet 相同）
#         self.time_embed = nn.Sequential(
#             nn.Linear(time_dim, time_dim * 4),
#             nn.SiLU(),
#             nn.Linear(time_dim * 4, time_dim),
#         )

#         # 初始卷積：noisy target + cond -> d_model channels
#         self.init_conv = nn.Conv2d(out_ch + cond_ch, d_model, kernel_size=3, padding=1)
#         self.time_proj = nn.Linear(time_dim, d_model)

#         # N 個 CSDI Block
#         self.blocks = nn.ModuleList([
#             CSDIBlock(d_model, spatial_size, n_heads, dropout)
#             for _ in range(n_layers)
#         ])

#         # 輸出
#         self.final_norm = nn.GroupNorm(8, d_model)
#         self.final_conv = nn.Sequential(
#             nn.SiLU(),
#             nn.Conv2d(d_model, out_ch, kernel_size=1),
#         )

#     def forward(self, x, cond, t, beta):
#         """
#         x    : (B, out_ch,  H, W) — noisy target
#         cond : (B, cond_ch, H, W) — 過去天氣場（條件）
#         t    : (B,)               — diffusion 時間步（long tensor）
#         beta : (T,) or None       — 保持介面與 UNet 一致，此模型不直接使用
#         """
#         B, C, H, W = x.shape
#         device = x.device

#         # 時間嵌入
#         t_emb = sinusoidal_embedding(t, self.time_dim, device)
#         t_emb = self.time_embed(t_emb)                              # (B, time_dim)

#         # 初始特徵
#         h = torch.cat([x, cond], dim=1)                            # (B, out_ch+cond_ch, H, W)
#         h = self.init_conv(h)                                       # (B, d_model, H, W)
#         h = h + self.time_proj(t_emb).unsqueeze(-1).unsqueeze(-1)  # 注入時間資訊

#         # N 個 CSDI Block（時間 + 空間 Self-Attention）
#         for block in self.blocks:
#             h = block(h)

#         # 輸出
#         h   = self.final_norm(h)
#         out = self.final_conv(h)                                    # (B, out_ch, H, W)
#         return out

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================================================
# <<< 時間嵌入（與 UNet 完全相同）>>>
# =========================================================
def sinusoidal_embedding(t, dim, device):
    half_dim = dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
    emb = t[:, None].float() * emb[None, :]
    emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
    return emb


# =========================================================
# <<< 特徵 Attention（對應 CSDI 的時間 Transformer）>>>
# =========================================================
class FeatureAttention(nn.Module):
    """
    對 d_model 個特徵維度做 Self-Attention。

    設計邏輯：
        (B, d_model, H, W)
        → (B, H*W, d_model)   H*W 個空間位置是 seq，d_model 是 token 維度
        → Self-Attention 讓每個空間位置的特徵維度互相溝通
        → (B, d_model, H, W)

    seq 長度 = H*W（但 token 維度小，計算量不大）
    OOM 風險：attention 矩陣 (B, H*W, H*W)，H*W=5400 → 還是大
    → 改成 (B*H*W, 1, d_model) 不行，seq=1 沒意義
    → 正確做法：把 H*W 當 batch，d_model 當 seq
      (B*H*W, d_model_seq=1, ...) 也不對

    最終正確：
        用 (B, H*W, d_model)，seq=H*W，dim=d_model
        attention 矩陣 = (B, H*W, H*W) 還是 OOM

    所以 Feature Attention 改成：
        (B*H*W, 1, d_model) → 每個位置獨立做 channel-wise attention 沒意義
        改成 channel attention（SE-style）：
        global avg pool → MLP → channel weight → multiply
        不是真正的 attention，但效果類似且記憶體友好
    """
    def __init__(self, d_model, reduction=4):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)   # global average pooling
        self.fc  = nn.Sequential(
            nn.Linear(d_model, d_model // reduction),
            nn.GELU(),
            nn.Linear(d_model // reduction, d_model),
            nn.Sigmoid(),
        )
        self.norm = nn.GroupNorm(8, d_model)

    def forward(self, x):
        # x: (B, d_model, H, W)
        B, C, H, W = x.shape
        w = self.gap(x).squeeze(-1).squeeze(-1)  # (B, d_model)
        w = self.fc(w).unsqueeze(-1).unsqueeze(-1)  # (B, d_model, 1, 1)
        return self.norm(x * w) + x              # channel re-weighting + residual


# =========================================================
# <<< Patch 空間 Attention（對應 CSDI 的特徵 Transformer）>>>
# =========================================================
class PatchSpatialAttention(nn.Module):
    """
    Patch-based 空間 Self-Attention：避免 H*W seq 過長造成 OOM。

    設計邏輯：
        把 H×W 切成 N 個 patch（每個 patch_size×patch_size）
        N = (H/p) * (W/p)
        H=30, W=180, p=6 → N = 5*30 = 150（可接受）

        (B, d_model, H, W)
        → 切 patch → (B, N, patch_dim)   patch_dim = p*p*d_model
        → Linear 投影 → (B, N, d_model)
        → Self-Attention（seq=150，dim=d_model，記憶體友好）
        → 投影回 patch_dim → 還原空間 → (B, d_model, H, W)
    """
    def __init__(self, d_model, patch_size=6, num_heads=4, dropout=0.1):
        super().__init__()
        self.patch_size = patch_size
        patch_dim       = patch_size * patch_size * d_model

        self.proj_in  = nn.Linear(patch_dim, d_model)
        self.norm     = nn.LayerNorm(d_model)
        self.attn     = nn.MultiheadAttention(d_model, num_heads,
                                              dropout=dropout, batch_first=True)
        self.ff       = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )
        self.proj_out = nn.Linear(d_model, patch_dim)

    def forward(self, x):
        B, C, H, W = x.shape
        p = self.patch_size

        # padding 讓 H, W 能被 p 整除
        pad_h = (p - H % p) % p
        pad_w = (p - W % p) % p
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        _, _, H2, W2 = x.shape
        nh, nw = H2 // p, W2 // p

        # 切 patch
        x_p = x.reshape(B, C, nh, p, nw, p)
        x_p = x_p.permute(0, 2, 4, 1, 3, 5).contiguous()  # (B, nh, nw, C, p, p)
        x_p = x_p.reshape(B, nh * nw, C * p * p)           # (B, N, patch_dim)

        # 投影 + attention
        x_p      = self.proj_in(x_p)                        # (B, N, d_model)
        normed   = self.norm(x_p)
        attn_out, _ = self.attn(normed, normed, normed)     # (B, N, d_model)
        x_p      = x_p + attn_out
        x_p      = x_p + self.ff(x_p)

        # 投影回空間
        x_p = self.proj_out(x_p)                            # (B, N, patch_dim)
        x_p = x_p.reshape(B, nh, nw, C, p, p)
        x_p = x_p.permute(0, 3, 1, 4, 2, 5).contiguous()  # (B, C, nh*p, nw*p)
        x_p = x_p.reshape(B, C, H2, W2)

        # 去掉 padding
        if pad_h > 0 or pad_w > 0:
            x_p = x_p[:, :, :H, :W]

        return x_p


# =========================================================
# <<< CSDI Block >>>
# =========================================================
class CSDIBlock(nn.Module):
    """
    一個 CSDI Block = 特徵 Attention + Patch 空間 Attention。
    對應原始 CSDI 的時間 Transformer + 特徵 Transformer。
    """
    def __init__(self, d_model, patch_size=6, num_heads=4, dropout=0.1):
        super().__init__()
        self.feature = FeatureAttention(d_model)
        self.spatial = PatchSpatialAttention(d_model, patch_size, num_heads, dropout)

    def forward(self, x):
        x = self.feature(x)
        x = self.spatial(x)
        return x


# =========================================================
# <<< CSDITransformer >>>
# =========================================================
class CSDITransformer(nn.Module):
    """
    CSDI-B Transformer 模型（Patch-based，避免 OOM）。

    介面與 UNet 完全相同：
        model(x_t, cond, t, beta) -> pred_noise

    config 需要包含：
        in_steps      : int   — 輸入時間步數（cond channel = in_steps * 2）
        target_steps  : int   — 預測時間步數（out_ch）
        d_model       : int   — Transformer hidden dim（預設 128）
        num_layers    : int   — CSDIBlock 數量（預設 6）
        num_heads     : int   — Attention head 數（預設 4）
        patch_size    : int   — 空間 patch 大小（預設 6）
                                H=30, W=180, p=6 → 150 patches（記憶體友好）
        dropout       : float — 預設 0.1
        time_dim      : int   — 預設 32
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        cond_ch    = config['in_steps'] * 2
        out_ch     = config['target_steps']
        d_model    = config.get('d_model',    128)
        n_layers   = config.get('num_layers',   6)
        n_heads    = config.get('num_heads',    4)
        patch_size = config.get('patch_size',   6)
        dropout    = config.get('dropout',    0.1)
        time_dim   = config.get('time_dim',    32)

        self.out_ch   = out_ch
        self.cond_ch  = cond_ch
        self.time_dim = time_dim

        # 時間嵌入（與 UNet 相同）
        self.time_embed = nn.Sequential(
            nn.Linear(time_dim, time_dim * 4),
            nn.SiLU(),
            nn.Linear(time_dim * 4, time_dim),
        )

        # 初始卷積：noisy target + cond → d_model channels
        self.init_conv = nn.Conv2d(out_ch + cond_ch, d_model, kernel_size=3, padding=1)
        self.time_proj = nn.Linear(time_dim, d_model)

        # N 個 CSDI Block
        self.blocks = nn.ModuleList([
            CSDIBlock(d_model, patch_size, n_heads, dropout)
            for _ in range(n_layers)
        ])

        # 輸出
        self.final_norm = nn.GroupNorm(8, d_model)
        self.final_conv = nn.Sequential(
            nn.SiLU(),
            nn.Conv2d(d_model, out_ch, kernel_size=1),
        )

    def forward(self, x, cond, t, beta):
        """
        x    : (B, out_ch,  H, W) — noisy target
        cond : (B, cond_ch, H, W) — 過去天氣場（條件）
        t    : (B,)               — diffusion 時間步（long tensor）
        beta : (T,) or None       — 保持介面與 UNet 一致，此模型不直接使用
        """
        B, C, H, W = x.shape
        device = x.device

        # 時間嵌入
        t_emb = sinusoidal_embedding(t, self.time_dim, device)
        t_emb = self.time_embed(t_emb)                              # (B, time_dim)

        # 初始特徵
        h = torch.cat([x, cond], dim=1)                            # (B, out_ch+cond_ch, H, W)
        h = self.init_conv(h)                                       # (B, d_model, H, W)
        h = h + self.time_proj(t_emb).unsqueeze(-1).unsqueeze(-1)  # 注入時間資訊

        # N 個 CSDI Block
        for block in self.blocks:
            h = block(h)

        # 輸出
        h   = self.final_norm(h)
        out = self.final_conv(h)                                    # (B, out_ch, H, W)
        return out