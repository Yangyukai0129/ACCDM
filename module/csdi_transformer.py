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
# <<< Patch 內部 Attention >>>
# =========================================================
class PatchInternalAttention(nn.Module):
    """
    讓每個 patch 內部的 p*p 個位置互相 attend（局部細節）。
    seq = p*p = 36，attention 矩陣 (B*N, 36, 36)，不會 OOM。
    """
    def __init__(self, d_model, patch_size=6, num_heads=4, dropout=0.1):
        super().__init__()
        self.patch_size = patch_size
        self.norm = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, num_heads,
                                          dropout=dropout, batch_first=True)
        self.ff   = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )

    def forward(self, x):
        B, C, H, W = x.shape
        residual = x                      # 先存原始 x，shape (B, C, H, W)
        p = self.patch_size

        # padding 讓 H, W 能被 p 整除
        pad_h = (p - H % p) % p
        pad_w = (p - W % p) % p
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        _, _, H2, W2 = x.shape
        nh, nw = H2 // p, W2 // p
        N = nh * nw

        # 切 patch：(B, C, H2, W2) → (B*N, p*p, C)
        x_p = x.reshape(B, C, nh, p, nw, p)
        x_p = x_p.permute(0, 2, 4, 3, 5, 1).contiguous()  # (B, nh, nw, p, p, C)
        x_p = x_p.reshape(B * N, p * p, C)                 # (B*N, p*p, C)

        # Self-Attention（patch 內）
        normed      = self.norm(x_p)
        attn_out, _ = self.attn(normed, normed, normed)     # (B*N, p*p, C)
        x_p         = x_p + attn_out
        x_p         = x_p + self.ff(x_p)

        # 還原空間
        x_p = x_p.reshape(B, nh, nw, p, p, C)
        x_p = x_p.permute(0, 5, 1, 3, 2, 4).contiguous()  # (B, C, nh*p, nw*p)
        x_p = x_p.reshape(B, C, H2, W2)

        # 去掉 padding
        if pad_h > 0 or pad_w > 0:
            x_p = x_p[:, :, :H, :W]                        # 還原成 (B, C, H, W)

        return x_p + residual                               # residual 用原始 x


# =========================================================
# <<< Patch 間 Attention >>>
# =========================================================
class PatchGlobalAttention(nn.Module):
    """
    讓 N 個 patch 之間互相 attend（全局空間關係）。
    seq = N = 150（H=30, W=180, p=6），attention 矩陣 (B, 150, 150)，記憶體友好。
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
        residual = x                      # 先存原始 x，shape (B, C, H, W)
        p = self.patch_size

        # padding
        pad_h = (p - H % p) % p
        pad_w = (p - W % p) % p
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        _, _, H2, W2 = x.shape
        nh, nw = H2 // p, W2 // p

        # 切 patch：(B, C, H2, W2) → (B, N, patch_dim)
        x_p = x.reshape(B, C, nh, p, nw, p)
        x_p = x_p.permute(0, 2, 4, 1, 3, 5).contiguous()  # (B, nh, nw, C, p, p)
        x_p = x_p.reshape(B, nh * nw, C * p * p)           # (B, N, patch_dim)

        # 投影 + Self-Attention（patch 間）
        x_p         = self.proj_in(x_p)                     # (B, N, d_model)
        normed      = self.norm(x_p)
        attn_out, _ = self.attn(normed, normed, normed)     # (B, N, d_model)
        x_p         = x_p + attn_out
        x_p         = x_p + self.ff(x_p)

        # 投影回 patch_dim 並還原空間
        x_p = self.proj_out(x_p)                            # (B, N, patch_dim)
        x_p = x_p.reshape(B, nh, nw, C, p, p)
        x_p = x_p.permute(0, 3, 1, 4, 2, 5).contiguous()  # (B, C, nh*p, nw*p)
        x_p = x_p.reshape(B, C, H2, W2)

        # 去掉 padding
        if pad_h > 0 or pad_w > 0:
            x_p = x_p[:, :, :H, :W]                        # 還原成 (B, C, H, W)

        return x_p + residual                               # residual 用原始 x


# =========================================================
# <<< CSDI Block >>>
# =========================================================
class CSDIBlock(nn.Module):
    """
    一個 CSDI Block = Patch 內部 Attention + Patch 間 Attention。
    兩層都有正確的 residual connection。
    """
    def __init__(self, d_model, patch_size=6, num_heads=4, dropout=0.1):
        super().__init__()
        self.internal = PatchInternalAttention(d_model, patch_size, num_heads, dropout)
        self.global_  = PatchGlobalAttention(d_model, patch_size, num_heads, dropout)

    def forward(self, x):
        x = self.internal(x)
        x = self.global_(x)
        return x


# =========================================================
# <<< CSDITransformer >>>
# =========================================================
class CSDITransformer(nn.Module):
    """
    CSDI-B Transformer 模型。

    介面與 UNet 完全相同：
        model(x_t, cond, t, beta) -> pred_noise

    config 需要包含：
        in_steps      : int   — 輸入時間步數（cond channel = in_steps * 2）
        target_steps  : int   — 預測時間步數（out_ch）
        d_model       : int   — Transformer hidden dim（預設 128）
        num_layers    : int   — CSDIBlock 數量（預設 6）
        num_heads     : int   — Attention head 數（預設 4）
        patch_size    : int   — patch 大小（預設 6）
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