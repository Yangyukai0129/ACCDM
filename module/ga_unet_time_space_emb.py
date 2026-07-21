import math
import torch
import torch.nn as nn
import torch.nn.functional as F
 
 
# =========================================================
# <<< 時間嵌入 >>>
# =========================================================
def sinusoidal_embedding(t, dim, device):
    half_dim = dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
    emb = t[:, None].float() * emb[None, :]
    emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
    return emb
 
 
# =========================================================
# <<< 基礎模組 >>>
# =========================================================
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        )
        self.shortcut = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )
 
    def forward(self, x):
        return self.shortcut(x) + self.block(x)
 
 
class CrossAttention2D(nn.Module):
    def __init__(self, query_channels, cond_channels, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim  = query_channels // num_heads
        assert query_channels % num_heads == 0
 
        self.to_q     = nn.Conv2d(query_channels, query_channels, kernel_size=1)
        self.to_k     = nn.Conv2d(cond_channels,  query_channels, kernel_size=1)
        self.to_v     = nn.Conv2d(cond_channels,  query_channels, kernel_size=1)
        self.out_proj = nn.Conv2d(query_channels, query_channels, kernel_size=1)
 
    def forward(self, x, cond):
        B, C, H, W    = x.shape
        _, Cc, Hc, Wc = cond.shape
 
        q = self.to_q(x).view(B, self.num_heads, self.head_dim, H * W)
        k = self.to_k(cond).view(B, self.num_heads, self.head_dim, Hc * Wc)
        v = self.to_v(cond).view(B, self.num_heads, self.head_dim, Hc * Wc)
 
        q = q.permute(0, 1, 3, 2)
        k = k.permute(0, 1, 3, 2)
        v = v.permute(0, 1, 3, 2)
 
        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = F.softmax(attn, dim=-1)
        out  = torch.matmul(attn, v)
        out  = out.permute(0, 1, 3, 2).contiguous().view(B, C, H, W)
        return self.out_proj(out) + x
 
 
# =========================================================
# <<< UNet >>>
# =========================================================
class UNet(nn.Module):
    """
    config 需要包含：
        in_steps      : int  — 輸入時間步數（window_day × 8）
        target_steps  : int  — 預測時間步數（3 × 8 = 24）
        depth         : int  — 下採樣層數
        base_channels : int  — 第一層通道數
        channel_mults : list — 各層通道倍率（長度 = depth）
        lat_min/lat_max/lon_min/lon_max : float — 經緯度範圍（用於空間編碼）
    time_dim 預設 32，可在 config 中覆蓋。
 
    新增編碼：
        1. 空間位置編碼（lat/lon → 2 channels concat 到輸入）
        2. 預測時間編碼（forecast_step → sinusoidal → 加到特徵）
    """
 
    def __init__(self, config):
        super().__init__()
        self.config = config
 
        in_ch   = config['in_steps']     * 2   # cond channel（t + z）
        cond_ch = config['in_steps']     * 2
        out_ch  = config['target_steps']        # 只預測 t
 
        self.out_ch   = out_ch
        self.cond_ch  = cond_ch
        self.time_dim = config.get('time_dim', 32)
 
        base  = config['base_channels']
        mults = config['channel_mults']
        depth = config['depth']
 
        # 空間位置編碼範圍
        self.lat_min = config.get('lat_min',  35.0)
        self.lat_max = config.get('lat_max',  50.0)
        self.lon_min = config.get('lon_min',  -5.0)
        self.lon_max = config.get('lon_max',  30.0)
 
        # ---- 時間嵌入（diffusion 時間步）----
        self.time_embed = nn.Sequential(
            nn.Linear(self.time_dim, self.time_dim * 4),
            nn.SiLU(),
            nn.Linear(self.time_dim * 4, self.time_dim),
        )
        self.time_proj = nn.Linear(self.time_dim, base)
 
        # ---- 預測時間嵌入（forecast step 0~23）----
        self.forecast_embed = nn.Sequential(
            nn.Linear(self.time_dim, self.time_dim * 4),
            nn.SiLU(),
            nn.Linear(self.time_dim * 4, self.time_dim),
        )
        self.forecast_proj = nn.Linear(self.time_dim, base)
 
        # ---- 初始卷積：noisy target + cond + 空間位置編碼(2ch) ----
        self.init_conv = nn.Conv2d(out_ch + cond_ch + 2, base, kernel_size=3, padding=1)
 
        # ---- 下採樣 ----
        self.down_blocks   = nn.ModuleList()
        self.down_pools    = nn.ModuleList()
        self.skip_channels = []
        cur_ch = base
 
        for i in range(depth):
            out = base * mults[i]
            self.down_blocks.append(nn.Sequential(
                ResidualBlock(cur_ch, out),
                ResidualBlock(out, out),
            ))
            self.skip_channels.append(out)
            if i < depth - 1:
                self.down_pools.append(nn.AvgPool2d(2, 2))
            cur_ch = out
 
        # ---- 瓶頸層 ----
        self.mid_block1 = ResidualBlock(cur_ch, cur_ch)
        self.mid_attn   = CrossAttention2D(query_channels=cur_ch, cond_channels=cond_ch)
        self.mid_block2 = ResidualBlock(cur_ch, cur_ch)
 
        # ---- 上採樣 ----
        self.up_blocks  = nn.ModuleList()
        self.up_samples = nn.ModuleList()
        for i in reversed(range(depth)):
            skip_ch = self.skip_channels[i]
            in_up   = cur_ch + skip_ch
            out_up  = base * mults[i - 1] if i > 0 else base
            self.up_blocks.append(nn.Sequential(
                ResidualBlock(in_up, skip_ch),
                ResidualBlock(skip_ch, out_up),
            ))
            self.up_samples.append(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
            )
            cur_ch = out_up
 
        self.final_conv = nn.Conv2d(base, out_ch, kernel_size=1)
 
    def _get_pos_enc(self, H, W, device):
        """
        建立空間位置編碼，shape: (1, 2, H, W)
        lat 正規化到 0~1（lat_min → 0，lat_max → 1）
        lon 正規化到 0~1
        注意：lat 資料從大到小排列（50→35），所以 linspace 從 1→0
        """
        lat = torch.linspace(1, 0, H, device=device)   # (H,) 50→35 對應 1→0
        lon = torch.linspace(0, 1, W, device=device)   # (W,) -5→30 對應 0→1
 
        lat_map = lat.unsqueeze(1).expand(H, W)   # (H, W)
        lon_map = lon.unsqueeze(0).expand(H, W)   # (H, W)
 
        pos_enc = torch.stack([lat_map, lon_map], dim=0).unsqueeze(0)  # (1, 2, H, W)
        return pos_enc
 
    def forward(self, x, cond, t, beta, forecast_step=None):
        """
        x             : (B, out_ch, H, W)    — noisy target
        cond          : (B, cond_ch, H, W)   — 條件資料
        t             : (B,)                 — diffusion 時間步
        beta          : (T,)                 — noise schedule
        forecast_step : (B,) or None         — 預測時間步（0~23）
                        None 時不加預測時間編碼
        """
        B, C, H, W = x.shape
        device = x.device
 
        # 1. Diffusion 時間嵌入
        t_emb = sinusoidal_embedding(t, self.time_dim, device)
        t_emb = self.time_embed(t_emb)   # (B, time_dim)
 
        # 2. 空間位置編碼
        pos_enc = self._get_pos_enc(H, W, device).expand(B, -1, -1, -1)  # (B, 2, H, W)
 
        # 3. 初始特徵：noisy target + cond + pos_enc
        h = torch.cat([x, cond, pos_enc], dim=1)
        h = self.init_conv(h)
 
        # 4. 加入 diffusion 時間編碼
        h = h + self.time_proj(t_emb).unsqueeze(-1).unsqueeze(-1)
 
        # 5. 加入預測時間編碼（如果有傳入）
        if forecast_step is not None:
            f_emb = sinusoidal_embedding(forecast_step, self.time_dim, device)
            f_emb = self.forecast_embed(f_emb)
            h = h + self.forecast_proj(f_emb).unsqueeze(-1).unsqueeze(-1)
 
        # 6. 下採樣
        skips    = []
        pool_idx = 0
        for i, block in enumerate(self.down_blocks):
            h = block(h)
            skips.append(h)
            if i < len(self.down_blocks) - 1:
                if h.shape[2] > 1 and h.shape[3] > 1:
                    h = self.down_pools[pool_idx](h)
                    pool_idx += 1
 
        # 7. 瓶頸
        h      = self.mid_block1(h)
        cond_r = F.interpolate(cond, size=h.shape[2:], mode='bilinear', align_corners=False)
        h      = self.mid_attn(h, cond_r)
        h      = self.mid_block2(h)
 
        # 8. 上採樣
        for i, (block, upsample) in enumerate(zip(self.up_blocks, self.up_samples)):
            skip = skips.pop()
            if h.shape[2] < skip.shape[2] or h.shape[3] < skip.shape[3]:
                h = upsample(h)
            if h.shape[2:] != skip.shape[2:]:
                h = F.interpolate(h, size=skip.shape[2:], mode='bilinear', align_corners=False)
            h = torch.cat([h, skip], dim=1)
            h = block(h)
 
        out = self.final_conv(h)
        if out.shape[2] != H or out.shape[3] != W:
            out = F.interpolate(out, size=(H, W), mode='bilinear', align_corners=False)
        return out