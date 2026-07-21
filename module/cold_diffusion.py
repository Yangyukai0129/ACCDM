"""
cold_diffusion.py
=================
Cold Diffusion 實作。

前向過程（退化）：
    x_t = (1 - t/T) * x0 + (t/T) * x_mean
    t=0 → 原始天氣場 x0
    t=T → 氣候平均場 x_mean

反向過程（還原）：
    模型預測擾動 deviation = x0 - x_mean
    再從 x_t 還原出 x0

Loss：
    MAE(pred_deviation, true_deviation)
    true_deviation = x0 - x_mean
"""

import os
import torch
import torch.nn as nn
import numpy as np


# =========================================================
# <<< 計算氣候平均場 >>>
# =========================================================
def compute_climatology(train_loader, norm_stats, device='cpu'):
    """
    從訓練集計算 target 的氣候平均場。
    回傳 x_mean: (target_steps, H, W)，反正規化後的值。
    """
    print("計算氣候平均場...")
    sum_t  = None
    count  = 0

    for cond, target, _, _ in train_loader:
        # 反正規化
        t_np = target.numpy() * norm_stats.t_std + norm_stats.t_mean  # (B, target_steps, H, W)
        if sum_t is None:
            sum_t = t_np.sum(axis=0)   # (target_steps, H, W)
        else:
            sum_t += t_np.sum(axis=0)
        count += t_np.shape[0]

    x_mean = sum_t / count   # (target_steps, H, W)
    print(f"  氣候平均場 shape: {x_mean.shape}")
    print(f"  溫度範圍: {x_mean.min():.2f} ~ {x_mean.max():.2f} K")
    return x_mean


# =========================================================
# <<< 前向退化過程 >>>
# =========================================================
def cold_forward(x0, x_mean, t, T):
    """
    Cold Diffusion 前向過程：線性插值趨向平均場。

    x0     : (B, C, H, W) 原始 target（正規化後）
    x_mean : (C, H, W) 氣候平均場（正規化後）
    t      : (B,) 時間步 0~T
    T      : int 總時間步數

    回傳 x_t : (B, C, H, W)
    """
    ratio = (t.float() / T)[:, None, None, None]   # (B, 1, 1, 1)
    x_mean_batch = x_mean.unsqueeze(0).expand_as(x0)
    x_t = (1 - ratio) * x0 + ratio * x_mean_batch
    return x_t


# =========================================================
# <<< 訓練函數 >>>
# =========================================================
def train_cold(model, train_loader, num_epochs, device,
               optimizer, x_mean_norm,
               train_loss_history=None,
               T=1000,
               use_checkpoint=False,
               checkpoint_dir='./checkpoints',
               checkpoint_interval=10,
               scheduler=None):
    """
    Cold Diffusion 訓練函數。

    Parameters
    ----------
    x_mean_norm : (target_steps, H, W) tensor — 正規化後的氣候平均場
    T           : int — 總時間步數
    """
    if train_loss_history is None:
        train_loss_history = []

    x_mean_norm = x_mean_norm.to(device)
    criterion = nn.MSELoss()
    # criterion   = nn.L1Loss()   # MAE loss

    if use_checkpoint:
        os.makedirs(checkpoint_dir, exist_ok=True)

    target_steps = model.out_ch

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        num_batches  = 0

        for cond, target, _, _ in train_loader:
            cond   = cond.to(device, non_blocking=True)     # (B, cond_ch, H, W)
            target = target.to(device, non_blocking=True)   # (B, target_steps, H, W)

            B = cond.shape[0]
            t = torch.randint(1, T + 1, (B,), device=device)   # t 從 1 開始避免 t=0

            # 前向退化：趨向平均場
            x_t = cold_forward(target, x_mean_norm, t, T)

            # 真實擾動
            true_deviation = target - x_mean_norm.unsqueeze(0)   # (B, target_steps, H, W)

            # 偽時間步（讓模型知道退化程度，用 t/T 正規化到 0~1）
            t_norm = (t.float() / T * 999).long()   # 映射到 0~999 讓 sinusoidal embedding 用

            optimizer.zero_grad(set_to_none=True)

            if use_checkpoint:
                from torch.utils.checkpoint import checkpoint as grad_ckpt
                pred_deviation = grad_ckpt(model, x_t, cond, t_norm, None,
                                           use_reentrant=False)
            else:
                pred_deviation = model(x_t, cond, t_norm, beta=None)

            loss = criterion(pred_deviation, true_deviation)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            num_batches  += 1

        avg_loss = running_loss / max(num_batches, 1)
        train_loss_history.append(avg_loss)
        print(f"  Epoch {epoch+1}/{num_epochs}  loss={avg_loss:.6f}")

        if scheduler is not None:
            scheduler.step()

        if use_checkpoint and (epoch + 1) % checkpoint_interval == 0:
            ckpt_path = os.path.join(checkpoint_dir, 'ckpt_cold_latest.pth')
            torch.save({
                'epoch'               : epoch + 1,
                'model_state_dict'    : model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss'                : avg_loss,
                'config'              : model.config,
                'x_mean_norm'         : x_mean_norm.cpu(),
                'T'                   : T,
            }, ckpt_path)
            print(f"  Checkpoint saved (epoch {epoch+1}): {ckpt_path}")

    return train_loss_history


# =========================================================
# <<< 推論函數 >>>
# =========================================================
@torch.no_grad()
# def cold_inference(model, cond, x_mean_norm, device, T=1000, num_steps=50):
#     """
#     Cold Diffusion 推論：從氣候平均場逐步還原天氣場。

#     cond        : (B, cond_ch, H, W)
#     x_mean_norm : (target_steps, H, W) 正規化後的氣候平均場
#     回傳 x0     : (B, target_steps, H, W)
#     """
#     x_mean_norm  = x_mean_norm.to(device)
#     B, _, H, W   = cond.shape
#     target_steps = model.out_ch

#     # 從氣候平均場開始（t=T）
#     x_t = x_mean_norm.unsqueeze(0).expand(B, -1, -1, -1).clone()

#     # 時間步從 T → 0
#     step_size = T // num_steps
#     tsteps    = list(range(T, 0, -step_size))

#     for idx, t_val in enumerate(tsteps):
#         t_norm = torch.full((B,), int(t_val / T * 999),
#                             device=device, dtype=torch.long)

#         pred_deviation = model(x_t, cond, t_norm, None)
#         x0_pred = x_mean_norm.unsqueeze(0) + pred_deviation
#         x0_pred = torch.clamp(x0_pred, -5, 5)

#         t_next = max(t_val - step_size, 0)
#         if t_next > 0:
#             ratio_next = t_next / T
#             x_t = ((1 - ratio_next) * x0_pred
#                     + ratio_next * x_mean_norm.unsqueeze(0))
#         else:
#             x_t = x0_pred

#         # ← 加這裡
#         if x_t.abs().max() > 100:
#             print(f"!!! 第 {idx} 步爆炸 t_val={t_val}, t_next={t_next}")
#             print(f"  x0_pred: {x0_pred.min():.4f} ~ {x0_pred.max():.4f}")
#             print(f"  ratio_next: {ratio_next if t_next > 0 else 'N/A'}")
#             print(f"  x_t: {x_t.min():.4f} ~ {x_t.max():.4f}")
#             break

#     return x_t
def cold_inference(model, cond, x_mean_norm, device, T=1000, num_steps=50):
    """
    Cold Diffusion 推論：從氣候平均場逐步還原天氣場。
 
    cond        : (B, cond_ch, H, W)
    x_mean_norm : (target_steps, H, W) 正規化後的氣候平均場
    回傳 x0     : (B, target_steps, H, W)
    """
    x_mean_norm  = x_mean_norm.to(device)
    B, _, H, W   = cond.shape
    target_steps = model.out_ch
 
    # 從氣候平均場開始（t=T）
    x_t = x_mean_norm.unsqueeze(0).expand(B, -1, -1, -1).clone()
 
    # 時間步從 T → 0
    step_size = T // num_steps
    tsteps    = list(range(T, 0, -step_size))
 
    for t_val in tsteps:
        t_norm = torch.full((B,), int(t_val / T * 999),
                            device=device, dtype=torch.long)
 
        # 模型預測擾動
        pred_deviation = model(x_t, cond, t_norm, beta=None)
 
        # 還原 x0 估計
        x0_pred = x_mean_norm.unsqueeze(0) + pred_deviation
 
        # 正確的 Cold Diffusion 更新公式（Bansal et al. 2022）：
        # x_{t-1} = x_t + (step_size/T) * pred_deviation
        t_next = max(t_val - step_size, 0)
        if t_next > 0:
            x_t = x_t + (step_size / T) * pred_deviation
        else:
            x_t = x0_pred
 
    return x_t