"""
csdi_diffusion.py
=================
CSDI-B：改編自 CSDI（Tashiro et al. 2021）的天氣預報版本。

任務設定：
    cond   : 過去天氣場（歷史觀測）(B, cond_ch, H, W)
    target : 未來天氣場（要預測的）(B, out_ch,  H, W)
    兩者解析度相同，差異是時間點不同

與原始 CSDI 的差異：
    原始 CSDI  → 同一序列內的插補（用遮罩區分已知/未知位置）
    此版本     → 時間預報（cond=過去, target=未來，不需要遮罩）

與 DDPM 的差異：
    DDPM      → UNet backbone，cond 在輸入層 channel concat
    此版本    → Transformer backbone（時間 + 空間 2D Self-Attention）
                cond 同樣 channel concat，但內部用 attention 捕捉時空關聯

前向過程（與 DDPM 完全相同）：
    x_t = sqrt(ᾱ_t) * x0 + sqrt(1 - ᾱ_t) * ε
    ε ~ N(0, I)

預測目標（與 DDPM 完全相同）：
    預測噪聲 ε

Loss：
    MSE(pred_noise, true_noise)
"""

import os
import torch
import torch.nn as nn


# =========================================================
# <<< Noise Schedule（與 DDPM 完全相同）>>>
# =========================================================
def make_beta_schedule(timesteps=1000, beta_start=1e-4, beta_end=0.02, device='cpu'):
    """
    建立線性 noise schedule。
    回傳 beta, alpha, alpha_cumprod，均在指定 device 上。
    """
    beta          = torch.linspace(beta_start, beta_end, timesteps, device=device)
    alpha         = 1.0 - beta
    alpha_cumprod = torch.cumprod(alpha, dim=0)
    return beta, alpha, alpha_cumprod


# =========================================================
# <<< 訓練函數 >>>
# =========================================================
def train_csdi(model, train_loader, num_epochs, device,
               optimizer, criterion,
               train_loss_history=None,
               beta=None, alpha=None, alpha_cumprod=None,
               use_checkpoint=False,
               checkpoint_dir='./checkpoints',
               checkpoint_interval=10,
               scheduler=None):
    """
    CSDI-B 訓練函數。

    訓練流程與 DDPM 完全相同：
        1. 對 target 加噪得到 x_t
        2. 模型接收 (x_t, cond, t) 預測噪聲 ε
        3. Loss = MSE(pred_noise, true_noise)

    與 DDPM train() 的唯一差異：
        - 模型內部是 Transformer（2D Self-Attention）而非 UNet
        - 模型介面保持一致：model(x_t, cond, t, beta)

    Parameters
    ----------
    model             : CSDITransformer（時間 + 空間 2D Self-Attention）
    train_loader      : DataLoader，每次回傳 (cond, target, year)
                        cond   = 過去天氣場 (B, cond_ch, H, W)
                        target = 未來天氣場 (B, out_ch,  H, W)
    num_epochs        : int
    device            : torch.device
    optimizer         : optimizer
    criterion         : loss function（通常是 MSELoss）
    train_loss_history: list，會 append 每個 epoch 的 avg loss
    beta/alpha/alpha_cumprod : noise schedule；None 時自動建立
    use_checkpoint    : 是否每 checkpoint_interval epoch 存檔
    checkpoint_dir    : 存檔路徑
    checkpoint_interval: int
    """
    if train_loss_history is None:
        train_loss_history = []

    # 建立或移至 device
    if beta is None or alpha is None or alpha_cumprod is None:
        beta, alpha, alpha_cumprod = make_beta_schedule(device=device)
    else:
        beta, alpha, alpha_cumprod = (
            beta.to(device), alpha.to(device), alpha_cumprod.to(device)
        )

    if use_checkpoint:
        os.makedirs(checkpoint_dir, exist_ok=True)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        num_batches  = 0

        for cond, target, _, _ in train_loader:
            cond   = cond.to(device, non_blocking=True)    # (B, cond_ch, H, W)
            target = target.to(device, non_blocking=True)  # (B, out_ch,  H, W)

            B = cond.shape[0]
            t = torch.randint(0, len(beta), (B,), device=device)

            # ---- 加噪（與 DDPM 完全相同）----
            noise = torch.randn_like(target)
            a_t   = alpha_cumprod[t][:, None, None, None]
            x_t   = torch.sqrt(a_t) * target + torch.sqrt(1 - a_t) * noise
            # cond（過去天氣場）不加噪，保持乾淨作為條件

            optimizer.zero_grad(set_to_none=True)

            # ---- 模型前向 ----
            # 模型介面與 DDPM 相同：model(x_t, cond, t, beta)
            # 差別在模型內部：Transformer 的 2D Self-Attention 處理時空關聯
            if use_checkpoint:
                from torch.utils.checkpoint import checkpoint as grad_ckpt
                pred_noise = grad_ckpt(model, x_t, cond, t, beta, use_reentrant=False)
            else:
                pred_noise = model(x_t, cond, t, beta)

            # ---- Loss（與 DDPM 完全相同）----
            loss = criterion(pred_noise, noise)
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
            ckpt_path = os.path.join(checkpoint_dir, 'ckpt_csdi_latest.pth')
            torch.save({
                'epoch'               : epoch + 1,
                'model_state_dict'    : model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss'                : avg_loss,
                'config'              : model.config,
            }, ckpt_path)
            print(f"  Checkpoint saved (epoch {epoch+1}): {ckpt_path}")

    return train_loss_history, beta.cpu(), alpha.cpu(), alpha_cumprod.cpu()


# =========================================================
# <<< DDIM 推論（公式與 DDPM 完全相同）>>>
# =========================================================
@torch.no_grad()
def csdi_inference(model, cond, beta, device, eta=0.0, num_steps=15):
    """
    CSDI-B 推論：從純噪聲出發，以過去天氣場為條件預測未來天氣場。

    推論公式與 DDPM ddim_inference 完全相同。
    差別只在模型內部用 Transformer 處理時空關聯。

    Parameters
    ----------
    model    : CSDITransformer
    cond     : (B, cond_ch, H, W) 過去天氣場
    beta     : (T,) noise schedule（CPU tensor）
    device   : torch.device
    eta      : 0.0 = deterministic DDIM；1.0 = DDPM-like stochastic
    num_steps: 推論步數（建議 15~50）

    Returns
    -------
    x0 : (B, out_ch, H, W) 預測的未來天氣場
    """
    beta          = beta.to(device)
    alpha         = 1.0 - beta
    alpha_cumprod = torch.cumprod(alpha, dim=0)

    B, _, H, W = cond.shape
    x_t = torch.randn(B, model.out_ch, H, W, device=device)  # 從純噪聲開始

    total  = len(beta)
    step   = total // num_steps
    tsteps = list(range(0, total, step))[::-1]   # T → 0

    for i, t_val in enumerate(tsteps):
        t_tensor   = torch.full((B,), t_val, device=device, dtype=torch.long)
        pred_noise = model(x_t, cond, t_tensor, beta)

        a_t = alpha_cumprod[t_val]
        x0  = (x_t - torch.sqrt(1 - a_t) * pred_noise) / torch.sqrt(a_t)

        if i < len(tsteps) - 1:
            t_next = tsteps[i + 1]
            a_next = alpha_cumprod[t_next]
            sigma  = eta * torch.sqrt((1 - a_next) / (1 - a_t) * (1 - a_t / a_next))
            noise  = sigma * torch.randn_like(x_t)
            x_t    = (torch.sqrt(a_next) * x0
                      + torch.sqrt(1 - a_next - sigma**2) * pred_noise
                      + noise)
        else:
            x_t = x0

    return x_t