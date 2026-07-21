import os
import torch
import torch.nn as nn


# =========================================================
# <<< Noise Schedule >>>
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
def train(model, train_loader, num_epochs, device,
          optimizer, criterion,
          train_loss_history=None,
          beta=None, alpha=None, alpha_cumprod=None,
          use_checkpoint=False,
          checkpoint_dir='./checkpoints',
          checkpoint_interval=10,
          scheduler=None):
    """
    Diffusion model 訓練函數。

    Parameters
    ----------
    model             : UNet
    train_loader      : DataLoader，每次回傳 (cond, target, year)
    num_epochs        : int
    device            : torch.device
    optimizer         : optimizer
    criterion         : loss function（通常是 MSELoss）
    train_loss_history: list，會 append 每個 epoch 的 avg loss
    beta/alpha/alpha_cumprod : noise schedule；None 時自動建立
    use_checkpoint    : 是否每 5 epoch 存檔
    checkpoint_dir    : 存檔路徑
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

        for cond, target, _ in train_loader:
            cond   = cond.to(device, non_blocking=True)    # (B, cond_ch, H, W)
            target = target.to(device, non_blocking=True)  # (B, out_ch,  H, W)

            B = cond.shape[0]
            t = torch.randint(0, len(beta), (B,), device=device)

            # 加噪
            noise = torch.randn_like(target)
            a_t   = alpha_cumprod[t][:, None, None, None]
            x_t   = torch.sqrt(a_t) * target + torch.sqrt(1 - a_t) * noise

            optimizer.zero_grad(set_to_none=True)

            if use_checkpoint:
                from torch.utils.checkpoint import checkpoint as grad_ckpt
                pred = grad_ckpt(model, x_t, cond, t, beta, use_reentrant=False)
            else:
                pred = model(x_t, cond, t, beta)

            loss = criterion(pred, noise)
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
            ckpt_path = os.path.join(checkpoint_dir, 'ckpt_latest.pth')
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
# <<< DDIM 推論 >>>
# =========================================================
@torch.no_grad()
def ddim_inference(model, cond, beta, device, eta=0.0, num_steps=15):
    """
    DDIM 快速推論。

    Parameters
    ----------
    model    : UNet
    cond     : (B, cond_ch, H, W)
    beta     : (T,) noise schedule（CPU tensor）
    device   : torch.device
    eta      : 0.0 = deterministic DDIM；1.0 = DDPM
    num_steps: 推論步數

    Returns
    -------
    x0 : (B, out_ch, H, W)
    """
    beta          = beta.to(device)
    alpha         = 1.0 - beta
    alpha_cumprod = torch.cumprod(alpha, dim=0)

    B, _, H, W = cond.shape
    x_t = torch.randn(B, model.out_ch, H, W, device=device)

    total  = len(beta)
    step   = total // num_steps
    tsteps = list(range(0, total, step))[::-1]

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
            x_t    = torch.sqrt(a_next) * x0 + torch.sqrt(1 - a_next - sigma**2) * pred_noise + noise
        else:
            x_t = x0

    return x_t