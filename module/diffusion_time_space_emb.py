import os
import torch
import torch.nn as nn
 
 
# =========================================================
# <<< Noise Schedule >>>
# =========================================================
def make_beta_schedule(timesteps=1000, beta_start=1e-4, beta_end=0.02, device='cpu'):
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
 
    if train_loss_history is None:
        train_loss_history = []
 
    if beta is None or alpha is None or alpha_cumprod is None:
        beta, alpha, alpha_cumprod = make_beta_schedule(device=device)
    else:
        beta, alpha, alpha_cumprod = (
            beta.to(device), alpha.to(device), alpha_cumprod.to(device)
        )
 
    if use_checkpoint:
        os.makedirs(checkpoint_dir, exist_ok=True)
 
    target_steps = model.out_ch   # 預測的時間步數（24）
 
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        num_batches  = 0
 
        for cond, target, _ in train_loader:
            cond   = cond.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
 
            B = cond.shape[0]
            t = torch.randint(0, len(beta), (B,), device=device)
 
            # 加噪
            noise = torch.randn_like(target)
            a_t   = alpha_cumprod[t][:, None, None, None]
            x_t   = torch.sqrt(a_t) * target + torch.sqrt(1 - a_t) * noise
 
            # 隨機抽一個 forecast_step（0 ~ target_steps-1）
            forecast_step = torch.randint(0, target_steps, (B,), device=device)
 
            optimizer.zero_grad(set_to_none=True)
 
            if use_checkpoint:
                from torch.utils.checkpoint import checkpoint as grad_ckpt
                pred = grad_ckpt(model, x_t, cond, t, beta, forecast_step,
                                 use_reentrant=False)
            else:
                pred = model(x_t, cond, t, beta, forecast_step)
 
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
    DDIM 推論。
    forecast_step 在每個 diffusion step 都用平均值（target_steps // 2）
    代表整體預測，不針對特定時間步。
    回傳 (B, target_steps, H, W)
    """
    beta          = beta.to(device)
    alpha         = 1.0 - beta
    alpha_cumprod = torch.cumprod(alpha, dim=0)
 
    B, _, H, W   = cond.shape
    target_steps = model.out_ch
 
    total  = len(beta)
    step   = total // num_steps
    tsteps = list(range(0, total, step))[::-1]
 
    x_t = torch.randn(B, target_steps, H, W, device=device)
 
    # 推論時用中間的 forecast_step 代表整體
    forecast_step = torch.full((B,), target_steps // 2,
                               device=device, dtype=torch.long)
 
    for i, t_val in enumerate(tsteps):
        t_tensor   = torch.full((B,), t_val, device=device, dtype=torch.long)
        pred_noise = model(x_t, cond, t_tensor, beta, forecast_step)
 
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