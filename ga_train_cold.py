import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os
 
from module.ga_unet_baseline import UNet
from module.cold_diffusion import compute_climatology, train_cold
from weather_dataset import build_dataset, build_norm_stats
 
# =========================================================
# <<< 設定區 >>>
# =========================================================
WINDOW_SIZE     = 5
TRAIN_YEARS     = range(1965, 2020)
OUTPUT_DIR      = 'output'
BATCH_SIZE      = 32
NUM_EPOCHS      = 30
LR              = 1e-4
CHECKPOINT_DIR  = './checkpoints'
SAVE_EVERY      = 10
T               = 1000   # Cold Diffusion 總時間步數
 
# BEST_CONFIG = {
#     'window_day'   : 5,
#     'depth'        : 2,
#     'base_channels': 128,
#     'channel_mults': [1, 2],
#     'in_steps'     : 40,    # 3 * 8
#     'target_steps' : 24,    # 3 * 8
# }
# BEST_CONFIG = {
#     'window_day'   : 6,
#     'depth'        : 5,
#     'base_channels': 128,
#     'channel_mults': [1, 2, 2, 4, 4],
#     'in_steps'     : 48,    # 3 * 8
#     'target_steps' : 24,    # 3 * 8
# }
# BEST_CONFIG = {
#     'window_day'   : 1,
#     'depth'        : 2,
#     'base_channels': 128,
#     'channel_mults': [1, 2],
#     'in_steps'     : 8,    # 3 * 8
#     'target_steps' : 24,    # 3 * 8
# }
# BEST_CONFIG = {
#     'window_day'   : 3,
#     'depth'        : 4,
#     'base_channels': 128,
#     'channel_mults': [1, 1, 1, 1],
#     'in_steps'     : 24,    # 3 * 8
#     'target_steps' : 24,    # 3 * 8
# }
# BEST_CONFIG = {
#     'window_day'   : 5,
#     'depth'        : 2,
#     'base_channels': 128,
#     'channel_mults': [1, 2],
#     'in_steps'     : 40,    # 3 * 8
#     'target_steps' : 24,    # 3 * 8
# }

# BEST_CONFIG = {
#     'window_day'   : 12,
#     'depth'        : 3,
#     'base_channels': 128,
#     'channel_mults': [1, 2, 8],
#     'in_steps'     : 96,    # 3 * 8
#     'target_steps' : 24,    # 3 * 8
# }
BEST_CONFIG = {
    'window_day'   : 5,
    'depth'        : 3,
    'base_channels': 128,
    'channel_mults': [1, 2, 8],
    'in_steps'     : 40,    # 3 * 8
    'target_steps' : 24,    # 3 * 8
}

# =========================================================
# <<< 主程式 >>>
# =========================================================
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用設備: {device}")
 
    # 1. 建立資料集
    print("\n載入資料...")
    norm_stats = build_norm_stats(OUTPUT_DIR, WINDOW_SIZE, TRAIN_YEARS)
    train_ds   = build_dataset(OUTPUT_DIR, WINDOW_SIZE, TRAIN_YEARS,
                               norm_stats=norm_stats)
    print(f"訓練集: {len(train_ds)} 筆")
 
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=0, pin_memory=True,
    )
 
    # 2. 計算氣候平均場（正規化後）
    print("\n計算氣候平均場...")
    # 先用 compute_climatology 算反正規化的平均場
    x_mean_denorm = compute_climatology(train_loader, norm_stats)  # (target_steps, H, W)
 
    # 再正規化回來供訓練使用
    x_mean_norm = torch.tensor(
        (x_mean_denorm - norm_stats.t_mean) / norm_stats.t_std,
        dtype=torch.float32
    )
    np.save(os.path.join(CHECKPOINT_DIR, 'x_mean_norm.npy'),
            x_mean_norm.numpy())
    print(f"  氣候平均場已儲存")
 
    # 3. 建立模型
    print("\n建立模型...")
    model = UNet(BEST_CONFIG).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型參數量: {total_params:,}")
 
    # 4. 優化器 + Scheduler
    optimizer = optim.AdamW(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=NUM_EPOCHS, eta_min=1e-6
    )
 
    # 5. 訓練
    print(f"\n開始 Cold Diffusion 訓練（{NUM_EPOCHS} epochs）...")
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
 
    loss_history = train_cold(
        model            = model,
        train_loader     = train_loader,
        num_epochs       = NUM_EPOCHS,
        device           = device,
        optimizer        = optimizer,
        x_mean_norm      = x_mean_norm,
        T                = T,
        use_checkpoint   = True,
        checkpoint_dir   = CHECKPOINT_DIR,
        checkpoint_interval = SAVE_EVERY,
        scheduler        = scheduler,
    )
 
    # 6. 儲存最終模型
    final_path = os.path.join(CHECKPOINT_DIR, 'ga_window5.pth')
    torch.save({
        'epoch'               : NUM_EPOCHS,
        'model_state_dict'    : model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config'              : BEST_CONFIG,
        'norm_stats'          : {
            't_mean': norm_stats.t_mean,
            't_std' : norm_stats.t_std,
            'z_mean': norm_stats.z_mean,
            'z_std' : norm_stats.z_std,
        },
        'x_mean_norm' : x_mean_norm,
        'T'           : T,
    }, final_path)
    print(f"\n最終模型已儲存：{final_path}")
 
    # 7. 繪製 loss 曲線
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, NUM_EPOCHS + 1), loss_history, 'b-o',
             linewidth=2, markersize=4)
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MAE)')
    plt.title('Cold Diffusion Training Loss Curve')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('train_cold_loss(mae).png', dpi=150)
    plt.show()
    print("Loss 曲線已儲存至 train_cold_loss.png")