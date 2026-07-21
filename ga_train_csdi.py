import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os

from module.csdi_transformer import CSDITransformer
from module.csdi_diffusion import train_csdi, make_beta_schedule
from weather_dataset import build_dataset, build_norm_stats

# =========================================================
# <<< 設定區 >>>
# =========================================================
WINDOW_SIZE     = 5
TRAIN_YEARS     = range(1965, 2020)
OUTPUT_DIR      = 'output'
BATCH_SIZE      = 32
NUM_EPOCHS      = 50
LR              = 1e-4
CHECKPOINT_DIR  = './checkpoints'
SAVE_EVERY      = 10

CSDI_CONFIG = {
    'in_steps'    : 40,        # WINDOW_SIZE(5) * 8 = 40，跟 BEST_CONFIG 一致
    'target_steps': 24,        # 3 * 8 = 24，跟 BEST_CONFIG 一致
    'patch_size': 6,   # 你的格網大小
    'd_model'     : 128,
    'num_layers'  : 6,
    'num_heads'   : 4,
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
    train_ds   = build_dataset(OUTPUT_DIR, WINDOW_SIZE, TRAIN_YEARS, norm_stats=norm_stats)
    print(f"訓練集: {len(train_ds)} 筆")

    train_loader = DataLoader(
        train_ds,
        batch_size  = BATCH_SIZE,
        shuffle     = True,
        num_workers = 0,
        pin_memory  = True,
    )

    # 2. 建立模型
    print("\n建立模型...")
    model = CSDITransformer(CSDI_CONFIG).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型參數量: {total_params:,}")

    # 3. 優化器 + Scheduler
    optimizer = optim.AdamW(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=NUM_EPOCHS, eta_min=1e-6
    )
    # criterion = nn.MSELoss()
    criterion = nn.L1Loss()

    # 4. 訓練
    print(f"\n開始訓練（{NUM_EPOCHS} epochs）...")
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    loss_history, beta, alpha, alpha_cumprod = train_csdi(
        model            = model,
        train_loader     = train_loader,
        num_epochs       = NUM_EPOCHS,
        device           = device,
        optimizer        = optimizer,
        criterion        = criterion,
        scheduler        = scheduler,
        use_checkpoint   = True,
        checkpoint_dir   = CHECKPOINT_DIR,
        checkpoint_interval = SAVE_EVERY,
    )

    # 5. 儲存最終模型
    final_path = os.path.join(CHECKPOINT_DIR, 'csdi.pth')
    torch.save({
        'model_state_dict'    : model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config'              : CSDI_CONFIG,
        'norm_stats'          : {
            't_mean': norm_stats.t_mean,
            't_std' : norm_stats.t_std,
            'z_mean': norm_stats.z_mean,
            'z_std' : norm_stats.z_std,
        },
        'beta'          : beta,
        'alpha'         : alpha,
        'alpha_cumprod' : alpha_cumprod,
    }, final_path)
    print(f"\n最終模型已儲存：{final_path}")

    # 6. 繪製 loss 曲線
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, NUM_EPOCHS + 1), loss_history, 'b-o', linewidth=2, markersize=4)
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title('Training Loss Curve')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('train_loss_csdi.png', dpi=150)
    plt.show()
    print("Loss 曲線已儲存至 train_loss.png")