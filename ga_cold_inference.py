import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
 
from module.ga_unet_baseline import UNet
from module.cold_diffusion import cold_inference
from weather_dataset import build_dataset, build_norm_stats, NormStats
import cartopy.crs as ccrs
import cartopy.feature as cfeature
 
# =========================================================
# <<< 設定區 >>>
# =========================================================
WINDOW_SIZE    = 5
TEST_YEARS     = range(2020, 2026)
TRAIN_YEARS    = range(1965, 2020)
OUTPUT_DIR     = 'output'
BATCH_SIZE     = 32
NUM_STEPS      = 100
MODEL_PATH     = './checkpoints/ga_window5.pth'
SAVE_DIR       = 'eval/eval_results_cold_window1'
 
TARGET_STEPS = {
    # ' 3hr': 0,
    '24hr': 7,
    '48hr': 15,
    '72hr': 23,
}
 
# =========================================================
# <<< 輔助函數 >>>
# =========================================================
def compute_rmse(pred, true):
    return np.sqrt(((pred - true) ** 2).mean(axis=(-2, -1)))
 
def compute_nrmse(pred, true):
    rmse = np.sqrt(((pred - true) ** 2).mean(axis=(-2, -1)))
    std  = true.std(axis=(-2, -1)) + 1e-8
    return rmse / std
 
def compute_mae(pred, true):
    return np.abs(pred - true).mean(axis=(-2, -1))
 
def compute_ssim_per_sample(pred, true):
    scores = []
    for i in range(len(pred)):
        p  = pred[i]
        t  = true[i]
        dr = max(t.max() - t.min(), 1e-6)
        scores.append(ssim(t, p, data_range=dr))
    return np.array(scores)
 
def compute_rmse_p95(pred, true, threshold):
    per_sample = []
    for i in range(len(true)):
        mask = true[i] > threshold
        if mask.sum() == 0:
            continue
        err = (pred[i][mask] - true[i][mask]) ** 2
        per_sample.append(np.sqrt(err.mean()))
    per_sample = np.array(per_sample)
    return per_sample.mean(), per_sample.std()
 
def compute_mae_p95(pred, true, threshold):
    per_sample = []
    for i in range(len(true)):
        mask = true[i] > threshold
        if mask.sum() == 0:
            continue
        err = np.abs(pred[i][mask] - true[i][mask])
        per_sample.append(err.mean())
    per_sample = np.array(per_sample)
    return per_sample.mean(), per_sample.std()
 
def compute_ssim_p95(pred, true, threshold):
    scores = []
    for i in range(len(true)):
        mask = true[i] > threshold
        if mask.sum() == 0:
            continue
        dr = max(true[i].max() - true[i].min(), 1e-6)
        _, ssim_map = ssim(true[i], pred[i], data_range=dr, full=True)
        #     ^^^^^^^^                              ^^^^^^^^^^
        #     多回傳一個「跟原圖同尺寸的 SSIM map」
        scores.append(ssim_map[mask].mean())
        #              ^^^^^^^^^^^^^^^^^^^^
        #              用 mask 只取極端事件區域的 SSIM 值再平均
    scores = np.array(scores)
    return scores.mean(), scores.std()
 
def compute_csi(pred, true, threshold):
    """
    CSI = TP / (TP + FP + FN)
    pred, true  : (N, H, W)
    threshold   : (H, W)
    """
    per_sample = []
    for i in range(len(true)):
        pred_hw = pred[i] > threshold
        true_hw = true[i] > threshold
 
        TP = (pred_hw & true_hw).sum()
        FP = (pred_hw & ~true_hw).sum()
        FN = (~pred_hw & true_hw).sum()
 
        denom = TP + FP + FN
        if denom == 0:
            continue
        per_sample.append(TP / denom)
 
    per_sample = np.array(per_sample)
    return per_sample.mean(), per_sample.std()
 
def print_metrics(results):
    header = (f"\n{'':>10} {'RMSE mean±std':>20} {'RMSE P95':>18} "
              f"{'nRMSE mean±std':>20} {'MAE mean±std':>20} {'MAE P95':>18} "
              f"{'SSIM mean±std':>20} {'SSIM P95':>20} {'CSI mean±std':>18}")
    print(header)
    print('-' * 140)
    for label, m in results.items():
        print(
            f"{label:>10}  "
            f"{m['RMSE']['mean']:.4f} ± {m['RMSE']['std']:.4f}   "
            f"{m['RMSE']['p95_mean']:.4f} ± {m['RMSE']['p95_std']:.4f}   "
            f"{m['nRMSE']['mean']:.4f} ± {m['nRMSE']['std']:.4f}   "
            f"{m['MAE']['mean']:.4f} ± {m['MAE']['std']:.4f}   "
            f"{m['MAE']['p95_mean']:.4f} ± {m['MAE']['p95_std']:.4f}   "
            f"{m['SSIM']['mean']:.4f} ± {m['SSIM']['std']:.4f}   "
            f"{m['SSIM']['p95_mean']:.4f} ± {m['SSIM']['p95_std']:.4f}   "
            f"{m['CSI']['mean']:.4f} ± {m['CSI']['std']:.4f}"
        )

# =========================================================
# <<< 主程式 >>>
# =========================================================
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用設備: {device}")
    os.makedirs(SAVE_DIR, exist_ok=True)
 
    # 1. 載入模型
    print("\n載入模型...")
    ckpt   = torch.load(MODEL_PATH, map_location=device)
    config = ckpt['config']
    model  = UNet(config).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
 
    if 'norm_stats' in ckpt:
        ns_dict    = ckpt['norm_stats']
        norm_stats = NormStats.__new__(NormStats)
        norm_stats.t_mean = ns_dict['t_mean']
        norm_stats.t_std  = ns_dict['t_std']
        norm_stats.z_mean = ns_dict['z_mean']
        norm_stats.z_std  = ns_dict['z_std']
    else:
        norm_stats = build_norm_stats(OUTPUT_DIR, WINDOW_SIZE, TRAIN_YEARS)
 
    # 2. 建立 test dataset
    print("載入 test 資料...")
    test_ds = build_dataset(
        output_dir  = OUTPUT_DIR,
        window_size = WINDOW_SIZE,
        years       = TEST_YEARS,
        norm_stats  = norm_stats,
    )
    print(f"Test 樣本數: {len(test_ds)}")
 
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
    )
 
    # 3. 推論
    print("\n開始 DDIM 推論...")
    all_pred_t = []
    all_true_t = []
    all_years  = []
    all_cond_t = []
    all_target_times = []
    x_mean_norm = ckpt['x_mean_norm'].to(device)
    T           = ckpt['T']
    print(f"x_mean_norm range: {x_mean_norm.min():.4f} ~ {x_mean_norm.max():.4f}")
    print(f"x_mean_norm shape: {x_mean_norm.shape}")
    print(f"T: {T}")
 
    with torch.no_grad():
        for i, (cond, target, year, target_times) in enumerate(test_loader):
            cond_np = cond.numpy()
            cond    = cond.to(device)
            pred = cold_inference(model, cond, x_mean_norm, device, T=T, num_steps=NUM_STEPS)

            print(f"batch {i}: pred range: {pred.min():.4f} ~ {pred.max():.4f}")
 
            all_pred_t.append(pred.cpu().numpy())
            all_true_t.append(target.numpy())
            all_years.append(np.array(year))
            all_cond_t.append(cond_np)
            all_target_times.append(target_times.numpy())
 
            if (i + 1) % 10 == 0:
                print(f"  已處理 {(i+1)*BATCH_SIZE} 筆...")
 
    all_pred_t = np.concatenate(all_pred_t, axis=0)
    all_true_t = np.concatenate(all_true_t, axis=0)
    all_years  = np.concatenate(all_years,  axis=0)
    all_cond_t = np.concatenate(all_cond_t, axis=0)
    all_target_times = np.concatenate(all_target_times, axis=0)  # (N, target_steps)
 
    # 4. 反正規化
    print("\n反正規化...")
    all_pred_t = all_pred_t * norm_stats.t_std + norm_stats.t_mean
    all_true_t = all_true_t * norm_stats.t_std + norm_stats.t_mean
 
    in_steps = config['in_steps']
    all_cond_t_denorm = all_cond_t[:, :in_steps] * norm_stats.t_std + norm_stats.t_mean

    print(f"all_pred_t range: {all_pred_t.min():.2f} ~ {all_pred_t.max():.2f}")
    print(f"all_true_t range: {all_true_t.min():.2f} ~ {all_true_t.max():.2f}")
    print(f"all_pred_t shape: {all_pred_t.shape}")
    print(f"all_true_t shape: {all_true_t.shape}")
 
    # 5. 儲存 .npy
    np.save(os.path.join(SAVE_DIR, 'pred_t.npy'), all_pred_t)
    np.save(os.path.join(SAVE_DIR, 'true_t.npy'), all_true_t)
    print(f"儲存至 {SAVE_DIR}")
 
    # 5.5 計算訓練集格點 P95 閾值
    print("\n計算訓練集格點 P95 閾值...")
    train_ds_p95 = build_dataset(
        output_dir=OUTPUT_DIR, window_size=WINDOW_SIZE,
        years=TRAIN_YEARS, norm_stats=norm_stats,
    )
    train_loader_p95 = DataLoader(
        train_ds_p95, batch_size=64, shuffle=False, num_workers=0
    )
    train_true_list = []
    with torch.no_grad():
        for _, target, _, _ in train_loader_p95:
            t_np = target.numpy() * norm_stats.t_std + norm_stats.t_mean
            train_true_list.append(t_np)
    train_true_all = np.concatenate(train_true_list, axis=0)   # (N_train, target_steps, H, W)
    threshold_all  = np.percentile(train_true_all, 95, axis=0) # (target_steps, H, W)
    print(f"  閾值 shape: {threshold_all.shape}")
    del train_true_all, train_true_list
 
    # 6. 計算指標
    print("\n計算指標...")
    results = {}
 
    rmse_all = np.array([compute_rmse(all_pred_t[:, s], all_true_t[:, s])
                         for s in range(all_pred_t.shape[1])])
    nrmse_all = np.array([compute_nrmse(all_pred_t[:, s], all_true_t[:, s])
                          for s in range(all_pred_t.shape[1])])
    mae_all  = np.array([compute_mae(all_pred_t[:, s],  all_true_t[:, s])
                         for s in range(all_pred_t.shape[1])])
    ssim_all = np.array([compute_ssim_per_sample(all_pred_t[:, s], all_true_t[:, s])
                         for s in range(all_pred_t.shape[1])])

    overall_rmse = rmse_all.mean(axis=0)
    overall_nrmse = nrmse_all.mean(axis=0)
    overall_mae  = mae_all.mean(axis=0)
    overall_ssim = ssim_all.mean(axis=0)
 
    # Overall P95（對所有時間步平均）
    p95_rmse_m_list, p95_rmse_s_list = [], []
    p95_mae_m_list,  p95_mae_s_list  = [], []
    p95_ssim_m_list, p95_ssim_s_list = [], []
    csi_m_list, csi_s_list = [], []
    for s in range(all_pred_t.shape[1]):
        rm, rs = compute_rmse_p95(all_pred_t[:, s], all_true_t[:, s], threshold_all[s])
        mm, ms = compute_mae_p95(all_pred_t[:, s],  all_true_t[:, s], threshold_all[s])
        sm, ss = compute_ssim_p95(all_pred_t[:, s], all_true_t[:, s], threshold_all[s])
        cm, cs = compute_csi(all_pred_t[:, s],      all_true_t[:, s], threshold_all[s])
        p95_rmse_m_list.append(rm); p95_rmse_s_list.append(rs)
        p95_mae_m_list.append(mm);  p95_mae_s_list.append(ms)
        p95_ssim_m_list.append(sm); p95_ssim_s_list.append(ss)
        csi_m_list.append(cm); csi_s_list.append(cs)
 
    results['Overall'] = {
    'RMSE' : {'mean': overall_rmse.mean(),  'std': overall_rmse.std(),
              'p95_mean': np.nanmean(p95_rmse_m_list), 'p95_std': np.nanmean(p95_rmse_s_list)},
    'nRMSE': {'mean': overall_nrmse.mean(), 'std': overall_nrmse.std()},  # ← 加這行
    'MAE'  : {'mean': overall_mae.mean(),   'std': overall_mae.std(),
              'p95_mean': np.nanmean(p95_mae_m_list),  'p95_std': np.nanmean(p95_mae_s_list)},
    'SSIM' : {'mean': overall_ssim.mean(),  'std': overall_ssim.std(),
              'p95_mean': np.nanmean(p95_ssim_m_list), 'p95_std': np.nanmean(p95_ssim_s_list)},
    'CSI'  : {'mean': np.nanmean(csi_m_list), 'std': np.nanmean(csi_s_list)},  # ← 加這行
    }
 
    for label, step in TARGET_STEPS.items():
        rmse_s  = compute_rmse(all_pred_t[:, step],  all_true_t[:, step])
        nrmse_s = compute_nrmse(all_pred_t[:, step], all_true_t[:, step])  # ← 加這行
        mae_s   = compute_mae(all_pred_t[:, step],   all_true_t[:, step])
        ssim_s  = compute_ssim_per_sample(all_pred_t[:, step], all_true_t[:, step])
        p95_rm, p95_rs = compute_rmse_p95(all_pred_t[:, step], all_true_t[:, step], threshold_all[step])
        p95_mm, p95_ms = compute_mae_p95(all_pred_t[:, step],  all_true_t[:, step], threshold_all[step])
        p95_sm, p95_ss = compute_ssim_p95(all_pred_t[:, step], all_true_t[:, step], threshold_all[step])
        csi_m, csi_s   = compute_csi(all_pred_t[:, step], all_true_t[:, step], threshold_all[step])  # ← 加這行

        results[label] = {
            'RMSE' : {'mean': rmse_s.mean(),  'std': rmse_s.std(),
                    'p95_mean': p95_rm, 'p95_std': p95_rs},
            'nRMSE': {'mean': nrmse_s.mean(), 'std': nrmse_s.std()},  # ← 加這行
            'MAE'  : {'mean': mae_s.mean(),   'std': mae_s.std(),
                    'p95_mean': p95_mm, 'p95_std': p95_ms},
            'SSIM' : {'mean': ssim_s.mean(),  'std': ssim_s.std(),
                    'p95_mean': p95_sm, 'p95_std': p95_ss},
            'CSI'  : {'mean': csi_m, 'std': csi_s},  # ← 加這行
        }
 
    print_metrics(results)
 
    # 6.5 Persistence Baseline
    print("\nPersistence Baseline:")
    last_input = all_cond_t_denorm[:, -1, :, :]
    print(f"\n{'':>10} {'RMSE mean±std':>20} {'MAE mean±std':>20}")
    print('-' * 55)
    for label, step in TARGET_STEPS.items():
        rmse_p = compute_rmse(last_input, all_true_t[:, step])
        mae_p  = compute_mae(last_input,  all_true_t[:, step])
        csi_p, csi_ps = compute_csi(last_input, all_true_t[:, step], threshold_all[step])  # ← 加這行
        print(f"{label:>10}  {rmse_p.mean():.4f} ± {rmse_p.std():.4f}   "
            f"{mae_p.mean():.4f} ± {mae_p.std():.4f}   "
            f"{csi_p:.4f} ± {csi_ps:.4f}")  # ← 加 CSI
 
    # 7. 每年抽 1 筆畫空間分布圖
    print("\n繪製空間分布圖（每年抽 1 筆）...")
    # plot_steps = [(' 3hr', 0), ('24hr', 7), ('48hr', 15), ('72hr', 23)]
    plot_steps = [('24hr', 7), ('48hr', 15), ('72hr', 23)]
    lon_ticks = list(range(-5, 31, 5))
    lat_ticks = list(range(35, 51, 5))
 
    for yr in sorted(set(all_years)):
        idx_yr     = np.where(all_years == yr)[0]
        sample_idx = np.random.choice(idx_yr)
 
        pred_sample = all_pred_t[sample_idx]
        true_sample = all_true_t[sample_idx]
 
        fig, axes = plt.subplots(3, 2, figsize=(22, 18),
                          subplot_kw={'projection': ccrs.PlateCarree()})

        # for row, (label, step) in enumerate(plot_steps):
        #     p = pred_sample[step]
        #     t = true_sample[step]
        #     d = p - t

        #     vmin = min(p.min(), t.min())
        #     vmax = max(p.max(), t.max())
        #     dmax = np.abs(d).max()

        #     lons = np.linspace(-180, 178, 180)    # 0°E ~ 358°E，間隔 2°
        #     lats = np.linspace(80, 20, 30)     # 80°N ~ 20°N，從大到小

        #     for ax, data, cmap, vlo, vhi, title, clabel in [
        #         (axes[row, 0], t, 'RdYlBu_r', vmin,  vmax,  f'True  ({label})', 'K'),
        #         (axes[row, 1], p, 'RdYlBu_r', vmin,  vmax,  f'Pred  ({label})', 'K'),
        #         # (axes[row, 2], d, 'bwr',      -dmax,  dmax,  f'Diff  ({label})', 'ΔK'),
        #     ]:
        #         im = ax.imshow(data, origin='upper', cmap=cmap,
        #             vmin=vlo, vmax=vhi,
        #             extent=[-180, 180, 20, 80],
        #             transform=ccrs.PlateCarree(),
        #             aspect='auto')
        #         ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
        #         ax.add_feature(cfeature.BORDERS, linewidth=0.5, linestyle=':')
        #         ax.set_extent([-180, 180, 20, 80], crs=ccrs.PlateCarree())
        #         ax.set_title(title)
        #         gl = ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.5)
        #         gl.top_labels = False
        #         gl.right_labels = False
        #         plt.colorbar(im, ax=ax, label=clabel, shrink=0.8)
        for row, (label, step) in enumerate(plot_steps):
            p = pred_sample[step]
            t = true_sample[step]

            vmin = min(p.min(), t.min())
            vmax = max(p.max(), t.max())

            for ax, data, show_cbar, title in [
                (axes[row, 0], t, False, f'True  ({label})'),
                (axes[row, 1], p, True,  f'Pred  ({label})'),
            ]:
                im = ax.imshow(data, origin='upper', cmap='RdYlBu_r',
                            vmin=vmin, vmax=vmax,
                            extent=[-180, 180, 20, 80],
                            transform=ccrs.PlateCarree(),
                            aspect='auto')
                ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
                ax.add_feature(cfeature.BORDERS, linewidth=0.5, linestyle=':')
                ax.set_extent([-180, 180, 20, 80], crs=ccrs.PlateCarree())
                ax.set_title(title, fontsize=18)
                gl = ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.5)
                gl.top_labels = False
                gl.right_labels = False
                gl.xlabel_style = {'size': 18}
                gl.ylabel_style = {'size': 18}

                cbar = plt.colorbar(im, ax=ax, label='K', shrink=0.8)
                cbar.ax.yaxis.label.set_size(18)
                if not show_cbar:
                    cbar.ax.set_visible(False)
 
        plt.suptitle(f'Temperature (T) — True vs Predicted (Year {yr}, Sample {sample_idx})',
                     fontsize=20)
        plt.tight_layout()
        save_path = os.path.join(SAVE_DIR, f'spatial_t_{yr}.png')
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"  {yr} 年圖已儲存：{save_path}")
 
    print("\n完成！")