import os
import numpy as np
import pandas as pd
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
MODEL_PATH     = './checkpoints/ga(mse)_model_cold_final(mse)_newcanon.pth'
SAVE_DIR       = 'eval/eval_results_cold_window5_newga(mse)'

TARGET_STEPS = {
    '24hr': 7,
    '48hr': 15,
    '72hr': 23,
}

HW_K = 3          # 熱浪連續天數門檻（對齊原始 heatwave_starts_95threshold_2deg_k3_1.nc 的定義）
HW_PERCENTILE = 95

# 網格座標：請務必核對與 t_data.npy / 原始 merged_2deg_t_full.nc 的 latitude/longitude 完全一致
LAT_ARR = np.linspace(80, 20, 30)
LON_ARR = np.linspace(-180, 178, 180)


# =========================================================
# <<< 輔助函數：指標計算 >>>
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
        scores.append(ssim_map[mask].mean())
    scores = np.array(scores)
    return scores.mean(), scores.std()

def compute_csi(pred, true, threshold):
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
# <<< 輔助函數：熱浪事件判定（對齊 heatwave_starts_95threshold_2deg_k3_1.nc 的邏輯）>>>
# =========================================================
def consecutive_days_start(binary_events, k):
    """
    binary_events : (n_days, H, W) int8/bool，每日是否超過門檻
    k             : 連續天數門檻

    Returns
    -------
    starts : (n_days, H, W) int8，1 = 這天是某次熱浪事件的「起始日」
    對應原始 xarray 版本：
        conv = events.rolling(k).sum(); is_event_end = conv>=k
        full_events = OR_{i=0}^{k-1} is_event_end.shift(-i)
        starts = full_events & ~full_events.shift(1)
    """
    n = binary_events.shape[0]
    binary_events = binary_events.astype(np.int16)

    conv = np.zeros_like(binary_events)
    for i in range(k - 1, n):
        conv[i] = binary_events[i - k + 1:i + 1].sum(axis=0)
    is_event_end = (conv >= k).astype(np.int8)

    full_events = np.zeros_like(is_event_end)
    for i in range(k):
        shifted = np.zeros_like(is_event_end)
        if i == 0:
            shifted = is_event_end
        else:
            shifted[:-i] = is_event_end[i:]
        full_events = full_events | shifted

    prev = np.zeros_like(full_events)
    prev[1:] = full_events[:-1]
    starts = full_events & (~prev.astype(bool))
    return starts.astype(np.int8)


def extract_heatwave_starts(daily_max, dates_sorted, threshold, k=3):
    """
    對「逐日排序、但可能含有日期缺口」的每日最高溫序列做熱浪起始日判定。
    在缺口處會把序列切段分開處理，避免跨越缺口誤判為連續。

    daily_max    : (n_days, H, W)  已按 dates_sorted 對應排序
    dates_sorted : list[datetime.date]，與 daily_max 第一維一一對應，需嚴格遞增
    threshold    : (H, W)
    k            : 連續天數門檻

    Returns
    -------
    starts_all : (n_days, H, W) int8，跟輸入等長，缺口不會被跨越判定為連續
    """
    n = len(dates_sorted)
    binary_events = (daily_max > threshold).astype(np.int8)
    starts_all = np.zeros_like(binary_events)

    # 找出連續無缺口的分段
    seg_start = 0
    for i in range(1, n + 1):
        gap = (i == n) or ((dates_sorted[i] - dates_sorted[i - 1]).days != 1)
        if gap:
            seg = binary_events[seg_start:i]
            if len(seg) >= k:
                starts_all[seg_start:i] = consecutive_days_start(seg, k)
            # 若這段長度不足 k 天，該段內不可能形成完整熱浪事件，維持全 0
            seg_start = i

    return starts_all


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
    # np.save(os.path.join(SAVE_DIR, 'pred_t.npy'), all_pred_t)
    # np.save(os.path.join(SAVE_DIR, 'true_t.npy'), all_true_t)
    # print(f"儲存至 {SAVE_DIR}")

    # 5.5 計算訓練集格點 P95 閾值（沿用原本 windowed 版本，供 RMSE/MAE/SSIM/CSI 的極端事件指標使用）
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
        for _, target, _, _ in train_loader_p95:   # dataset 現在多回傳 target_times，需一併解包
            t_np = target.numpy() * norm_stats.t_std + norm_stats.t_mean
            train_true_list.append(t_np)
    train_true_all = np.concatenate(train_true_list, axis=0)   # (N_train, target_steps, H, W)
    threshold_all  = np.percentile(train_true_all, 95, axis=0) # (target_steps, H, W)
    print(f"  閾值 shape: {threshold_all.shape}")
    del train_true_all, train_true_list

    # =====================================================
    # 5.6 計算「熱浪判定用」的全期每日最高溫 P95 閾值
    #     （對齊 heatwave_starts_95threshold_2deg_k3_1.nc 的定義：
    #      用「全部年份」的每日最高溫算 95th percentile，而非只用訓練集）
    # =====================================================
    print("\n計算全期每日最高溫 P95 閾值（用於熱浪事件判定）...")
    raw_t_data = np.load(os.path.join(OUTPUT_DIR, 'raw', 't_data.npy'), mmap_mode='r')
    raw_times  = pd.to_datetime(np.load(os.path.join(OUTPUT_DIR, 'raw', 'times.npy'), allow_pickle=True))

    df_time_all = pd.DataFrame({'idx': np.arange(len(raw_times)), 'date': raw_times.date})
    daily_max_list = []
    dates_all_sorted = sorted(df_time_all['date'].unique())
    for date in dates_all_sorted:
        idxs = df_time_all.loc[df_time_all['date'] == date, 'idx'].values
        daily_max_list.append(raw_t_data[idxs].max(axis=0))
    full_daily_max = np.stack(daily_max_list, axis=0)   # (N_all_days, H, W)

    threshold_daily = np.percentile(full_daily_max, HW_PERCENTILE, axis=0)   # (H, W)
    print(f"  熱浪閾值 shape: {threshold_daily.shape}")
    del raw_t_data, full_daily_max, daily_max_list

    # =====================================================
    # 5.7 建立「滾動序列」（每次 inference 只取 lead=3hr 的第一步）
    #     因為 inference stride = 3hr = 資料時間解析度，
    #     這樣重建的序列每個真實時刻恰好只有一筆預測、無重疊，
    #     且用的都是誤差最小的最短前置時間結果。
    #     接著套用與真實熱浪一致的「連續 k=3 天」判定，萃取推論熱浪事件的起始日。
    # =====================================================
    print("\n建立滾動序列並萃取推論熱浪事件起始日...")
    series_pred  = all_pred_t[:, 0]                                    # (N, H, W)
    series_times = pd.to_datetime(all_target_times[:, 0], unit='ns')   # (N,)

    df_time = pd.DataFrame({'idx': np.arange(len(series_times)), 'date': series_times.date})
    dates_sorted = sorted(df_time['date'].unique())

    daily_max_list = []
    for date in dates_sorted:
        idxs = df_time.loc[df_time['date'] == date, 'idx'].values
        daily_max_list.append(series_pred[idxs].max(axis=0))
    pred_daily_max = np.stack(daily_max_list, axis=0)   # (n_days, H, W)，依日期升冪排列

    starts_pred = extract_heatwave_starts(
        daily_max    = pred_daily_max,
        dates_sorted = dates_sorted,
        threshold    = threshold_daily,
        k            = HW_K,
    )   # (n_days, H, W)

    records = []
    for d, date in enumerate(dates_sorted):
        lat_idx, lon_idx = np.where(starts_pred[d] == 1)
        for la, lo in zip(lat_idx, lon_idx):
            records.append({
                'lat': LAT_ARR[la],
                'lon': LON_ARR[lo],
                'year': pd.Timestamp(date).year,
                'month': pd.Timestamp(date).month,
                'date': date,
            })

    df_pred_heatwave = pd.DataFrame(records)
    if len(df_pred_heatwave) > 0:
        df_pred_heatwave = df_pred_heatwave[['lat', 'lon', 'year', 'month', 'date']].drop_duplicates()

    save_path_heatwave = os.path.join(SAVE_DIR, 'pred_heatwave_starts.csv')
    df_pred_heatwave.to_csv(save_path_heatwave, index=False)
    print(f"  推論熱浪事件起始日數: {len(df_pred_heatwave)}，已儲存至 {save_path_heatwave}")