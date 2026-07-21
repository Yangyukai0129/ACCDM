"""
weather_dataset.py
==================
從預先計算的起始索引動態切取樣本，支援正規化與多年份合併。

目錄結構（由 slice_window_index.py 產生）：
  output/raw/t_data.npy
  output/raw/z_data.npy
  output/raw/times.npy
  output/window_{win}/{year}/starts.npy
  output/window_{win}/{year}/times.npy
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, ConcatDataset


# =========================================================
# <<< 正規化統計量（從訓練集計算，供所有 split 共用）>>>
# =========================================================
class NormStats:
    """
    計算並保存正規化統計量（mean / std）。
    只應從訓練集計算，再傳給驗證集與測試集使用。
    """
    def __init__(self, t_data_path, z_data_path, starts_list):
        """
        Parameters
        ----------
        t_data_path : str           — output/raw/t_data.npy 路徑
        z_data_path : str           — output/raw/z_data.npy 路徑
        starts_list : list[ndarray] — 多個年份的 starts 陣列（全域 index）
        """
        print("計算正規化統計量（從訓練集抽樣）...")
        t_data = np.load(t_data_path, mmap_mode='r')
        z_data = np.load(z_data_path, mmap_mode='r')

        # 合併所有起始 index，最多取 2000 筆估計統計量
        all_starts = np.concatenate(starts_list)
        if len(all_starts) > 2000:
            all_starts = all_starts[np.random.choice(len(all_starts), 2000, replace=False)]

        t_samples = np.stack([t_data[i] for i in all_starts], axis=0)
        z_samples = np.stack([z_data[i] for i in all_starts], axis=0)

        self.t_mean = float(t_samples.mean())
        self.t_std  = float(t_samples.std()) + 1e-8
        self.z_mean = float(z_samples.mean())
        self.z_std  = float(z_samples.std()) + 1e-8

        print(f"  t: mean={self.t_mean:.4f}, std={self.t_std:.4f}")
        print(f"  z: mean={self.z_mean:.4f}, std={self.z_std:.4f}")

    def normalize_t(self, x): return (x - self.t_mean) / self.t_std
    def normalize_z(self, x): return (x - self.z_mean) / self.z_std
    def denormalize_t(self, x): return x * self.t_std + self.t_mean
    def denormalize_z(self, x): return x * self.z_std + self.z_mean


# =========================================================
# <<< 單一年份 Dataset >>>
# =========================================================
class WeatherDataset(Dataset):
    """
    根據 starts.npy 的全域索引，從原始資料動態切取樣本。

    Parameters
    ----------
    starts_path  : str        — starts.npy 路徑
    t_data_path  : str        — output/raw/t_data.npy 路徑
    z_data_path  : str        — output/raw/z_data.npy 路徑
    input_steps  : int        — 輸入時間步數（window_day × 8）
    target_steps : int        — 預測時間步數（target_days × 8）
    norm_stats   : NormStats  — 正規化統計量（None 表示不正規化）
    """

    def __init__(self, starts_path, t_data_path, z_data_path, times_path,
                 input_steps, target_steps, norm_stats=None, year=None):
        self.starts       = np.load(starts_path)                 # (N,)
        self.t_data       = np.load(t_data_path, mmap_mode='r')  # (T, lat, lon)
        self.z_data       = np.load(z_data_path, mmap_mode='r')
        self.input_steps  = input_steps
        self.target_steps = target_steps
        self.norm_stats   = norm_stats
        self.year = year
        self.times        = np.load(times_path, allow_pickle=True)

        # max_valid = len(self.t_data) - input_steps - target_steps
        # valid_mask = self.starts <= max_valid
        # if valid_mask.sum() < len(self.starts):
        #     print(f"  [{year}] 過濾掉 {(~valid_mask).sum()} 個超出範圍的樣本")
        # self.starts = self.starts[valid_mask]

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, idx):
        s  = int(self.starts[idx])
        e  = s + self.input_steps
        e2 = e + self.target_steps

        # 動態切取（.copy() 讓 mmap 資料轉為實際記憶體）
        X_t = self.t_data[s:e].copy().astype(np.float32)    # (input_steps, lat, lon)
        X_z = self.z_data[s:e].copy().astype(np.float32)
        Y_t = self.t_data[e:e2].copy().astype(np.float32)   # (target_steps, lat, lon)
        Y_z = self.z_data[e:e2].copy().astype(np.float32)

        # 正規化
        if self.norm_stats is not None:
            ns  = self.norm_stats
            X_t = ns.normalize_t(X_t)
            X_z = ns.normalize_z(X_z)
            Y_t = ns.normalize_t(Y_t)
            Y_z = ns.normalize_z(Y_z)

        # 合併 t、z 成單一 tensor，channel 軸在前
        # cond   shape: (input_steps*2,  lat, lon)
        # target shape: (target_steps*2, lat, lon)
        cond   = torch.from_numpy(np.concatenate([X_t, X_z], axis=0))
        target = torch.from_numpy(np.concatenate([Y_t], axis=0))
        target_times = self.times[e:e2].astype('datetime64[ns]').astype(np.int64)

        return cond, target, self.year, torch.from_numpy(target_times)


# =========================================================
# <<< 多年份合併（對外主要介面）>>>
# =========================================================
def build_dataset(output_dir, window_size, years,
                  hours_per_day=8, target_days=3,
                  norm_stats=None):
    """
    合併多個年份的 WeatherDataset，回傳 ConcatDataset。

    Parameters
    ----------
    output_dir   : str       — output 根目錄
    window_size  : int       — 視窗大小（天數）
    years        : iterable  — 要合併的年份
    hours_per_day: int
    target_days  : int
    norm_stats   : NormStats — 傳入訓練集統計量；None 表示不正規化

    Returns
    -------
    ConcatDataset | None
    """
    input_steps  = window_size * hours_per_day
    target_steps = target_days * hours_per_day
    t_data_path  = os.path.join(output_dir, 'raw', 't_data.npy')
    z_data_path  = os.path.join(output_dir, 'raw', 'z_data.npy')
    times_path   = os.path.join(output_dir, 'raw', 'times.npy')

    datasets = []
    for year in years:
        starts_path = os.path.join(output_dir, f'window_{window_size}', str(year), 'starts.npy')
        if not os.path.exists(starts_path):
            continue
        ds = WeatherDataset(
            starts_path  = starts_path,
            t_data_path  = t_data_path,
            z_data_path  = z_data_path,
            times_path   = times_path,
            input_steps  = input_steps,
            target_steps = target_steps,
            norm_stats   = norm_stats,
            year = year
        )
        datasets.append(ds)

    if not datasets:
        print(f"[警告] window_{window_size}：找不到任何年份資料，請先執行 slice_window_index.py")
        return None

    return ConcatDataset(datasets)


def build_norm_stats(output_dir, window_size, train_years):
    """
    從訓練集年份計算 NormStats。
    應在 build_dataset 之前呼叫，將回傳值傳給 build_dataset 的 norm_stats 參數。
    """
    t_data_path = os.path.join(output_dir, 'raw', 't_data.npy')
    z_data_path = os.path.join(output_dir, 'raw', 'z_data.npy')

    starts_list = []
    for year in train_years:
        p = os.path.join(output_dir, f'window_{window_size}', str(year), 'starts.npy')
        if os.path.exists(p):
            starts_list.append(np.load(p))

    if not starts_list:
        raise FileNotFoundError(
            f"找不到 window_{window_size} 的訓練資料，請先執行 slice_window_index.py"
        )

    return NormStats(t_data_path, z_data_path, starts_list)


# =========================================================
# <<< 使用範例 >>>
# =========================================================
# if __name__ == '__main__':
#     from torch.utils.data import DataLoader

#     OUTPUT_DIR  = 'output'
#     WINDOW_SIZE = 3
#     TRAIN_YEARS = list(range(1965, 2018))
#     VAL_YEARS   = list(range(2018, 2021))

#     # 1. 從訓練集計算正規化統計量
#     norm_stats = build_norm_stats(OUTPUT_DIR, WINDOW_SIZE, TRAIN_YEARS)

#     # 2. 建立訓練集與驗證集（驗證集共用訓練集統計量）
#     train_ds = build_dataset(OUTPUT_DIR, WINDOW_SIZE, TRAIN_YEARS, norm_stats=norm_stats)
#     val_ds   = build_dataset(OUTPUT_DIR, WINDOW_SIZE, VAL_YEARS,   norm_stats=norm_stats)

#     print(f"訓練集: {len(train_ds)} 筆，驗證集: {len(val_ds)} 筆")

#     # 3. DataLoader
#     loader = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=2)
#     cond, target = next(iter(loader))
#     print(f"cond shape:   {cond.shape}")    # (4, input_steps*2, lat, lon)
#     print(f"target shape: {target.shape}")  # (4, target_steps*2, lat, lon)