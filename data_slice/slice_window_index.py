"""
slice_window_index.py
=====================
只儲存滑動視窗的「起始索引」，不儲存實際資料。
輸出：
  output/raw/t_data.npy         ← 整份 t 原始資料（一次性儲存）
  output/raw/z_data.npy         ← 整份 z 原始資料（一次性儲存）
  output/raw/times.npy          ← 對應時間戳
  output/window_{win}/{year}/starts.npy  ← 每個樣本的全域起始 index
  output/window_{win}/{year}/times.npy   ← 每個樣本對應的起始時間字串
"""

import xarray as xr
import numpy as np
import os

# === 設定 ===
INPUT_FILE_T  = 'raw_data/merged_2deg_t(1965-2025).nc'
INPUT_FILE_Z  = 'raw_data/merged_2deg_z(1965-2025).nc'
OUTPUT_DIR    = 'output'
RAW_DIR       = os.path.join(OUTPUT_DIR, 'raw')
# WINDOW_SIZES  = range(3, 11)
WINDOW_SIZES  = range(1, 17)
TARGET_DAYS   = 3
HOURS_PER_DAY = 8
TARGET_MONTHS = [6, 7, 8]

# === 載入資料 ===
print("載入資料中...")
ds_t = xr.open_dataset(INPUT_FILE_T)
ds_z = xr.open_dataset(INPUT_FILE_Z)

# t_data = ds_t['t'].values[:, 0, :, :]   # shape: (T, lat, lon)
# z_data = ds_z['z'].values[:, 0, :, :]
t_data = ds_t['t'].values
z_data = ds_z['z'].values
times  = ds_t['valid_time'].values

print(f"原始資料 shape: {t_data.shape}，共 {len(times)} 個時間步")

# === 儲存完整原始資料（只做一次）===
os.makedirs(RAW_DIR, exist_ok=True)
raw_t_path = os.path.join(RAW_DIR, 't_data.npy')
raw_z_path = os.path.join(RAW_DIR, 'z_data.npy')
raw_time_path = os.path.join(RAW_DIR, 'times.npy')

if not os.path.exists(raw_t_path):
    print("儲存完整 t_data...")
    np.save(raw_t_path, t_data)
if not os.path.exists(raw_z_path):
    print("儲存完整 z_data...")
    np.save(raw_z_path, z_data)
if not os.path.exists(raw_time_path):
    np.save(raw_time_path, times)

print(f"原始資料已存至 {RAW_DIR}")

# === 取得所有年份 ===
years = sorted(set(
    int(str(t)[:4]) for t in times
    if int(str(t)[5:7]) in TARGET_MONTHS
))
print(f"資料涵蓋年份: {years[0]} ~ {years[-1]}")

# === 主迴圈：只存索引 ===
for year in years:
    mask = np.array([
        int(str(t)[:4]) == year and int(str(t)[5:7]) in TARGET_MONTHS
        for t in times
    ])
    idx_year = np.where(mask)[0]   # 全域 index

    if len(idx_year) == 0:
        print(f"{year}: 無資料，跳過")
        continue

    n_steps = len(idx_year)
    print(f"\n{year}: 共 {n_steps} 個時間步 ({n_steps // HOURS_PER_DAY} 天)")

    for win in WINDOW_SIZES:
        input_steps  = win * HOURS_PER_DAY
        target_steps = TARGET_DAYS * HOURS_PER_DAY
        total_steps  = input_steps + target_steps

        if n_steps < total_steps:
            print(f"  window_{win}: 時間步不足，跳過")
            continue

        # 只記錄全域起始 index
        starts     = []
        time_strs  = []

        for local_start in range(0, n_steps - total_steps + 1):
            global_start = idx_year[local_start]   # 對應到完整陣列的位置
            starts.append(global_start)
            time_strs.append(str(times[global_start])[:19])

        starts_arr = np.array(starts, dtype=np.int32)
        time_arr   = np.array(time_strs)

        save_dir = os.path.join(OUTPUT_DIR, f'window_{win}', str(year))
        os.makedirs(save_dir, exist_ok=True)
        np.save(os.path.join(save_dir, 'starts.npy'), starts_arr)
        np.save(os.path.join(save_dir, 'times.npy'),  time_arr)

        print(f"  window_{win}: {len(starts)} 筆索引（X={input_steps}步, Y={target_steps}步）→ {save_dir}")

print("\n全部完成！")
print(f"原始資料路徑: {RAW_DIR}/t_data.npy, z_data.npy")
print("訓練時請使用 weather_dataset.py 中的 WeatherDataset 動態讀取。")