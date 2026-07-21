# X_t.npy → (samples, win×8, lat, lon) 例如 window_3 是 (N, 24, 61, 141)
# Y_t.npy → (samples, 24, lat, lon) 固定 3day
# time.npy → (samples,) 每筆的 input 起始日，例如 '1965-06-01'

import xarray as xr
import numpy as np
import os
from datetime import timedelta
 
# === 設定 ===
INPUT_FILE_T = 'raw_data/0.25_t.nc'
INPUT_FILE_Z = 'raw_data/0.25_z.nc'
OUTPUT_DIR = 'output'
WINDOW_SIZES = range(3, 11)   # 3 到 10 day（input）
TARGET_DAYS = 3               # 預測目標天數
HOURS_PER_DAY = 8             # 每天 8 個時間步（每 3 小時）
TARGET_MONTHS = [6, 7, 8]     # 只取 6、7、8 月
 
# === 載入資料 ===
print("載入資料中...")
ds_t = xr.open_dataset(INPUT_FILE_T)
ds_z = xr.open_dataset(INPUT_FILE_Z)
 
# 確認變數名稱
print("變數t:", list(ds_t.data_vars))
print("維度t:", dict(ds_t.dims))
print("變數z:", list(ds_t.data_vars))
print("維度z:", dict(ds_t.dims))

# 壓掉 pressure_level 維度（只有 1 層） 
t_data = ds_t['t'].values[:, 0, :, :]   # shape: (time, lat, lon)
z_data = ds_z['z'].values[:, 0, :, :]

# 時間用 t 檔案的就好（兩個應該一樣）
times = ds_t['valid_time'].values
 
# === 取得所有年份 ===
years = sorted(set(
    int(str(t)[:4]) for t in times
    if int(str(t)[5:7]) in TARGET_MONTHS
))
print(f"資料涵蓋年份: {years[0]} ~ {years[-1]}")
 
# === 主迴圈：逐年、逐視窗長度切割 ===
for year in years:
    # 找出該年 6~8 月的時間索引
    mask = np.array([
        int(str(t)[:4]) == year and int(str(t)[5:7]) in TARGET_MONTHS
        for t in times
    ])
    idx_year = np.where(mask)[0]
 
    if len(idx_year) == 0:
        print(f"{year}: 無資料，跳過")
        continue
 
    t_year = t_data[idx_year]   # shape: (N_steps, lat, lon)
    z_year = z_data[idx_year]
    times_year = times[idx_year]
 
    n_steps = len(idx_year)
    print(f"\n{year}: 共 {n_steps} 個時間步 ({n_steps // HOURS_PER_DAY} 天)")
 
    for win in WINDOW_SIZES:
        input_steps = win * HOURS_PER_DAY
        target_steps = TARGET_DAYS * HOURS_PER_DAY
        total_steps = input_steps + target_steps
 
        X_t_list, X_z_list = [], []
        Y_t_list, Y_z_list = [], []
        time_list = []
 
        # 滑動視窗（以天為單位）
        for start_day in range(0, (n_steps - total_steps) // HOURS_PER_DAY + 1):
            start = start_day * HOURS_PER_DAY
            end = start + total_steps
 
            if end > n_steps:
                break
 
            X_t_list.append(t_year[start : start + input_steps])
            X_z_list.append(z_year[start : start + input_steps])
            Y_t_list.append(t_year[start + input_steps : end])
            Y_z_list.append(z_year[start + input_steps : end])
 
            # 記錄 input 起始時間（字串格式）
            time_str = str(times_year[start])[:10]  # 'YYYY-MM-DD'
            time_list.append(time_str)
 
        if len(X_t_list) == 0:
            print(f"  window_{win}: 樣本數為 0，跳過")
            continue
 
        # 轉成 numpy array
        # shape: (samples, days*8, lat, lon)
        X_t = np.array(X_t_list)
        X_z = np.array(X_z_list)
        Y_t = np.array(Y_t_list)
        Y_z = np.array(Y_z_list)
        time_arr = np.array(time_list)
 
        # 建立輸出資料夾
        save_dir = os.path.join(OUTPUT_DIR, f'window_{win}', str(year))
        os.makedirs(save_dir, exist_ok=True)
 
        # 儲存
        np.save(os.path.join(save_dir, 'X_t.npy'), X_t)
        np.save(os.path.join(save_dir, 'X_z.npy'), X_z)
        np.save(os.path.join(save_dir, 'Y_t.npy'), Y_t)
        np.save(os.path.join(save_dir, 'Y_z.npy'), Y_z)
        np.save(os.path.join(save_dir, 'time.npy'), time_arr)
 
        print(f"  window_{win}: {len(X_t_list)} 筆樣本，"
              f"X shape={X_t.shape}, Y shape={Y_t.shape} → 已儲存至 {save_dir}")
 
print("\n全部完成！")