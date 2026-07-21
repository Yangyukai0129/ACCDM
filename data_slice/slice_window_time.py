import xarray as xr
import numpy as np
import os

# === 設定 ===
INPUT_FILE_T = 'raw_data/0.25_t.nc'
INPUT_FILE_Z = 'raw_data/0.25_z.nc'
OUTPUT_DIR = 'output'
WINDOW_SIZES = range(3, 11)
TARGET_DAYS = 3
HOURS_PER_DAY = 8
TARGET_MONTHS = [6, 7, 8]

# === 載入資料 ===
print("載入資料中...")
ds_t = xr.open_dataset(INPUT_FILE_T)
ds_z = xr.open_dataset(INPUT_FILE_Z)

print("變數t:", list(ds_t.data_vars))
print("維度t:", dict(ds_t.dims))
print("變數z:", list(ds_z.data_vars))
print("維度z:", dict(ds_z.dims))

t_data = ds_t['t'].values[:, 0, :, :]
z_data = ds_z['z'].values[:, 0, :, :]
times = ds_t['valid_time'].values

# === 取得所有年份 ===
years = sorted(set(
    int(str(t)[:4]) for t in times
    if int(str(t)[5:7]) in TARGET_MONTHS
))
print(f"資料涵蓋年份: {years[0]} ~ {years[-1]}")

# === 主迴圈：逐年、逐視窗長度切割 ===
for year in years:
    mask = np.array([
        int(str(t)[:4]) == year and int(str(t)[5:7]) in TARGET_MONTHS
        for t in times
    ])
    idx_year = np.where(mask)[0]

    if len(idx_year) == 0:
        print(f"{year}: 無資料，跳過")
        continue

    t_year = t_data[idx_year]
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

        # 以時間步滑動（每次移動 1 個時間步）
        # end 不超過 n_steps，確保年跟年不重疊
        for start in range(0, n_steps - total_steps + 1):
            end = start + total_steps

            X_t_list.append(t_year[start : start + input_steps])
            X_z_list.append(z_year[start : start + input_steps])
            Y_t_list.append(t_year[start + input_steps : end])
            Y_z_list.append(z_year[start + input_steps : end])

            # 記錄 input 起始時間（含小時）
            time_str = str(times_year[start])[:19]  # 'YYYY-MM-DDTHH:MM:SS'
            time_list.append(time_str)

        if len(X_t_list) == 0:
            print(f"  window_{win}: 樣本數為 0，跳過")
            continue

        X_t = np.array(X_t_list)
        X_z = np.array(X_z_list)
        Y_t = np.array(Y_t_list)
        Y_z = np.array(Y_z_list)
        time_arr = np.array(time_list)

        save_dir = os.path.join(OUTPUT_DIR, f'window_{win}', str(year))
        os.makedirs(save_dir, exist_ok=True)

        np.savez_compressed(
            os.path.join(save_dir, 'data.npz'),
            X_t=X_t, X_z=X_z,
            Y_t=Y_t, Y_z=Y_z,
            time=time_arr
        )

        print(f"  window_{win}: {len(X_t_list)} 筆樣本，"
              f"X shape={X_t.shape}, Y shape={Y_t.shape} → 已儲存至 {save_dir}")

print("\n全部完成！")