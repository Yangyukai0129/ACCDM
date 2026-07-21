import random
import re
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

# =========================================================
# <<< 可調參數設定區 >>>
# =========================================================

# GA 參數
POPULATION_SIZE = 20 #10
GENERATIONS     = 20
NUM_PARENTS     = 4
NUM_ELITES      = 2
ALPHA           = 0.0   # 深度懲罰係數，調大會傾向淺層
# α 太小（如 0.001）→ 懲罰幾乎無感，GA 會偏好深層
# α 太大（如 1.0）→ 懲罰蓋過 Loss，GA 永遠選最淺層

# 變異率
PM_DEPTH  = 0.01         # 深度基因變異率（bits 4-6）
PM_PARAM  = 0.1          # 參數基因變異率（其餘 bits，含填充區，不再排除）

# 資料設定
TRAIN_YEARS  = range(1965, 2020)
OUTPUT_DIR   = 'output'
WINDOW_SIZES = list(range(1, 17))   # 1~16 day

# Proxy 訓練設定
PROXY_SAMPLES = 1000 #500
PROXY_EPOCHS  = 20 #20
PROXY_BATCH   = 16

# =========================================================
# <<< 染色體編碼常數 >>>
# =========================================================
S_BITS = 4   # bits 0-3：輸入索引（window 選擇）
D_BITS = 3   # bits 4-6：深度
G_BITS = 2   # 每層參數位元數

# g(1) → base_channels（PHI_MAP）
PHI_MAP = {
    '00': 16,
    '01': 32,
    '10': 64,
    '11': 128,
}

# g(2)~g(depth) → 累乘倍率（MU_MAP）
# 四種位元組合皆為合法倍率，'00'（×1，持平）不再是填充區佔位，
# 也不會被 canonicalize 強制轉換成其他值。
# 填充區與活躍區的區分，完全由 depth 基因在 decode 當下即時判斷，
# 不依賴任何位元值本身。
MU_MAP = {
    '00': 1,   # ×1
    '01': 2,   # ×2
    '10': 4,   # ×4
    '11': 8,   # ×8
}

MAX_LAYERS        = 8
CHROMOSOME_LENGTH = S_BITS + D_BITS + G_BITS * MAX_LAYERS  # 4+3+2*8 = 23 bits
C_MAX_CHANNELS    = 1024   # 通道數硬上限

# =========================================================
# <<< 輔助函數 >>>
# =========================================================
def bin_to_int(b): return int(b, 2)
def int_to_bin(n, bits): return format(n, f'0{bits}b')

def get_data_dir(window_day):
    return os.path.join(OUTPUT_DIR, f'window_{window_day}')

# =========================================================
# <<< Canonicalize >>>
# =========================================================
def canonicalize(chrom):
    """
    Canonicalize 只做一件事：依 depth 基因，把染色體『切』到只剩活躍區，
    填充區直接捨棄、不讀取、不覆寫。

    - 不修改 depth 基因本身（不做跨親代 clip，公式(4)已移除）。
    - depth 僅做全域範圍檢查 [2, MAX_LAYERS]，這跟 decode() 內部的
      clamp 邏輯一致，純粹是全域邊界保護，不涉及任何親代資訊。
    - 回傳值長度會隨 depth 而變化（不是固定的 CHROMOSOME_LENGTH），
      只適合當作「活躍基因內容」的代表值（例如 evaluate_fitness 的
      cache key），不能再拿去做固定長度的位元操作。
    """
    d_val = 1 + bin_to_int(chrom[S_BITS:S_BITS + D_BITS])
    d_val = max(2, min(d_val, MAX_LAYERS))  # 全域邊界，非跨親代 clip

    active_len = S_BITS + D_BITS + d_val * G_BITS
    return chrom[:active_len]

# =========================================================
# <<< 解碼染色體 >>>
# =========================================================
def decode(chrom):
    chrom = canonicalize(chrom)

    # 輸入索引 → window day
    s_idx   = min(bin_to_int(chrom[0:S_BITS]), len(WINDOW_SIZES) - 1)
    win_day = WINDOW_SIZES[s_idx]

    # 深度
    depth = 1 + bin_to_int(chrom[S_BITS:S_BITS + D_BITS])
    depth = max(2, min(depth, MAX_LAYERS))

    # base_channels（g(1)）
    g1       = chrom[S_BITS + D_BITS: S_BITS + D_BITS + G_BITS]
    base_ch  = PHI_MAP.get(g1, 32)

    # channel_mults（g(2)~g(depth)，累乘；允許持平，不要求嚴格遞增）
    channel_mults = [1]
    current_mult  = 1
    for k in range(1, depth):
        idx    = S_BITS + D_BITS + k * G_BITS
        gk     = chrom[idx:idx + G_BITS]
        factor = MU_MAP.get(gk, 1)
        current_mult *= factor
        safe_max = max(1, C_MAX_CHANNELS // base_ch)
        channel_mults.append(min(current_mult, safe_max))

    return {
        'window_day'   : win_day,
        'data_dir'     : get_data_dir(win_day),
        'depth'        : depth,
        'base_channels': base_ch,
        'channel_mults': channel_mults,
        'in_steps'     : win_day * 8,    # 每天 8 個時間步
        'target_steps' : 3 * 8,          # 固定預測 3 天
    }

# =========================================================
# <<< 隨機產生個體 >>>
# =========================================================
def random_individual():
    # 不再提前呼叫 canonicalize；族群中儲存的染色體全程維持原始未遮罩狀態，
    # canonicalize 只在 decode / evaluate_fitness 當下即時計算。
    return ''.join(random.choice('01') for _ in range(CHROMOSOME_LENGTH))

# =========================================================
# <<< 交叉（區塊對齊）>>>
# =========================================================
def crossover(p1, p2):
    boundaries = [0, S_BITS, S_BITS + D_BITS]
    for i in range(MAX_LAYERS):
        boundaries.append(S_BITS + D_BITS + i * G_BITS)
    boundaries.append(CHROMOSOME_LENGTH)
    boundaries = sorted(set(boundaries))

    num_points = 1
    points = sorted(random.sample(boundaries[1:-1], k=min(num_points, len(boundaries) - 2)))

    c1, c2 = list(p1), list(p2)
    swap = False
    prev = 0
    for pt in points + [CHROMOSOME_LENGTH]:
        if swap:
            c1[prev:pt], c2[prev:pt] = c2[prev:pt], c1[prev:pt]
        swap = not swap
        prev = pt

    return ''.join(c1), ''.join(c2)  # 不在這裡 canonicalize

# =========================================================
# <<< 變異（深度區與參數區分開變異率）>>>
# =========================================================
def mutate(chrom):
    # 不再排除填充區：填充區的定義只在 decode 前才由 canonicalize 判斷，
    # 突變階段的染色體本身沒有「填充區」跟「活躍區」之分。
    c = list(chrom)
    for i in range(CHROMOSOME_LENGTH):
        pm = PM_DEPTH if S_BITS <= i < S_BITS + D_BITS else PM_PARAM
        if random.random() < pm:
            c[i] = '1' if c[i] == '0' else '0'
    return ''.join(c)

# =========================================================
# <<< Rank-based Selection >>>
# =========================================================
def rank_selection(pop_with_fitness, num_parents):
    # 過濾掉 fitness = -inf
    valid = [(c, f) for c, f in pop_with_fitness if f != -float('inf')]
    if not valid:
        print("!!! 所有個體 fitness 為 -inf，隨機選取父母")
        return [c for c, _ in random.sample(pop_with_fitness, k=num_parents)]

    # 依 fitness 由小到大排序（rank 1 = 最差，rank N = 最好）
    valid.sort(key=lambda x: x[1])
    n = len(valid)

    # 權重：rank 越高權重越大
    weights = [i + 1 for i in range(n)]           # 1, 2, ..., n
    total   = sum(weights)
    probs   = [w / total for w in weights]

    # 依權重抽樣
    selected = random.choices([c for c, _ in valid], weights=probs, k=num_parents)
    return selected

# =========================================================
# <<< Proxy 訓練與 Fitness 評估 >>>
# =========================================================
dataset_cache = {}
norm_cache     = {}
phenotype_cache = {}

def evaluate_fitness(chrom, device):
    # canonicalize 後的（變動長度）活躍區字串，同時作為 cache key：
    # depth、活躍基因皆相同的染色體會命中同一筆快取，不受填充區雜訊影響。
    cache_key = canonicalize(chrom)

    if cache_key in phenotype_cache:
        f = phenotype_cache[cache_key]
        print(f"  [快取] fitness={f:.6f}")
        return f

    cfg = decode(chrom)
    print(f"\n  評估: window={cfg['window_day']}day, depth={cfg['depth']}, "
          f"base_ch={cfg['base_channels']}, mults={cfg['channel_mults']}")

    try:
        # 載入資料（有快取則不重複載入）
        data_dir = cfg['data_dir']
        from weather_dataset import build_dataset, build_norm_stats

        if data_dir not in dataset_cache:
            win = cfg['window_day']
            # 同一個 window_day 的 norm_stats 只算一次
            if win not in norm_cache:
                norm_cache[win] = build_norm_stats(OUTPUT_DIR, win, TRAIN_YEARS)
            dataset_cache[data_dir] = build_dataset(
                output_dir  = OUTPUT_DIR,
                window_size = win,
                years       = TRAIN_YEARS,
                norm_stats  = norm_cache[win],
            )
        dataset = dataset_cache[data_dir]

        if len(dataset) == 0:
            raise ValueError(f"資料集為空：{data_dir}")

        # 隨機抽樣 proxy subset
        indices = random.sample(range(len(dataset)), k=min(len(dataset), PROXY_SAMPLES))
        subset  = Subset(dataset, indices)
        loader  = DataLoader(subset, batch_size=PROXY_BATCH, shuffle=True, num_workers=0)

        # 建立模型（需要ga_unet.py 提供 UNet）
        from module.ga_unet_baseline import UNet
        from module.diffusion_baseline import make_beta_schedule
        model     = UNet(cfg).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        criterion = nn.MSELoss()
        beta, _, alpha_cumprod = make_beta_schedule(device=device)

        model.train()
        final_loss = None
        for epoch in range(PROXY_EPOCHS):
            epoch_loss = 0.0
            for cond, target, _, _ in loader:
                cond   = cond.to(device)
                target = target.to(device)
                B = cond.shape[0]
                t = torch.randint(0, len(beta), (B,), device=device)

                # 加噪
                noise = torch.randn_like(target)
                a_t   = alpha_cumprod[t][:, None, None, None]
                x_t   = torch.sqrt(a_t) * target + torch.sqrt(1 - a_t) * noise

                optimizer.zero_grad(set_to_none=True)
                pred = model(x_t, cond, t, beta)
                loss = criterion(pred, noise)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            final_loss = epoch_loss / len(loader)

        rmse    = final_loss ** 0.5
        fitness = -(rmse + ALPHA * cfg['depth'])

        del model, optimizer, loader, subset
        torch.cuda.empty_cache()

        print(f"  RMSE={rmse:.6f}, depth_penalty={ALPHA * cfg['depth']:.4f}, fitness={fitness:.6f}")
        phenotype_cache[cache_key] = fitness
        return fitness

    except Exception as e:
        import traceback
        print(f"  !!! 評估失敗: {e}")
        traceback.print_exc()
        phenotype_cache[cache_key] = -float('inf')
        return -float('inf')

# =========================================================
# <<< 收斂曲線 >>>
# =========================================================
def plot_convergence(history):
    import matplotlib.pyplot as plt

    generations = list(range(1, len(history["best"]) + 1))

    # 轉回 fitness（負號）
    best_fitness = [-v for v in history["best"]]
    avg_fitness  = [-v for v in history["avg"]]

    plt.figure(figsize=(10, 5))
    plt.plot(generations, best_fitness, "r-o", linewidth=2, markersize=6, label="Best fitness")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.title("GA Convergence Curve")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig("ga_adjust2_convergence_mse.png", dpi=150)
    plt.show()
    print("Convergence curve saved to ga_convergence.png")

# =========================================================
# <<< 主 GA 流程 >>>
# =========================================================
def genetic_algorithm(device='cuda'):
    print(f"=== 開始 GA 最佳化（共 {GENERATIONS} 代，族群大小 {POPULATION_SIZE}）===")

    population = [random_individual() for _ in range(POPULATION_SIZE)]

    best_chrom   = None
    best_fitness = -float('inf')
    history = {'best': [], 'avg': []}

    for gen in range(GENERATIONS):
        print(f"\n{'='*50}")
        print(f"第 {gen+1}/{GENERATIONS} 代")

        # 評估 fitness（canonicalize 只發生在 evaluate_fitness / decode 內部）
        pop_with_fitness = []
        for chrom in population:
            f = evaluate_fitness(chrom, device)
            pop_with_fitness.append((chrom, f))
            if f > best_fitness:
                best_fitness = f
                best_chrom   = chrom

        # 統計
        valid_f = [f for _, f in pop_with_fitness if f != -float('inf')]
        if valid_f:
            gen_best = max(valid_f)
            gen_avg  = np.mean(valid_f)
        else:
            gen_best = gen_avg = float('inf')

        history['best'].append(-best_fitness)   # 轉回 RMSE+penalty
        history['avg'].append(gen_avg if gen_avg == float('inf') else -gen_avg)

        print(f"\n本代最佳 fitness={gen_best:.6f}, 平均={gen_avg:.6f}")
        print(f"歷代最佳 fitness={best_fitness:.6f}")

        # 精英保留
        pop_with_fitness.sort(key=lambda x: x[1], reverse=True)
        next_pop = [chrom for chrom, _ in pop_with_fitness[:NUM_ELITES]]

        # Rank-based selection + 交叉 + 變異 → 填滿族群
        # 交叉、突變後直接存入下一代，不再呼叫 canonicalize
        # （跨親代 depth clip / 公式(4) 已移除）
        parents = rank_selection(pop_with_fitness, NUM_PARENTS)
        while len(next_pop) < POPULATION_SIZE:
            p1, p2 = random.sample(parents, 2)
            c1, c2 = crossover(p1, p2)
            c1, c2 = mutate(c1), mutate(c2)
            next_pop.append(c1)
            if len(next_pop) < POPULATION_SIZE:
                next_pop.append(c2)

        population = next_pop

    print("\n=== GA 結束 ===")
    best_cfg = decode(best_chrom)
    print(f"最佳設定: window={best_cfg['window_day']}day, depth={best_cfg['depth']}, "
          f"base_ch={best_cfg['base_channels']}, mults={best_cfg['channel_mults']}")
    print(f"最佳 fitness={best_fitness:.6f}")

    # 繪製收斂曲線
    plot_convergence(history)

    return best_cfg, history

# =========================================================
# <<< 主程式入口 >>>
# =========================================================
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用設備: {device}")

    best_cfg, history = genetic_algorithm(device=device)

    print("\n最終最佳設定:")
    print(best_cfg)