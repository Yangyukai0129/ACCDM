import numpy as np
import os

OUTPUT_DIR = 'output'

for win in range(3, 11):
    total = 0
    for year in sorted(os.listdir(f'{OUTPUT_DIR}/window_{win}')):
        path = f'{OUTPUT_DIR}/window_{win}/{year}/time.npy'
        t = np.load(path, allow_pickle=True)
        total += len(t)
    print(f'window_{win}: {total} 筆')