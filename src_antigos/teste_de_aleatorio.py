import numpy as np
rng = np.random.default_rng(102)

idx_0 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40]
idx_1 = [101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120]

idx_0_amostrado = rng.choice(idx_0, size=len(idx_1), replace=False)

idx_final = np.concatenate([idx_1, idx_0_amostrado])
rng.shuffle(idx_final)

print(f"Índices embaralhados: {idx_final}")
