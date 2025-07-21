import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm
import pandas as pd


# Exemplo com poucas linhas (substitua por seu X real)
# X = np.load("seuarquivo.npz")['data']
#X = np.random.randint(0, 10, size=(1000, 20))  # cuidado com tamanhos grandes!

dados = np.load('dados_filtrados.npz', allow_pickle=True,mmap_mode='r')
X = dados['data']
df = pd.DataFrame(X)
print(df.head(10))
y = dados['classes']

# colunas = dados['column_names']

# indices_lidos = []
# with open('indices_2.txt', 'r') as f:
#     for line in f:
#         indices_lidos.append(int(line.strip()))

# X = np.delete(X, indices_lidos, axis=0)
# y = np.delete(y, indices_lidos, axis=0)

# np.savez_compressed("dados_filtrados_novos.npz", data=X, classes=y, column_names=colunas)