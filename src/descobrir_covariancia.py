import os
import gc
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm

import csv

# Caminho para o arquivo compactado
<<<<<<< HEAD
# CAMINHO_ARQUIVO = '../dados/amex-1M-[intents-permissions-opcodes-apicalls].npz'
CAMINHO_ARQUIVO = '../dados/amostras_sem_duplicidades.npz'
=======
CAMINHO_ARQUIVO = '../dados/amostras_reduzidas.npz'
>>>>>>> 62d1ff5026a1a25ea7f9fc0291b7a72ab0df2284

# Carrega os dados com mmap_mode para uso mais leve de memória
dados = np.load(CAMINHO_ARQUIVO, allow_pickle=True)

print(dados.files)
# Extração dos arrays principais
X = dados['data']
y = dados['classes']
colunas = dados['column_names']

print(f"X é um {type(X)} (Shape:  {X.shape})")
print(f"y é um {type(y)} (Shape: {y.shape})")
print(f"colunas é um {type(colunas)} (Shape: ({colunas.shape})")

print(f"O tipo de dados dentro de X é: {X.dtype}")

nome_grupo = "apicalls"
idx_features = [i for i, nome in enumerate(colunas) if nome.startswith(f"{nome_grupo}::")]
print(len(idx_features))

# caminho = os.path.join('.', 'idx_features.csv')
# print(caminho)
# with open(caminho, 'w', newline='') as csvfile:
#     writer = csv.writer(csvfile)
#     for s in idx_features:
#         writer.writerow([s]) 

X = X[:,idx_features]
colunas = colunas[idx_features]

print(f"X é um {type(X)} (Shape:  {X.shape})")
print(f"colunas é um {type(colunas)} (Shape: ({colunas.shape})")

gc.collect()

highly_correlated_pairs = []
threshold = 0.90

n_cols = X.shape[1]

# Iterar sobre todos os pares de colunas
# Usamos tqdm para acompanhar o progresso, o que será lento
agora = datetime.now().strftime('%d%m%Y_%H%M')
print(agora)
for i in tqdm(range(n_cols)):
    agora = datetime.now().strftime('%d%m%Y_%H%M')
    print(f"{i} - agora: {agora}", end=' ; ')
    for j in tqdm(range(i + 1, n_cols)):
        col_i = X[:, i]
        col_j = X[:, j]
        # print(".",end='')
        # Calcular a correlação de Pearson entre as duas colunas
        # np.corrcoef retorna a matriz de correlação, pegamos o valor off-diagonal
        # Adicionado tratamento para colunas com desvio padrão zero
        if np.std(col_i) == 0 or np.std(col_j) == 0:
            correlation = np.nan # ou 0, dependendo de como quer tratar
        else:
            correlation_matrix = np.corrcoef(col_i, col_j)
            correlation = correlation_matrix[0, 1]

        # Verificar se a correlação é alta e não é NaN
        if not np.isnan(correlation) and abs(correlation) > threshold:
            highly_correlated_pairs.append((i, j, correlation))

print("Pares de colunas altamente correlacionadas (correlação > 0.90):")
for pair in highly_correlated_pairs:
    print(f"Colunas {pair[0]} e {pair[1]} com correlação: {pair[2]:.4f}")

# Nome do arquivo CSV para salvar os resultados
csv_filename = 'highly_correlated_pairs.csv'

# Abrir o arquivo CSV em modo de escrita
with open(csv_filename, 'w', newline='') as csvfile:
    # Criar um escritor CSV
    writer = csv.writer(csvfile)

    # Escrever o cabeçalho
    writer.writerow(['Coluna 1', 'Coluna 2', 'Correlação'])

    # Escrever cada par de colunas e sua correlação
    for pair in highly_correlated_pairs:
        writer.writerow([pair[0], pair[1], pair[2]])

print(f"Pares altamente correlacionados salvos em '{csv_filename}'")