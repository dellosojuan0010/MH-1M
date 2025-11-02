import os
import gc
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm

import csv

# Caminho para o arquivo compactado
CAMINHO_ARQUIVO = '../dados/amex-1M-[intents-permissions-opcodes-apicalls].npz'

# Carrega os dados com mmap_mode para uso mais leve de memória
dados = np.load(CAMINHO_ARQUIVO, allow_pickle=True)

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

caminho = os.path.join('.', 'idx_features.csv')
print(caminho)
with open(caminho, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    for s in idx_features:
        writer.writerow([s]) 

X = X[:,idx_features]
colunas = colunas[idx_features]

print(f"X é um {type(X)} (Shape:  {X.shape})")
print(f"colunas é um {type(colunas)} (Shape: ({colunas.shape})")

gc.collect()

lista_remover = []

# Itera sobre cada linha do array com um índice
for i in tqdm(range(X.shape[0])):
    # Considera que a linha i é unico
    print(i, end=' ')
    if i in lista_remover:
        break
    for j in tqdm(range(i+1, X.shape[0])):
        if j not in lista_remover:
            a_remover = 1            
            for k in range(X.shape[1]):
                if X[i][k] != X[j][k]:
                    a_remover = 0
                    break
            if a_remover == 1:
                lista_remover.append(j)


# Nome do arquivo CSV para salvar os resultados
csv_filename = 'amostras_a_remover.csv'

# Abrir o arquivo CSV em modo de escrita
with open(csv_filename, 'w', newline='') as csvfile:
    # Criar um escritor CSV
    writer = csv.writer(csvfile)

    # Escrever o cabeçalho (opcional)
    writer.writerow(['idxs'])

    # Escrever cada inteiro da lista Z em uma nova linha
    for inteiro in lista_remover:
        writer.writerow([inteiro])