import numpy as np

# Array de exemplo
arr = np.array([
    [1, 2],
    [3, 4],
    [5, 6],
    [1, 2],
    [3, 4],
    [7, 8]
])

duplicated_indices = []
n = len(arr)

for i in range(n):
    for j in range(i + 1, n):
        if np.array_equal(arr[i], arr[j]):
            if j not in duplicated_indices:
                duplicated_indices.append(j)

# Complemento: índices das linhas únicas e primeiras ocorrências
todos_indices = set(range(n))
indices_complemento = sorted(todos_indices - set(duplicated_indices))

print("Índices do complemento (únicos e primeiras ocorrências):", indices_complemento)

# 💾 Salvar a lista como array numpy
np.save("indices_complemento.npy", np.array(indices_complemento))
print("Arquivo 'indices_complemento.npy' salvo com sucesso.")

import numpy as np
import os
import pandas as pd
import gc
# Caminho para o arquivo .npz
CAMINHO_ARQUIVO = os.path.join("..", "dados", "amex-1M-[intents-permissions-opcodes-apicalls].npz")

# Carregar os dados
dados = np.load(CAMINHO_ARQUIVO, allow_pickle=True)
X = dados['data']
y = dados['classes']
colunas = dados['column_names']

# Detectar duplicatas
duplicated_indices = []
n = len(X)

for i in range(n):
    for j in range(i + 1, n):
        igual = True
        for k in range(0, len(X[i])):
            if X[i][k]!=X[j][k]:
                igual = False
        if igual == True:
            if j not in duplicated_indices:
                duplicated_indices.append(j)
                print('.',end='')

# Complemento: índices únicos e primeiras ocorrências
todos_indices = set(range(n))
indices_complemento = sorted(todos_indices - set(duplicated_indices))
print('\nFiltrando')
# Subconjunto dos dados
X_filtrado = X[indices_complemento]
y_filtrado = y[indices_complemento]
print('Salvando')
# Caminho de saída
ARQUIVO_SAIDA = "dados_sem_duplicatas.npz"

# Salvar novo arquivo .npz
np.savez(ARQUIVO_SAIDA, data=X_filtrado, classes=y_filtrado, column_names=colunas)
print(f"✅ Arquivo salvo: {ARQUIVO_SAIDA}")
print(f"Instâncias mantidas: {len(indices_complemento)} de {len(X)} totais")
