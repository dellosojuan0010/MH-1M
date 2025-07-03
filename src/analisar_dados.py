import numpy as np
import pandas as pd
import os

# Caminho relativo para acessar o arquivo de dados
CAMINHO_ARQUIVO = os.path.join("..", "dados", "amex-1M-[intents-permissions-opcodes-apicalls].npz")

# Verifica se o arquivo existe
if not os.path.exists(CAMINHO_ARQUIVO):
    print(f"Arquivo não encontrado: {CAMINHO_ARQUIVO}")
    exit(1)

# Carrega o arquivo
print(f"Carregando arquivo: {CAMINHO_ARQUIVO}")
dados = np.load(CAMINHO_ARQUIVO, allow_pickle=True, mmap_mode='r')

print("Chaves encontradas no arquivo:")
print(dados.files)

X = dados['data']
print(f"Shape da matriz 'data': {X.shape}")
print(f"Tipo de dado da matriz 'data': {X.dtype}")

# # Análise da chave 'data'
# if 'data' in dados:
#     print("\nAnalisando chave: 'data'")
#     X = dados['data']

#     print(f"Shape: {X.shape}")
#     print(f"Tipo de dado: {X.dtype}")

#     if np.issubdtype(X.dtype, np.number):
#         df = pd.DataFrame(X)
#         print("\nEstatísticas descritivas:")
#         print(df.describe())
#     else:
#         print("A matriz 'data' não é numérica.")
# else:
#     print("A chave 'data' não foi encontrada no arquivo.")

# # Análise da chave 'classes'
# if 'classes' in dados:
#     y = dados['classes']
#     unicos, contagens = np.unique(y, return_counts=True)
#     print("\nDistribuição das classes:")
#     for valor, qtd in zip(unicos, contagens):
#         print(f"Classe {valor}: {qtd} instâncias")

# #
