import numpy as np
import os
import gc
from tqdm import tqdm
import pandas as pd


# Caminho para o arquivo .npz
#print("Definindo caminho para o arquivo")
#CAMINHO_ARQUIVO = os.path.join("..", "dados", "amex-1M-[intents-permissions-opcodes-apicalls].npz")
# CAMINHO_ARQUIVO = os.path.join("..", "dados", "amostras_balanceadas.npz")
CAMINHO_ARQUIVO = os.path.join("..", "dados", "amostras_balanceadas_apicalls.npz")
#CAMINHO_ARQUIVO = os.path.join("..", "dados", "dados_filtrados.npz")


# Carrega o arquivo .npz
print("Abrindo o arquivo")
dados = np.load(CAMINHO_ARQUIVO, allow_pickle=True, mmap_mode='r')
print(f"Arquivos: {dados.files}")

print(f"Shape dos dados: {dados['data'].shape}")
print(f"Shape das classes: {dados['classes'].shape}")
print(f"Reshape das classes: {dados['classes'].reshape(-1, 1).shape}")

# Acessa diretamente os dados
print("Carregando os dados")
X = dados['data']
y = dados['classes']
colunas = dados['column_names']


# df = pd.DataFrame(X) # Não necessário agora pois vamos usar ainda o X

# Escreve encontra os duplicados em blocos de 100000
# chunk_size = 100000
# n_linhas = X.shape[0]
# indices = []
# for i in tqdm(range(0, n_linhas, chunk_size)):
#     fim = min(i + chunk_size, n_linhas)
#     chunk = X[i:fim]
#     df_bloco = pd.DataFrame(chunk)
#     duplicados = df_bloco[df_bloco.duplicated()].index
#     duplicados = duplicados + i
#     indices.extend(duplicados)    

# print(len(indices))

# Escreve o primeiro arquivo de indices a serem eliminados
# with open('indices.txt', 'w') as f:
#   for item in indices:
#     f.write("%s\n" % item)

# Carrega os indices do primeiro arquivo de indices
# print("Abrindo o primeiros indices a serem eliminados")
# indices_lidos = []
# with open('indices.txt', 'r') as f:
#     for line in f:
#         indices_lidos.append(int(line.strip()))

# print(f"Número de índices lidos do arquivo: {len(indices_lidos)}")

# print("Removendo os primeiros registros duplicados")
# X = np.delete(X, indices_lidos, axis=0)
# y = np.delete(y, indices_lidos, axis=0)
# print(X.shape)

# gc.collect()


# Escreve encontra os duplicados em blocos de 200000
# chunk_size = 200000
# n_linhas = X.shape[0]
# indices = []
# for i in tqdm(range(0, n_linhas, chunk_size)):
#     fim = min(i + chunk_size, n_linhas)
#     chunk = X[i:fim]
#     df_bloco = pd.DataFrame(chunk)
#     duplicados = df_bloco[df_bloco.duplicated()].index
#     duplicados = duplicados + i
#     indices.extend(duplicados)    

# print(len(indices))

# Escreve o primeiro arquivo de indices a serem eliminados
# with open('indices_2.txt', 'w') as f:
#   for item in indices:
#     f.write("%s\n" % item)


# Carrega os indices do primeiro arquivo de indices
# print("Abrindo segundo arquivo de indices a serem eliminados")
# indices_lidos = []
# with open('indices_2.txt', 'r') as f:
#     for line in f:
#         indices_lidos.append(int(line.strip()))

# print(f"Número de índices lidos do arquivo: {len(indices_lidos)}")

# print("Removendo o segundo grupo de duplicados")
# X = np.delete(X, indices_lidos, axis=0)
# y = np.delete(y, indices_lidos, axis=0)
# print(X.shape)

# gc.collect()

# print("Criando dataframe")
# df = pd.DataFrame(X)
# print("Descobrindo duplicado nos restantes")
# duplicados = df[df.duplicated()].index
# print("Eliminando todos os duplicados restantes")
# X = np.delete(X, duplicados, axis = 0)
# y = np.delete(y, duplicados, axis = 0)
# print("Salvando os dados não duplicados")
# np.savez_compressed("dados_filtrados.npz", data=X, classes=y, column_names=colunas)
# print("Pronto")
# # Filtra colunas com namespace 'apicalls::'
# colunas_apicalls = [i for i, nome in enumerate(colunas) if nome.startswith("apicalls::")]

# # Extrai submatriz apenas com colunas 'apicalls'
# X_apicalls = X[:, colunas_apicalls]
# nomes_apicalls = [colunas[i] for i in colunas_apicalls]

# # Verifica se a chave 'classes' existe para adicionar como coluna final
# if 'classes' in dados:
#     y = dados['classes'].reshape(-1, 1)  # transforma em coluna
#     X_apicalls_com_classe = np.hstack((X_apicalls, y))
#     nomes_apicalls_com_classe = nomes_apicalls + ['classe']
#     print("✅ Submatriz com apicalls + classe criada com sucesso.")
# else:
#     X_apicalls_com_classe = X_apicalls
#     nomes_apicalls_com_classe = nomes_apicalls
#     print("⚠️ Classe não encontrada no arquivo. Apenas apicalls disponíveis.")

# # Exibe informações
# print(f"🔢 Shape da matriz resultante: {X_apicalls_com_classe.shape}")
# print(f"📋 Primeiros nomes de colunas: {nomes_apicalls_com_classe[:5]} ...")