import numpy as np
import os
import gc
from tqdm import tqdm
import pandas as pd


# Caminho para o arquivo .npz
CAMINHO_ARQUIVO = os.path.join("..", "dados", "amex-1M-[intents-permissions-opcodes-apicalls].npz")
#CAMINHO_ARQUIVO = os.path.join("..", "dados", "amostras_balanceadas.npz")
#CAMINHO_ARQUIVO = os.path.join("..", "dados", "amostras_balanceadas_apicalls.npz")

# Carrega o arquivo .npz
dados = np.load(CAMINHO_ARQUIVO, allow_pickle=True, mmap_mode='r')
print(f"Arquivos: {dados.files}")

print(f"🔢 Shape dos dados: {dados['data'].shape}")
print(f"🔢 Shape das classes: {dados['classes'].shape}")
print(f"🔢 Reshape das classes: {dados['classes'].reshape(-1, 1).shape}")

# Acessa diretamente os dados
X = dados['data']
y = dados['classes']
colunas = dados['column_names']

df = pd.DataFrame(X)

chunk_size = 100000
n_linhas = X.shape[0]
indices = []
for i in tqdm(range(0, n_linhas, chunk_size)):
    fim = min(i + chunk_size, n_linhas)
    chunk = X[i:fim]
    df_bloco = pd.DataFrame(chunk)
    duplicados = df_bloco[df_bloco.duplicated()].index
    duplicados = duplicados + i
    indices.extend(duplicados)    

print(len(indices))

#     # Aqui você insere a lógica de processamento
#     processar(bloco)

# gc.collect()
X_filtrado = np.delete(X, indices, axis=0)
y_filtrado = np.delete(y, indices, axis=0)

# print(X_filtrado.shape)


                                                                                                                                                                                                                                                                                                                                                                                          

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
