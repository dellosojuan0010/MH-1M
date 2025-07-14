import numpy as np
import os

# Caminho para o arquivo .npz
#CAMINHO_ARQUIVO = os.path.join("..", "dados", "amex-1M-[intents-permissions-opcodes-apicalls].npz")
#CAMINHO_ARQUIVO = os.path.join("..", "dados", "amostras_balanceadas.npz")
CAMINHO_ARQUIVO = os.path.join("..", "dados", "amostras_balanceadas_apicalls.npz")

# Carrega o arquivo .npz
dados = np.load(CAMINHO_ARQUIVO, allow_pickle=True)
print(f"Arquivos: {dados.files}")

print(f"🔢 Shape dos dados: {dados['data'].shape}")
print(f"🔢 Shape das classes: {dados['classes'].shape}")
print(f"🔢 Reshape das classes: {dados['classes'].reshape(-1, 1).shape}")

# Acessa diretamente os dados
X = dados['data'][0:1000, :]  # Exemplo: pega as primeiras 1000 amostras

y = dados['classes'].reshape(-1, 1)[0:1000]  # Exemplo: pega as primeiras 1000 classes

                                                                                                                                                                                                                                                                                                                                                                                          

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
