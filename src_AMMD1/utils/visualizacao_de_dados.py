
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import random
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

print("Definindo caminho de entrada...")
CAMINHO_ARQUIVO_ENTRADA = os.path.join("..", "..", "dados", "amex-1M-[intents-permissions-opcodes-apicalls].npz")
print(f"Arquivo de entrada: {CAMINHO_ARQUIVO_ENTRADA}")

print("Abrindo arquivo...")
dados = np.load(CAMINHO_ARQUIVO_ENTRADA, allow_pickle=True, mmap_mode='r')
print(dados.files)

print("Acessando dados do arquivo...")
X = dados['data']
y = dados['classes']
colunas = dados['column_names']

print(f"X: {X.shape}")
print(f"y: {y.shape}")
print(f"colunas: {colunas.shape}")

print("Definindo localização das amostras de aplicativos benignos e de malwares...")
idx_benignos = np.where(y==0)[0]
idx_malwares = np.where(y==1)[0]
print(len(idx_benignos), len(idx_malwares))

print("Definindo os índices de cada grupo de features...")
idx_permissions = [i for i, nome in enumerate(colunas) if nome.startswith("permissions::")]
idx_intents = [i for i, nome in enumerate(colunas) if nome.startswith("intents::")]
idx_opcodes = [i for i, nome in enumerate(colunas) if nome.startswith("opcodes::")]
idx_apicalls = [i for i, nome in enumerate(colunas) if nome.startswith("apicalls::")]
print("Índices de grupos de features definidos")

# === ATÉ A LINHA 37 AS INSTRUÇÕES SÃO AS MESMAS DO SCRIPT prepara_base_separa_treino_teste.py

# Calcular a quantidade de malwares e benignos dentro de blocos sequênciais com
# o objetivo de observar a frequência de distribuição dos malwares dentre os
# benignos em gráfico de barra

# Define o tamanho do chunk
chunk_size = 1000

# Calcula o número de chunks
num_chunks = len(y) // chunk_size + (len(y) % chunk_size != 0)

# Inicializa listas para armazenar as contagens de benignos e malwares por chunk
contagem_benignos = []
contagem_malwares = []

# Itera sobre os chunks
for i in range(num_chunks):
    # Define o limite inicial e final de cada chunks
    start = i * chunk_size
    end = min((i + 1) * chunk_size, len(y))
    chunk_y = y[start:end]

    # Conta o número de benignos (0) e malwares (1) no chunk
    benignos_no_chunk = np.sum(chunk_y == 0)
    malwares_no_chunk = np.sum(chunk_y == 1)

    # Adiciona em listas as quantidades de benignos e malwares para cada chunk
    contagem_benignos.append(benignos_no_chunk)
    contagem_malwares.append(malwares_no_chunk)

# Cria o histograma
plt.figure(figsize=(10, 5))
plt.bar(range(num_chunks), contagem_benignos, label='Benignos', alpha=0.7, color="green")
plt.bar(range(num_chunks), contagem_malwares, label='Malwares', alpha=0.7, color="red", bottom=contagem_benignos)

# Configurações do gráfico
plt.xlabel('Chunk')
plt.ylabel('Número de Amostras')
plt.title(f'Distribuição de Benignos e Malwares por Chunk ({chunk_size} amostras)')
plt.legend()
plt.grid(axis='y', alpha=0.75)
plt.show()

# === FIM DO HISTOGRAMA DE DISTRIBUIÇÃO DE BENIGNOS E MALWARES POR BLOCO DE 1000 AMOSTRAS

# Define o tamanho do chunk
chunk_size = 1000

# Calcula a quantidade de uso de features em usados por instâncias em blocos
# sequenciais para observar se existe uma concentração de uso de recursos.

# Calcula o número de chunks
num_chunks = X.shape[1] // chunk_size + (X.shape[1] % chunk_size != 0)

# Inicializa listas para armazenar as contagens de benignos e malwares por chunk
contagem = []

# Itera sobre os chunks
for i in range(num_chunks):
    start = i * chunk_size
    end = min((i + 1) * chunk_size, X.shape[1])

    # Conta o número de features usadas em grupos de features
    contagem.append(np.sum(X[:,start:end], axis=0).sum())


# Cria o gráfico de linha
plt.figure(figsize=(12, 6))
plt.plot(range(num_chunks), contagem, label='Distribuição de Uso de Features')

# Configurações do gráfico
plt.xlabel('Chunk')
plt.ylabel('Número de Features')
plt.title(f'Distribuição de Features por Chunk ({chunk_size} features)')
plt.legend()
plt.grid(axis='y', alpha=0.75)
plt.show()

# === FIM DE VISUALIZAÇÃO DE DISTRIBUIÇÃO DE FEATURES

# Calcula a quantidade de uso de features em usados por instâncias em blocos
# sequenciais para observar se existe uma concentração de uso de recursos
# por benignos.

# Calcula o número de chunks
num_chunks = X.shape[1] // chunk_size + (X.shape[1] % chunk_size != 0)

# Inicializa listas para armazenar as contagens de benignos e malwares por chunk
contagem_benignos = []
contagem_malwares = []

# Itera sobre os chunks
for i in range(num_chunks):
    start = i * chunk_size
    end = min((i + 1) * chunk_size, X.shape[1])

    # Conta o número de features usadas em grupos de features
    contagem_benignos.append(np.sum(X[idx_benignos,start:end], axis=0).sum())
    contagem_malwares.append(np.sum(X[idx_malwares,start:end], axis=0).sum())

# Cria o gráfico de linha
plt.figure(figsize=(12, 6))
plt.plot(range(num_chunks), contagem_benignos, label='Benignos', color = 'green')
plt.plot(range(num_chunks), contagem_malwares, label='Malwares', color = 'red')
# Configurações do gráfico
plt.xlabel('Chunk')
plt.ylabel('Número de Features')
plt.title(f'Distribuição de Features por Chunk ({chunk_size} features)')
plt.legend()
plt.grid(axis='y', alpha=0.75)
plt.show()

# === VISUALIZAÇÃO DE PERMISSIONS

# Define o tamanho do chunk
chunk_size = 10

# Calcula a quantidade de uso de features permissions (por blocos) usados por instâncias
# para observar se existe uma concentração de uso de recursos por benignos

# Calcula o número de chunks
num_chunks = len(idx_permissions) // chunk_size + (len(idx_permissions) % chunk_size != 0)

# Inicializa listas para armazenar as contagens de benignos e malwares por chunk
contagem = []

# Itera sobre os chunks
for i in range(num_chunks):
    start = i * chunk_size
    end = min((i + 1) * chunk_size, len(idx_permissions))

    # Conta o número de features usadas em grupos de features
    contagem.append(np.sum(X[:,start:end], axis=0).sum())

# Cria o gráfico de linha
plt.figure(figsize=(12, 6))
plt.plot(range(num_chunks), contagem, label='Geral', color = 'blue')
# Configurações do gráfico
plt.xlabel('Chunk')
plt.ylabel('Número de permissions')
plt.title(f'Distribuição de Features por Chunk ({chunk_size} permissions)')
plt.legend()
plt.grid(axis='y', alpha=0.75)
plt.show()

# === VISUALIZAÇÃO DE PERMISSIONS EM CLASSES BENIGNOS E MALWARES

# Calcula a quantidade de uso de features permissions (por blocos) usados por instâncias
# para observar se existe uma concentração de uso de recursos por benignos

# Calcula o número de chunks
num_chunks = len(idx_permissions) // chunk_size + (len(idx_permissions) % chunk_size != 0)

# Inicializa listas para armazenar as contagens de benignos e malwares por chunk
contagem_benignos = []
contagem_malwares = []

# Itera sobre os chunks
for i in range(num_chunks):
    start = i * chunk_size
    end = min((i + 1) * chunk_size, len(idx_permissions))

    # Conta o número de features usadas em grupos de features
    contagem_benignos.append(np.sum(X[idx_benignos,start:end], axis=0).sum())
    contagem_malwares.append(np.sum(X[idx_malwares,start:end], axis=0).sum())

max_contagem_benignos = np.max(contagem_benignos).max()
max_contagem_malwares = np.max(contagem_malwares).max()

norm_contagem_benignos = contagem_benignos/max_contagem_benignos + 1
norm_contagem_malwares = contagem_malwares/max_contagem_malwares - 1

# Cria o gráfico de linha
plt.figure(figsize=(12, 6))
plt.plot(range(num_chunks), norm_contagem_benignos, label='Benignos', color='green')
plt.plot(range(num_chunks), norm_contagem_malwares, label='Malwares', color='red')
# Configurações do gráfico
plt.xlabel('Chunk')
plt.ylabel('Número de permissions')
plt.title(f'Distribuição de Features por Chunk ({chunk_size} permissions)')
plt.legend()
plt.grid(axis='y', alpha=0.75)
plt.show()

# === INTENTS

# Define o tamanho do chunk
chunk_size = 10

# Calcula a quantidade de uso de features intens usados por instâncias
# para observar se existe uma concentração de uso de recursos por benignos

# Calcula o número de chunks
num_chunks = len(idx_intents) // chunk_size + (len(idx_intents) % chunk_size != 0)

# Inicializa listas para armazenar as contagens de benignos e malwares por chunk
contagem = []

# Itera sobre os chunks
for i in range(num_chunks):
    start = i * chunk_size
    end = min((i + 1) * chunk_size, len(idx_intents))

    # Conta o número de features usadas em grupos de features
    contagem.append(np.sum(X[:,start:end], axis=0).sum())

# Cria o gráfico de linha
plt.figure(figsize=(12, 6))
plt.plot(range(num_chunks), contagem, label='Geral', color = 'blue')
# Configurações do gráfico
plt.xlabel('Chunk')
plt.ylabel('Número de intents')
plt.title(f'Distribuição de Features por Chunk ({chunk_size} intents)')
plt.legend()
plt.grid(axis='y', alpha=0.75)
plt.show()


# Calcula a quantidade de uso de features permissions usados por instâncias
# para observar se existe uma concentração de uso de recursos por benignos

# Calcula o número de chunks
num_chunks = len(idx_intents) // chunk_size + (len(idx_intents) % chunk_size != 0)

# Inicializa listas para armazenar as contagens de benignos e malwares por chunk
contagem_benignos = []
contagem_malwares = []

# Itera sobre os chunks
for i in range(num_chunks):
    start = i * chunk_size
    end = min((i + 1) * chunk_size, len(idx_intents))

    # Conta o número de features usadas em grupos de features
    contagem_benignos.append(np.sum(X[idx_benignos,start:end], axis=0).sum())
    contagem_malwares.append(np.sum(X[idx_malwares,start:end], axis=0).sum())

max_contagem_benignos = np.max(contagem_benignos).max()
max_contagem_malwares = np.max(contagem_malwares).max()

norm_contagem_benignos = contagem_benignos/max_contagem_benignos + 1
norm_contagem_malwares = contagem_malwares/max_contagem_malwares - 1

# Cria o gráfico de linha
plt.figure(figsize=(12, 6))
plt.plot(range(num_chunks), norm_contagem_benignos, label='Benignos', color = 'green')
plt.plot(range(num_chunks), norm_contagem_malwares, label='Malwares', color = 'red')
# Configurações do gráfico
plt.xlabel('Chunk')
plt.ylabel('Número de intents')
plt.title(f'Distribuição de Features por Chunk ({chunk_size} intents)')
plt.legend()
plt.grid(axis='y', alpha=0.75)
plt.show()

# === OPCODES

# Define o tamanho do chunk
chunk_size = 10

# Calcula a quantidade de uso de features intens usados por instâncias
# para observar se existe uma concentração de uso de recursos por benignos

# Calcula o número de chunks
num_chunks = len(idx_opcodes) // chunk_size + (len(idx_opcodes) % chunk_size != 0)

# Inicializa listas para armazenar as contagens de benignos e malwares por chunk
contagem = []

# Itera sobre os chunks
for i in range(num_chunks):
    start = i * chunk_size
    end = min((i + 1) * chunk_size, len(idx_opcodes))

    # Conta o número de features usadas em grupos de features
    contagem.append(np.sum(X[:,start:end], axis=0).sum())

# Cria o gráfico de linha
plt.figure(figsize=(12, 6))
plt.plot(range(num_chunks), contagem, label='Geral', color = 'blue')
# Configurações do gráfico
plt.xlabel('Chunk')
plt.ylabel('Número de opcodes')
plt.title(f'Distribuição de Features por Chunk ({chunk_size} opcodes)')
plt.legend()
plt.grid(axis='y', alpha=0.75)
plt.show()

# DIVISAO EM CLASSES

# Calcula a quantidade de uso de features permissions usados por instâncias
# para observar se existe uma concentração de uso de recursos por benignos

# Calcula o número de chunks
num_chunks = len(idx_opcodes) // chunk_size + (len(idx_opcodes) % chunk_size != 0)

# Inicializa listas para armazenar as contagens de benignos e malwares por chunk
contagem_benignos = []
contagem_malwares = []

# Itera sobre os chunks
for i in range(num_chunks):
    start = i * chunk_size
    end = min((i + 1) * chunk_size, len(idx_opcodes))

    # Conta o número de features usadas em grupos de features
    contagem_benignos.append(np.sum(X[idx_benignos,start:end], axis=0).sum())
    contagem_malwares.append(np.sum(X[idx_malwares,start:end], axis=0).sum())

max_contagem_benignos = np.max(contagem_benignos).max()
max_contagem_malwares = np.max(contagem_malwares).max()

norm_contagem_benignos = contagem_benignos/max_contagem_benignos + 1
norm_contagem_malwares = contagem_malwares/max_contagem_malwares - 1

# Cria o gráfico de linha
plt.figure(figsize=(12, 6))
plt.plot(range(num_chunks), norm_contagem_benignos, label='Benignos', color = 'green')
plt.plot(range(num_chunks), norm_contagem_malwares, label='Malwares', color = 'red')
# Configurações do gráfico
plt.xlabel('Chunk')
plt.ylabel('Número de opcodes')
plt.title(f'Distribuição de Features por Chunk ({chunk_size} opcodes)')
plt.legend()
plt.grid(axis='y', alpha=0.75)
plt.show()

# API CALLS

# Define o tamanho do chunk
chunk_size = 100

# Calcula a quantidade de uso de features intens usados por instâncias
# para observar se existe uma concentração de uso de recursos por benignos

# Calcula o número de chunks
num_chunks = len(idx_apicalls) // chunk_size + (len(idx_apicalls) % chunk_size != 0)

# Inicializa listas para armazenar as contagens de benignos e malwares por chunk
contagem = []

# Itera sobre os chunks
for i in range(num_chunks):
    start = i * chunk_size
    end = min((i + 1) * chunk_size, len(idx_apicalls))

    # Conta o número de features usadas em grupos de features
    contagem.append(np.sum(X[:,start:end], axis=0).sum())

# Cria o gráfico de linha
plt.figure(figsize=(12, 6))
plt.plot(range(num_chunks), contagem, label='Geral', color = 'blue')
# Configurações do gráfico
plt.xlabel('Chunk')
plt.ylabel('Número de apicalls')
plt.title(f'Distribuição de Features por Chunk ({chunk_size} apicalls)')
plt.legend()
plt.grid(axis='y', alpha=0.75)
plt.show()

# Calcula a quantidade de uso de features permissions usados por instâncias
# para observar se existe uma concentração de uso de recursos por benignos

# Calcula o número de chunks
num_chunks = len(idx_apicalls) // chunk_size + (len(idx_apicalls) % chunk_size != 0)

# Inicializa listas para armazenar as contagens de benignos e malwares por chunk
contagem_benignos = []
contagem_malwares = []

# Itera sobre os chunks
for i in range(num_chunks):
    start = i * chunk_size
    end = min((i + 1) * chunk_size, len(idx_apicalls))

    # Conta o número de features usadas em grupos de features
    contagem_benignos.append(np.sum(X[idx_benignos,start:end], axis=0).sum())
    contagem_malwares.append(np.sum(X[idx_malwares,start:end], axis=0).sum())

max_contagem_benignos = np.max(contagem_benignos).max()
max_contagem_malwares = np.max(contagem_malwares).max()

norm_contagem_benignos = contagem_benignos/max_contagem_benignos# + 1
norm_contagem_malwares = contagem_malwares/max_contagem_malwares# - 1

# Cria o gráfico de linha
plt.figure(figsize=(12, 6))
plt.plot(range(num_chunks), norm_contagem_benignos, label='Benignos', color = 'green')
plt.plot(range(num_chunks), norm_contagem_malwares, label='Malwares', color = 'red')
# Configurações do gráfico
plt.xlabel('Chunk')
plt.ylabel('Número de apicalls')
plt.title(f'Distribuição de Features por Chunk ({chunk_size} apicalls)')
plt.legend()
plt.grid(axis='y', alpha=0.75)
plt.show()