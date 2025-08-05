
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

# Define uma lista de api calls
print("Criando um dicionário de grupos de API Calls...")
apicalls = {}
for i, c in enumerate(colunas[idx_apicalls]):  
  nome_do_grupo = c.split(".")[0]  
  if nome_do_grupo not in apicalls:
    apicalls[nome_do_grupo] = [idx_apicalls[i]]
  else:
    apicalls[nome_do_grupo].append(idx_apicalls[i])

print(f"Total de API calls grupos únicos: {len(apicalls)}")

print("Criando uma estrutura para manter os valores de API Calls agrupados...")
X_apicalls = np.zeros((X.shape[0],len(apicalls)),dtype=np.int8)
print(X_apicalls.shape)

print("Iterando sobre os grupos e somando as colunas de cada grupo...")
for i, grupo in enumerate(apicalls):
    idxs_grupo = apicalls[grupo]           # pega os índices das colunas do grupo
    soma = np.sum(X[:, idxs_grupo], axis=1)     # soma horizontalmente por linha
    X_apicalls[:,i] = soma             # adiciona o vetor soma na lista

print("Removendo as colunas de API Calls originais...")
X = np.delete(X, idx_apicalls, axis=1)
colunas = np.delete(colunas, idx_apicalls)
print(X.shape)
print(colunas.shape)

print("Concatenando as novas colunas calculadas...")
X = np.concatenate((X, X_apicalls), axis=1)
colunas = np.concatenate((colunas, list(apicalls.keys())))
print(X.shape)
print(colunas.shape)

# === SIMILAR AO 

# Split the data into 70% training and 30% testing
print("Separando dados de treino(70%) e teste (30%)...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

print("Data split and saved successfully!")
print(f"Training data shape: {X_train.shape}, {y_train.shape}")
print(f"Testing data shape: {X_test.shape}, {y_test.shape}")

print("Salvando em CSV...");
CAMINHO_ARQUIVO_SAIDA_COLUNAS = os.path.join("..", "..", "dados", "colunas_apicalls_agrupadas.csv")

CAMINHO_ARQUIVO_SAIDA_DADOS_TREINO = os.path.join("..", "..", "dados", "dados_treino_apicalls_agrupadas.csv")
CAMINHO_ARQUIVO_SAIDA_CLASSES_TREINO = os.path.join("..", "..", "dados", "classes_treino_apicalls_agrupadas.csv")

CAMINHO_ARQUIVO_SAIDA_DADOS_TESTE = os.path.join("..", "..", "dados", "dados_teste_apicalls_agrupadas.csv")
CAMINHO_ARQUIVO_SAIDA_CLASSES_TESTE = os.path.join("..", "..", "dados", "classes_teste_apicalls_agrupadas.csv")

CAMINHO_ARQUIVO_SAIDA_TREINO_TESTE_COMPRIMIDO = os.path.join("..","..","dados","dados_treino_teste_apicalls_agrupadas.npz")

print(f"Salvando arquivo de colunas em {CAMINHO_ARQUIVO_SAIDA_COLUNAS}...")
np.save_to_csv(CAMINHO_ARQUIVO_SAIDA_COLUNAS, colunas, delimiter=',')

# Save the training data
print(f"Salvando arquivo de dados de treino em {CAMINHO_ARQUIVO_SAIDA_DADOS_TREINO}...")
np.save_to_csv(CAMINHO_ARQUIVO_SAIDA_DADOS_TREINO, X_train, delimiter=',')
print(f"Salvando arquivo de classes de treino em {CAMINHO_ARQUIVO_SAIDA_CLASSES_TREINO}...")
np.save_to_csv(CAMINHO_ARQUIVO_SAIDA_CLASSES_TREINO, y_train, delimiter=',')

# Save the testing data
print(f"Salvando arquivo de dados de treino em {CAMINHO_ARQUIVO_SAIDA_DADOS_TESTE}...")
np.save_to_csv(CAMINHO_ARQUIVO_SAIDA_DADOS_TESTE, X_test, delimiter=',')
print(f"Salvando arquivo de classes de treino em {CAMINHO_ARQUIVO_SAIDA_CLASSES_TESTE}...")
np.save_to_csv(CAMINHO_ARQUIVO_SAIDA_CLASSES_TESTE, y_test, delimiter=',')

print(f"Salvando arquivo todos os comprimidos de treino em {CAMINHO_ARQUIVO_SAIDA_TREINO_TESTE_COMPRIMIDO}...")
np.savez_compressed(
    CAMINHO_ARQUIVO_SAIDA_TREINO_TESTE_COMPRIMIDO,
    X_train=X_train,
    X_test=X_test,
    y_train=y_train,
    y_test=y_test,
    column_names=colunas
 )

print("Finalizado!")