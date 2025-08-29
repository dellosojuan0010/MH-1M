# === APENAS LOCALIZA FEATURES QUE PODERIA MSER CONSIDERADAS INUTEIS
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

# === O QUE SÃO DEFINIDAS FEATURES INUTEIS??? --- falta implementar isso ainda.