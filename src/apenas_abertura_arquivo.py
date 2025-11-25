import os
import gc
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm

import csv

# Caminho para o arquivo compactado
# CAMINHO_ARQUIVO = '../dados/amex-1M-[intents-permissions-opcodes-apicalls].npz'
CAMINHO_ARQUIVO = '../dados/sem_duplicidades_amostras_features_repetidas_removidas.npz'
print("Arquivo: ", CAMINHO_ARQUIVO)

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

benignos = len(np.where(y==0)[0])
malwares = len(np.where(y==1)[0])

print(f"Quantidade de Benignos: {benignos}")
print(f"Quantidade de Malwares: {malwares}")