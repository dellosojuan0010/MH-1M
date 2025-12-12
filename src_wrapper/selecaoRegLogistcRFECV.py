
import os
import gc
from datetime import datetime
from math import trunc

import numpy as np
import pandas as pd

from sklearn.datasets import make_classification
from sklearn.feature_selection import RFECV

import matplotlib.pyplot as plt

#from tqdm.notebook import tqdm

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif, chi2

from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression




# Caminho para o arquivo compactado
CAMINHO_ARQUIVO = '../dados/mh1m_balanceados_reduzidos_shap.npz'

# Carrega os dados com mmap_mode para uso mais leve de memória
dados = np.load(CAMINHO_ARQUIVO, allow_pickle=True)

# Extração dos arrays principais
#X = dados['data']
#y = dados['classes']
colunas = dados['column_names']
#print(f"Dados: X={X.shape}, y={y.shape}, colunas={colunas.shape}")

qtdIntents = 0
qtdPermissions = 0
qtdOpcodes = 0
qtdApicalls = 0
for c in colunas:
  if c.startswith("intents::"):
    qtdIntents += 1
  elif c.startswith("permissions::"):
    qtdPermissions += 1
  elif c.startswith("opcodes::"):
    qtdOpcodes += 1
  else:
    qtdApicalls += 1
print("Intents: ", qtdIntents)
print("Permissions", qtdPermissions)
print("Opcodes", qtdOpcodes)
print("Apicalls", qtdApicalls)


# Caminho para o arquivo compactado
CAMINHO_ARQUIVO = '../dados/mh1m_balanceadas.npz'

# Carrega os dados com mmap_mode para uso mais leve de memória
dados = np.load(CAMINHO_ARQUIVO, allow_pickle=True)

# Extração dos arrays principais
X = dados['data']
y = dados['classes']
colunas = dados['column_names']


grupos = ["intents", "permissions", "opcodes", "apicalls", "permissions_opcodes", "todas"]
qtde_features = [qtdIntents, qtdPermissions, qtdOpcodes, qtdApicalls, qtdPermissions + qtdOpcodes, qtdIntents + qtdPermissions + qtdOpcodes + qtdApicalls]
k = 0
for nome_grupo in grupos:
    print(f"\n\n--- Processando grupo: {nome_grupo} ---\n")

    if nome_grupo == "permissions_opcodes":
        idx_permissions = [i for i, nome in enumerate(colunas) if nome.startswith("permissions::")]
        idx_opcodes = [i for i, nome in enumerate(colunas) if nome.startswith("opcodes::")]
        idx_features = idx_permissions + idx_opcodes
        df = pd.DataFrame(X[:, idx_features], columns=np.array(colunas)[idx_features])
        df['classe'] = y
    elif nome_grupo == "todas":
        df = pd.DataFrame(X, columns=np.array(colunas))
        df['classe'] = y
    else:
        idx_features = [i for i, nome in enumerate(colunas) if nome.startswith(f"{nome_grupo}::")]
        df = pd.DataFrame(X[:, idx_features], columns=np.array(colunas)[idx_features])
        df['classe'] = y


    print("DataFrames criados:")
    print(f" - df : {df.shape}")

    X_ = df.drop(columns=['classe']).values.astype(np.int8)
    y_ = df['classe'].values.astype(np.int8)
    feature_cols = df.drop(columns=['classe']).columns
    # Modelo base
    estimator = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        n_jobs=-1,       # paralelização dentro da RandomForest
        random_state=42
    )

    rfe = RFE(
        estimator=estimator,
        n_features_to_select=qtde_features[k],
        verbose=1
    )

    X_selected = rfe.fit_transform(X_, y_)

    mask = rfe.get_support()
    selected_columns = np.array(feature_cols)[mask]

    print("Shape original:", X_.shape)
    print("Shape reduzido :", X_selected.shape)
    print("Colunas selecionadas:", selected_columns)

    # usar o estimator treinado dentro do RFE
    score = rfe.estimator_.score(X_selected, y_)
    print("Score:", score)

    df_selected = pd.DataFrame({
        "feature": selected_columns,
        "pontuacao": score
    })

    # Criar um nome de arquivo organizado por grupo e timestamp
    nome_arquivo = f"features_selecionadas_{nome_grupo}.csv"

    # Salvar em CSV
    df_selected.to_csv(nome_arquivo, index=False, encoding="utf-8")

    print(f"Features selecionadas salvas em: {nome_arquivo}")