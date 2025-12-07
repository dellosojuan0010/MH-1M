# Parte 1 - Importação das bibliotecas

import os
import gc
from datetime import datetime

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import shap
from tqdm import tqdm

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier

import joblib

# Função de avaliação de valores SHAP
def cria_valores_shap(df, modelo_nome, nome_grupo, n_splits=5):

    X = df.drop(columns=['classe']).values.astype(np.float32)
    y = df['classe'].astype(np.int32).values  # garante inteiros 0/1
    input_dim = X.shape[1]

    print(f"\nIniciando avaliação para o grupo de features: '{nome_grupo}' com {X.shape[1]} atributos e {X.shape[0]} instâncias.")
    
    print(f"\nModelo: {modelo_nome} - Total de folds: {n_splits}")

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    folds = list(skf.split(X, y))

    for fold in range(n_splits):
        if fold != 2:
            continue
        train_idx, test_idx = folds[fold]
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        CAMINHO_PASTA_MODELO = f'../src_inicial/{nome_grupo}/{modelo_nome}/resultados/'
        CAMINHO_MODELO = os.path.join(CAMINHO_PASTA_MODELO,f'{nome_grupo}__{modelo_nome}__fold{fold+1}.joblib')
        modelo = joblib.load(CAMINHO_MODELO)

        print(f"Criando SHAP para o modelo {modelo_nome} - Fold {fold+1}")    
        explainer = shap.TreeExplainer(modelo, data=X_train, model_output='probability')

        shap_values = explainer(X_test)

        shap_exp = shap_values  # só para semântica
        shap_mat = shap_exp.values.astype(np.float32)             # (n, d)
        base_values = np.array(shap_exp.base_values).astype(np.float32)  # (n,)
        feature_names = np.array([c for c in df.columns if c != 'classe'])
        X_explicado = shap_exp.data.astype(np.float32)            # (n, d)

        # salva tudo em um único arquivo compacto
        CAMINHO_SAIDA = f'./{nome_grupo}/{modelo_nome}/'
        os.makedirs(CAMINHO_SAIDA, exist_ok=True)
        saida_npz = os.path.join(CAMINHO_SAIDA,f'{nome_grupo}__{modelo_nome}__fold{fold+1}_SHAP.npz')
        
        np.savez_compressed(
            saida_npz,
            shap_values=shap_mat,
            base_values=base_values,
            feature_names=feature_names,
            X_explicado=X_explicado,
            test_index=test_idx.astype(np.int64),
            y_test=y_test.astype(np.int32)
        )


# Parte 2 - Abertura do arquivo, recuperação dos dados e embaralhamento
CAMINHO_ARQUIVO = "../dados/mh1m_balanceadas.npz"

# Carrega os dados com mmap_mode para uso mais leve de memória
dados = np.load(CAMINHO_ARQUIVO, allow_pickle=True)

# Extração dos arrays principais
X = dados['data']
y = dados['classes']
colunas = dados['column_names']

# Embaralhar X e y
rng = np.random.default_rng(42)  # garante reprodutibilidade
idx_final = rng.permutation(X.shape[0])  # embaralha os índices

X = X[idx_final]
y = y[idx_final]

print(f"Dados embaralhados: X={X.shape}, y={y.shape}")

# modelos = ["RandomForest", "XGBoost"]
modelos = ["XGBoost"]
grupos = ["intents", "permissions", "opcodes", "apicalls"]
# grupos = ["opcodes", "apicalls"]

for modelo_nome in modelos:
    for nome_grupo in grupos:
        # modelo_nome = "RandomForest"
        # Parte 3 - Separar as colunas das features e criar os DataFrames
        # Identificar colunas por namespace
        # nome_grupo = "intents"
        idx_features = [i for i, nome in enumerate(colunas) if nome.startswith(f"{nome_grupo}::")]

        df = pd.DataFrame(X[:, idx_features], columns=np.array(colunas)[idx_features])
        df['classe'] = y

        print("DataFrames criados:")
        print(f" - df : {df.shape}")

        cria_valores_shap(df, modelo_nome, nome_grupo, n_splits=5)
        del df
        gc.collect()

