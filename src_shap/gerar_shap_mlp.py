from google.colab import drive

import os
import gc
from datetime import datetime

import numpy as np
import pandas as pd
from math import trunc

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, losses
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LeakyReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

import matplotlib.pyplot as plt

from tqdm.notebook import tqdm

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

import shap

from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, chi2

# Função de avaliação de valores SHAP
def cria_valores_shap(df, modelo_nome, nome_grupo, threshold, n_splits=5):

    X = df.drop(columns=['classe']).values.astype(np.int8)
    y = df['classe'].astype(np.int8).values
    input_dim = X.shape[1]

    print(f"\nIniciando avaliação para o grupo de features: '{nome_grupo}' com {X.shape[1]} atributos e {X.shape[0]} instâncias.")
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    folds = list(skf.split(X,y))

    for fold in range(n_splits):

        train_idx, test_idx = folds[fold]
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        print(f"\nCarregar o modelo MLP Keras.")

        CAMINHO_PASTA_MODELO = f'./{nome_grupo}/{modelo_nome}_limiar{trunc(threshold*10)}/resultados/'
        CAMINHO_MODELO = os.path.join(CAMINHO_PASTA_MODELO,f'{nome_grupo}__{modelo_nome}__fold{fold+1}.keras')
        # caminho_modelo = os.path.join(pasta_entrada_modelo, f"modelo_mlp_{nome_grupo}_{fold+1}.keras")
        modelo = tf.keras.models.load_model(CAMINHO_MODELO)

        min_max_evals = 2 * X.shape[1] + 1
        # masker = shap.maskers.Independent(X_train)  # 50–200 costuma ser ok
        # explainer = shap.Explainer(modelo, masker)

        # shap_values = explainer(X_test[0,:], max_evals=min_max_evals)

        explainer = shap.Explainer(modelo, X_train)

        shap_values = explainer(X_test,max_evals=min_max_evals)


        shap_exp = shap_values
        shap_mat = shap_exp.values.astype(np.float32)            # (n, d)
        base_values = np.array(shap_exp.base_values).astype(np.float32)  # (n,)
        feature_names = np.array([c for c in df.columns if c != 'classe'])
        X_explicado = shap_exp.data.astype(np.float32)           # (n, d)


        # salva pacote completo .npz
        CAMINHO_SAIDA = f'./{nome_grupo}/{modelo_nome}_limiar{trunc(threshold*10)}/'
        os.makedirs(CAMINHO_SAIDA, exist_ok=True)
        saida_npz = os.path.join(CAMINHO_SAIDA,f"{nome_grupo}__{modelo_nome}__fold{fold+1}_SHAP.npz")

        np.savez_compressed(
            saida_npz,
            shap_values=shap_mat,
            base_values=base_values,
            feature_names=feature_names,
            X_explicado=X_explicado,
            test_index=test_idx.astype(np.int64),
            y_test=y_test.astype(np.int32)
        )
        del shap_values, shap_exp, shap_mat, base_values, feature_names, X_explicado
        del explainer
        gc.collect()


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

modelos = ["mlp"]
threshold = 0.8
grupos = ["apicalls"]

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

        cria_valores_shap(df, modelo_nome, nome_grupo, threshold=threshold, n_splits=5)
        del df
        gc.collect()

