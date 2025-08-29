# === Importação das bibliotecas

#from google.colab import drive

import os
import gc
import datetime


import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, losses

import matplotlib.pyplot as plt

from tqdm.notebook import tqdm

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from xgboost import XGBClassifier

# === Parte 1 - Abertura do arquivo de dados, recuperação e embaralhamento

# Acesso ao drive pessoal
#drive.mount('/content/drive')

# Caminho para o arquivo compactado
#CAMINHO_ARQUIVO = '/content/drive/MyDrive/dados_MH1M/dados_undersampling_duplicados_eliminados_para_autoencoder.npz'
CAMINHO_ARQUIVO = os.path.join("..", "dados", "dados_undersampling_duplicados_eliminados_para_autoencoder.npz")
# Carrega os dados com mmap_mode para uso mais leve de memória
dados = np.load(CAMINHO_ARQUIVO, allow_pickle=True, mmap_mode='r')

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

# === Parte 2 e 3 - Separar as colunas das features e criar os dataframes
# Identificar colunas por namespace
idx_permissions = [i for i, nome in enumerate(colunas) if nome.startswith("permissions::")]
idx_intents     = [i for i, nome in enumerate(colunas) if nome.startswith("intents::")]
idx_opcodes     = [i for i, nome in enumerate(colunas) if nome.startswith("opcodes::")]
idx_apicalls    = [i for i, nome in enumerate(colunas) if nome.startswith("apicalls::")]

# Criação dos DataFrames
df_all  = pd.DataFrame(X, columns=colunas)
df_all['classe'] = y

df_p    = pd.DataFrame(X[:, idx_permissions], columns=np.array(colunas)[idx_permissions])
df_p['classe'] = y

df_i    = pd.DataFrame(X[:, idx_intents], columns=np.array(colunas)[idx_intents])
df_i['classe'] = y

df_op   = pd.DataFrame(X[:, idx_opcodes], columns=np.array(colunas)[idx_opcodes])
df_op['classe'] = y

df_api  = pd.DataFrame(X[:, idx_apicalls], columns=np.array(colunas)[idx_apicalls])
df_api['classe'] = y

print("DataFrames criados:")
print(f" - df_all: {df_all.shape}")
print(f" - df_p  : {df_p.shape}")
print(f" - df_i  : {df_i.shape}")
print(f" - df_op : {df_op.shape}")
print(f" - df_api: {df_api.shape}")

# === Parte 4 - 
def avaliar_modelos_em_dataframe(df, nome_grupo, n_splits=5):
    X = df.drop(columns=['classe']).values
    y = df['classe'].values

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    modelos = {
        'MLP': MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, random_state=42, verbose=False),
        'SVM': SVC(kernel='rbf', probability=True, random_state=42, verbose=False),
        'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
        'XGBoost': XGBClassifier(n_estimators=100, verbosity=0, use_label_encoder=False, random_state=42)
    }

    resultados = []

    for modelo_nome, modelo in tqdm(modelos.items(), desc=f"[{nome_grupo}] Modelos"):
        for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
            print(f"Treinando {modelo_nome} - Fold {fold}")
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            if modelo_nome in ['MLP', 'SVM']:
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)

            if modelo_nome == 'MLP':
                modelo.set_params(verbose=True)
            elif modelo_nome == 'SVM':
                modelo.set_params(verbose=True)
            elif modelo_nome == 'XGBoost':
                modelo.set_params(verbosity=1)

            modelo.fit(X_train, y_train)
            y_pred = modelo.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

            for classe in ['0', '1']:
                resultados.append({
                    'grupo_de_features': nome_grupo,
                    'modelo': modelo_nome,
                    'classe': classe,
                    'precision': report[classe]['precision'],
                    'recall': report[classe]['recall'],
                    'f1_score': report[classe]['f1-score'],
                    'support': report[classe]['support'],
                    'fold': fold,
                    'accuracy_geral': acc
                })

    return pd.DataFrame(resultados)

# === Parte 5 - Executar o modelo e recuperar os resultados para cada Dataframe
# 5. Executar para todos os grupos
df_resultados = pd.concat([
    avaliar_modelos_em_dataframe(df_all, 'all'),
    avaliar_modelos_em_dataframe(df_p, 'permissions'),
    avaliar_modelos_em_dataframe(df_i, 'intents'),
    avaliar_modelos_em_dataframe(df_op, 'opcodes'),
    avaliar_modelos_em_dataframe(df_api, 'apicalls')
], ignore_index=True)

# === Parte 6 - Exportar e printar os resultados resumidos
caminho_saida = '/content/drive/MyDrive/dados_MH1M/resultados_modelos.csv'
df_resultados.to_csv(caminho_saida, index=False)

print("\nResumo final (médias por grupo, modelo e classe):")
print(df_resultados.groupby(['grupo_de_features', 'modelo', 'classe'])[['precision', 'recall', 'f1_score']].mean().round(4))