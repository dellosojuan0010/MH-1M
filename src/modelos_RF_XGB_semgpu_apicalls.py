###
#
# ESSE MODELO SERIA PARA USAR TENSORFLOW, A GPU DO DESKTOP DO LAB SÓ RODOU COM PYTORCH
#
###


# Parte 1 - Importação das bibliotecas

#from google.colab import drive

import os
import gc
from datetime import datetime

import numpy as np
import pandas as pd

# import tensorflow as tf
# from tensorflow.keras import layers, models, optimizers, losses
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Dropout, LeakyReLU
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.callbacks import EarlyStopping

import matplotlib.pyplot as plt

from tqdm import tqdm

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier
#from sklearn.neural_network import MLPClassifier
#from sklearn.svm import SVC

from xgboost import XGBClassifier


# Parte 2 - Abertura do arquivo, recuperacão dos dados e embaralhamento

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

# Parte 3 - Separar as colunas das features e criar os DataFrames

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


# Parte 4 - Criação da função para MLP Keras
# def definir_mlp_keras(input_dim):
#     if input_dim >= 23000:
#         units = [8192, 4096, 2048, 1024, 512]
#     elif input_dim > 10000:
#         units = [4096, 2048, 1024, 512]
#     elif input_dim > 1000:
#         units = [1024, 512, 256]
#     else:
#         units = [256, 128]

#     model = Sequential()
#     model.add(Dense(units[0], input_shape=(input_dim,)))
#     model.add(LeakyReLU(alpha=0.01))
#     for u in units[1:]:
#         model.add(Dense(u))
#         model.add(LeakyReLU(alpha=0.01))
#     model.add(Dense(1, activation='sigmoid'))

#     model.compile(optimizer=Adam(0.001), loss='binary_crossentropy', metrics=['accuracy'])
#     return model


# Parte 5 - Função para definir modelos adaptados

# 5. Função para definir modelos adaptados apenas que usaram o sklearn
def definir_modelos_sklearn(input_dim):
    if input_dim > 20000:
        #svm = SVC(kernel='rbf', C=0.5, gamma='scale', probability=True, random_state=42, verbose=True)
        rf = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42, n_jobs=-1)
        xgb = XGBClassifier(n_estimators=200, max_depth=10, learning_rate=0.05, verbosity=1, use_label_encoder=False, random_state=42, n_jobs=-1)
    elif input_dim > 400:
        #svm = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42, verbose=True)
        rf = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
        xgb = XGBClassifier(n_estimators=150, max_depth=8, learning_rate=0.07, verbosity=1, use_label_encoder=False, random_state=42, n_jobs=-1)
    else:
        #svm = SVC(kernel='rbf', C=2.0, gamma='auto', probability=True, random_state=42, verbose=True)
        rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
        xgb = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, verbosity=1, use_label_encoder=False, random_state=42, n_jobs=-1)

    #return {'SVM': svm, 'RandomForest': rf, 'XGBoost': xgb}
    return {'RandomForest': rf, 'XGBoost': xgb}



# Parte 6 - Definir função de avaliação

def avaliar_modelos_em_dataframe(df, nome_grupo, n_splits=5):

    X = df.drop(columns=['classe']).values.astype(np.float32)
    y = df['classe'].astype(np.float32).values
    input_dim = X.shape[1]

    print(f"\nIniciando avaliação para o grupo de features: '{nome_grupo}' com {X.shape[1]} atributos e {X.shape[0]} instâncias.")

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    print(f"\nDefinindo os modelo RandomForest e XGBoost.")
    modelos = definir_modelos_sklearn(input_dim)
    resultados = []

    for modelo_nome, modelo in tqdm(modelos.items(), desc=f"[{nome_grupo}] Modelos sklearn"):

        print(f"\nModelo: {modelo_nome} - Total de folds: {n_splits}")

        for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
            print(f"Treinando {modelo_nome} - Fold {fold}")
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # if modelo_nome == 'SVM':
            #     scaler = StandardScaler()
            #     X_train = scaler.fit_transform(X_train)
            #     X_test = scaler.transform(X_test)

            modelo.fit(X_train, y_train)
            y_pred = modelo.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            report = classification_report(y_test.astype(int), y_pred.astype(int), output_dict=True, zero_division=0)
            print(report)

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
    print(f"Finalizado modelo {modelo_nome} para '{nome_grupo}'")

    # # MLP Keras
    # print(f"\nIniciando MLP (Keras) para o grupo: {nome_grupo} com arquitetura dinâmica.")

    # for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
    #     print(f"Treinando MLP (Keras) - Fold {fold}")
    #     X_train, X_test = X[train_idx], X[test_idx]
    #     y_train, y_test = y[train_idx], y[test_idx]

    #     print(f"\nDefinindo o modelo MLP Keras.")
    #     model = definir_mlp_keras(input_dim)
    #     model.fit(X_train, y_train, epochs=20, batch_size=256, verbose=1, validation_split=0.1)

    #     y_pred_prob = model.predict(X_test).flatten()
    #     thresholds = [0.2, 0.4, 0.5, 0.6, 0.8]
    #     for t in thresholds:
    #         y_pred = (y_pred_prob >= t).astype(int)
    #         acc = accuracy_score(y_test, y_pred)
    #         report = classification_report(y_test.astype(int), y_pred.astype(int), output_dict=True, zero_division=0)
    #         print(report)

    #         for classe in ['0', '1']:
    #             resultados.append({
    #                 'grupo_de_features': nome_grupo,
    #                 'modelo': f"MLP_Keras_thr{t}",
    #                 'classe': classe,
    #                 'precision': report[classe]['precision'],
    #                 'recall': report[classe]['recall'],
    #                 'f1_score': report[classe]['f1-score'],
    #                 'support': report[classe]['support'],
    #                 'fold': fold,
    #                 'accuracy_geral': acc
    #             })
    #         print(f"→ Threshold {t:.1f} | Fold {fold} | Acc: {acc:.4f}")

    print(f"\nAvaliação concluída para o grupo: {nome_grupo}")
    return pd.DataFrame(resultados)


# Parte 7 - Executar o modelo e recuperar os resultados para cada DataFrame

# 7. Executar para todos os grupos
df_resultados = pd.concat([
    #avaliar_modelos_em_dataframe(df_p, 'permissions'),
    #avaliar_modelos_em_dataframe(df_i, 'intents'),
    #avaliar_modelos_em_dataframe(df_op, 'opcodes'),
    avaliar_modelos_em_dataframe(df_api, 'apicalls'),
    #avaliar_modelos_em_dataframe(df_all, 'all')
    ], ignore_index=True)


# Parte 8 - Exportar e printar os resultados resumidos

# Parte 8.1 - Criação das pastas de resultados
agora = datetime.now().strftime('%d%m%Y_%H%M')
pasta_saida = os.path.join("..", "resultados", f"resultado_RF_XGB_sem_gpu_apicalls_{agora}")
os.makedirs(pasta_saida, exist_ok=True)

# Parte 8.2 - Exportar os dados
caminho_saida = os.path.join(pasta_saida, 'resultados_modelos.csv')
df_resultados.to_csv(caminho_saida, index=False)

# Parte 8.3 - Exibir resumo final das métricas
print("\nResumo final (médias por grupo, modelo e classe):")
print(df_resultados.groupby(['grupo_de_features', 'modelo', 'classe'])[['precision', 'recall', 'f1_score']].mean().round(4))

# Calcula e salva o resumo
resumo = df_resultados.groupby(['grupo_de_features', 'modelo', 'classe'])[['precision', 'recall', 'f1_score']].mean().round(4)
resumo.to_csv(os.path.join(pasta_saida, 'resumo_resultados.csv'))

# Exibe o resumo no terminal
print("\nResumo final (médias por grupo, modelo e classe):")
print(resumo)