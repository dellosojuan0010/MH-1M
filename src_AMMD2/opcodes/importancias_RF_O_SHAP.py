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
        train_idx, test_idx = folds[fold]
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        modelo = joblib.load(f'{nome_grupo}__{modelo_nome}__fold{fold+1}.joblib')

        print(f"Criando SHAP para o modelo {modelo_nome} - Fold {fold+1}")    
        explainer = shap.TreeExplainer(modelo, data=X_train, model_output='probability')

        shap_values = explainer(X_test)

        shap_exp = shap_values  # só para semântica
        shap_mat = shap_exp.values.astype(np.float32)             # (n, d)
        base_values = np.array(shap_exp.base_values).astype(np.float32)  # (n,)
        feature_names = np.array([c for c in df.columns if c != 'classe'])
        X_explicado = shap_exp.data.astype(np.float32)            # (n, d)

        # salva tudo em um único arquivo compacto
        saida_npz = f'{nome_grupo}__{modelo_nome}__fold{fold+1}_SHAP.npz'

        np.savez_compressed(
            saida_npz,
            shap_values=shap_mat,
            base_values=base_values,
            feature_names=feature_names,
            X_explicado=X_explicado,
            test_index=test_idx.astype(np.int64),
            y_test=y_test.astype(np.int32)
        )


modelo_nome = "RandomForest"
nome_grupo = "opcodes"
# Parte 2 - Abertura do arquivo, recuperação dos dados e embaralhamento
CAMINHO_ARQUIVO = os.path.join("..","..", "dados", "dados_undersampling_duplicados_eliminados.npz")

# Caminho para o arquivo compactado

PASTA_MODELO = "."

PASTA_DADOS_SHAP = os.path.join(".","SHAP")

PASTA_SAIDA = os.path.join(".","SHAP")

# cria_valores_shap(df, modelo_nome, nome_grupo, n_splits=5)

n_folds = 5
for i in range(n_folds):
    fold = i+1
    caminho_arquivo_shap = os.path.join(PASTA_DADOS_SHAP,f"{nome_grupo}__{modelo_nome}__fold{fold}_SHAP.npz")
    print(caminho_arquivo_shap)
    # Carregar os dados SHAP salvos
    shap_values_carregados = np.load(caminho_arquivo_shap, allow_pickle=True)
    print(shap_values_carregados.files)
    # Extrair os dados
    shap_values_values = shap_values_carregados['shap_values']
    shap_values_base_values = shap_values_carregados['base_values']
    shap_values_data = shap_values_carregados['X_explicado']
    shap_values_feature_names = shap_values_carregados['feature_names']
    # output_names_carregados = dados_shap_carregados['output_names']

    print("Dados SHAP carregados com sucesso!")
    print(f"Shape values carregados: {shap_values_values.shape}")
    print(f"Base values carregados: {shap_values_base_values.shape}")
    print(f"Data carregados: {shap_values_data.shape}")
    print(f"Quantidade de Features carregadas: {len(shap_values_feature_names)}")
    # print(f"Quantidade de Nome de Classes carregadas: {len(output_names_carregados)}")
    print()
    print(shap_values_feature_names[0:5])

    # (n_amostras, n_features, n_classes)
    vals = shap_values_values[:, :, 1]  # pega a classe 1
    # vals = np.mean(np.abs(vals), axis=2)  # agrega sobre classes

    # Importância global = média do valor absoluto por feature nas amostras
    importancias = np.abs(vals).mean(axis=0)  # (n_features,)

    # Transforma em Series (liga valores aos nomes das features)
    importances = pd.Series(importancias, index=shap_values_feature_names.tolist())

    # # Ordena mantendo os nomes
    importances = importances.sort_values(ascending=False)

    # # Seleciona top 20 (ou top k)
    top_features = importances.head(20).index.tolist()

    # print(top_features)
    print(importances.head(20))

    arquivo = os.path.join(PASTA_SAIDA,f"rank_importancias_{modelo_nome}_shap_fold{fold}.csv")
    importances.to_csv(arquivo, header=["valor_importancia_media"], index_label="feature_name")