# Parte 1 - Importação das bibliotecas
import os
import gc
from datetime import datetime

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from tqdm import tqdm

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier

import joblib

import shap

# Função para definir modelos adaptados -> retorna um modelo
def definir_modelos_sklearn(input_dim, nome_modelo):
    if nome_modelo == "RandomForest":
        if input_dim > 20000:
            return RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42)
        elif input_dim > 400:
            return RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42)
        else:
            return RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    elif nome_modelo == "XGBoost":
        if input_dim > 20000:
            return XGBClassifier(n_estimators=200, max_depth=10, learning_rate=0.05, verbosity=1, use_label_encoder=False, random_state=42)
        elif input_dim > 400:
            return XGBClassifier(n_estimators=150, max_depth=8, learning_rate=0.07, verbosity=1, use_label_encoder=False, random_state=42)
        else:
            return XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, verbosity=1, use_label_encoder=False, random_state=42)

    raise ValueError(f"nome_modelo inválido: {nome_modelo}")

# Função de avaliação (agora salvando matriz de confusão por experimento)
def treina_modelo(df, modelo, nome_modelo, nome_grupo, n_splits=5, pasta_saida="resultado"):

    X = df.drop(columns=['classe']).values.astype(np.float32)
    y = df['classe'].astype(np.int32).values  # garante inteiros 0/1
    input_dim = X.shape[1]

    print(f"\nIniciando avaliação para o grupo de features: '{nome_grupo}' com {X.shape[1]} atributos e {X.shape[0]} instâncias.")

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)    
    print(f"\nModelo: {nome_modelo} - Total de folds: {n_splits}")

    # Pasta para matrizes de confusão
    pasta_cm = os.path.join(pasta_saida,"matrizes_confusao")
    os.makedirs(pasta_cm, exist_ok=True)
  
    resultados = []

    # acumuladores para a MATRIZ MACRO (agregada) do modelo
    # y_true_all_folds = []
    # y_pred_all_folds = []
    idx_fold = 1
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
        if fold != idx_fold:
            continue
        print(f" # Treinando {nome_modelo} - Fold {fold}")
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        nome_base = f"{nome_grupo}__{nome_modelo}__fold{fold}"

        modelo.fit(X_train, y_train)
        joblib.dump(modelo, os.path.join(pasta_saida,nome_base+'pre.joblib'))
        y_pred = modelo.predict(X_test).astype(int)

        # acumula para a matriz macro
        # y_true_all_folds.append(y_test)
        # y_pred_all_folds.append(y_pred)

        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        print(report)

        # ===== MATRIZ DE CONFUSÃO (2x2 com labels fixos) =====
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

        # Salvar CSV da matriz de confusão do experimento (fold)            
        caminho_cm_csv = os.path.join(pasta_cm, f"{nome_base}.csv")
        pd.DataFrame(cm, index=["true_0", "true_1"], columns=["pred_0", "pred_1"]).to_csv(caminho_cm_csv, index=True)

        # Salvar figura da matriz de confusão do experimento
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['benigno', 'maligno'])
        fig, ax = plt.subplots(figsize=(4, 4), dpi=120)
        disp.plot(ax=ax, values_format='d', colorbar=False)
        ax.set_title(f"{nome_grupo} - {nome_modelo} - Fold {fold}")
        fig.tight_layout()
        caminho_cm_png = os.path.join(pasta_cm, f"{nome_base}.png")
        fig.savefig(caminho_cm_png)
        plt.close(fig)

        # Registra métricas por classe
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
                'accuracy_geral': acc,
                # opcional: caminho dos artefatos do fold
                'cm_csv': caminho_cm_csv if classe == '0' else '',
                'cm_png': caminho_cm_png if classe == '0' else ''
            })

        # =========================
        # MATRIZ DE CONFUSÃO MACRO
        # =========================
        y_true_all = np.concatenate(y_true_all_folds)
        y_pred_all = np.concatenate(y_pred_all_folds)

        cm_macro = confusion_matrix(y_true_all, y_pred_all, labels=[0, 1])  # soma/aggregado dos folds
        cm_macro_norm = cm_macro.astype(float) / cm_macro.sum(axis=1, keepdims=True)  # normalizada por linha

        # salvar CSVs macro
        base_macro = f"{nome_grupo}__{modelo_nome}__MACRO"
        caminho_cm_macro_csv = os.path.join(pasta_cm, f"{base_macro}.csv")
        caminho_cm_macro_norm_csv = os.path.join(pasta_cm, f"{base_macro}__NORMALIZADA.csv")

        pd.DataFrame(cm_macro, index=["true_0", "true_1"], columns=["pred_0", "pred_1"]).to_csv(caminho_cm_macro_csv, index=True)
        pd.DataFrame(cm_macro_norm, index=["true_0", "true_1"], columns=["pred_0", "pred_1"]).to_csv(caminho_cm_macro_norm_csv, index=True)

        # salvar figuras macro (absoluta e normalizada)
        # absoluta
        fig, ax = plt.subplots(figsize=(4, 4), dpi=120)
        ConfusionMatrixDisplay(confusion_matrix=cm_macro, display_labels=[0, 1]).plot(ax=ax, values_format='d', colorbar=False)
        ax.set_title(f"{nome_grupo} - {modelo_nome} - MACRO (absoluta)")
        fig.tight_layout()
        caminho_cm_macro_png = os.path.join(pasta_cm, f"{base_macro}.png")
        fig.savefig(caminho_cm_macro_png)
        plt.close(fig)

        # normalizada por linha
        fig, ax = plt.subplots(figsize=(4, 4), dpi=120)
        ConfusionMatrixDisplay.from_predictions(
            y_true_all, y_pred_all, display_labels=[0, 1],
            normalize='true', values_format='.2f', ax=ax, colorbar=False
        )
        ax.set_title(f"{nome_grupo} - {modelo_nome} - MACRO (normalizada)")
        fig.tight_layout()
        caminho_cm_macro_norm_png = os.path.join(pasta_cm, f"{base_macro}__NORMALIZADA.png")
        fig.savefig(caminho_cm_macro_norm_png)
        plt.close(fig)

        print(f"Finalizado modelo {modelo_nome} para '{nome_grupo}'")
        print(f"-> MACRO salva: {caminho_cm_macro_csv} | {caminho_cm_macro_png}")
        print(f"-> MACRO NORMALIZADA salva: {caminho_cm_macro_norm_csv} | {caminho_cm_macro_norm_png}")

    print(f"\nAvaliação concluída para o grupo: {nome_grupo}")
    return pd.DataFrame(resultados)

# Função que abre os dados e retorna os valores originais de X, y, e nome das colunas
def carregar_dados():
    ## Abertura do arquivo, recuperação dos dados e embaralhamento
    CAMINHO_ARQUIVO = os.path.join("..","..", "dados", "dados_undersampling_duplicados_eliminados.npz")
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
    return X, y, colunas

# Função que recebe os dados para selecionar apenas as features definidas pelo namespace e devolve o dataframe.
def seleciona_dados(X, y, colunas, namespace_feature):
    idx_features = [i for i, nome in enumerate(colunas) if nome.startswith(f"{namespace_feature}::")]
    df = pd.DataFrame(X[:, idx_features], columns=np.array(colunas)[idx_features])
    df['classe'] = y
    print(f"DataFrame criados: {df.shape}")    
    return df

def carrega_modelo(namespace_feature, nome_modelo, fold):
    # Abrir modelo salvo
    modelo_carregado = joblib.load(f"{namespace_feature}__{nome_modelo}__fold{fold}.joblib")
    # nome_modelo_carregado = inferir_nome_modelo(modelo_carregado)
    # print(f"Modelo carregado: {nome_modelo_carregado} ({ARQ_MODELO})")    
    return modelo_carregado

def funcao_predicao(modelo, Xbatch):
    Xbatch = np.asarray(Xbatch, dtype=np.float32)
    return modelo.predict(Xbatch, verbose=0).ravel()


# Função principal
def main():

    # Carrega os dados utilizados nos treinos
    X, y, colunas = carregar_dados()
    
    ## Separar as colunas das features e criar os DataFrames
    # Identificar colunas por namespace ("intents::", "permissions::","opcodes::", "apicalls::")
    namespace_feature = "permissions"
    df = seleciona_dados(X, y, colunas, namespace_feature)
    
    gc.collect()

    ## Carrega modelo
    nome_modelo = "RandomForest"
    fold = 1
    modelo = carrega_modelo(namespace_feature, nome_modelo, fold)

    # X e df já preparados; modelo (RF/XGB) já carregado
    X_ns = df.drop(columns='classe').values
    y_ns = df['classe'].values
    
    # Divisão treino/teste (mantendo features certas)
    X_train, X_test, y_train, y_test = train_test_split(
        X_ns,
        y_ns,
        test_size=0.3,
        random_state=42
    )

    explainer = shap.Explainer(modelo, X_train)

    # SHAP de todas as amostras de uma vez
    shap_values = explainer.shap_values(X_test)

    print(f"Shape values: {shap_values.values.shape}")
    print(f"Base values: {shap_values.base_values.shape}")
    print(f"Data: {shap_values.data.shape}")
    print(f"Nome das Features: {len(shap_values.feature_names)}")
    print(f"Nome das Classes: {len(shap_values.output_names)}")

    np.savez_compressed(
        f"shap_{nome_modelo}_{namespace_feature}_fold{fold}.npz",
        shap_values = shap_values.values,
        base_values = shap_values.base_values,
        data = shap_values.data,
        feature_names = shap_values.feature_names,
        output_names = shap_values.output_names
    )   



# Ponto de entrada
if __name__ == "__main__":
    main()

