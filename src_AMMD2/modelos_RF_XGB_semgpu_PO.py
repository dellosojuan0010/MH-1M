###
#
# ESSE MODELO SERIA PARA USAR TENSORFLOW, A GPU DO DESKTOP DO LAB SÓ RODOU COM PYTORCH
#
###

# Parte 1 - Importação das bibliotecas

import os
import gc
from datetime import datetime

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from tqdm import tqdm

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier


# Parte 2 - Abertura do arquivo, recuperação dos dados e embaralhamento

CAMINHO_ARQUIVO = os.path.join("..", "dados", "dados_undersampling_duplicados_eliminados.npz")

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

# ======= Parte 8 - Criação das pastas de resultados (FEITO ANTES DA AVALIAÇÃO) =======

agora = datetime.now().strftime('%d%m%Y_%H%M')
pasta_saida = os.path.join("..", "resultadosAMMD2", "amostras_reduzidas_balanceadas", f"resultado_RF_XGB_PO_{agora}")
os.makedirs(pasta_saida, exist_ok=True)


# Parte 3 - Separar as colunas das features e criar os DataFrames

# Identificar colunas por namespace
idx_permissions = [i for i, nome in enumerate(colunas) if nome.startswith("permissions::")]
idx_opcodes     = [i for i, nome in enumerate(colunas) if nome.startswith("opcodes::")]

df_po    = pd.DataFrame(X[:, idx_permissions + idx_opcodes], columns=np.array(colunas)[idx_permissions + idx_opcodes])
df_po['classe'] = y

print("DataFrames criados:")
print(f" - df_po : {df_po.shape}")


# Parte 5 - Função para definir modelos adaptados

def definir_modelos_sklearn(input_dim):
    if input_dim > 20000:
        rf = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42)
        xgb = XGBClassifier(n_estimators=200, max_depth=10, learning_rate=0.05, verbosity=1, use_label_encoder=False, random_state=42)
    elif input_dim > 400:
        rf = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42)
        xgb = XGBClassifier(n_estimators=150, max_depth=8, learning_rate=0.07, verbosity=1, use_label_encoder=False, random_state=42)
    else:
        rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        xgb = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, verbosity=1, use_label_encoder=False, random_state=42)

    return {'RandomForest': rf, 'XGBoost': xgb}


# Parte 6 - Definir função de avaliação (agora salvando matriz de confusão por experimento)

def avaliar_modelos_em_dataframe(df, nome_grupo, pasta_saida, n_splits=5):

    X = df.drop(columns=['classe']).values.astype(np.float32)
    y = df['classe'].astype(np.int32).values  # garante inteiros 0/1
    input_dim = X.shape[1]

    print(f"\nIniciando avaliação para o grupo de features: '{nome_grupo}' com {X.shape[1]} atributos e {X.shape[0]} instâncias.")

    # Pasta para matrizes de confusão
    pasta_cm = os.path.join(pasta_saida, "matrizes_confusao")
    os.makedirs(pasta_cm, exist_ok=True)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    print(f"\nDefinindo os modelos RandomForest e XGBoost.")
    modelos = definir_modelos_sklearn(input_dim)
    resultados = []

    for modelo_nome, modelo in tqdm(modelos.items(), desc=f"[{nome_grupo}] Modelos sklearn"):

        print(f"\nModelo: {modelo_nome} - Total de folds: {n_splits}")

        # acumuladores para a MATRIZ MACRO (agregada) do modelo
        y_true_all_folds = []
        y_pred_all_folds = []

        for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
            print(f"Treinando {modelo_nome} - Fold {fold}")
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            modelo.fit(X_train, y_train)
            y_pred = modelo.predict(X_test).astype(int)

            # acumula para a matriz macro
            y_true_all_folds.append(y_test)
            y_pred_all_folds.append(y_pred)

            acc = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
            print(report)

            # ===== MATRIZ DE CONFUSÃO (2x2 com labels fixos) =====
            cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

            # Salvar CSV da matriz de confusão do experimento (fold)
            nome_base = f"{nome_grupo}__{modelo_nome}__fold{fold}"
            caminho_cm_csv = os.path.join(pasta_cm, f"{nome_base}.csv")
            pd.DataFrame(cm, index=["true_0", "true_1"], columns=["pred_0", "pred_1"]).to_csv(caminho_cm_csv, index=True)

            # Salvar figura da matriz de confusão do experimento
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['benigno', 'maligno'])
            fig, ax = plt.subplots(figsize=(4, 4), dpi=120)
            disp.plot(ax=ax, values_format='d', colorbar=False)
            ax.set_title(f"{nome_grupo} - {modelo_nome} - Fold {fold}")
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



# Parte 7 - Executar o modelo e recuperar os resultados para cada DataFrame

df_resultados = pd.concat([
    avaliar_modelos_em_dataframe(df_po, 'permissions_opcodes', pasta_saida),
], ignore_index=True)


# Parte 8.2 - Exportar os dados consolidados
caminho_saida = os.path.join(pasta_saida, 'resultados_modelos.csv')
df_resultados.to_csv(caminho_saida, index=False)

# Parte 8.3 - Exibir e salvar resumo final das métricas
print("\nResumo final (médias por grupo, modelo e classe):")
resumo = df_resultados.groupby(['grupo_de_features', 'modelo', 'classe'])[['precision', 'recall', 'f1_score']].mean().round(4)
print(resumo)

resumo.to_csv(os.path.join(pasta_saida, 'resumo_resultados.csv'))

print(f"\nArquivos salvos em: {pasta_saida}")
print("• CSV por experimento da matriz de confusão em: pasta 'matrizes_confusao/'")
print("• PNG por experimento da matriz de confusão em: pasta 'matrizes_confusao/'")