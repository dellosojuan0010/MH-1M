# Importação das bibliotecas
import sys
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import time
from datetime import datetime
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from xgboost import XGBClassifier

# Definição do path para importação de DatasetSelector
diretorio_atual = os.path.dirname(__file__)
diretorio_raiz = os.path.abspath(os.path.join(diretorio_atual, '..'))
sys.path.append(diretorio_raiz)
from dataset_selector import DatasetSelector

# Analisa os argumentos
parser = argparse.ArgumentParser(description="Treinamento XGBoost com seleção de namespaces")
parser.add_argument('--namespaces', nargs='+', required=True,
                    help='Lista de namespaces a serem utilizados (ex: permissions intents opcodes)')
parser.add_argument('--top_features', type=int, default=-1,
                    help='Número de features mais importantes a serem usadas (use -1 para não usar)')
parser.add_argument('--grid_search_f', action='store_true',
                    help='Se especificado, executa GridSearchCV para o modelo completo (todas as features)')
parser.add_argument('--grid_search_topf', action='store_true',
                    help='Se especificado, executa GridSearchCV para o modelo com top_features')
args = parser.parse_args()

# Exibe modo de execução
print("\n===== CONFIGURAÇÃO DO EXPERIMENTO =====")
print(f"Namespaces selecionados: {args.namespaces}")
print(f"Top features ativado: {'Sim' if args.top_features > 0 else 'Não'}")
print(f"GridSearch completo: {'Ativado' if args.grid_search_f else 'Desativado'}")
print(f"GridSearch top_features: {'Ativado' if args.grid_search_topf else 'Desativado'}")
print("=======================================\n")

# === Carregamento de dados ===
ds = DatasetSelector()
X, feature_names, y = ds.get_data_by_namespaces(args.namespaces)
y = y.astype(int)

# === Verificação ===
print(f"Dados carregados: {X.shape}, Classes: {y.shape}")
unique, counts = np.unique(y, return_counts=True)
for label, count in zip(unique, counts):
    print(f"Classe {label}: {count} instâncias")

# === Divisão treino/validação/teste ===
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=42, stratify=y_train_val)

# === Hiperparâmetros padrão ===
parametros_padrao_geral = {
    'n_estimators': 300,
    'max_depth': 10,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'eval_metric': 'logloss',
    'verbosity': 1,
    'n_jobs': 3
}

parametros_padrao_top_features = {
    'n_estimators': 300,
    'max_depth': 100,
    'learning_rate': 0.05,
    'subsample': 1.0,
    'eval_metric': 'logloss',
    'verbosity': 1,
    'n_jobs': 3
}

# === GridSearch se ativado para modelo completo ===
if args.grid_search_f:
    print("\nExecutando GridSearchCV para encontrar melhores hiperparâmetros (modelo completo)...")
    grid_params = {
        'n_estimators': [50, 100, 200, 300],
        'max_depth': [50, 100, 150, 300],
        'learning_rate': [0.05, 0.1],
        'subsample': [0.8, 1.0]
    }
    modelo_grid = GridSearchCV(XGBClassifier(eval_metric='logloss'),
                               grid_params, scoring='f1', cv=3, verbose=1, n_jobs=3)
    start = time.time()
    modelo_grid.fit(X_train, y_train)
    end = time.time()
    modelo = modelo_grid.best_estimator_
    best_params = modelo_grid.best_params_
    print(f"Melhores parâmetros encontrados: {best_params}")
else:
    start = time.time()
    modelo = XGBClassifier(**parametros_padrao_geral)
    modelo.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=True)
    end = time.time()
    best_params = parametros_padrao_geral

print(f"Treinamento concluído em {end - start:.2f} segundos.")

# === Curva de aprendizado ===
eval_results = modelo.evals_result()
if 'validation_0' in eval_results:
    plt.figure()
    plt.plot(eval_results['validation_0']['logloss'], label='Validação')
    if 'validation_1' in eval_results:
        plt.plot(eval_results['validation_1']['logloss'], label='Validação Extra')
    plt.xlabel('Épocas')
    plt.ylabel('Logloss')
    plt.title('Curva de Aprendizado - Modelo Completo')
    plt.legend()
    plt.tight_layout()

# === Avaliação ===
y_pred = modelo.predict(X_test)
relatorio = classification_report(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Benigno', 'Malware'])
disp.plot(cmap=plt.cm.Blues)
plt.title("Matriz de Confusão - XGBoost")
plt.tight_layout()

# === Organiza diretórios de saída ===
tag = "_".join(args.namespaces)
modelo_nome = "XGBoost"
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
nome_rodada = f"{tag}_{modelo_nome}_{timestamp}"

pasta_resultados = os.path.join(diretorio_raiz, 'resultados')
pasta_rodada = os.path.join(pasta_resultados, nome_rodada)
pasta_imagens = os.path.join(pasta_rodada, 'imagens')
os.makedirs(pasta_imagens, exist_ok=True)

# === Salva imagens ===
disp.figure_.savefig(os.path.join(pasta_imagens, f"matriz_confusao_completo_{tag}.png"))
if 'validation_0' in eval_results:
    plt.savefig(os.path.join(pasta_imagens, f"curva_aprendizado_completo_{tag}.png"))
    plt.close()

# === Salva relatório txt ===
log_path = os.path.join(pasta_rodada, f"relatorio_experimento_{tag}.txt")
with open(log_path, "w") as f:
    f.write("=== RELATÓRIO DO EXPERIMENTO ===\n")
    f.write(f"Data e hora: {datetime.now()}\n")
    f.write(f"Namespaces utilizados: {args.namespaces}\n\n")
    f.write(f">> Modelo completo:\n")
    f.write(f"- Nº de instâncias: {X.shape[0]}\n")
    f.write(f"- Nº de features: {X.shape[1]}\n")
    f.write(f"- Tempo de treino: {end - start:.2f} s\n")
    f.write(f"- Hiperparâmetros: {best_params}\n")
    f.write(f"- Relatório de classificação:\n{relatorio}\n")

# === Salva métricas em CSV ===
metricas_dict = {
    "Modelo": ["Completo"],
    "Accuracy": [accuracy_score(y_test, y_pred)],
    "Precision": [precision_score(y_test, y_pred)],
    "Recall": [recall_score(y_test, y_pred)],
    "F1-Score": [f1_score(y_test, y_pred)]
}

# === Curva de aprendizado para top_features ===
if args.top_features > 0:
    print(f"\nSelecionando top {args.top_features} features com maior variância...")
    variancias = np.var(X, axis=0)
    top_indices = np.argsort(variancias)[-args.top_features:]
    X_top = X[:, top_indices]

    X_train_val_top, X_test_top, y_train_val_top, y_test_top = train_test_split(X_top, y, test_size=0.2, random_state=42, stratify=y)
    X_train_top, X_val_top, y_train_top, y_val_top = train_test_split(X_train_val_top, y_train_val_top, test_size=0.25, random_state=42, stratify=y_train_val_top)

    if args.grid_search_topf:
        print("\nExecutando GridSearchCV para top features...")
        modelo_grid_top = GridSearchCV(XGBClassifier(eval_metric='logloss'),
                                       grid_params, scoring='f1', cv=3, verbose=1, n_jobs=-1)
        start_top = time.time()
        modelo_grid_top.fit(X_train_top, y_train_top)
        end_top = time.time()
        modelo_top = modelo_grid_top.best_estimator_
        best_params_top = modelo_grid_top.best_params_
    else:
        start_top = time.time()
        modelo_top = XGBClassifier(**parametros_padrao_top_features)
        modelo_top.fit(X_train_top, y_train_top, eval_set=[(X_val_top, y_val_top)], verbose=True)
        end_top = time.time()
        best_params_top = parametros_padrao_top_features

    y_pred_top = modelo_top.predict(X_test_top)
    report_top = classification_report(y_test_top, y_pred_top)
    cm_top = confusion_matrix(y_test_top, y_pred_top)
    disp_top = ConfusionMatrixDisplay(confusion_matrix=cm_top, display_labels=['Benigno', 'Malware'])
    disp_top.plot(cmap=plt.cm.Oranges)
    plt.title(f"Matriz de Confusão - XGBoost ({args.top_features} Features)")
    plt.tight_layout()
    disp_top.figure_.savefig(os.path.join(pasta_imagens, f"matriz_confusao_top{args.top_features}_{tag}.png"))

    eval_results_top = modelo_top.evals_result()
    if 'validation_0' in eval_results_top:
        plt.figure()
        plt.plot(eval_results_top['validation_0']['logloss'], label='Validação')
        if 'validation_1' in eval_results_top:
            plt.plot(eval_results_top['validation_1']['logloss'], label='Validação Extra')
        plt.xlabel('Épocas')
        plt.ylabel('Logloss')
        plt.title(f'Curva de Aprendizado - Top {args.top_features} Features')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(pasta_imagens, f"curva_aprendizado_top{args.top_features}_{tag}.png"))
        plt.close()

    with open(log_path, "a") as f:
        f.write(f"\n>> Modelo com as {args.top_features} features mais variáveis:\n")
        f.write(f"- Nº de features: {args.top_features}\n")
        f.write(f"- Tempo de treino: {end_top - start_top:.2f} s\n")
        f.write(f"- Hiperparâmetros: {best_params_top}\n")
        f.write(f"- Relatório de classificação:\n{report_top}\n")

    metricas_dict["Modelo"].append(f"Top{args.top_features}")
    metricas_dict["Accuracy"].append(accuracy_score(y_test_top, y_pred_top))
    metricas_dict["Precision"].append(precision_score(y_test_top, y_pred_top))
    metricas_dict["Recall"].append(recall_score(y_test_top, y_pred_top))
    metricas_dict["F1-Score"].append(f1_score(y_test_top, y_pred_top))

csv_path = os.path.join(pasta_rodada, f"metricas_{tag}.csv")
pd.DataFrame(metricas_dict).to_csv(csv_path, index=False)

print(f"\nResultados salvos em: {pasta_rodada}")
