# svm_kernel_rbf_experimento.py

import sys
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import time
from datetime import datetime
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
from sklearn.svm import SVC

# === Definir diretório raiz para imports personalizados ===
diretorio_atual = os.path.dirname(__file__)
diretorio_raiz = os.path.abspath(os.path.join(diretorio_atual, '..'))
sys.path.append(diretorio_raiz)
from dataset_selector import DatasetSelector

# === Argumentos de execução ===
parser = argparse.ArgumentParser(description="Treinamento SVM com seleção de namespaces")
parser.add_argument('--namespaces', nargs='+', required=True,
                    help='Lista de namespaces a serem utilizados (ex: permissions intents opcodes)')
parser.add_argument('--top_features', type=int, default=-1,
                    help='Número de features mais importantes a serem usadas (use -1 para não usar)')
parser.add_argument('--grid_search_f', action='store_true',
                    help='Se especificado, executa GridSearchCV para o modelo completo')
parser.add_argument('--grid_search_topf', action='store_true',
                    help='Se especificado, executa GridSearchCV para o modelo com top_features')
args = parser.parse_args()

# === Exibir configuração ===
print("\n===== CONFIGURAÇÃO DO EXPERIMENTO =====")
print(f"Namespaces selecionados: {args.namespaces}")
print(f"Top features ativado: {'Sim' if args.top_features > 0 else 'Não'}")
print(f"GridSearch completo: {'Ativado' if args.grid_search_f else 'Desativado'}")
print(f"GridSearch top_features: {'Ativado' if args.grid_search_topf else 'Desativado'}")
print("=======================================\n")

# === Carregar dados ===
ds = DatasetSelector()
X, feature_names, y = ds.get_data_by_namespaces(args.namespaces)
y = y.astype(int)

# === Verificação básica ===
print(f"Dados carregados: {X.shape}, Classes: {y.shape}")
unique, counts = np.unique(y, return_counts=True)
for label, count in zip(unique, counts):
    print(f"Classe {label}: {count} instâncias")

# === Divisão dos dados ===
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=42, stratify=y_train_val)

# === Escalonamento ===
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# === Parâmetros para GridSearch ===
grid_params = {
    'C': [0.1, 1, 10],
    'gamma': ['scale', 'auto'],
    'kernel': ['rbf']
}

# === Treinamento modelo completo ===
if args.grid_search_f:
    print("\nExecutando GridSearchCV para SVM completo...")
    modelo_grid = GridSearchCV(SVC(), grid_params, scoring='f1', cv=3, verbose=1, n_jobs=-1)
    start = time.time()
    modelo_grid.fit(X_train, y_train)
    end = time.time()
    modelo = modelo_grid.best_estimator_
    best_params = modelo_grid.best_params_
else:
    start = time.time()
    modelo = SVC(kernel='rbf', C=1.0, gamma='scale')
    modelo.fit(X_train, y_train)
    end = time.time()
    best_params = {'kernel': 'rbf', 'C': 1.0, 'gamma': 'scale'}

print(f"Treinamento concluído em {end - start:.2f} segundos.")

# === Avaliação modelo completo ===
y_pred = modelo.predict(X_test)
relatorio = classification_report(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Benigno', 'Malware'])
disp.plot(cmap=plt.cm.Blues)
plt.title("Matriz de Confusão - SVM")
plt.tight_layout()

# === Organização de saída ===
tag = "_".join(args.namespaces)
modelo_nome = "SVM"
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
nome_rodada = f"{tag}_{modelo_nome}_{timestamp}"

pasta_resultados = os.path.join(diretorio_raiz, 'resultados')
pasta_rodada = os.path.join(pasta_resultados, nome_rodada)
pasta_imagens = os.path.join(pasta_rodada, 'imagens')
os.makedirs(pasta_imagens, exist_ok=True)

# === Salvar saída ===
disp.figure_.savefig(os.path.join(pasta_imagens, f"matriz_confusao_completo_{tag}.png"))

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

metricas_dict = {
    "Modelo": ["Completo"],
    "Accuracy": [accuracy_score(y_test, y_pred)],
    "Precision": [precision_score(y_test, y_pred)],
    "Recall": [recall_score(y_test, y_pred)],
    "F1-Score": [f1_score(y_test, y_pred)]
}

# === Top features ===
if args.top_features > 0:
    print(f"\nSelecionando top {args.top_features} features com maior variância...")
    variancias = np.var(X, axis=0)
    top_indices = np.argsort(variancias)[-args.top_features:]
    X_top = X[:, top_indices]

    X_train_val_top, X_test_top, y_train_val_top, y_test_top = train_test_split(X_top, y, test_size=0.2, random_state=42, stratify=y)
    X_train_top, X_val_top, y_train_top, y_val_top = train_test_split(X_train_val_top, y_train_val_top, test_size=0.25, random_state=42, stratify=y_train_val_top)

    scaler_top = StandardScaler()
    X_train_top = scaler_top.fit_transform(X_train_top)
    X_test_top = scaler_top.transform(X_test_top)

    if args.grid_search_topf:
        print("\nExecutando GridSearchCV para top features...")
        modelo_grid_top = GridSearchCV(SVC(), grid_params, scoring='f1', cv=3, verbose=1, n_jobs=-1)
        start_top = time.time()
        modelo_grid_top.fit(X_train_top, y_train_top)
        end_top = time.time()
        modelo_top = modelo_grid_top.best_estimator_
        best_params_top = modelo_grid_top.best_params_
    else:
        start_top = time.time()
        modelo_top = SVC(kernel='rbf', C=1.0, gamma='scale')
        modelo_top.fit(X_train_top, y_train_top)
        end_top = time.time()
        best_params_top = {'kernel': 'rbf', 'C': 1.0, 'gamma': 'scale'}

    y_pred_top = modelo_top.predict(X_test_top)
    report_top = classification_report(y_test_top, y_pred_top)
    cm_top = confusion_matrix(y_test_top, y_pred_top)
    disp_top = ConfusionMatrixDisplay(confusion_matrix=cm_top, display_labels=['Benigno', 'Malware'])
    disp_top.plot(cmap=plt.cm.Oranges)
    plt.title(f"Matriz de Confusão - SVM ({args.top_features} Features)")
    plt.tight_layout()
    disp_top.figure_.savefig(os.path.join(pasta_imagens, f"matriz_confusao_top{args.top_features}_{tag}.png"))

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

# === Salvar métricas CSV ===
csv_path = os.path.join(pasta_rodada, f"metricas_{tag}.csv")
pd.DataFrame(metricas_dict).to_csv(csv_path, index=False)

print(f"\nResultados salvos em: {pasta_rodada}")
