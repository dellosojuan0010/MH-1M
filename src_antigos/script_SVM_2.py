# sgd_sparse_experimento.py

import sys
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import time
from datetime import datetime
import pandas as pd
from sklearn.model_selection import train_test_split
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
from sklearn.linear_model import SGDClassifier
from scipy.sparse import csr_matrix

# === Definição do path para importação de DatasetSelector ===
diretorio_atual = os.path.dirname(__file__)
diretorio_raiz = os.path.abspath(os.path.join(diretorio_atual, '..'))
sys.path.append(diretorio_raiz)
from dataset_selector import DatasetSelector

# === Argumentos ===
parser = argparse.ArgumentParser(description="Treinamento SGD com dados esparsos e seleção de namespaces")
parser.add_argument('--namespaces', nargs='+', required=True,
                    help='Lista de namespaces a serem utilizados (ex: permissions intents opcodes)')
parser.add_argument('--top_features', type=int, default=-1,
                    help='Número de features mais importantes a serem usadas (use -1 para não usar)')
args = parser.parse_args()

# === Exibe configuração ===
print("\n===== CONFIGURAÇÃO DO EXPERIMENTO =====")
print(f"Namespaces selecionados: {args.namespaces}")
print(f"Top features ativado: {'Sim' if args.top_features > 0 else 'Não'}")
print("=======================================\n")

# === Carregar dados ===
ds = DatasetSelector()
X, feature_names, y = ds.get_data_by_namespaces(args.namespaces)
y = y.astype(int)

# === Seleção de top features se especificado ===
if args.top_features > 0:
    print(f"Selecionando top {args.top_features} features com maior variância...")
    variancias = np.var(X, axis=0)
    top_indices = np.argsort(variancias)[-args.top_features:]
    X = X[:, top_indices]
    feature_names = feature_names[top_indices]

# === Converter para formato esparso ===
X_sparse = csr_matrix(X)

# === Divisão dos dados ===
X_train, X_test, y_train, y_test = train_test_split(
    X_sparse, y, test_size=0.2, random_state=42, stratify=y
)

# === Treinamento com SGDClassifier ===
modelo = SGDClassifier(loss='hinge', max_iter=1000, tol=1e-3, n_jobs=-1)

print("\nTreinando modelo SGDClassifier com dados esparsos...")
start = time.time()
modelo.fit(X_train, y_train)
end = time.time()
print(f"Treinamento concluído em {end - start:.2f} segundos.")

# === Avaliação ===
y_pred = modelo.predict(X_test)
relatorio = classification_report(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Benigno', 'Malware'])
disp.plot(cmap=plt.cm.Blues)
plt.title("Matriz de Confusão - SGDClassifier")
plt.tight_layout()

# === Organização das saídas ===
tag = "_".join(args.namespaces)
modelo_nome = "SGDClassifier"
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
nome_rodada = f"{tag}_{modelo_nome}_{timestamp}"

pasta_resultados = os.path.join(diretorio_raiz, 'resultados')
pasta_rodada = os.path.join(pasta_resultados, nome_rodada)
pasta_imagens = os.path.join(pasta_rodada, 'imagens')
os.makedirs(pasta_imagens, exist_ok=True)

# === Salvar imagem da matriz de confusão ===
disp.figure_.savefig(os.path.join(pasta_imagens, f"matriz_confusao_{tag}.png"))

# === Relatório em TXT ===
log_path = os.path.join(pasta_rodada, f"relatorio_experimento_{tag}.txt")
with open(log_path, "w") as f:
    f.write("=== RELATÓRIO DO EXPERIMENTO ===\n")
    f.write(f"Data e hora: {datetime.now()}\n")
    f.write(f"Namespaces utilizados: {args.namespaces}\n\n")
    f.write(f">> Modelo: SGDClassifier\n")
    f.write(f"- Nº de instâncias: {X.shape[0]}\n")
    f.write(f"- Nº de features: {X.shape[1]}\n")
    f.write(f"- Tempo de treino: {end - start:.2f} s\n")
    f.write(f"- Relatório de classificação:\n{relatorio}\n")

# === Métricas em CSV ===
metricas_dict = {
    "Modelo": ["SGDClassifier"],
    "Accuracy": [accuracy_score(y_test, y_pred)],
    "Precision": [precision_score(y_test, y_pred)],
    "Recall": [recall_score(y_test, y_pred)],
    "F1-Score": [f1_score(y_test, y_pred)]
}
csv_path = os.path.join(pasta_rodada, f"metricas_{tag}.csv")
pd.DataFrame(metricas_dict).to_csv(csv_path, index=False)

print(f"\nResultados salvos em: {pasta_rodada}")
