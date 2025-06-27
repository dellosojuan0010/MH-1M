# svm_kernel_rbf_minilotes_balanceado.py

import sys
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import time
from datetime import datetime
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedShuffleSplit
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

# === Caminho para DatasetSelector ===
diretorio_atual = os.path.dirname(__file__)
diretorio_raiz = os.path.abspath(os.path.join(diretorio_atual, '..'))
sys.path.append(diretorio_raiz)
from dados.dataset_selector import DatasetSelector

# === Argumentos ===
parser = argparse.ArgumentParser(description="Treinamento SVM com kernel RBF e minilotes balanceados")
parser.add_argument('--namespaces', nargs='+', required=True,
                    help='Lista de namespaces a serem utilizados (ex: permissions intents opcodes)')
parser.add_argument('--minilote_tamanho', type=int, default=50000,
                    help='Tamanho de cada minilote balanceado')
args = parser.parse_args()

# === Carregamento dos dados ===
ds = DatasetSelector()
X, feature_names, y = ds.get_data_by_namespaces(args.namespaces)
y = y.astype(int)

print(f"Total de amostras: {X.shape[0]}, Features: {X.shape[1]}")
unique, counts = np.unique(y, return_counts=True)
for label, count in zip(unique, counts):
    print(f"Classe {label}: {count} instâncias")

# === Mini-lotes balanceados com StratifiedShuffleSplit ===
minilote_tamanho = args.minilote_tamanho
sss = StratifiedShuffleSplit(n_splits=int(X.shape[0] / minilote_tamanho), train_size=minilote_tamanho, random_state=42)

resultados = []
lote_idx = 1

for train_index, _ in sss.split(X, y):
    print(f"\n== Treinando minilote {lote_idx} ==")
    X_lote = X[train_index]
    y_lote = y[train_index]

    # === Mostrar balanceamento do minilote ===
    classes_lote, counts_lote = np.unique(y_lote, return_counts=True)
    print("Distribuição de classes neste minilote:")
    for classe, count in zip(classes_lote, counts_lote):
        print(f"  Classe {classe}: {count} instâncias")

    # === Escalonamento ===
    scaler = StandardScaler()
    X_lote = scaler.fit_transform(X_lote)

    # === Treinamento ===
    modelo = SVC(kernel='rbf', C=1.0, gamma='scale')
    start = time.time()
    modelo.fit(X_lote, y_lote)
    end = time.time()

    # === Avaliação no próprio lote ===
    y_pred = modelo.predict(X_lote)
    relatorio = classification_report(y_lote, y_pred, output_dict=True)
    cm = confusion_matrix(y_lote, y_pred)

    # === Salvando resultados ===
    resultados.append({
        "Lote": lote_idx,
        "Amostras": len(y_lote),
        "Accuracy": accuracy_score(y_lote, y_pred),
        "Precision": precision_score(y_lote, y_pred),
        "Recall": recall_score(y_lote, y_pred),
        "F1-Score": f1_score(y_lote, y_pred),
        "Tempo": end - start
    })

    # === Plot matriz de confusão ===
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Benigno', 'Malware'])
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"Matriz de Confusão - Lote {lote_idx}")
    plt.tight_layout()

    # === Diretórios ===
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = "_".join(args.namespaces)
    nome_rodada = f"{tag}_SVM_Lote{lote_idx}_{timestamp}"
    pasta_rodada = os.path.join(diretorio_raiz, 'resultados', nome_rodada)
    os.makedirs(pasta_rodada, exist_ok=True)

    # === Salvar imagem ===
    plt.savefig(os.path.join(pasta_rodada, f"matriz_confusao_lote{lote_idx}.png"))
    plt.close()

    # === Salvar relatório texto ===
    with open(os.path.join(pasta_rodada, f"relatorio_lote{lote_idx}.txt"), "w") as f:
        f.write("=== RELATÓRIO DO MINILOTE ===\n")
        f.write(f"Data e hora: {datetime.now()}\n")
        f.write(f"Namespaces utilizados: {args.namespaces}\n\n")
        f.write(f">> Minilote {lote_idx}:\n")
        f.write(f"- Nº de instâncias: {X_lote.shape[0]}\n")
        f.write(f"- Nº de features: {X_lote.shape[1]}\n")
        f.write(f"- Tempo de treino: {end - start:.2f} s\n")
        f.write(f"- Accuracy: {accuracy_score(y_lote, y_pred):.4f}\n")
        f.write(f"- Precision: {precision_score(y_lote, y_pred):.4f}\n")
        f.write(f"- Recall: {recall_score(y_lote, y_pred):.4f}\n")
        f.write(f"- F1-Score: {f1_score(y_lote, y_pred):.4f}\n")

    lote_idx += 1

# === Salvar resumo CSV ===
resumo_df = pd.DataFrame(resultados)
resumo_path = os.path.join(diretorio_raiz, 'resultados', f"resumo_minilotes_{'_'.join(args.namespaces)}.csv")
os.makedirs(os.path.dirname(resumo_path), exist_ok=True)
resumo_df.to_csv(resumo_path, index=False)

print("\nProcessamento finalizado. Resumo salvo em:")
print(resumo_path)
