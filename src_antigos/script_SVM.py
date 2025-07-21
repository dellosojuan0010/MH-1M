# Importação das bibliotecas
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
from sklearn.svm import SVC
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# Define o path para importação de DatasetSelector
diretorio_atual = os.path.dirname(__file__)
diretorio_raiz = os.path.join(diretorio_atual, '..')
caminho_absoluto = os.path.abspath(diretorio_raiz)
sys.path.append(caminho_absoluto)
from dados.dataset_selector import DatasetSelector

# Argumentos
parser = argparse.ArgumentParser(description="Treinamento SVM com seleção de namespaces")
parser.add_argument('--namespaces', nargs='+', required=True,
                    help='Lista de namespaces a serem utilizados (ex: permissions intents opcodes)')
parser.add_argument('--top_features', type=int, default=-1,
                    help='Número de features mais importantes a serem usadas (use -1 para não usar)')
args = parser.parse_args()

# Carregamento de dados
#caminho_npz = "../dados/amex-1M-[intents-permissions-opcodes-apicalls].npz"
ds = DatasetSelector()
print(f"Selecionando namespaces: {args.namespaces}")
X, feature_names, y = ds.get_data_by_namespaces(args.namespaces)
y = y.astype(int)

# Verificação
print(f"Dados carregados: {X.shape}, Classes: {y.shape}")
unique, counts = np.unique(y, return_counts=True)
for label, count in zip(unique, counts):
    print(f"Classe {label}: {count} instâncias")

# Divisão treino/teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Escalonamento
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Modelo completo
modelo = SVC(kernel='rbf', C=1.0, gamma='scale', verbose=True)
print("\nTreinando modelo SVM com todas as features selecionadas...")
start_full = time.time()
modelo.fit(X_train_scaled, y_train)
end_full = time.time()
print(f"Treinamento concluído em {end_full - start_full:.2f} segundos.")

# Avaliação completa
y_pred = modelo.predict(X_test_scaled)
relatorio = classification_report(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Benigno', 'Malware'])
disp.plot(cmap=plt.cm.Blues)
plt.title("Matriz de Confusão - SVM")
plt.tight_layout()

# Organiza diretórios de saída
tag = "_".join(args.namespaces)
modelo_nome = "SVM"
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
nome_rodada = f"{tag}_{modelo_nome}_{timestamp}"

pasta_resultados = os.path.join(os.path.dirname(__file__), 'resultados')
pasta_rodada = os.path.join(pasta_resultados, nome_rodada)
pasta_imagens = os.path.join(pasta_rodada, 'imagens')
os.makedirs(pasta_imagens, exist_ok=True)

# Salva imagem do modelo completo
disp.figure_.savefig(os.path.join(pasta_imagens, f"matriz_confusao_completo_{tag}.png"))

# Salva relatório txt
log_path = os.path.join(pasta_rodada, f"relatorio_experimento_{tag}.txt")
with open(log_path, "w") as f:
    f.write("=== RELATÓRIO DO EXPERIMENTO ===\n")
    f.write(f"Data e hora: {datetime.now()}\n")
    f.write(f"Namespaces utilizados: {args.namespaces}\n\n")

    f.write(">> Modelo completo:\n")
    f.write(f"- Nº de instâncias: {X.shape[0]}\n")
    f.write(f"- Nº de features: {X.shape[1]}\n")
    f.write(f"- Tempo de treino: {end_full - start_full:.2f} s\n")
    f.write(f"- Kernel: rbf, C=1.0, gamma='scale'\n")
    f.write(f"- Relatório de classificação:\n{relatorio}\n")

# Cria CSV de métricas
metricas_dict = {
    "Modelo": ["Completo"],
    "Accuracy": [accuracy_score(y_test, y_pred)],
    "Precision": [precision_score(y_test, y_pred)],
    "Recall": [recall_score(y_test, y_pred)],
    "F1-Score": [f1_score(y_test, y_pred)]
}

# Modelo com seleção de features, se solicitado
if args.top_features > 0:
    print(f"\nSelecionando top {args.top_features} features com maior variância...")
    variancias = np.var(X, axis=0)
    top_indices = np.argsort(variancias)[-args.top_features:]
    X_top = X[:, top_indices]

    X_train_top, X_test_top, y_train_top, y_test_top = train_test_split(
        X_top, y, test_size=0.2, random_state=42, stratify=y
    )

    X_train_top_scaled = scaler.fit_transform(X_train_top)
    X_test_top_scaled = scaler.transform(X_test_top)

    model_top = SVC(kernel='rbf', C=1.0, gamma='scale', verbose=False)
    print(f"Treinando modelo SVM com as {args.top_features} features mais variáveis...")
    start_top = time.time()
    model_top.fit(X_train_top_scaled, y_train_top)
    end_top = time.time()
    print(f"Treinamento concluído em {end_top - start_top:.2f} segundos.")

    y_pred_top = model_top.predict(X_test_top_scaled)
    report_top = classification_report(y_test_top, y_pred_top)
    cm_top = confusion_matrix(y_test_top, y_pred_top)
    disp_top = ConfusionMatrixDisplay(confusion_matrix=cm_top, display_labels=['Benigno', 'Malware'])
    disp_top.plot(cmap=plt.cm.Oranges)
    plt.title(f"Matriz de Confusão - SVM ({args.top_features} Features)")
    plt.tight_layout()

    # Salva imagem
    disp_top.figure_.savefig(os.path.join(pasta_imagens, f"matriz_confusao_top{args.top_features}_{tag}.png"))

    # Atualiza CSV
    metricas_dict["Modelo"].append(f"Top{args.top_features}")
    metricas_dict["Accuracy"].append(accuracy_score(y_test_top, y_pred_top))
    metricas_dict["Precision"].append(precision_score(y_test_top, y_pred_top))
    metricas_dict["Recall"].append(recall_score(y_test_top, y_pred_top))
    metricas_dict["F1-Score"].append(f1_score(y_test_top, y_pred_top))

    # Atualiza TXT
    with open(log_path, "a") as f:
        f.write(f"\n>> Modelo com as {args.top_features} features mais variáveis:\n")
        f.write(f"- Nº de features: {args.top_features}\n")
        f.write(f"- Tempo de treino: {end_top - start_top:.2f} s\n")
        f.write(f"- Kernel: rbf, C=1.0, gamma='scale'\n")
        f.write(f"- Relatório de classificação:\n{report_top}\n")

# Salva métricas CSV
csv_path = os.path.join(pasta_rodada, f"metricas_{tag}.csv")
pd.DataFrame(metricas_dict).to_csv(csv_path, index=False)

print(f"\nResultados salvos em: {pasta_rodada}")
