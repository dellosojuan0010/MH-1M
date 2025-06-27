import sys
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score
import pandas as pd
from datetime import datetime

# ========== Caminhos ==========
diretorio_atual = os.path.dirname(__file__)
diretorio_raiz = os.path.abspath(os.path.join(diretorio_atual, '..'))
sys.path.append(diretorio_raiz)

try:
    from dataset_selector import DatasetSelector
except Exception as e:
    print("❌ Erro ao importar DatasetSelector:", e)
    sys.exit(1)

# ========== Argumentos ==========
parser = argparse.ArgumentParser(description="Clusterização DBSCAN com análise binária supervisionada")
parser.add_argument('--namespaces', nargs='+', default=['permissions', 'intents'],
                    help='Lista de namespaces (ex: permissions intents opcodes)')
parser.add_argument('--eps', type=float, default=0.5, help='Distância máxima para vizinhança')
parser.add_argument('--min_samples', type=int, default=5, help='Mínimo de amostras por cluster')
args = parser.parse_args()

print("\n=== INICIANDO DBSCAN ===")
print(f"Namespaces: {args.namespaces}")
print(f"Parâmetros: eps={args.eps}, min_samples={args.min_samples}")
print("========================\n")

# ========== Carrega os dados ==========
try:
    ds = DatasetSelector()
    X, feature_names, y = ds.get_data_by_namespaces(args.namespaces)
    y = y.astype(int)
    print(f"Dados carregados: {X.shape[0]} instâncias, {X.shape[1]} features")
except Exception as e:
    print("❌ Erro ao carregar dados:", e)
    sys.exit(1)

# ========== Pré-processamento ==========
print("Normalizando dados...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ========== Redução com PCA ==========
print("Reduzindo dimensionalidade com PCA para visualização...")
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# ========== Clusterização DBSCAN ==========
print("Executando DBSCAN...")
try:
    dbscan = DBSCAN(eps=args.eps, min_samples=args.min_samples)
    predicted_labels = dbscan.fit_predict(X_scaled)
except Exception as e:
    print("❌ Erro ao executar DBSCAN:", e)
    sys.exit(1)

# ========== Avaliação ==========
n_clusters = len(set(predicted_labels)) - (1 if -1 in predicted_labels else 0)
n_noise = list(predicted_labels).count(-1)
ari_score = adjusted_rand_score(y, predicted_labels)

print(f"\nClusters detectados: {n_clusters}")
print(f"Instâncias como ruído: {n_noise}")
print(f"Adjusted Rand Index (ARI): {ari_score:.4f}")

# ========== Visualização ==========
print("Gerando gráfico...")
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=predicted_labels, cmap='tab10', s=10)
plt.title(f"DBSCAN - {n_clusters} clusters\nRuído: {n_noise}")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")

plt.subplot(1, 2, 2)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='coolwarm', s=10)
plt.title("Rótulos reais")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")

plt.tight_layout()

# ========== Salvamento ==========
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
tag = "_".join(args.namespaces)
saida_dir = os.path.join(diretorio_raiz, 'resultados_cluster', f'dbscan_{tag}_{timestamp}')
os.makedirs(saida_dir, exist_ok=True)

try:
    plt.savefig(os.path.join(saida_dir, f"dbscan_vs_true_{tag}.png"))
    print(f"Gráfico salvo em: {saida_dir}")
    plt.close()
except Exception as e:
    print("❌ Erro ao salvar gráfico:", e)

try:
    df = pd.DataFrame({'DBSCAN_Label': predicted_labels, 'Classe_Real': y})
    df.to_csv(os.path.join(saida_dir, f"dbscan_labels_{tag}.csv"), index=False)
    print(f"Labels salvos em: {saida_dir}")
except Exception as e:
    print("❌ Erro ao salvar CSV:", e)

print("\n✅ Script finalizado com sucesso.")
