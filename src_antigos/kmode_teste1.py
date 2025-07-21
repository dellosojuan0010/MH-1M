import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from kmodes.kmodes import KModes
from datetime import datetime

# === Adiciona o caminho para DatasetSelector ===
diretorio_atual = os.path.dirname(__file__)
diretorio_raiz = os.path.abspath(os.path.join(diretorio_atual, '..'))
sys.path.append(diretorio_raiz)
from dados.dataset_selector import DatasetSelector

# === Argumentos de linha de comando ===
parser = argparse.ArgumentParser(description="Clusterização com K-Modes para atributos categóricos")
parser.add_argument('--namespaces', nargs='+', required=True,
                    help='Lista de namespaces (ex: permissions intents opcodes)')
parser.add_argument('--classe0', type=int, required=True,
                    help='Número de amostras da classe 0')
parser.add_argument('--classe1', type=int, required=True,
                    help='Número de amostras da classe 1')
parser.add_argument('--saida', type=str, default="resultados_kmodes",
                    help='Diretório para salvar os resultados')
args = parser.parse_args()

# === Criação do diretório de saída ===
os.makedirs(args.saida, exist_ok=True)

# === Carregamento dos dados ===
print("🔍 Carregando dados...")
selector = DatasetSelector()
X_bin, colunas, y = selector.select_random_classes_custom(
    namespaces=args.namespaces,
    samples_per_class={0: args.classe0, 1: args.classe1},
    random_state=42
)

# === Conversão para categorias ===
X_cat = X_bin.astype(str)

# === K-Modes ===
print("⚙️ Executando K-Modes...")
km = KModes(n_clusters=2, init='Huang', n_init=5, verbose=1, random_state=42)
labels = km.fit_predict(X_cat)

# === Avaliação ===
ari = adjusted_rand_score(y, labels)
nmi = normalized_mutual_info_score(y, labels)

print(f"✅ ARI: {ari:.4f}")
print(f"✅ NMI: {nmi:.4f}")

# === PCA para visualização ===
print("📊 Gerando visualização com PCA...")
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_bin)

# === Plot comparativo ===
fig, axs = plt.subplots(1, 2, figsize=(12, 5))
axs[0].scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='tab10', s=5)
axs[0].set_title("K-Modes - 2 Clusters")

axs[1].scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='bwr', s=5)
axs[1].set_title("Rótulos Reais")

for ax in axs:
    ax.set_xlabel("PCA 1")
    ax.set_ylabel("PCA 2")

plt.tight_layout()
nome_base = "-".join(args.namespaces)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
nome_arquivo_img = f"{args.saida}/kmodes_{nome_base}_{timestamp}.png"
plt.savefig(nome_arquivo_img, dpi=300)
print(f"📷 Gráfico salvo em: {nome_arquivo_img}")

# === Salvar rótulos e métricas ===
np.savez_compressed(f"{args.saida}/kmodes_labels_{nome_base}_{timestamp}.npz",
                    labels=labels, y=y, columns=colunas)

with open(f"{args.saida}/avaliacao_kmodes_{nome_base}_{timestamp}.txt", "w") as f:
    f.write(f"Namespaces: {args.namespaces}\n")
    f.write(f"Amostras por classe: classe0={args.classe0}, classe1={args.classe1}\n")
    f.write(f"ARI: {ari:.4f}\n")
    f.write(f"NMI: {nmi:.4f}\n")
print(f"📝 Avaliação salva.")
