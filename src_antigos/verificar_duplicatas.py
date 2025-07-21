# verificar_duplicatas_minilotes.py

import sys
import os
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

# === Caminho para DatasetSelector ===
diretorio_atual = os.path.dirname(__file__)
diretorio_raiz = os.path.abspath(os.path.join(diretorio_atual, '..'))
sys.path.append(diretorio_raiz)
from dados.dataset_selector import DatasetSelector

# === Argumentos ===
parser = argparse.ArgumentParser(description="Verificação de duplicatas por minilote")
parser.add_argument('--namespaces', nargs='+', required=True,
                    help='Namespaces usados (ex: permissions intents)')
parser.add_argument('--minilote_tamanho', type=int, default=50000,
                    help='Tamanho de cada minilote')
args = parser.parse_args()

# === Carregamento dos dados ===
ds = DatasetSelector()
X, feature_names, y = ds.get_data_by_namespaces(args.namespaces)
y = y.astype(int)
X = X.astype(np.float32)

print(f"Total de amostras: {X.shape[0]}, Features: {X.shape[1]}")

# === Divisão em minilotes balanceados ===
sss = StratifiedShuffleSplit(n_splits=int(X.shape[0] / args.minilote_tamanho),
                             train_size=args.minilote_tamanho,
                             random_state=42)

# === Diretório de saída ===
saida_dir = os.path.join(diretorio_raiz, 'resultados', f"duplicatas_minilotes_{'_'.join(args.namespaces)}")
os.makedirs(saida_dir, exist_ok=True)

# === Processamento por minilote ===
for i, (train_index, _) in enumerate(sss.split(X, y), 1):
    print(f"\n>> Minilote {i}")
    X_lote = X[train_index]
    y_lote = y[train_index]
    
    # Combina features + rótulo
    Xy_lote = np.hstack((X_lote, y_lote.reshape(-1, 1)))
    Xy_view = Xy_lote.view([('', Xy_lote.dtype)] * Xy_lote.shape[1])
    
    _, idx, counts = np.unique(Xy_view, return_index=True, return_counts=True)
    duplicadas = Xy_lote[idx[counts > 1]]
    repeticoes = counts[counts > 1]

    if len(duplicadas) > 0:
        df = pd.DataFrame(duplicadas)
        df.columns = [*feature_names, 'class']
        df['repeticoes'] = repeticoes

        nome_csv = os.path.join(saida_dir, f"duplicatas_lote{i}.csv")
        df.to_csv(nome_csv, index=False)
        print(f"  - {len(df)} duplicatas salvas em: {nome_csv}")
    else:
        print("  - Nenhuma duplicata encontrada.")

print("\nFinalizado.")
