
from pathlib import Path
import os
import pandas as pd
import numpy as np

import csv


modelo_nome = 'XGBoost'
nome_grupo = 'apicalls'
n_folds = 1
CAMINHO_ARQUIVO = os.path.join("..", "dados", "dados_undersampling_duplicados_eliminados.npz")
CAMINHO_ARQUIVO_RANK = os.path.join(".",f"{nome_grupo}","SHAP",)

# Carrega os dados com mmap_mode para uso mais leve de memória
dados = np.load(CAMINHO_ARQUIVO, allow_pickle=True)

# Extração dos arrays principais
# X = dados['data']
# y = dados['classes']
colunas = dados['column_names']

idx_features_importantes_folds = []
feature_names_importantes_folds = []

for i in range(n_folds):
  fold = i+1
  arquivo = os.path.join(CAMINHO_ARQUIVO_RANK,f"rank_importancias_{modelo_nome}_shap_fold{fold}.csv")

  df = pd.read_csv(arquivo)
  feature_names = df['feature_name'].values
  importancias = df['valor_importancia_media'].values

  idx_features_importantes = np.where(importancias > 0)[0]

  feature_names_importantes = feature_names[idx_features_importantes]

  indices_gerais = [colunas.tolist().index(feature_name) for feature_name in feature_names_importantes]
  idx_features_importantes_folds.append(indices_gerais)
  feature_names_importantes_folds.append(feature_names_importantes)

  np.savez_compressed(
          os.path.join(CAMINHO_ARQUIVO_RANK,f'importancias_finais_com_idx_{modelo_nome}_{nome_grupo}_fold{fold}.npz'),
          feature_names=feature_names_importantes,
          importancias=importancias[idx_features_importantes],
          indices_gerais=indices_gerais
  )

  idx_features_importantes_unicos = set()

for lista in idx_features_importantes_folds:
    idx_features_importantes_unicos.update(lista)

idxs = sorted(idx_features_importantes_unicos)

arquivo = os.path.join(CAMINHO_ARQUIVO_RANK,f"importantes_unicas_{modelo_nome}_{nome_grupo}_shap_folds.csv")

with open(arquivo, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["idx"])  # Cabeçalho opcional
    for v in idxs:
        writer.writerow([v])

