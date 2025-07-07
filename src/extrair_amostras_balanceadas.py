import numpy as np
import pandas as pd
import os

# Caminho do arquivo de entrada
CAMINHO_ARQUIVO = os.path.join("..", "dados", "amex-1M-[intents-permissions-opcodes-apicalls].npz")

print("🔄 Carregando arquivo .npz...")
dados = np.load(CAMINHO_ARQUIVO, allow_pickle=True)

X = dados['data']
y = dados['classes']
colunas = dados['column_names']

print("✅ Arquivo carregado com sucesso.")
print(f"📐 Shape dos dados: X = {X.shape}, y = {y.shape}")
print(f"📊 Número total de instâncias: {len(y)}")
print(f"🔍 Valores únicos em y (classes): {np.unique(y)}")

# Contagem de cada classe
valores, contagens = np.unique(y, return_counts=True)
for valor, contagem in zip(valores, contagens):
    classe = "Malware" if valor == 1 else "Não Malware"
    print(f"   - {classe} ({valor}): {contagem} instâncias")

# Índices por classe
idx_malware = np.where(y == 1)[0]
idx_nao_malware = np.where(y == 0)[0]

print(f"\n🔢 Malware encontrados: {len(idx_malware)}")
print(f"🔢 Não-malware encontrados: {len(idx_nao_malware)}")

# Amostragem balanceada
qtd = min(len(idx_malware), len(idx_nao_malware))
print(f"\n⚖️ Selecionando {qtd} instâncias de cada classe para balanceamento...")

rng = np.random.default_rng(seed=42)
idx_nao_malware_sample = rng.choice(idx_nao_malware, size=qtd, replace=False)
idx_malware_sample = idx_malware[:qtd]

# Combina e embaralha
idx_total = np.concatenate([idx_malware_sample, idx_malware_sample])
rng.shuffle(idx_total)

# Dados finais
X_bal = X[idx_total]
y_bal = y[idx_total]
feature_names = np.array(colunas)

print(f"📐 Novo shape dos dados balanceados: X = {X_bal.shape}, y = {y_bal.shape}")
print("💾 Salvando arrays .npz...")

np.savez("amostras_balanceadas.npz", data=X_bal, classes=y_bal, column_names=feature_names)
print("✅ Arquivo salvo: amostras_balanceadas.npz")
