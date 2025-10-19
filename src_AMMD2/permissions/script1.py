# -*- coding: utf-8 -*-
# MLP + SHAP + seleção de Top-N + novo MLP (genérico: n_features, n_samples)
# Dependências: pip install shap tensorflow scikit-learn

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import shap

# ===================== PARÂMETROS =====================
N_FEATURES = 21      # nº de features (binárias)
N_SAMPLES  = 1000    # nº de amostras
N_TOP      = 5       # nº de features a selecionar
EPOCHS     = 80
BATCH_SIZE = 32
SEED       = 42
# ======================================================

np.random.seed(SEED)
tf.random.set_seed(SEED)

# 1) DADOS BINÁRIOS + RÓTULOS SINTÉTICOS
X = np.random.randint(0, 2, size=(N_SAMPLES, N_FEATURES)).astype(np.float32)

# Gera rótulos com uma relação não-trivial (linear + interação simples)
w = np.random.uniform(-1.0, 1.0, size=(N_FEATURES,))
logits = X @ w + 0.75*(X[:, :min(4, N_FEATURES)].sum(axis=1))  # leve viés nas primeiras
probs = 1/(1+np.exp(-logits))
y = (probs > np.median(probs)).astype(np.float32)  # balancear classes

# 2) MLP BASE
model = keras.Sequential([
    layers.Input(shape=(N_FEATURES,)),
    layers.Dense(32, activation='relu'),
    layers.Dense(16, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer=keras.optimizers.Adam(0.01),
              loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X, y, epochs=EPOCHS, batch_size=BATCH_SIZE,
          validation_split=0.2, verbose=0)

def f_predict(Xbatch):
    Xbatch = np.asarray(Xbatch, dtype=np.float32)
    return model.predict(Xbatch, verbose=0).ravel()

# 3) EXPLICABILIDADE: SHAP
# Para custo controlado: background reduzido com k-means
k_bg = min(50, X.shape[0])  # até 50 pontos de fundo
X_bg = shap.kmeans(X, k_bg)

# Tenta GradientExplainer (mais rápido p/ redes); se não der, cai p/ KernelExplainer
shap_values_subset = None
idx_explain = np.random.choice(X.shape[0], size=min(200, X.shape[0]), replace=False)
X_explain = X[idx_explain]

try:
    explainer = shap.GradientExplainer(model, X_bg)
    sv = explainer.shap_values(X_explain)
    # alguns wrappers retornam lista; normaliza para np.array shape (n, d)
    if isinstance(sv, list):
        sv = sv[0]
    shap_values_subset = np.array(sv)
except Exception:
    explainer = shap.KernelExplainer(f_predict, X_bg)
    # nsamples ~ proporcional ao nº de features p/ custo razoável
    ns = min(1000, 2 * N_FEATURES)
    sv = explainer.shap_values(X_explain, nsamples=ns)
    shap_values_subset = np.array(sv)

# Importância global = média do |SHAP| no subconjunto
importances = np.mean(np.abs(shap_values_subset), axis=0)

# 4) SELEÇÃO TOP-N
top_idx = np.argsort(importances)[::-1][:N_TOP]
print("Importâncias (|SHAP| médio):", np.round(importances, 4))
print(f"Top-{N_TOP} features (índices):", top_idx.tolist())

# 5) NOVO MLP COM FEATURES SELECIONADAS
X_red = X[:, top_idx]
model2 = keras.Sequential([
    layers.Input(shape=(N_TOP,)),
    layers.Dense(16, activation='relu'),
    layers.Dense(8, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])
model2.compile(optimizer=keras.optimizers.Adam(0.01),
               loss='binary_crossentropy', metrics=['accuracy'])
model2.fit(X_red, y, epochs=EPOCHS, batch_size=BATCH_SIZE,
           validation_split=0.2, verbose=0)

# 6) COMPARAÇÃO DE DESEMPENHO (no mesmo conjunto, demo)
acc_base = ((f_predict(X) >= 0.5).astype(np.float32) == y).mean()
acc_red  = ((model2.predict(X_red, verbose=0).ravel() >= 0.5).astype(np.float32) == y).mean()
print(f"Acurácia MLP base (todas features): {acc_base:.3f}")
print(f"Acurácia MLP (Top-{N_TOP} via SHAP): {acc_red:.3f}")
