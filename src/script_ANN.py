#!/usr/bin/env python3
import sys, os, argparse, time
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    accuracy_score, precision_score, recall_score, f1_score
)

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

# Path para DatasetSelector
diretorio_atual = os.path.dirname(__file__)
diretorio_raiz = os.path.abspath(os.path.join(diretorio_atual, '..'))
sys.path.append(diretorio_raiz)
from dataset_selector import DatasetSelector

# Diretório base para resultados
resultados_dir = os.path.join(diretorio_raiz, 'resultados')
os.makedirs(resultados_dir, exist_ok=True)

# Argumentos
parser = argparse.ArgumentParser(description="Treinamento MLP com seleção de namespaces")
parser.add_argument('--namespaces', nargs='+', required=True,
                    help='Lista de namespaces a serem utilizados')
parser.add_argument('--top_features', type=int, default=-1,
                    help='Número de features mais importantes a serem usadas (use -1 para não usar)')
parser.add_argument('--epochs', type=int, default=50,
                    help='Número de épocas de treinamento')
parser.add_argument('--batch_size', type=int, default=32,
                    help='Tamanho do batch')
args = parser.parse_args()

print("\n=== CONFIGURAÇÃO ===")
print(f"Namespaces: {args.namespaces}")
print(f"Top features: {args.top_features}")
print(f"Epochs: {args.epochs}, Batch size: {args.batch_size}")
print("====================\n")

# Carregamento de dados
ds = DatasetSelector()
X, feature_names, y = ds.get_data_by_namespaces(args.namespaces)
y = y.astype(int)

# Escalonamento
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Seleção top features (se solicitado)
if args.top_features > 0:
    variancias = np.var(X, axis=0)
    idx = np.argsort(variancias)[-args.top_features:]
    X = X[:, idx]
    feature_names = feature_names[idx]

# Divisão treino/validação/teste
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.25, random_state=42, stratify=y_train_val)

# Construção do modelo MLP
def build_mlp(input_dim):
    model = models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )
    return model

model = build_mlp(X_train.shape[1])
model.summary()

# Verificar disponibilidade de GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"GPUs disponíveis: {[gpu.name for gpu in gpus]}")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
else:
    print("Nenhuma GPU encontrada, usando CPU.")

# Callbacks
es = callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
mc = callbacks.ModelCheckpoint(
    filepath=os.path.join(resultados_dir, 'best_mlp.h5'),
    monitor='val_loss', save_best_only=True
)

# Treinamento
start = time.time()
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=args.epochs,
    batch_size=args.batch_size,
    callbacks=[es, mc],
    verbose=1  # Progress bar por época
)
end = time.time()
print(f"\nTreino concluído em {end-start:.2f}s")

# Curva de aprendizado
plt.figure()
plt.plot(history.history['loss'], label='Treino')
plt.plot(history.history['val_loss'], label='Validação')
plt.xlabel('Época')
plt.ylabel('Loss')
plt.title('Curva de Aprendizado - MLP')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(resultados_dir, f'curva_mlp_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'))
plt.close()

# Avaliação final
y_pred_prob = model.predict(X_test).ravel()
y_pred = (y_pred_prob >= 0.5).astype(int)

print("\n--- Relatório de Classificação ---")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=['Benigno', 'Malware'])
disp.plot(cmap=plt.cm.Blues)
plt.title('Matriz de Confusão - MLP')
plt.tight_layout()
plt.savefig(os.path.join(resultados_dir, f'matriz_confusao_mlp_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'))
plt.close()

# Salvamento de resultados
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
tag = "_".join(args.namespaces)
out_dir = os.path.join(resultados_dir, f"{tag}_MLP_{timestamp}")
os.makedirs(out_dir, exist_ok=True)

# Salva modelo
model.save(os.path.join(out_dir, 'mlp_model.h5'))

# Salva relatório txt
with open(os.path.join(out_dir, 'relatorio_mlp.txt'), 'w') as f:
    f.write(f"Treino MLP - {timestamp}\n")
    f.write(f"Namespaces: {args.namespaces}\n")
    f.write(f"Epochs: {len(history.history['loss'])}\n")
    f.write(classification_report(y_test, y_pred))

# Salva métricas em CSV
metrics_dict = {
    'Accuracy': [accuracy_score(y_test, y_pred)],
    'Precision': [precision_score(y_test, y_pred)],
    'Recall': [recall_score(y_test, y_pred)],
    'F1': [f1_score(y_test, y_pred)]
}
pd.DataFrame(metrics_dict).to_csv(os.path.join(out_dir, 'metricas_mlp.csv'), index=False)

print(f"\nResultados salvos em {out_dir}")
