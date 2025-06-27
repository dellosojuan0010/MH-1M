import os
import sys
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split

# === Configura o path para importar DatasetSelector ===
diretorio_atual = os.path.dirname(__file__)
diretorio_raiz = os.path.abspath(os.path.join(diretorio_atual, '..'))
sys.path.append(diretorio_raiz)
from dataset_selector import DatasetSelector

# === Argumentos ===
parser = argparse.ArgumentParser(description="Treinamento com embeddings supervisionados (TF 1.15)")
parser.add_argument('--namespaces', nargs='+', required=True,
                    help='Namespaces de features: permissions, intents, opcodes, apicalls')
parser.add_argument('--embedding_dim', type=int, default=1000,
                    help='Tamanho da camada de embedding (padrão: 1000)')
parser.add_argument('--output_dir', type=str, default='resultados_embeddings',
                    help='Diretório de saída para os arquivos gerados')
args = parser.parse_args()

# === Carrega dados ===
ds = DatasetSelector()
X, y = ds.select_by_namespaces(args.namespaces)

print("✅ Dados carregados:", X.shape, "classes:", np.unique(y))

# === Divide os dados ===
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, stratify=y, random_state=42)

# === Importa TensorFlow 1.15 ===
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout

input_dim = X.shape[1]
embedding_dim = args.embedding_dim

# === Define o modelo ===
inputs = Input(shape=(input_dim,))
x = Dense(4096, activation='relu')(inputs)
x = Dropout(0.3)(x)
x = Dense(1024, activation='relu')(x)
embedding = Dense(embedding_dim, activation='linear', name='embedding_layer')(x)
x = Dense(128, activation='relu')(embedding)
output = Dense(1, activation='sigmoid')(x)

model = Model(inputs=inputs, outputs=output)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# === Treina o modelo ===
model.fit(X_train, y_train,
          epochs=30,
          batch_size=512,
          validation_data=(X_val, y_val),
          verbose=1)

# === Extrai embeddings para todo o conjunto ===
embedding_model = Model(inputs=model.input, outputs=model.get_layer('embedding_layer').output)
X_embeddings = embedding_model.predict(X, batch_size=512)

# === Salva os resultados ===
timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
os.makedirs(args.output_dir, exist_ok=True)
base_name = f"embeddings_{embedding_dim}d_{timestamp}"

np.save(os.path.join(args.output_dir, f"{base_name}.npy"), X_embeddings)
np.save(os.path.join(args.output_dir, f"{base_name}_labels.npy"), y)

# Também salva como CSV
df = pd.DataFrame(X_embeddings)
df['label'] = y
df.to_csv(os.path.join(args.output_dir, f"{base_name}.csv"), index=False)

print("✅ Embeddings salvos em:", args.output_dir)
