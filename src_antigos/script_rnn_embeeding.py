import os
import sys
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split

# === Caminho para DatasetSelector ===
diretorio_atual = os.path.dirname(__file__)
diretorio_raiz = os.path.abspath(os.path.join(diretorio_atual, '..'))
sys.path.append(diretorio_raiz)
from dataset_selector import DatasetSelector

# === Argumentos ===
parser = argparse.ArgumentParser(description="Treinamento de rede supervisionada com camada de embeddings")
parser.add_argument('--namespaces', nargs='+', required=True,
                    help='Lista de namespaces a serem utilizados (ex: permissions intents opcodes apicalls)')
parser.add_argument('--embedding_dim', type=int, default=1000,
                    help='Tamanho da camada de embedding (default: 1000)')
parser.add_argument('--output_dir', type=str, default='resultados_embeddings',
                    help='Diretório onde os arquivos serão salvos')
args = parser.parse_args()

# === Carrega os dados ===
ds = DatasetSelector()
X, y = ds.get_data_by_namespaces(args.namespaces)

print(f"Total de amostras: {X.shape[0]}, Total de features: {X.shape[1]}")

# === Divide os dados para treinamento ===
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, stratify=y, random_state=42)

# === Modelo com camada de embedding ===
input_dim = X.shape[1]
embedding_dim = args.embedding_dim

inputs = Input(shape=(input_dim,))
x = Dense(4096, activation='relu')(inputs)
x = Dropout(0.3)(x)
x = Dense(1024, activation='relu')(x)
embedding = Dense(embedding_dim, activation='linear', name='embedding_layer')(x)
x = Dense(128, activation='relu')(embedding)
output = Dense(1, activation='sigmoid')(x)

model = Model(inputs=inputs, outputs=output)
model.compile(optimizer=Adam(1e-4), loss='binary_crossentropy', metrics=['accuracy'])

# === Treinamento ===
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

model.fit(X_train, y_train,
          validation_data=(X_val, y_val),
          epochs=30,
          batch_size=512,
          callbacks=[early_stop],
          verbose=3)

# === Extração dos embeddings ===
embedding_model = Model(inputs=model.input, outputs=model.get_layer('embedding_layer').output)
X_embeddings = embedding_model.predict(X, batch_size=512)

# === Salva os resultados ===
timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
os.makedirs(args.output_dir, exist_ok=True)
base_name = f"embeddings_{embedding_dim}d_{timestamp}"

# Arquivos
np.save(os.path.join(args.output_dir, f"{base_name}.npy"), X_embeddings)
np.save(os.path.join(args.output_dir, f"{base_name}_labels.npy"), y)

# Também salva como CSV
df = pd.DataFrame(X_embeddings)
df['label'] = y
df.to_csv(os.path.join(args.output_dir, f"{base_name}.csv"), index=False)

print(f"Embeddings salvos em: {args.output_dir}/{base_name}.[npy/csv]")
