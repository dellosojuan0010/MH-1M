import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
import os

class AutoencoderEmbedding:
    def __init__(self, input_dim, bottleneck_dim=6000, hidden_ratio=0.5, max_iter=100):
        self.input_dim = input_dim
        self.bottleneck_dim = bottleneck_dim
        self.hidden_dim = int(input_dim * hidden_ratio)
        self.max_iter = max_iter
        self.scaler = MinMaxScaler()
        self.model = MLPRegressor(
            hidden_layer_sizes=(self.hidden_dim, self.bottleneck_dim, self.hidden_dim),
            activation='relu',
            solver='adam',
            max_iter=self.max_iter,
            random_state=42
        )

    def fit(self, X):
        print("🔄 Normalizando os dados...")
        X_scaled = self.scaler.fit_transform(X)

        print("🚀 Treinando autoencoder...")
        self.model.fit(X_scaled, X_scaled)

        self.X_scaled = X_scaled
        print("✅ Treinamento concluído.")

    def transform(self, X=None):
        if X is None:
            X = self.X_scaled
        else:
            X = self.scaler.transform(X)

        # Camada oculta 1
        W1, b1 = self.model.coefs_[0], self.model.intercepts_[0]
        H1 = np.maximum(0, np.dot(X, W1) + b1)  # ReLU

        # Bottleneck
        W2, b2 = self.model.coefs_[1], self.model.intercepts_[1]
        Z = np.maximum(0, np.dot(H1, W2) + b2)  # ReLU

        return Z

    def save_embeddings(self, embeddings, path_npy="embeddings.npy", path_csv="embeddings.csv"):
        print(f"💾 Salvando embeddings em '{path_npy}' e '{path_csv}'...")
        np.save(path_npy, embeddings)
        pd.DataFrame(embeddings).to_csv(path_csv, index=False)
        print("✅ Embeddings salvos com sucesso.")

# =================== USO ===================

if __name__ == "__main__":
    # Exemplo: use seus dados reais aqui
    n_amostras = 1000
    n_features = 22394
    X = np.random.rand(n_amostras, n_features)

    autoenc = AutoencoderEmbedding(input_dim=n_features, bottleneck_dim=6000, max_iter=100)
    autoenc.fit(X)

    embeddings = autoenc.transform()

    print("\n✅ Embeddings gerados:")
    print("Shape:", embeddings.shape)

    # Salvando os embeddings
    autoenc.save_embeddings(embeddings, path_npy="meus_embeddings.npy", path_csv="meus_embeddings.csv")
