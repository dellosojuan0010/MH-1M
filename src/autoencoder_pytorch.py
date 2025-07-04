import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os

class Autoencoder(nn.Module):
    def __init__(self, input_dim, bottleneck_dim=6000, hidden_ratio=0.5):
        super(Autoencoder, self).__init__()
        hidden_dim = int(input_dim * hidden_ratio)

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, bottleneck_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon, z

def treinar_autoencoder(X, input_dim, bottleneck_dim=6000, hidden_ratio=0.5, batch_size=128, num_epochs=10, device='cpu'):
    model = Autoencoder(input_dim, bottleneck_dim, hidden_ratio).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Normaliza
    X = X.astype(np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    X = (X - X.min()) / (X.max() - X.min() + 1e-8)

    # Cria DataLoader
    dataset = TensorDataset(torch.tensor(X))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    print(f"🚀 Treinando autoencoder: {input_dim} → {bottleneck_dim}")
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        for batch in loader:
            inputs = batch[0].to(device)
            optimizer.zero_grad()
            outputs, _ = model(inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * inputs.size(0)

        avg_loss = total_loss / len(dataset)
        print(f"📅 Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.6f}")

    return model

def extrair_embeddings(model, X, device='cpu', batch_size=512):
    model.eval()
    X = torch.tensor(X.astype(np.float32)).to(device)
    loader = DataLoader(X, batch_size=batch_size)
    embeddings = []

    with torch.no_grad():
        for batch in loader:
            _, z = model(batch)
            embeddings.append(z.cpu().numpy())

    return np.vstack(embeddings)

# ========== USO ==========
if __name__ == "__main__":

    # === Carregar dados ===
    from dataset_selector import DatasetSelector
    ds = DatasetSelector()
    #X, feature_names, y = ds.get_data_by_namespaces(['apicalls'])
    X, feature_names, y = ds.select_random_classes(['apicalls'],total_samples=119094)

    print(f"Quantidade de cada classe: {np.unique(y)}")

    model = treinar_autoencoder(
        X, input_dim=X.shape[1], bottleneck_dim=6000,
        hidden_ratio=0.3, batch_size=256, num_epochs=100
    )

    print("🎯 Extraindo embeddings...")
    embeddings = extrair_embeddings(model, X)

    np.save("embeddings.npy", embeddings)
    print(f"✅ Embeddings salvos! Shape: {embeddings.shape}")
