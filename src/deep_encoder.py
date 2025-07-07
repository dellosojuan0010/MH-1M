import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import datetime
import os
import matplotlib.pyplot as plt

class DeepAutoencoder(nn.Module):
    def __init__(self, input_dim, bottleneck_dim=700):
        super(DeepAutoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 8192),
            nn.ReLU(),
            nn.Linear(8192, 4096),
            nn.ReLU(),
            nn.Linear(4096, 2048),
            nn.ReLU(),
            nn.Linear(2048, bottleneck_dim),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, 2048),
            nn.ReLU(),
            nn.Linear(2048, 4096),
            nn.ReLU(),
            nn.Linear(4096, 8192),
            nn.ReLU(),
            nn.Linear(8192, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon, z

def treinar_autoencoder(X, input_dim, bottleneck_dim=700, batch_size=128, num_epochs=10, device='cpu'):
    model = DeepAutoencoder(input_dim, bottleneck_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    X = X.astype(np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    X = (X - X.min()) / (X.max() - X.min() + 1e-8)

    dataset = TensorDataset(torch.tensor(X))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    loss_por_epoca = []

    print(f"🚀 Treinando Deep Autoencoder: {input_dim} → {bottleneck_dim}")
    try:
        for epoch in range(num_epochs):
            model.train()
            total_loss = 0.0
            for i, batch in enumerate(loader):
                inputs = batch[0].to(device)
                optimizer.zero_grad()
                outputs, _ = model(inputs)
                loss = criterion(outputs, inputs)

                if torch.isnan(loss):
                    print(f"❌ Loss virou NaN no batch {i+1}. Abortando treinamento.")
                    return model

                loss.backward()
                optimizer.step()
                total_loss += loss.item() * inputs.size(0)

            avg_loss = total_loss / len(dataset)
            loss_por_epoca.append(avg_loss)
            print(f"[{datetime.datetime.now()}] 📅 Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.6f}")

        # 🔄 Salvar histórico da loss
        np.savetxt("loss_por_epoca.csv", loss_por_epoca, delimiter=",")
        print("✅ Loss por época salva em 'loss_por_epoca.csv'")

        try:
            plt.figure()
            plt.plot(range(1, num_epochs + 1), loss_por_epoca, marker='o')
            plt.xlabel("Época")
            plt.ylabel("Loss (MSE)")
            plt.title("Curva de perda do Autoencoder")
            plt.grid(True)
            plt.savefig("curva_loss.png")
            plt.close()
            print("📊 Gráfico da loss salvo em 'curva_loss.png'")
        except Exception as e:
            print("⚠️ Não foi possível salvar gráfico da loss:", e)

    except Exception as e:
        print(f"❌ Erro durante o treinamento: {e}")
    
    return model

def extrair_embeddings(model, X, device='cpu', batch_size=512):
    model.eval()
    X = torch.tensor(X.astype(np.float32)).to(device)
    loader = DataLoader(X, batch_size=batch_size, num_workers=0)
    embeddings = []

    with torch.no_grad():
        for i, batch in enumerate(loader):
            _, z = model(batch)
            embeddings.append(z.cpu().numpy())

    return np.vstack(embeddings)

# ========== USO ==========
if __name__ == "__main__":
    from dataset_selector import DatasetSelector
    ds = DatasetSelector()

    X, feature_names, y = ds.select_random_classes(['apicalls'], total_samples=119094)
    print(f"✅ Dados carregados: X={X.shape}, y={y.shape}")
    print(f"🔍 Classes únicas: {np.unique(y)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️ Usando dispositivo: {device.upper()}")

    model = treinar_autoencoder(
        X, input_dim=X.shape[1], bottleneck_dim=700,
        batch_size=64, num_epochs=20, device=device
    )

    print("🎯 Extraindo embeddings...")
    embeddings = extrair_embeddings(model, X, device=device)

    np.save("deep_embeddings.npy", embeddings)
    print(f"✅ Embeddings salvos! Shape: {embeddings.shape}")
