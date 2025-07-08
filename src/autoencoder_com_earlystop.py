import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import datetime
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import copy

class DeepAutoencoder(nn.Module):
    def __init__(self, input_dim, bottleneck_dim=1500):
        super(DeepAutoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 11197),
            nn.LeakyReLU(0.01),
            nn.Linear(11197, 6000),
            nn.LeakyReLU(0.01),
            nn.Linear(6000, bottleneck_dim),
            nn.LeakyReLU(0.01)
        )

        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, 6000),
            nn.LeakyReLU(0.01),
            nn.Linear(6000, 11197),
            nn.LeakyReLU(0.01),
            nn.Linear(11197, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon, z

def treinar_autoencoder(X, input_dim, bottleneck_dim=1500, batch_size=128, num_epochs=10, device='cpu'):
    model = DeepAutoencoder(input_dim, bottleneck_dim).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    X = X.astype(np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    X = (X - X.min()) / (X.max() - X.min() + 1e-8)

    dataset = TensorDataset(torch.tensor(X))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=12, pin_memory=True)

    loss_por_epoca = []
    best_loss = float('inf')
    best_model = None
    paciencia = 5
    contador = 0

    print(f"Iniciando treinamento: {input_dim} → {bottleneck_dim}")
    try:
        for epoch in range(num_epochs):
            model.train()
            total_loss = 0.0

            for i, batch in enumerate(tqdm(loader, desc=f"Época {epoch+1}/{num_epochs}", unit="batch")):
                inputs = batch[0].to(device, non_blocking=True)
                optimizer.zero_grad()
                outputs, _ = model(inputs)
                loss = criterion(outputs, inputs)

                if torch.isnan(loss):
                    print(f"Loss virou NaN no batch {i+1}. Abortando.")
                    return model

                loss.backward()
                optimizer.step()
                total_loss += loss.item() * inputs.size(0)

            avg_loss = total_loss / len(dataset)
            loss_por_epoca.append(avg_loss)
            print(f"[{datetime.datetime.now()}] Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.6f}")

            if avg_loss < best_loss:
                best_loss = avg_loss
                best_model = copy.deepcopy(model)
                contador = 0
            else:
                contador += 1
                if contador >= paciencia:
                    print(f"Early stopping: perda não melhorou nas últimas {paciencia} épocas.")
                    break

        np.savetxt("loss_por_epoca.csv", loss_por_epoca, delimiter=",")
        print("Loss por época salva: 'loss_por_epoca.csv'")

        try:
            plt.figure()
            plt.plot(range(1, len(loss_por_epoca) + 1), loss_por_epoca, marker='o')
            plt.xlabel("Época")
            plt.ylabel("Loss (BCE)")
            plt.title("Curva de perda do Autoencoder")
            plt.grid(True)
            plt.savefig("curva_loss.png")
            plt.close()
            print("Gráfico da loss salvo em 'curva_loss.png'")
        except Exception as e:
            print("Não foi possível salvar gráfico da loss:", e)

    except Exception as e:
        print(f"Erro durante treinamento: {e}")

    return best_model

def extrair_embeddings(model, X, device='cpu', batch_size=512):
    model.eval()
    X = torch.tensor(X.astype(np.float32)).to(device)
    loader = DataLoader(X, batch_size=batch_size, num_workers=12)
    embeddings = []

    with torch.no_grad():
        for batch in loader:
            _, z = model(batch)
            embeddings.append(z.cpu().numpy())

    return np.vstack(embeddings)

if __name__ == "__main__":
    print("Carregando dados...")
    caminho_arquivo = os.path.join("..", "dados", "amostras_balanceadas_apicalls.npz")
    dados = np.load(caminho_arquivo, allow_pickle=True)
    X = dados['data']
    y = dados['classes']
    colunas = dados['column_names']

    print(f"Dados carregados: X={X.shape}, y={y.shape}")
    print(f"Classes únicas: {np.unique(y)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        print("GPU disponível! Usando CUDA.")
    else:
        print("GPU não disponível. Usando CPU.")
        raise RuntimeError("GPU não disponível. Treinamento não pode prosseguir.")

    print(f"Usando dispositivo: {device.upper()}")

    model = treinar_autoencoder(
        X, input_dim=X.shape[1], bottleneck_dim=3000,
        batch_size=384, num_epochs=20, device=device
    )

    print("Extraindo embeddings...")
    embeddings = extrair_embeddings(model, X, device=device)
    np.save("deep_embeddings.npy", embeddings)
    print(f"Embeddings salvos: 'deep_embeddings.npy' (shape: {embeddings.shape})")
