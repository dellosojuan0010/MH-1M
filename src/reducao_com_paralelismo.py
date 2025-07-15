import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm
import multiprocessing

# Exemplo com poucas linhas (substitua por seu X real)
# X = np.load("seuarquivo.npz")['data']
#X = np.random.randint(0, 10, size=(1000, 20))  # cuidado com tamanhos grandes!

dados = np.load('dados_filtrados.npz', allow_pickle=True,mmap_mode='r')
X = dados['data']
y = dados['classes']
colunas = dados['column_names']

def encontrar_duplicatas_para_i(i, X, existentes):
    duplicados = []
    for j in range(i + 1, X.shape[0]):
        if j in existentes:
            continue
        if np.array_equal(X[i], X[j]):
            duplicados.append(j)
    return duplicados

if __name__ == "__main__":
    num_cores = multiprocessing.cpu_count()

    # Lista paralela dos resultados
    resultados = Parallel(n_jobs=num_cores)(
        delayed(encontrar_duplicatas_para_i)(i, X, set())  # set vazio por simplicidade
        for i in tqdm(range(X.shape[0]), desc="Procurando duplicatas")
    )

    # Flatten e remover repetições
    indices_duplicados = sorted(set(j for sublist in resultados for j in sublist))
    
    np.save("indices_duplicados.npy", np.array(indices_duplicados))
    print(f"\n🔍 Total de duplicatas encontradas: {len(indices_duplicados)}")
    print("Exemplos de índices duplicados:", indices_duplicados[:10])
