import numpy as np
import os

# Caminho relativo para o arquivo
CAMINHO_ARQUIVO = os.path.join("..", "dados", "amex-1M-[intents-permissions-opcodes-apicalls].npz")

# Carregar o arquivo
dados = np.load(CAMINHO_ARQUIVO, allow_pickle=True)

# Verificar chaves
assert 'data' in dados and 'column_names' in dados, "Arquivo .npz deve conter 'data' e 'column_names'"

X = dados['data']
colunas = dados['column_names']

# Verificar quais colunas pertencem ao namespace 'apicalls'
colunas_apicalls = [i for i, nome in enumerate(colunas) if nome.startswith("apicalls::")]

# Verificar quais colunas têm apenas zeros
colunas_zeros = []
for i in colunas_apicalls:
    if np.all(X[:, i] == 0):
        colunas_zeros.append(colunas[i])

# Exibir resultados
print(f"Total de colunas no namespace 'apicalls': {len(colunas_apicalls)}")
print(f"Colunas 'apicalls' com apenas zeros: {len(colunas_zeros)}")
print("\nNomes das colunas:")
for nome in colunas_zeros:
    print(nome)
