
# Script com XGBoost incremental em minilotes (sem RandomForest)

# Também morreu com o recurso que temos

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay

from xgboost import XGBClassifier

# Parte 1 - Parâmetros
CAMINHO_ARQUIVO = os.path.join("..", "dados", "amex-1M-[intents-permissions-opcodes-apicalls].npz")
agora = datetime.now().strftime('%d%m%Y_%H%M')
pasta_saida = os.path.join("..", "resultados", "incremental", f"resultado_XGB_INCREMENTAL_API_{agora}")
os.makedirs(pasta_saida, exist_ok=True)

# Parte 2 - Carregamento dos dados
print("Carregando arquivo...")
dados = np.load(CAMINHO_ARQUIVO, allow_pickle=True, mmap_mode='r')
X = dados['data']
y = dados['classes']
colunas = dados['column_names']

print("Separando features de apicalls...")
idx_apicalls = [i for i, nome in enumerate(colunas) if nome.startswith("apicalls::")]
X_apicalls = X[:, idx_apicalls]

print("Dividindo treino e teste...")
X_train, X_test, y_train, y_test = train_test_split(X_apicalls, y, test_size=0.2, stratify=y, random_state=42)

# Parte 3 - Treinamento incremental
def treinar_xgb_em_lotes(X_train, y_train, lote=100000):
    modelo = None
    for i in range(0, X_train.shape[0], lote):
        fim = min(i + lote, X_train.shape[0])
        print(f"Treinando lote {i} até {fim}")
        x_lote = X_train[i:fim]
        y_lote = y_train[i:fim]

        modelo = XGBClassifier(
            n_estimators=50,
            max_depth=10,
            learning_rate=0.05,
            verbosity=1,
            use_label_encoder=False,
            random_state=42,
            n_jobs=-1
        )
        modelo.fit(x_lote, y_lote, xgb_model=modelo if i > 0 else None)

    return modelo

print("Treinando modelo XGBoost com minilotes...")
modelo = treinar_xgb_em_lotes(X_train, y_train)

# Parte 4 - Avaliação
print("Avaliando modelo...")
y_pred = modelo.predict(X_test)
acc = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
print(report)

# Matriz de confusão
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Benigno', 'Malware'])
disp.plot(cmap='Blues')
plt.title("Matriz de Confusão - apicalls - XGB Incremental")
plt.tight_layout()
plt.savefig(os.path.join(pasta_saida, 'matriz_confusao_apicalls_xgb_incremental.png'))
plt.close()

# Parte 5 - Salvamento dos resultados
df_resultados = []
for classe in ['0', '1']:
    df_resultados.append({
        'grupo_de_features': 'apicalls',
        'modelo': 'XGBoost_Incremental',
        'classe': classe,
        'precision': report[classe]['precision'],
        'recall': report[classe]['recall'],
        'f1_score': report[classe]['f1-score'],
        'support': report[classe]['support'],
        'accuracy_geral': acc
    })
df_resultados = pd.DataFrame(df_resultados)
df_resultados.to_csv(os.path.join(pasta_saida, 'resultados_modelo_incremental.csv'), index=False)

print("Finalizado com sucesso.")
