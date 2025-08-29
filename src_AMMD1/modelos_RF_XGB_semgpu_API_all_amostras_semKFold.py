# Parte 1 - Importação das bibliotecas

import os
import gc
from datetime import datetime

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier
# from sklearn.svm import SVC
from xgboost import XGBClassifier


# Parte 7 - Executar os modelos

agora = datetime.now().strftime('%d%m%Y_%H%M')
pasta_saida = os.path.join("..", "resultados", "todas_as_amostras", f"resultado_RF_XGB_API_{agora}")
os.makedirs(pasta_saida, exist_ok=True)

# Parte 2 - Abertura do arquivo, recuperação dos dados e embaralhamento

CAMINHO_ARQUIVO = os.path.join("..", "dados", "amex-1M-[intents-permissions-opcodes-apicalls].npz")
print("Carregando arquivo...")
dados = np.load(CAMINHO_ARQUIVO, allow_pickle=True, mmap_mode='r')

print("Extraindo dados...")
X = dados['data']
y = dados['classes']
colunas = dados['column_names']

print("Embaralhando...")
#rng = np.random.default_rng(42)
#idx_final = rng.permutation(X.shape[0])
#X = X[idx_final]
#y = y[idx_final]

print(f"Dados embaralhados: X={X.shape}, y={y.shape}")

# Parte 3 - Separar as colunas das features e criar os DataFrames
print("Separando os grupos de features...")
idx_permissions = [i for i, nome in enumerate(colunas) if nome.startswith("permissions::")]
idx_intents     = [i for i, nome in enumerate(colunas) if nome.startswith("intents::")]
idx_opcodes     = [i for i, nome in enumerate(colunas) if nome.startswith("opcodes::")]
idx_apicalls    = [i for i, nome in enumerate(colunas) if nome.startswith("apicalls::")]

print("Criando os Dataframes")
# df_all  = pd.DataFrame(X, columns=colunas)
# df_all['classe'] = y

# df_p    = pd.DataFrame(X[:, idx_permissions], columns=np.array(colunas)[idx_permissions])
# df_p['classe'] = y

# df_i    = pd.DataFrame(X[:, idx_intents], columns=np.array(colunas)[idx_intents])
# df_i['classe'] = y

# df_op   = pd.DataFrame(X[:, idx_opcodes], columns=np.array(colunas)[idx_opcodes])
# df_op['classe'] = y

df_api  = pd.DataFrame(X[:, idx_apicalls], columns=np.array(colunas)[idx_apicalls])
df_api['classe'] = y

print("DataFrames criados:")
# print(f" - df_all: {df_all.shape}")
# print(f" - df_p  : {df_p.shape}")
# print(f" - df_i  : {df_i.shape}")
# print(f" - df_op : {df_op.shape}")
print(f" - df_api: {df_api.shape}")


# Parte 5 - Função para definir modelos
def definir_modelos_sklearn(input_dim):
    if input_dim > 20000:
        # svm = SVC(kernel='rbf', C=0.5, gamma='scale', probability=True, random_state=42, verbose=True)
        rf  = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42, n_jobs=-1)
        xgb = XGBClassifier(n_estimators=300, max_depth=15, learning_rate=0.05, verbosity=2, use_label_encoder=False, random_state=42, n_jobs=-1)
    elif input_dim > 400:
        # svm = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42, verbose=True)
        rf  = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
        xgb = XGBClassifier(n_estimators=50, max_depth=10, learning_rate=0.05, verbosity=2, use_label_encoder=False, random_state=42, n_jobs=-1)
    else:
        # svm = SVC(kernel='rbf', C=2.0, gamma='auto', probability=True, random_state=42, verbose=True)
        rf  = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
        xgb = XGBClassifier(n_estimators=25, max_depth=6, learning_rate=0.05, verbosity=2, use_label_encoder=False, random_state=42, n_jobs=-1)

    # return {'SVM': svm, 'RandomForest': rf, 'XGBoost': xgb}
    return {'RandomForest': rf, 'XGBoost': xgb}


# Parte 6 - Função de avaliação com matriz de confusão
def avaliar_modelos_em_dataframe(df, nome_grupo, test_size=0.2):

    X = df.drop(columns=['classe']).values.astype(np.int8)
    y = df['classe'].astype(np.int8).values
    input_dim = X.shape[1]

    print(f"\nIniciando avaliação para o grupo de features: '{nome_grupo}' com {X.shape[1]} atributos e {X.shape[0]} instâncias.")

    # Divisão simples treino/teste
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=42
    )

    modelos = definir_modelos_sklearn(input_dim)
    resultados = []

    for modelo_nome, modelo in modelos.items():
        print(f"\nTreinando modelo: {modelo_nome}")
        modelo.fit(X_train, y_train)
        y_pred = modelo.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test.astype(int), y_pred.astype(int), output_dict=True, zero_division=0)
        print(report)

        # Matriz de confusão
        cm = confusion_matrix(y_test.astype(int), y_pred.astype(int))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Benigno', 'Malware'])
        disp.plot(cmap='Blues')
        plt.title(f"Matriz de Confusão - {nome_grupo} - {modelo_nome}")
        plt.tight_layout()
        plt.savefig(os.path.join(pasta_saida, f'matriz_confusao_{nome_grupo}_{modelo_nome}.png'))
        plt.close()

        for classe in ['0', '1']:
            resultados.append({
                'grupo_de_features': nome_grupo,
                'modelo': modelo_nome,
                'classe': classe,
                'precision': report[classe]['precision'],
                'recall': report[classe]['recall'],
                'f1_score': report[classe]['f1-score'],
                'support': report[classe]['support'],
                'fold': 1,
                'accuracy_geral': acc
            })

    print(f"Finalizado modelo {modelo_nome} para '{nome_grupo}'")
    print(f"Avaliação concluída para o grupo: {nome_grupo}")
    return pd.DataFrame(resultados)


# # Parte 7 - Executar os modelos

# agora = datetime.now().strftime('%d%m%Y_%H%M')
# pasta_saida = os.path.join("..", "resultados", "todas_as_amostras", f"resultado_RF_XGB_{agora}")
# os.makedirs(pasta_saida, exist_ok=True)

df_resultados = pd.concat([
    # avaliar_modelos_em_dataframe(df_p, 'permissions'),
    # avaliar_modelos_em_dataframe(df_i, 'intents'),
    # avaliar_modelos_em_dataframe(df_op, 'opcodes'),
    avaliar_modelos_em_dataframe(df_api, 'apicalls')
    # avaliar_modelos_em_dataframe(df_all, 'all'),
], ignore_index=True)


# Parte 8 - Exportar e printar os resultados

caminho_saida = os.path.join(pasta_saida, 'resultados_modelos.csv')
df_resultados.to_csv(caminho_saida, index=False)

resumo = df_resultados.groupby(['grupo_de_features', 'modelo', 'classe'])[['precision', 'recall', 'f1_score']].mean().round(4)
resumo.to_csv(os.path.join(pasta_saida, 'resumo_resultados.csv'))

print("\nResumo final (médias por grupo, modelo e classe):")
print(resumo)
