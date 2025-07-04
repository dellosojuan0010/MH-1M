import numpy as np

class DatasetSelector:
    def __init__(self):
        # Abrir arquivo
        caminho_arquivo = "../dados/amex-1M-[intents-permissions-opcodes-apicalls].npz"
        self.dados = np.load(caminho_arquivo, allow_pickle=True)
        # Capturando os dados
        self.data = self.dados['data']
        self.column_names = self.dados['column_names']
        self.classes = self.dados['classes']
        
        # Criação de um dicionário para possa manter os índices das features indicado por namespaces.
        # Exemplo: self.namespaces_indices = {'permissions':[0, 1, 2, 3, 4], 'intents':[5, 6, 7]}
        self.namespace_indices = {}
        
        # Junta em uma lista  combinada (zip) com as sequeências de índices (0,1, 2, 3 ...) e os nomes das colunas 
        for i, nome in zip(range(len(self.column_names)), self.column_names):
            # Se não tiver a "::" forçar um erro para não confundir quais features estão ou não sendo utilizadas.
            if "::" not in nome:
                raise ValueError(f"Coluna '{nome}' não segue o padrão de namespace com '::'")
            # Pega apenas o namespaces para ser utilizado a chave no dicionário namespace_indices
            ns = nome.split("::")[0]
            # Se não tem um namespace ainda no dicionário namespaces_indices adiciona uma chave com uma lista vazia
            if ns not in self.namespace_indices:
                self.namespace_indices[ns] = []
            # Adiciona o indice da feature visitada na lista do dicionário
            self.namespace_indices[ns].append(i)

    # Retorna as chaves do dicionário namespaces_indices (grupos de features)
    def get_available_namespaces(self):
        return sorted(self.namespace_indices.keys())

    # Pega os dados a partir de namespaces fornecidos
    def get_data_by_namespaces(self, namespaces):
        # Cria uma lista de indices vazia para guardar os indices das colunas que serão utilizadas
        indices = []
        for ns in namespaces:
            # Se tiver um namespace não mapeado gera interrupção do algoritmo
            if ns not in self.namespace_indices:
                raise ValueError(f"Namespace '{ns}' não encontrado. Use: {self.get_available_namespaces()}")            
            # Concatena os índices de dados que serão utilizados
            indices.extend(self.namespace_indices[ns])
        # Atribue em X todas as linhas de todos os indices existentes na lista de índices
        X = self.data[:, indices]
        # Pega os nomes das features originais mas somentes das colunas selecionadas para X
        cols = self.column_names[indices]
        # Retorna todas as classes (saídas / atributo alvo) de todas as instâncias
        return X, cols, self.classes

    # Pega um montante de amostras
    def select_random_classes(self, namespaces, total_samples=1000, random_state=42):
        # Objeto com funções para gerar escolhas aleatórias e embaralhamento com base em uma semente
        rng = np.random.default_rng(seed=random_state)
        
        # Captura todas amostras de namespaces especificados e retornado por get_data_by_namespaces
        X, cols, y = self.get_data_by_namespaces(namespaces)
        
        # goodware == 0
        # malware == 1
        # classe_unicas = [0, 1]
        classes_unicas = np.unique(y)
        # Se por acaso não for detectado exatamente duas classes será considerando um problema e o algoritmo irá parar
        if len(classes_unicas) != 2:
            print(classes_unicas)
            #raise ValueError("O dataset deve conter exatamente duas classes.")
        
        # a quantidade de amostras deve ser igual para cada classe
        amostras_por_classe = total_samples // 2
        idx_class_0 = np.where(y == classes_unicas[0])[0]
        idx_class_1 = np.where(y == classes_unicas[1])[0]

        if len(idx_class_0) < amostras_por_classe or len(idx_class_1) < amostras_por_classe:
            raise ValueError("Não há instâncias suficientes em uma das classes para o total solicitado.")

        idx_0 = rng.choice(idx_class_0, amostras_por_classe, replace=False)
        idx_1 = rng.choice(idx_class_1, amostras_por_classe, replace=False)

        indices_finais = np.concatenate([idx_0, idx_1])
        rng.shuffle(indices_finais)

        return X[indices_finais], cols, y[indices_finais]

    def select_random_classes_custom(self, namespaces, samples_per_class: dict, random_state=42):
        """
        Retorna subconjuntos aleatórios com quantidades específicas por classe.

        Parâmetros:
        - namespaces: lista de namespaces a serem usados
        - samples_per_class: dicionário no formato {classe: quantidade}
        - random_state: semente para reprodutibilidade

        Retorna: X_selected, colunas, y_selected
        """
        rng = np.random.default_rng(seed=random_state)
        X, cols, y = self.get_data_by_namespaces(namespaces)

        indices_selecionados = []

        for classe, qtd in samples_per_class.items():
            idx_classe = np.where(y == classe)[0]
            if len(idx_classe) < qtd:
                raise ValueError(f"Classe {classe} possui apenas {len(idx_classe)} instâncias disponíveis, mas {qtd} foram solicitadas.")
            idx_amostrados = rng.choice(idx_classe, qtd, replace=False)
            indices_selecionados.extend(idx_amostrados)

        rng.shuffle(indices_selecionados)
        return X[indices_selecionados], cols, y[indices_selecionados]
