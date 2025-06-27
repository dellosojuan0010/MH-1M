import numpy as np

class DatasetSelector:
    def __init__(self):
        caminho_arquivo = "../dados/amex-1M-[intents-permissions-opcodes-apicalls].npz"
        self.dados = np.load(caminho_arquivo, allow_pickle=True)
        self.data = self.dados['data']
        self.column_names = self.dados['column_names']
        self.classes = self.dados['classes']
        
        self.namespace_indices = {}
        for i, nome in enumerate(self.column_names):
            if "::" in nome:
                ns = nome.split("::")[0]
                if ns not in self.namespace_indices:
                    self.namespace_indices[ns] = []
                self.namespace_indices[ns].append(i)

    def get_available_namespaces(self):
        return sorted(self.namespace_indices.keys())

    def get_data_by_namespaces(self, namespaces):
        if isinstance(namespaces, str):
            namespaces = [namespaces]
        
        indices = []
        for ns in namespaces:
            if ns not in self.namespace_indices:
                raise ValueError(f"Namespace '{ns}' não encontrado. Use: {self.get_available_namespaces()}")
            indices.extend(self.namespace_indices[ns])
        
        X = self.data[:, indices]
        cols = self.column_names[indices]
        return X, cols, self.classes

    def select_random_classes(self, namespaces, total_samples=1000, random_state=42):
        rng = np.random.default_rng(seed=random_state)
        X, cols, y = self.get_data_by_namespaces(namespaces)
        
        classes_unicas = np.unique(y)
        if len(classes_unicas) != 2:
            raise ValueError("O dataset deve conter exatamente duas classes.")

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
