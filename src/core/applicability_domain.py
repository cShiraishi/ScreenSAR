import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

class ApplicabilityDomain:
    def __init__(self, k_neighbors=5, z_threshold=3.0, metric='jaccard'):
        """
        Inicializa o calculador de Domínio de Aplicabilidade.
        
        Args:
            k_neighbors (int): Número de vizinhos para calcular distância média.
            z_threshold (float): Limite Z-score para definir outliers e cutoff.
            metric (str): Métrica de distância ('jaccard' para bits fingerprints, 'euclidean' para contínuos).
                          Nota: scikit-learn usa 'jaccard' como dissimilaridade, perfeito para Tanimoto Distance.
        """
        self.k = k_neighbors
        self.z = z_threshold
        self.metric = metric
        self.model_nn = None
        self.threshold_AD = None
        self.mean_dist_train = None
        self.std_dist_train = None
        self.training_distances = None

    def fit(self, X):
        """
        Treina o AD com os dados de treinamento (Fingerprints).
        
        Args:
            X (array-like): Matriz de fingerprints do treino.
        """
        X = np.array(X)
        
        # 1. Ajustar Nearest Neighbors no próprio treino
        self.model_nn = NearestNeighbors(n_neighbors=self.k + 1, metric=self.metric, n_jobs=-1)
        self.model_nn.fit(X)
        
        # 2. Calcular distâncias para os k vizinhos (excluindo o próprio ponto -> índice 0)
        # kneighbors retorna distâncias ordenadas
        distances, _ = self.model_nn.kneighbors(X)
        
        # Média das distâncias para os k vizinhos mais próximos (colunas 1 a k)
        # Coluna 0 é a distância para si mesmo (0.0)
        mean_distances = np.mean(distances[:, 1:], axis=1)
        
        self.training_distances = mean_distances
        self.mean_dist_train = np.mean(mean_distances)
        self.std_dist_train = np.std(mean_distances)
        
        # 3. Definir Limiar de Aplicabilidade (AD Threshold)
        # D_crit = <D> + Z * std(D)
        self.threshold_AD = self.mean_dist_train + (self.z * self.std_dist_train)
        
        return self

    def detect_outliers(self):
        """
        Identifica outliers no conjunto de treinamento baseado no limiar calculado.
        
        Returns:
            list: Índices dos outliers no array original X usado no fit.
        """
        if self.training_distances is None:
            raise ValueError("O modelo AD ainda não foi treinado. Execute .fit(X) primeiro.")
            
        outlier_indices = np.where(self.training_distances > self.threshold_AD)[0]
        return outlier_indices

    def predict(self, X_new):
        """
        Verifica se novos compostos estão dentro do domínio.
        
        Args:
            X_new (array-like): Fingerprints dos novos compostos.
            
        Returns:
            tuple: (is_inside_array, distances_array)
                   is_inside_array: Boolean array (True=Inside, False=Outside)
                   distances_array: Average distances to k nearest neighbors in training set
        """
        if self.model_nn is None:
            raise ValueError("O modelo AD ainda não foi treinado.")
            
        X_new = np.array(X_new)
        
        # Para novos pontos, buscamos os K vizinhos no treino
        distances, _ = self.model_nn.kneighbors(X_new, n_neighbors=self.k)
        
        # Média das k distâncias
        mean_dists = np.mean(distances, axis=1)
        
        # Verificar condição
        is_inside = mean_dists <= self.threshold_AD
        
        return is_inside, mean_dists
