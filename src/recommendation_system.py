"""
Recommendation System based on Node2Vec Embeddings
Author: Vaibhavi Kolipaka
"""
import numpy as np
from typing import Dict, List, Tuple

class RecommendationSystem:
    """
    Uses stock embeddings and correlation data to recommend diversified portfolios.
    """
    
    def __init__(self, embeddings: Dict[str, np.ndarray], corr_matrix: np.ndarray, nodes: List[str]):
        """
        :param embeddings: Dictionary of {stock_id: embedding_vector}.
        :param corr_matrix: The original correlation matrix (for diversification score calculation).
        :param nodes: List of stock IDs (ordered as in corr_matrix).
        """
        self.embeddings = embeddings
        self.corr_matrix = corr_matrix
        self.nodes = nodes
        self.node_to_index = {node: i for i, node in enumerate(self.nodes)}

    def calculate_cosine_similarity(self, u: str, v: str) -> float:
        """Calculates cosine similarity between two stock embeddings."""
        vec_u = self.embeddings.get(u)
        vec_v = self.embeddings.get(v)
        
        if vec_u is None or vec_v is None:
            return 0.0
            
        dot_product = np.dot(vec_u, vec_v)
        norm_u = np.linalg.norm(vec_u)
        norm_v = np.linalg.norm(vec_v)
        
        if norm_u == 0 or norm_v == 0:
            return 0.0
            
        return dot_product / (norm_u * norm_v)

    def calculate_average_embedding_similarity(self, graph) -> float:
        """
        Calculates the average cosine similarity between stocks and their direct neighbors.
        This is one of the 'Accuracy and Structure' metrics.
        """
        total_similarity = 0.0
        edge_count = 0
        
        for u in self.nodes:
            for v in graph.get_neighbors(u):
                # Ensure we only count each unique edge once (if graph is undirected)
                if self.node_to_index[u] < self.node_to_index[v]: 
                    total_similarity += self.calculate_cosine_similarity(u, v)
                    edge_count += 1
        
        return total_similarity / edge_count if edge_count > 0 else 0.0

    def recommend_diversified_portfolio(self, current_portfolio: List[str], k: int = 5) -> List[str]:
        """
        Recommends k new stocks that are maximally different (low similarity) from the
        current portfolio to increase diversification.
        """
        if not self.embeddings:
            return []

        candidates = list(set(self.nodes) - set(current_portfolio))
        if not candidates:
            return []

        # Calculate the average similarity of each candidate stock to the current portfolio
        candidate_dissimilarity = {}
        for candidate in candidates:
            similarity_sum = 0
            for stock in current_portfolio:
                similarity_sum += self.calculate_cosine_similarity(candidate, stock)
            
            avg_similarity = similarity_sum / len(current_portfolio)
            # Dissimilarity = 1 - avg_similarity (Maximizing dissimilarity)
            candidate_dissimilarity[candidate] = 1 - avg_similarity 

        # Select the top k stocks with the highest dissimilarity (most diversified)
        recommended = sorted(candidate_dissimilarity, key=candidate_dissimilarity.get, reverse=True)[:k]
        return recommended

    def calculate_diversification_score(self, portfolio: List[str]) -> float:
        """
        Calculates the Recommendation Accuracy metric: Average pairwise correlation 
        within the recommended portfolio. Lower score means better diversification.
        """
        if len(portfolio) < 2:
            return 0.0

        correlations = []
        indices = [self.node_to_index[stock] for stock in portfolio]
        
        # Calculate the average of all unique pairwise correlations (off-diagonal)
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                idx_i = indices[i]
                idx_j = indices[j]
                # Use the absolute value to measure risk/clustering regardless of direction
                correlations.append(abs(self.corr_matrix[idx_i, idx_j])) 

        return np.mean(correlations) if correlations else 0.0