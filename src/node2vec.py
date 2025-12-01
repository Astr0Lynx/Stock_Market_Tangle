"""
Node2Vec Implementation for Stock Network Embeddings
Author: Vaibhavi Kolipaka
"""
import numpy as np
import random
from typing import Dict, List, Tuple

# Assume the Graph object from src/graph.py is available.
# Graph class should provide methods: get_nodes(), get_neighbors(node), get_weight(u, v)

class Node2Vec:
    """
    Implements Node2Vec with biased random walks and a skip-gram-like embedding layer.
    """
    
    def __init__(self, graph, walk_length: int = 80, num_walks: int = 10, 
                 p: float = 1.0, q: float = 1.0, embedding_dim: int = 64, 
                 window_size: int = 10, epochs: int = 5, learning_rate: float = 0.01):
        """
        :param graph: The correlation Graph object.
        :param walk_length: Length of each random walk.
        :param num_walks: Number of walks starting from each node.
        :param p: Return parameter (inverse of likelihood of immediately revisiting a node).
        :param q: In-out parameter (inverse of likelihood of exploring unvisited parts).
        :param embedding_dim: Dimensionality of the stock embeddings.
        :param window_size: Context window size for skip-gram training.
        :param epochs: Number of training iterations.
        :param learning_rate: Learning rate for embedding updates.
        """
        self.graph = graph
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.p = p
        self.q = q
        self.dim = embedding_dim
        self.window_size = window_size
        self.epochs = epochs
        self.lr = learning_rate
        self.nodes = list(graph.get_nodes())
        self.node_to_index = {node: i for i, node in enumerate(self.nodes)}
        self.num_nodes = len(self.nodes)

        # Initialize embeddings randomly (using numpy)
        self.embeddings = np.random.uniform(-1, 1, (self.num_nodes, self.dim))
        # Initialize context embeddings (output layer weights)
        self.context_weights = np.random.uniform(-1, 1, (self.num_nodes, self.dim))

    def _get_transition_prob(self, current_node: str, previous_node: str) -> Dict[str, float]:
        """
        Calculates the biased transition probabilities for the next step (next_node).
        """
        probs = {}
        for neighbor in self.graph.get_neighbors(current_node):
            weight = self.graph.get_weight(current_node, neighbor) # Correlation weight
            
            # Distance d_tx: 0 (if neighbor == prev), 1 (if neighbor is connected to prev), 2 (otherwise)
            if neighbor == previous_node:
                # Return parameter p
                alpha = 1 / self.p
            elif self.graph.has_edge(previous_node, neighbor):
                # Distance 1
                alpha = 1.0
            else:
                # Distance 2 (In-out parameter q)
                alpha = 1 / self.q
            
            probs[neighbor] = alpha * weight
        
        # Normalize probabilities
        total = sum(probs.values())
        if total == 0:
            return {n: 0.0 for n in self.graph.get_neighbors(current_node)}
            
        return {n: prob / total for n, prob in probs.items()}

    def _biased_random_walk(self, start_node: str) -> List[str]:
        """Generates a single biased random walk."""
        walk = [start_node]
        
        while len(walk) < self.walk_length:
            current = walk[-1]
            neighbors = self.graph.get_neighbors(current)
            
            if not neighbors:
                break
            
            # Determine previous node for bias calculation
            previous = walk[-2] if len(walk) > 1 else None
            
            if previous is None:
                # First step, uniform distribution based on edge weight
                weights = [self.graph.get_weight(current, n) for n in neighbors]
                probs = np.array(weights) / sum(weights) if sum(weights) > 0 else None
                next_node = np.random.choice(neighbors, p=probs)
            else:
                # Biased step
                transition_probs = self._get_transition_prob(current, previous)
                nodes, probs = zip(*transition_probs.items())
                probs = np.array(probs)
                
                # Sample the next node
                next_node = np.random.choice(nodes, p=probs)
                
            walk.append(next_node)
            
        return walk

    def generate_walks(self) -> List[List[str]]:
        """Generates the full corpus of random walks."""
        all_walks = []
        for _ in range(self.num_walks):
            # Shuffle nodes to ensure walks start from different points
            random.shuffle(self.nodes)
            for node in self.nodes:
                all_walks.append(self._biased_random_walk(node))
        return all_walks

    def _skip_gram_update(self, target_idx: int, context_idx: int):
        """
        Performs a single step of gradient descent for the skip-gram model.
        Uses Negative Sampling (simplified/simulated for this implementation).
        """
        v_t = self.embeddings[target_idx]
        u_c = self.context_weights[context_idx]

        # 1. Calculate Score and Error (Simplified Sigmoid Loss)
        score = np.dot(v_t, u_c)
        # Use simple sigmoid activation and error for demonstration
        prediction = 1.0 / (1.0 + np.exp(-score))
        error = 1.0 - prediction # Target is 1 for context words
        
        # 2. Gradient Calculation (simulated)
        # Gradient of V_t (target embedding)
        grad_v_t = error * u_c
        # Gradient of U_c (context weight)
        grad_u_c = error * v_t

        # 3. Update Embeddings (using stochastic gradient descent)
        self.embeddings[target_idx] += self.lr * grad_v_t
        self.context_weights[context_idx] += self.lr * grad_u_c
        
        # NOTE: For a complete implementation, this would involve negative sampling
        # which requires multiple updates (1 positive, K negative samples) per step.

    def learn_embeddings(self) -> Dict[str, np.ndarray]:
        """Trains the embeddings using the generated walks and simplified skip-gram."""
        
        # 1. Generate corpus
        walks = self.generate_walks()
        
        # 2. Train embeddings
        for _ in range(self.epochs):
            random.shuffle(walks)
            for walk in walks:
                for i, target_node in enumerate(walk):
                    target_idx = self.node_to_index[target_node]
                    
                    # Iterate through context window
                    for j in range(max(0, i - self.window_size), min(len(walk), i + self.window_size + 1)):
                        if i == j:
                            continue
                        
                        context_node = walk[j]
                        context_idx = self.node_to_index[context_node]
                        
                        self._skip_gram_update(target_idx, context_idx)
                        
        # 3. Return final embeddings
        final_embeddings = {node: self.embeddings[self.node_to_index[node]] for node in self.nodes}
        return final_embeddings