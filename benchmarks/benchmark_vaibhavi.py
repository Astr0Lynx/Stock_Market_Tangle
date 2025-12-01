"""
Benchmark script for Node2Vec and Recommendation System
Author: Vaibhavi Kolipaka
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import time
import psutil
import json
from typing import Dict, List, Tuple
import numpy as np

# --- 1. Import Shared and Custom Modules ---
from data_generation import StockDataGenerator
from graph import build_graph_from_correlation
from node2vec import Node2Vec
from recommendation_system import RecommendationSystem

# --- 2. Helper Functions ---
def get_memory_usage():
    """Get current memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def get_base_portfolio(nodes: List[str], size: int = 5) -> List[str]:
    """Generates a random base portfolio for recommendation testing."""
    if len(nodes) < size:
        return nodes
    return np.random.choice(nodes, size, replace=False).tolist()

# --- 3. Benchmark Runner ---
def main():
    """Run Node2Vec Benchmarks across all scenarios."""
    
    print("\n" + "="*70)
    print("VAIBHAVI'S NODE2VEC & RECOMMENDATION BENCHMARKS")
    print("="*70 + "\n")
    
    # Setup
    generator = StockDataGenerator(seed=42)
    sizes = [50, 100, 200]
    scenarios = ["stable", "normal", "volatile", "crash"]
    results = []
    
    # Configuration for Node2Vec (Fixed for benchmarking, could be varied for custom analysis)
    P_N2V = 1.0
    Q_N2V = 0.5 # Tendency to explore outside local cluster
    EMBEDDING_DIM = 64
    
    for size in sizes:
        for scenario in scenarios:
            
            print(f"\n{'='*70}")
            print(f"Testing: {size} stocks, {scenario} scenario | Dim: {EMBEDDING_DIM}")
            print('='*70)
            
            # --- Data Generation and Graph Building (Shared Responsibility) ---
            returns, corr_matrix, stock_attrs = generator.generate_dataset(size, scenario=scenario)
            threshold = 0.5 if scenario != "crash" else 0.3
            graph = build_graph_from_correlation(corr_matrix, stock_attrs, threshold)
            nodes = list(graph.get_nodes())
            
            # --- 1. Benchmark Node2Vec Embedding ---
            print("  1. Running Node2Vec Embeddings...")
            mem_before = get_memory_usage()
            start_time = time.time()
            
            n2v = Node2Vec(graph, p=P_N2V, q=Q_N2V, embedding_dim=EMBEDDING_DIM)
            embeddings = n2v.learn_embeddings()
            
            end_time = time.time()
            mem_after = get_memory_usage()
            
            runtime_n2v = end_time - start_time
            memory_n2v = mem_after - mem_before

            # --- 2. Calculate Recommendation Metrics ---
            print("  2. Calculating Recommendation Metrics...")
            recommender = RecommendationSystem(embeddings, corr_matrix, nodes)
            
            # Metric 1: Accuracy and Structure (Embedding Similarity)
            avg_sim = recommender.calculate_average_embedding_similarity(graph)
            
            # Metric 2: Financial Relevance (Diversification Score)
            # Use a random 10% of the nodes as the starting portfolio
            start_portfolio = get_base_portfolio(nodes, size=max(2, size // 10)) 
            recommended_stocks = recommender.recommend_diversified_portfolio(start_portfolio, k=5)
            
            # The lower the score, the better the recommendation (lower internal correlation)
            diversification_score = recommender.calculate_diversification_score(recommended_stocks) 
            
            # --- 3. Record Results ---
            results.append({
                'algorithm': 'Node2Vec',
                'scenario': scenario,
                'num_stocks': size,
                'num_edges': graph.num_edges,
                'runtime_seconds': runtime_n2v,
                'memory_mb': memory_n2v,
                'embedding_dim': EMBEDDING_DIM,
                'avg_embedding_similarity': avg_sim,
                'recommendation_accuracy': diversification_score, # Lower is better
            })
            
            print(f"    ✓ Runtime: {runtime_n2v*1000:.2f}ms | Memory: {memory_n2v:.2f}MB")
            print(f"    ✓ Avg Embedding Similarity (Accuracy): {avg_sim:.4f}")
            print(f"    ✓ Diversification Score (Financial Relevance): {diversification_score:.4f} (Lower is better)")
            
    # --- 4. Save Results ---
    os.makedirs('results', exist_ok=True)
    with open('results/vaibhavi_benchmarks.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n✓ Benchmarks complete! Results saved to results/vaibhavi_benchmarks.json")
    print("Run 'python visualize_results.py' to generate charts.")

if __name__ == "__main__":
    main()