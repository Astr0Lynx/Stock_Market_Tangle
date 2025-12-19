# Stock Market Tangle 📈

> A comprehensive graph algorithm analysis framework for modeling stock market correlation networks and detecting market fragmentation in real-time.

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## 🎯 Overview

Stock Market Tangle implements and benchmarks **7 graph algorithms** to analyze stock correlation networks across different market conditions (stable, volatile, crash, normal). The project demonstrates practical applications of graph theory in financial analysis, particularly for:

- **Real-time fragmentation detection** using Union-Find (sub-millisecond performance)
- **Contagion risk assessment** via BFS shortest paths
- **Market influence ranking** with PageRank
- **Community detection** using Louvain and Girvan-Newman methods
- **Stock similarity embeddings** through Node2Vec

### Key Results

| Algorithm | Runtime (500 stocks) | Use Case | Complexity |
|-----------|---------------------|----------|------------|
| **DFS** | 1-5ms | Component discovery | O(V+E) |
| **Union-Find** | 0.7-1.3ms | Fragmentation detection | O(α(n)) ≈ O(1) |
| **BFS** | 2-25ms | Contagion paths | O(V+E) |
| **PageRank** | 8-105ms | Influence ranking | O(k·E) |
| **Louvain** | 20-100ms | Community detection | O(V log V) |
| **Node2Vec** | 1.7-24s | Embeddings | O(walks·length) |
| **Girvan-Newman** | 150ms+ | Hierarchical clustering | O(V²E) |

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- pip package manager

### Installation

```bash
# Clone repository
git clone https://github.com/Astr0Lynx/Stock_Market_Tangle.git
cd Stock_Market_Tangle

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Running Algorithms

```bash
# Generate sample stock correlation data
python src/data_generation.py

# Run individual algorithms
python src/union_find.py      # Fastest - fragmentation detection
python src/bfs.py              # Shortest paths / contagion
python src/dfs.py              # Component discovery
python src/pagerank.py         # Market influence ranking
python src/louvain.py          # Community detection
python src/node2vec.py         # Stock embeddings
python src/girvan_newman.py    # Hierarchical clustering
```

### Benchmarking & Visualization

```bash
# Run comprehensive benchmarks (60+ test scenarios)
python benchmarks.py

# Generate performance visualizations
python visualize_results.py

# View results in results/ directory
ls results/*/  # Algorithm-specific charts and metrics
```

---

## 📊 Features

### 1. **Algorithm Implementations**
- **Union-Find**: Path compression + union by rank optimization
- **BFS**: Multi-source shortest path analysis
- **DFS**: Connected component traversal
- **PageRank**: Iterative influence propagation
- **Louvain**: Modularity-optimized community detection
- **Girvan-Newman**: Edge betweenness-based divisive clustering
- **Node2Vec**: Biased random walks + Skip-gram embeddings

### 2. **Benchmarking System**
- 3 graph sizes: 100, 250, 500 stocks
- 4 market scenarios: stable, normal, volatile, crash
- Precision timing with `time.perf_counter()`
- Memory profiling with `psutil`
- Automated JSON result export

### 3. **Visualizations**
- Runtime comparison (2×2 subplot grids)
- Memory usage analysis
- Component distribution charts
- Graph density plots
- Scalability curves (linear vs quadratic)
- Summary tables with quality metrics

### 4. **Market Scenarios**
| Scenario | Threshold | Description | Expected Structure |
|----------|-----------|-------------|-------------------|
| **Crash** | 0.10 | Panic selling | 1 giant component |
| **Stable** | 0.40 | Normal correlation | Few large sectors |
| **Normal** | 0.44 | Moderate fragmentation | ~40-70 components |
| **Volatile** | 0.68 | High uncertainty | Near-complete isolation |

---

## 📁 Repository Structure

```
Stock_Market_Tangle/
├── src/                      # Algorithm implementations
│   ├── union_find.py         # Disjoint-set union (O(α(n)))
│   ├── bfs.py                # Breadth-first search
│   ├── dfs.py                # Depth-first search
│   ├── pagerank.py           # Influence ranking
│   ├── louvain.py            # Community detection
│   ├── node2vec.py           # Graph embeddings
│   ├── girvan_newman.py      # Hierarchical clustering
│   ├── data_generation.py    # Stock correlation generator
│   └── graph.py              # Graph data structure
├── results/                  # Benchmark outputs & visualizations
│   ├── union_find/           # Algorithm-specific charts
│   ├── bfs/
│   └── ...
├── benchmarks.py             # Comprehensive benchmark runner
├── visualize_results.py      # Chart generation system
├── analysis_report.md        # Detailed performance analysis
├── TESTCASES.md              # Standardized test specifications
└── requirements.txt          # Python dependencies
```

---

## 🧪 Testing

```bash
# Run all test cases
pytest

# Run specific algorithm tests
pytest src/test_union_find.py

# View standardized test cases
cat TESTCASES.md
```

---

## 📈 Performance Analysis

### Scalability Results (Stable Market, 500 stocks)

**Speed Champions (≤10ms):**
- DFS: 5.0ms - Component discovery
- Union-Find: 1.2ms - Fragmentation detection  
- BFS: 25ms - Contagion analysis

**Production Ready (10-100ms):**
- PageRank: 10-105ms - Converges in 13-68 iterations
- Louvain: 20-100ms - Near-linear scaling

**Batch Processing (>100ms):**
- Node2Vec: 1.7-24s - Training-dominated
- Girvan-Newman: 150ms+ - O(V²E) limits use

### Memory Efficiency
All algorithms maintain **<1MB memory overhead** for 500-stock graphs, making them suitable for embedded/real-time systems.

For detailed analysis, see [analysis_report.md](analysis_report.md).

---

## 🎓 Educational Value

This project demonstrates:
- **Algorithm complexity analysis**: O(1) vs O(n) vs O(n²) practical differences
- **Optimization techniques**: Path compression, union by rank, lazy evaluation
- **Benchmarking best practices**: Reproducible timing, memory profiling, statistical analysis
- **Real-world graph applications**: Financial networks, social networks, infrastructure modeling

---

## 📚 Documentation

- **[TESTCASES.md](TESTCASES.md)** - Standardized test specifications for all algorithms
- **[analysis_report.md](analysis_report.md)** - Comprehensive performance analysis with tables and insights

---

## 🛠️ Tech Stack

- **Python 3.8+**: Core language
- **NumPy**: Numerical operations & matrix computations
- **Matplotlib**: Visualization generation
- **psutil**: Memory profiling
- **pytest**: Testing framework

---

## 🤝 Contributing

Contributions welcome! Areas for enhancement:
- Additional graph algorithms (Dijkstra, A*, Kruskal)
- GPU acceleration for large graphs
- Interactive visualization dashboard
- Real-time stock data integration

---

## 📄 License

This project is licensed under the MIT License - see LICENSE file for details.

---

## 👤 Author

**Guntesh Singh**  
- GitHub: [@Astr0Lynx](https://github.com/Astr0Lynx)
- Project: Algorithm Analysis & Design Course Work

---

## 🙏 Acknowledgments

Developed as part of the **Algorithm Analysis and Design** course project (Team: Queue-ties).

Special thanks to collaborators: Avani, Khushi, Saanvi, Vaibhavi for algorithm implementations and testing contributions.
course-project-queue-ties/
├── src/                          # Shared modules (main branch)
│   ├── data_generation.py        # Stock data generator (Guntesh)
│   └── graph.py                  # Graph representation (Guntesh)
├── benchmarks/                   # Benchmark scripts
├── tests/                        # Unit tests
├── results/                      # Benchmark results (gitignored)
├── visualize_results.py          # Dynamic visualization script
├── USAGE_GUIDE.md                # Usage instructions
├── TESTCASES.md                  # Standardized test cases
└── requirements.txt              # Python dependencies
```

---

## Team Members & Algorithms

- **Guntesh Singh** - Data Foundation, Union-Find, BFS
- **Avani Sood** - Girvan-Newman
- **Khushi Dhingra** - DFS, PageRank
- **Saanvi Jain** - Louvain
- **Vaibhavi Kolipaka** - Node2Vec, Recommendations

---

## Notes 

- Each algorithm is implemented in a separate file to ensure modularity.
- The repository includes detailed documentation for setup, usage, and testing.
- Please refer to the `USAGE_GUIDE.md` for instructions on running benchmarks and visualizing results.
- Test cases are standardized and can be found in `TESTCASES.md`.
- If you encounter any issues, feel free to reach out to the team members listed above.

