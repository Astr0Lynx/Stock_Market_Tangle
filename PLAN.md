The division groups related algorithms logically to minimize overlap and maximize efficiency. Algorithms are assigned as follows:
Member 1: Union-Find and BFS (connected components and traversal).
Member 2: DFS and PageRank (traversal and centrality).
Member 3: Girvan-Newman (community detection).
Member 4: Louvain (community detection).
Member 5: Node2Vec (embeddings/recommendations).
Additional tasks like data generation, simulations, and report integration are distributed to balance workload. All code must be in Python (or agreed language), modular (one file per algorithm + helpers), with docstrings, comments, and no external libraries for core logic (e.g., no NetworkX for algorithm implementations, but okay for basic data structures or synthetic data generation). Use a shared GitHub repo for version control.



Step 1: Overall Team Setup and Shared Responsibilities
Team Lead Role: Rotate weekly (e.g., start with Avani Sood) to coordinate meetings, track progress, and ensure deadlines are met.

Shared Tasks (All Members):
Weekly meetings (2-3 hours) to integrate code, review progress, and ensure consistency (e.g., uniform coding style, shared helper functions for graph representation).
High-level review of each other's code and report sections for cross-knowledge.
GitHub management: Commit frequently with clear messages; use branches for individual work.
Reproducibility: Contribute to the README.md (setup instructions, dependencies like numpy/pandas for data, how to run benchmarks).
Testing: Write unit tests for your algorithms; contribute to a shared benchmarking script (e.g., measure runtime/memory with timeit/psutil across scenarios).
Presentation: All prepare slides for their sections; practice as a group for the 15-min delivery.
Bonus Disclosure: As a group, decide on bonuses (e.g., advanced metrics like modularity stability in crashes) and document in the report.

Timeline Suggestion (assuming 4-6 weeks to deadline):
Week 1: Data generation, algorithm implementations.
Week 2: Simulations, metrics, visualizations.
Week 3: Comparisons, report drafting.
Week 4: Integration, testing, presentation prep, final reviews.



Step 2: Individual Assignments
Each member is primarily responsible for 1-2 algorithms (implementation from scratch, theoretical explanation, complexity analysis, implementation details). They also handle related experiments/metrics, report subsections, and presentation slides. For algorithms, implement core logic manually (e.g., Union-Find with path compression/union-by-rank; no pre-built functions). Use adjacency lists for graphs (e.g., dict of dicts in Python).


Guntesh Singh (Focus: Data Foundation and Connected Components/Traversals):
Code Implementation:
Generate synthetic data (stock returns, correlations, traits like volatility/stability) using numpy/pandas.
Build the initial graph (nodes with attributes, edges based on correlation threshold).
Implement Union-Find from scratch (for market segments/connected components).
Implement BFS from scratch (for shortest paths/correlation chains, prioritizing low-volatility routes).
Create helper functions for graph representation (shared with team).
Experimental Setup and Results:
Define datasets (synthetic graphs of varying sizes/densities).
Run benchmarks for Union-Find and BFS (e.g., runtime on stable/volatile graphs).
Measure metrics: Number of components (Union-Find), average path length (BFS).
Report Sections:
Lead Introduction (problem definition, objectives, real-world relevance like portfolio optimization).
Algorithm Descriptions and Implementation Details for Union-Find and BFS (theory, O-time/space, data structures like queues/lists).
Experimental Setup (hardware/software, datasets).
Presentation:
Slides on intro, data generation, Union-Find/BFS results (graphs/charts).
Rationale for Assignment: Handles foundational data, ensuring the graph is ready for others. ~20% of algorithms.


Khushi Dhingra (Focus: Traversals and Centrality):
Code Implementation:
Implement DFS from scratch (for clusters/dependencies, e.g., recursive or stack-based).
Implement PageRank from scratch (iterative random walk with damping, weighted by correlations).
Integrate disruption simulations (stable/volatile/crash scenarios) into the benchmarking harness.
Experimental Setup and Results:
Simulate market disruptions (edge removals based on volatility).
Benchmarks for DFS and PageRank (e.g., cluster sizes, rank stability across scenarios).
Metrics: Wall-clock time, memory usage, influence scores (PageRank).
Report Sections:
Algorithm Descriptions and Implementation Details for DFS and PageRank (theory, complexities like O(V+E) for DFS).
Part of Results & Analysis: Compare DFS/BFS traversals, PageRank influence in disruptions.
Presentation:
Slides on DFS/PageRank overviews, disruption simulations, key findings (e.g., charts of centrality changes).
Rationale for Assignment: Builds on traversals (pairing with BFS) and adds influence analysis. ~20% of algorithms.


Avani Sood (Focus: Community Detection - Girvan-Newman):
Code Implementation:
Implement Girvan-Newman from scratch (betweenness centrality calculation, edge removal for clusters).
Add modularity calculation as a metric (implement manually or use basic math).
Experimental Setup and Results:
Run on all scenarios; compare with Louvain.
Metrics: Modularity score, community sizes, bridge stocks identified.
Efficiency: Runtime/scalability for varying graph sizes.
Report Sections:
Algorithm Descriptions and Implementation Details for Girvan-Newman (theory, O((V+E)^2) complexity challenges).
Part of Results & Analysis: Sector cohesion, accuracy in detecting clusters vs. theory.
Presentation:
Slides on Girvan-Newman overview, community results (visualizations like network layouts).
Rationale for Assignment: Specialized in one complex algorithm; focuses on depth. ~15% of algorithms.


Saanvi Jain (Focus: Community Detection - Louvain):
Code Implementation:
Implement Louvain from scratch (modularity optimization, node grouping phases).
Adjust for edge weights (tag similarity/correlation).
Experimental Setup and Results:
Run on all scenarios; compare with Girvan-Newman.
Metrics: Modularity, community density, financial relevance (e.g., sector merging in volatility).
Accuracy: Path metrics if integrated with traversals.
Report Sections:
Algorithm Descriptions and Implementation Details for Louvain (theory, heuristic complexity).
Part of Results & Analysis: Trade-offs with Girvan-Newman (speed vs. accuracy).
Presentation:
Slides on Louvain overview, comparisons with other community algos.
Rationale for Assignment: Pairs with Avani for community focus; allows direct comparison. ~15% of algorithms.


Vaibhavi Kolipaka (Focus: Embeddings, Recommendations, and Wrap-Up):
Code Implementation:
Implement Node2Vec from scratch (biased random walks, embeddings via skip-gram-like model; use numpy for vectors, no full ML libs for core).
Create recommendation system (similarity-based portfolios using embeddings).
Lead the test/benchmarking harness (scripts to run all algos, generate data for reports).
Experimental Setup and Results:
Dynamic adjustments for market cycles.
Metrics: Recommendation accuracy (e.g., diversification score), embedding similarity.
Overall comparisons: Aggregate efficiency/accuracy/financial relevance across all algos.
Report Sections:
Algorithm Descriptions and Implementation Details for Node2Vec (theory, O(walks * length) complexity).
Lead Conclusion (summaries, limitations like synthetic data biases, future ideas).
Results & Analysis: Full algorithm comparisons (tables/graphs for 3 dimensions).
References and Bonus Disclosure.
Presentation:
Slides on Node2Vec/recommendations, overall comparisons, conclusion.
Rationale for Assignment: Handles the most advanced algo; wraps up analysis. ~15% of algorithms.
Step 3: Integration and Quality Checks
Code Integration: Guntesh provides the base graph; others pull it and add their algos. Use a main.py to run everything.
Report Compilation: Each writes their sections; Vaibhavi compiles into a single PDF (use LaTeX or Word for professionalism). Aim for 15-20 pages.
Visualizations: Use matplotlib for graphs/charts (e.g., runtime vs. graph size); assign based on sections (e.g., Khushi/Saanvi for community visuals).
Risk Mitigation: If someone falls behind, reassign minor tasks. Test code early for "from-scratch" adherence.
Bonus Opportunities: Implement extra metrics (e.g., centrality stability); label clearly in report/presentation.
This division ensures coverage of all requirements, balanced workloads (~20% each), and preparedness for evaluation phases. Adjust based on strengths (e.g., swap if someone prefers a different algo).

