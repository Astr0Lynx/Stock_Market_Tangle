"""
Visualization Script for Benchmark Results
Author: Guntesh Singh
Description: Generate charts and graphs from benchmark data for report and presentation

Usage:
    python visualize_results.py                    # Auto-detect all algorithms
    python visualize_results.py <algorithm_name>   # Filter by specific algorithm
    
Examples:
    python visualize_results.py union_find
    python visualize_results.py girvan_newman
    python visualize_results.py bfs
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import os
import sys


def normalize_algorithm_name(name):
    """Normalize algorithm name for matching (lowercase, replace spaces/hyphens with underscores)."""
    return name.lower().replace(' ', '_').replace('-', '_')


def load_benchmark_results(filepath=None, algorithm_filter=None):
    """Load benchmark results from JSON file.

    If `filepath` is None, look for algorithm-specific file first (e.g., results/<algorithm>_benchmarks.json),
    then fall back to any file ending with `_benchmarks.json`.
    
    Args:
        filepath: Explicit path to JSON file
        algorithm_filter: If provided, prefer loading results/<algorithm>_benchmarks.json
    
    Returns:
        List of result dictionaries, optionally filtered by algorithm
    """
    if filepath:
        with open(filepath, 'r') as f:
            all_results = json.load(f)
    else:
        # Try algorithm-specific file first
        if algorithm_filter:
            algo_file = f'results/{normalize_algorithm_name(algorithm_filter)}_benchmarks.json'
            if os.path.exists(algo_file):
                with open(algo_file, 'r') as f:
                    all_results = json.load(f)
                print(f"✓ Loaded from {algo_file}")
                return all_results
        
        # Auto-discover any benchmark file
        candidates = []
        if os.path.exists('results'):
            for fn in os.listdir('results'):
                if fn.endswith('_benchmarks.json'):
                    candidates.append(os.path.join('results', fn))
        
        if not candidates:
            raise FileNotFoundError('No benchmark json file found in results/. Run benchmarks first.')
        
        with open(candidates[0], 'r') as f:
            all_results = json.load(f)
        print(f"✓ Loaded from {candidates[0]}")
    
    # Filter by algorithm if specified
    if algorithm_filter:
        normalized_filter = normalize_algorithm_name(algorithm_filter)
        filtered = [r for r in all_results if normalize_algorithm_name(r.get('algorithm', '')) == normalized_filter]
        if not filtered:
            print(f"⚠ Warning: No results found for algorithm '{algorithm_filter}'")
            print(f"  Available algorithms: {', '.join(set(r.get('algorithm', 'Unknown') for r in all_results))}")
            sys.exit(1)
        return filtered
    
    return all_results


def plot_runtime_by_algorithm(results, output_dir='results'):
    """Plot runtime vs graph size for each detected algorithm and scenario.

    Produces one PNG per algorithm and a combined comparison plot.
    """
    algorithms = sorted(set(r['algorithm'] for r in results))
    scenarios = sorted(set(r.get('scenario', 'default') for r in results))

    # Per-algorithm plots
    for algo in algorithms:
        algo_results = [r for r in results if r['algorithm'] == algo]
        sizes = sorted(set(r['num_stocks'] for r in algo_results))

        fig, ax = plt.subplots(figsize=(8, 5))
        for scenario in scenarios:
            runtimes = []
            for size in sizes:
                match = [r for r in algo_results if r['num_stocks'] == size and r.get('scenario') == scenario]
                runtimes.append(match[0]['runtime_seconds'] if match else 0)
            ax.plot(sizes, runtimes, marker='o', label=scenario.capitalize())

        ax.set_xlabel('Number of Stocks')
        ax.set_ylabel('Runtime (s)')
        ax.set_title(f'{algo} Runtime by Scenario')
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.tight_layout()
        out = f"{output_dir}/{algo.replace(' ', '_').lower()}_runtime.png"
        plt.savefig(out, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved: {out}")

    # Combined comparison: runtime per algorithm for each scenario
    sizes = sorted(set(r['num_stocks'] for r in results))
    fig, axes = plt.subplots(1, len(scenarios), figsize=(5 * len(scenarios), 4), squeeze=False)
    for j, scenario in enumerate(scenarios):
        ax = axes[0][j]
        for algo in algorithms:
            vals = []
            for size in sizes:
                match = [r for r in results if r['algorithm'] == algo and r['num_stocks'] == size and r.get('scenario') == scenario]
                vals.append(match[0]['runtime_seconds'] if match else 0)
            ax.plot(sizes, vals, marker='o', label=algo)
        ax.set_title(scenario.capitalize())
        ax.set_xlabel('Number of Stocks')
        ax.set_ylabel('Runtime (s)')
        ax.grid(True, alpha=0.3)
        ax.legend()

    plt.tight_layout()
    out = f"{output_dir}/runtime_comparison_all_algorithms.png"
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {out}")


def plot_components_analysis(results, output_dir='results'):
    """Plot number of connected components when available in results.

    This function looks for any algorithm outputs that include `num_components`.
    """
    entries = [r for r in results if 'num_components' in r]
    if not entries:
        print('⚠ No num_components data found; skipping components analysis')
        return

    algorithms = sorted(set(r['algorithm'] for r in entries))
    sizes = sorted(set(r['num_stocks'] for r in entries))
    scenarios = sorted(set(r.get('scenario', 'default') for r in entries))

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(sizes))
    width = 0.15

    for i, scenario in enumerate(scenarios):
        comps = []
        for size in sizes:
            matches = [r for r in entries if r['num_stocks'] == size and r.get('scenario') == scenario]
            # If multiple algorithms provide components, sum or pick first
            comps.append(sum([m['num_components'] for m in matches]) if matches else 0)
        ax.bar(x + i * width, comps, width, label=scenario.capitalize())

    ax.set_xlabel('Graph Size')
    ax.set_ylabel('Number of Components')
    ax.set_title('Connected Components by Scenario')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(sizes)
    ax.legend()
    ax.grid(True, alpha=0.3)

    out = f"{output_dir}/components_analysis.png"
    plt.tight_layout()
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {out}")


def plot_path_length_analysis(results, output_dir='results'):
    """Plot average path length when present in result entries (e.g., BFS/DFS)."""
    entries = [r for r in results if 'avg_path_length' in r]
    if not entries:
        print('⚠ No avg_path_length data found; skipping path length analysis')
        return

    algorithms = sorted(set(r['algorithm'] for r in entries))
    sizes = sorted(set(r['num_stocks'] for r in entries))
    scenarios = sorted(set(r.get('scenario', 'default') for r in entries))

    for algo in algorithms:
        algo_entries = [r for r in entries if r['algorithm'] == algo]
        fig, ax = plt.subplots(figsize=(8, 5))
        for scenario in scenarios:
            vals = []
            for size in sizes:
                match = [r for r in algo_entries if r['num_stocks'] == size and r.get('scenario') == scenario]
                vals.append(match[0]['avg_path_length'] if match else 0)
            ax.plot(sizes, vals, marker='o', label=scenario)
        ax.set_title(f'{algo} - Avg Path Length by Scenario')
        ax.set_xlabel('Number of Stocks')
        ax.set_ylabel('Avg Path Length')
        ax.grid(True, alpha=0.3)
        ax.legend()
        out = f"{output_dir}/{algo.replace(' ', '_').lower()}_path_length.png"
        plt.tight_layout()
        plt.savefig(out, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved: {out}")


def plot_graph_density(results, output_dir='results'):
    """
    Plot graph density across different scenarios.
    """
    uf_results = [r for r in results if r['algorithm'] == 'Union-Find']
    
    sizes = sorted(set(r['num_stocks'] for r in uf_results))
    scenarios = ['stable', 'normal', 'volatile', 'crash']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for scenario in scenarios:
        densities = []
        for size in sizes:
            matching = [r for r in uf_results 
                       if r['num_stocks'] == size and r['scenario'] == scenario]
            if matching:
                r = matching[0]
                # Calculate density
                n = r['num_stocks']
                e = r['num_edges']
                density = (2 * e) / (n * (n - 1)) if n > 1 else 0
                densities.append(density)
            else:
                densities.append(0)
        
        ax.plot(sizes, densities, marker='D', 
                label=scenario.capitalize(), linewidth=2, markersize=8)
    
    ax.set_xlabel('Number of Stocks', fontsize=12)
    ax.set_ylabel('Graph Density', fontsize=12)
    ax.set_title('Graph Density by Market Scenario', 
                 fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/graph_density.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir}/graph_density.png")
    plt.close()


def plot_scalability(results, output_dir='results'):
    """
    Plot scalability: runtime vs graph size and edges.
    """
    uf_results = [r for r in results if r['algorithm'] == 'Union-Find']
    bfs_results = [r for r in results if r['algorithm'] == 'BFS']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Union-Find: runtime vs edges
    edges = [r['num_edges'] for r in uf_results]
    runtimes = [r['runtime_seconds'] * 1000 for r in uf_results]  # Convert to ms
    colors = [{'stable': 'blue', 'normal': 'green', 
               'volatile': 'orange', 'crash': 'red'}[r['scenario']] 
              for r in uf_results]
    
    ax1.scatter(edges, runtimes, c=colors, alpha=0.6, s=100)
    ax1.set_xlabel('Number of Edges', fontsize=12)
    ax1.set_ylabel('Runtime (milliseconds)', fontsize=12)
    ax1.set_title('Union-Find Scalability', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='blue', label='Stable'),
                      Patch(facecolor='green', label='Normal'),
                      Patch(facecolor='orange', label='Volatile'),
                      Patch(facecolor='red', label='Crash')]
    ax1.legend(handles=legend_elements)
    
    # BFS: runtime vs edges
    edges = [r['num_edges'] for r in bfs_results]
    runtimes = [r['runtime_seconds'] * 1000 for r in bfs_results]  # Convert to ms
    colors = [{'stable': 'blue', 'normal': 'green', 
               'volatile': 'orange', 'crash': 'red'}[r['scenario']] 
              for r in bfs_results]
    
    ax2.scatter(edges, runtimes, c=colors, alpha=0.6, s=100)
    ax2.set_xlabel('Number of Edges', fontsize=12)
    ax2.set_ylabel('Runtime (milliseconds)', fontsize=12)
    ax2.set_title('BFS Scalability', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(handles=legend_elements)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/scalability_analysis.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir}/scalability_analysis.png")
    plt.close()


def plot_component_size_distribution(results, output_dir='results'):
    """
    Plot distribution of component sizes for largest graph.
    """
    uf_results = [r for r in results if r['algorithm'] == 'Union-Find']
    
    # Get largest graph for each scenario (200 stocks)
    scenarios = ['stable', 'normal', 'volatile', 'crash']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for i, scenario in enumerate(scenarios):
        matching = [r for r in uf_results 
                   if r['num_stocks'] == 200 and r['scenario'] == scenario]
        
        if matching and matching[0]['component_sizes']:
            sizes = matching[0]['component_sizes']
            
            # Create histogram
            axes[i].hist(sizes, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
            axes[i].set_xlabel('Component Size', fontsize=11)
            axes[i].set_ylabel('Frequency', fontsize=11)
            axes[i].set_title(f'{scenario.capitalize()} Market - Component Distribution', 
                            fontsize=12, fontweight='bold')
            axes[i].grid(True, alpha=0.3, axis='y')
            
            # Add statistics
            avg_size = np.mean(sizes)
            max_size = max(sizes)
            axes[i].axvline(avg_size, color='red', linestyle='--', 
                          label=f'Avg: {avg_size:.1f}')
            axes[i].axvline(max_size, color='green', linestyle='--', 
                          label=f'Max: {max_size}')
            axes[i].legend()
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/component_distribution.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir}/component_distribution.png")
    plt.close()


def create_summary_table(results, output_dir='results'):
    """
    Create a summary table image.
    """
    uf_results = [r for r in results if r['algorithm'] == 'Union-Find']
    bfs_results = [r for r in results if r['algorithm'] == 'BFS']
    
    # Prepare data
    scenarios = ['stable', 'normal', 'volatile', 'crash']
    sizes = [50, 100, 200]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Union-Find table
    uf_data = []
    for scenario in scenarios:
        row = [scenario.capitalize()]
        for size in sizes:
            matching = [r for r in uf_results 
                       if r['num_stocks'] == size and r['scenario'] == scenario]
            if matching:
                r = matching[0]
                row.append(f"{r['num_components']}\n({r['runtime_seconds']*1000:.2f}ms)")
            else:
                row.append("N/A")
        uf_data.append(row)
    
    ax1.axis('tight')
    ax1.axis('off')
    table1 = ax1.table(cellText=uf_data,
                       colLabels=['Scenario', '50 stocks', '100 stocks', '200 stocks'],
                       cellLoc='center',
                       loc='center',
                       colWidths=[0.25, 0.25, 0.25, 0.25])
    table1.auto_set_font_size(False)
    table1.set_fontsize(10)
    table1.scale(1, 2)
    
    # Style header
    for i in range(4):
        table1[(0, i)].set_facecolor('#4CAF50')
        table1[(0, i)].set_text_props(weight='bold', color='white')
    
    ax1.set_title('Union-Find: Components (Runtime)', 
                  fontsize=14, fontweight='bold', pad=20)
    
    # BFS table
    bfs_data = []
    for scenario in scenarios:
        row = [scenario.capitalize()]
        for size in sizes:
            matching = [r for r in bfs_results 
                       if r['num_stocks'] == size and r['scenario'] == scenario]
            if matching:
                r = matching[0]
                row.append(f"{r['avg_path_length']:.2f}\n({r['runtime_seconds']*1000:.2f}ms)")
            else:
                row.append("N/A")
        bfs_data.append(row)
    
    ax2.axis('tight')
    ax2.axis('off')
    table2 = ax2.table(cellText=bfs_data,
                       colLabels=['Scenario', '50 stocks', '100 stocks', '200 stocks'],
                       cellLoc='center',
                       loc='center',
                       colWidths=[0.25, 0.25, 0.25, 0.25])
    table2.auto_set_font_size(False)
    table2.set_fontsize(10)
    table2.scale(1, 2)
    
    # Style header
    for i in range(4):
        table2[(0, i)].set_facecolor('#2196F3')
        table2[(0, i)].set_text_props(weight='bold', color='white')
    
    ax2.set_title('BFS: Avg Path Length (Runtime)', 
                  fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/summary_table.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir}/summary_table.png")
    plt.close()


def main():
    """Generate all visualizations."""
    # Parse command-line arguments
    algorithm_filter = None
    if len(sys.argv) > 1:
        if sys.argv[1] in ['-h', '--help']:
            print(__doc__)
            sys.exit(0)
        algorithm_filter = sys.argv[1]
        # Remove .py extension if provided
        if algorithm_filter.endswith('.py'):
            algorithm_filter = algorithm_filter[:-3]
    
    print("="*70)
    print("GENERATING VISUALIZATIONS")
    if algorithm_filter:
        print(f"Algorithm Filter: {algorithm_filter}")
    print("="*70)
    
    # Determine output directory
    if algorithm_filter:
        output_dir = f'results/{normalize_algorithm_name(algorithm_filter)}'
    else:
        output_dir = 'results'
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load results
    print("\nLoading benchmark results...")
    results = load_benchmark_results(algorithm_filter=algorithm_filter)
    print(f"✓ Loaded {len(results)} benchmark results")
    
    # Generate all plots
    print("\nGenerating charts...")
    # runtime plots for detected algorithms
    plot_runtime_by_algorithm(results, output_dir)
    plot_components_analysis(results, output_dir)
    plot_path_length_analysis(results, output_dir)
    plot_graph_density(results, output_dir)
    plot_scalability(results, output_dir)
    plot_component_size_distribution(results, output_dir)
    create_summary_table(results, output_dir)
    
    print("\n" + "="*70)
    print("VISUALIZATION COMPLETE!")
    print("="*70)
    if algorithm_filter:
        print(f"\nVisualization files saved to '{output_dir}/'")
    else:
        print(f"\nGenerated visualization files in '{output_dir}/':")
        print("  - Per-algorithm runtime plots")
        print("  - Combined runtime comparison")
        print("  - Components analysis")
        print("  - Path length analysis")
        print("  - Graph density")
        print("  - Scalability analysis")
        print("  - Component distribution")
        print("  - Summary table")
    print("\nThese can be used in your report and presentation!")
    print("="*70)


if __name__ == "__main__":
    main()
