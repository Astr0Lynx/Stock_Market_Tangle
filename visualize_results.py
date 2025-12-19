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

import json  # For reading/writing JSON benchmark files
import matplotlib.pyplot as plt  # For creating charts and plots
import numpy as np  # For numerical operations like mean, array manipulation
import os  # For file/directory operations and path handling
import sys  # For command-line arguments and program exit


def normalize_algorithm_name(name):
    """Normalize algorithm name for matching (lowercase, replace spaces/hyphens with underscores)."""
    return name.lower().replace(' ', '_').replace('-', '_')  # Convert to lowercase and replace spaces/hyphens with underscores for consistent matching


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
    if filepath:  # If explicit file path was provided
        with open(filepath, 'r') as f:  # Open the file for reading
            all_results = json.load(f)  # Parse JSON content into Python list
    else:  # No explicit path, so auto-discover benchmark files
        # Try algorithm-specific file first
        if algorithm_filter:  # If user wants results for specific algorithm
            algo_file = f'results/{normalize_algorithm_name(algorithm_filter)}_benchmarks.json'  # Build expected filename
            if os.path.exists(algo_file):  # Check if that specific file exists
                with open(algo_file, 'r') as f:  # Open the algorithm-specific file
                    all_results = json.load(f)  # Load its benchmark data
                print(f"[OK] Loaded from {algo_file}")  # Inform user which file was loaded
                return all_results  # Return early since we found the specific file
        
        # Auto-discover any benchmark file in results directory
        candidates = []  # List to store discovered benchmark files
        if os.path.exists('results'):  # Check if results directory exists
            for fn in os.listdir('results'):  # Loop through all files in results/
                if fn.endswith('_benchmarks.json'):  # Check if filename matches pattern
                    candidates.append(os.path.join('results', fn))  # Add full path to candidates list
        
        if not candidates:  # If no benchmark files were found
            raise FileNotFoundError('No benchmark json file found in results/. Run benchmarks first.')  # Raise error with helpful message
        
        with open(candidates[0], 'r') as f:  # Open first discovered benchmark file
            all_results = json.load(f)  # Parse its JSON content
        print(f"✓ Loaded from {candidates[0]}")  # Show user which file was used
    
    # Filter by algorithm if specified
    if algorithm_filter:  # User wants to see only specific algorithm's results
        normalized_filter = normalize_algorithm_name(algorithm_filter)  # Normalize user's input for comparison
        filtered = [r for r in all_results if normalize_algorithm_name(r.get('algorithm', '')) == normalized_filter]  # Keep only matching results
        if not filtered:  # If no results match the filter
            print(f"⚠ Warning: No results found for algorithm '{algorithm_filter}'")  # Warn user
            print(f"  Available algorithms: {', '.join(set(r.get('algorithm', 'Unknown') for r in all_results))}")  # Show what's actually available
            sys.exit(1)  # Exit with error code
        return filtered  # Return only the filtered results
    
    return all_results  # Return all results if no filter specified


def plot_runtime_by_algorithm(results, output_dir='results'):
    """Plot runtime vs graph size for each detected algorithm and scenario.

    Produces one PNG per algorithm with 2x2 subplot grid (one per scenario).
    """
    algorithms = sorted(set(r['algorithm'] for r in results))  # Extract unique algorithm names and sort them
    scenarios = sorted(set(r.get('scenario', 'default') for r in results))  # Extract unique scenario names from data

    # Per-algorithm plots with 2x2 subplots
    # Create one chart per algorithm with 2x2 subplots (one for each scenario)
    for algo in algorithms:  # Loop through each algorithm that has results
        algo_results = [r for r in results if r['algorithm'] == algo]  # Filter results to get only this algorithm's data
        sizes = sorted(set(r['num_stocks'] for r in algo_results))  # Extract all graph sizes tested for this algorithm

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))  # Create figure with 2x2 grid of subplots (12x10 inches)
        axes = axes.flatten()  # Convert 2D array of axes to 1D for easier iteration
        
        # Plot each scenario in a separate subplot
        for idx, scenario in enumerate(scenarios):  # Loop through each market scenario with index
            ax = axes[idx]  # Get the subplot for this scenario
            runtimes = []  # List to store runtime values for each graph size
            for size in sizes:  # Loop through each graph size
                # Find the benchmark result for this size and scenario
                match = [r for r in algo_results if r['num_stocks'] == size and r.get('scenario') == scenario]  # Filter for matching size and scenario
                runtimes.append(match[0]['runtime_seconds'] * 1000 if match else 0)  # Convert seconds to milliseconds, use 0 if no data
            
            ax.plot(sizes, runtimes, marker='o', linewidth=2, markersize=8, color=['#2E86AB', '#A23B72', '#F18F01', '#C73E1D'][idx])  # Draw line with markers using scenario-specific color
            ax.set_xlabel('Number of Stocks', fontsize=10)  # Label x-axis
            ax.set_ylabel('Runtime (milliseconds)', fontsize=10)  # Label y-axis
            ax.set_title(f'{scenario.capitalize()} Market', fontsize=12, fontweight='bold')  # Set subplot title
            ax.grid(True, alpha=0.3)  # Add semi-transparent grid for readability
            
            # Add value labels on points
            for x, y in zip(sizes, runtimes):  # Loop through data points
                ax.annotate(f'{y:.2f}', (x, y), textcoords="offset points", xytext=(0,5), ha='center', fontsize=8)  # Add text label above each point showing exact value

        fig.suptitle(f'{algo} - Runtime by Market Scenario', fontsize=14, fontweight='bold')  # Add overall title to the entire figure
        plt.tight_layout()  # Adjust subplot spacing to prevent overlaps
        out = f"{output_dir}/{algo.replace(' ', '_').lower()}_runtime.png"  # Build output filename with algorithm name
        plt.savefig(out, dpi=300, bbox_inches='tight')  # Save figure as high-resolution PNG (300 DPI)
        plt.close()  # Close figure to free memory
        print(f"[OK] Saved: {out}")  # Confirm file was saved


def plot_memory_by_algorithm(results, output_dir='results'):
    """Plot memory usage vs graph size for each detected algorithm and scenario.

    Produces one PNG per algorithm with 2x2 subplot grid (one per scenario).
    """
    algorithms = sorted(set(r['algorithm'] for r in results))  # Get unique algorithm names sorted alphabetically
    scenarios = sorted(set(r.get('scenario', 'default') for r in results))  # Get unique scenario names from results

    # Per-algorithm plots with 2x2 subplots
    for algo in algorithms:  # Create separate chart for each algorithm
        algo_results = [r for r in results if r['algorithm'] == algo]  # Get only this algorithm's results
        sizes = sorted(set(r['num_stocks'] for r in algo_results))  # Get sorted list of graph sizes

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))  # Create 2x2 grid of subplots
        axes = axes.flatten()  # Flatten to 1D array for easier access
        
        for idx, scenario in enumerate(scenarios):  # Iterate through scenarios with index
            ax = axes[idx]  # Select subplot for current scenario
            memory_usage = []  # List to collect memory measurements
            for size in sizes:  # Go through each graph size
                match = [r for r in algo_results if r['num_stocks'] == size and r.get('scenario') == scenario]  # Find matching result
                memory_usage.append(match[0]['memory_mb'] if match else 0)  # Extract memory value in MB, default to 0
            
            ax.plot(sizes, memory_usage, marker='s', linewidth=2, markersize=8, color=['#2E86AB', '#A23B72', '#F18F01', '#C73E1D'][idx])  # Plot line with square markers
            ax.set_xlabel('Number of Stocks', fontsize=10)  # Set x-axis label
            ax.set_ylabel('Memory Usage (MB)', fontsize=10)  # Set y-axis label
            ax.set_title(f'{scenario.capitalize()} Market', fontsize=12, fontweight='bold')  # Set subplot title
            ax.grid(True, alpha=0.3)  # Add faint grid lines
            
            # Add value labels on points
            for x, y in zip(sizes, memory_usage):  # Pair up x and y coordinates
                ax.annotate(f'{y:.2f}', (x, y), textcoords="offset points", xytext=(0,5), ha='center', fontsize=8)  # Add text showing exact memory value

        fig.suptitle(f'{algo} - Memory Usage by Market Scenario', fontsize=14, fontweight='bold')
        plt.tight_layout()
        out = f"{output_dir}/{algo.replace(' ', '_').lower()}_memory.png"
        plt.savefig(out, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[OK] Saved: {out}")


def plot_components_analysis(results, output_dir='results'):
    """Plot number of connected components when available in results.

    This function looks for any algorithm outputs that include `num_components`.
    """
    entries = [r for r in results if 'num_components' in r]  # Filter to results that have component count data
    if not entries:  # If no results have component data
        print('[WARNING] No num_components data found; skipping components analysis')  # Warn user
        return  # Exit function early

    algorithms = sorted(set(r['algorithm'] for r in entries))  # Get unique algorithms with component data
    sizes = sorted(set(r['num_stocks'] for r in entries))  # Get graph sizes that have component data
    scenarios = sorted(set(r.get('scenario', 'default') for r in entries))  # Get scenarios present in data

    fig, ax = plt.subplots(figsize=(10, 6))  # Create single plot (10x6 inches)
    x = np.arange(len(sizes))  # Create array of x positions for bar groups
    width = 0.15  # Width of each bar in the grouped bar chart

    for i, scenario in enumerate(scenarios):  # Loop through each scenario with its index
        comps = []  # List to store component counts for this scenario
        for size in sizes:  # Go through each graph size
            matches = [r for r in entries if r['num_stocks'] == size and r.get('scenario') == scenario]  # Find results for this size+scenario
            # If multiple algorithms provide components, sum or pick first
            comps.append(sum([m['num_components'] for m in matches]) if matches else 0)  # Sum components from all matching results
        bars = ax.bar(x + i * width, comps, width, label=scenario.capitalize())  # Create bar group offset by scenario index
        
        # Add value labels on top of each bar
        for bar in bars:  # Loop through each bar in this scenario's group
            height = bar.get_height()  # Get bar height (component count)
            if height > 0:  # Only add label if bar is visible
                ax.text(bar.get_x() + bar.get_width()/2., height,  # Position text at bar center
                       f'{int(height)}',  # Display count as integer
                       ha='center', va='bottom', fontsize=9, fontweight='bold')  # Center align, place above bar

    ax.set_xlabel('Graph Size')  # Label x-axis with graph size
    ax.set_ylabel('Number of Components')  # Label y-axis with component count
    ax.set_title('Connected Components by Scenario')  # Set chart title
    ax.set_xticks(x + width * 1.5)  # Position x-axis ticks at center of bar groups
    ax.set_xticklabels(sizes)  # Set x-axis labels to actual graph sizes
    ax.legend()  # Display legend showing scenario colors
    ax.grid(True, alpha=0.3)  # Add subtle horizontal grid lines

    out = f"{output_dir}/components_analysis.png"  # Build output file path
    plt.tight_layout()  # Adjust layout to prevent label cutoff
    plt.savefig(out, dpi=300, bbox_inches='tight')  # Save as high-resolution PNG
    plt.close()  # Close figure to free memory
    print(f"[OK] Saved: {out}")  # Confirm save success


def plot_path_length_analysis(results, output_dir='results'):
    """Plot average path length when present in result entries (e.g., BFS/DFS)."""
    entries = [r for r in results if 'avg_path_length' in r]  # Filter to results containing path length data
    if not entries:  # If no path length data found
        # Silently skip if no path length data (e.g., Union-Find doesn't track paths)
        return  # Exit function without error

    algorithms = sorted(set(r['algorithm'] for r in entries))  # Get unique algorithms that have path data
    sizes = sorted(set(r['num_stocks'] for r in entries))  # Get graph sizes with path data
    scenarios = sorted(set(r.get('scenario', 'default') for r in entries))  # Get scenarios present in path data

    for algo in algorithms:  # Create one chart per algorithm
        algo_entries = [r for r in entries if r['algorithm'] == algo]  # Filter for this algorithm's data
        fig, ax = plt.subplots(figsize=(8, 5))  # Create single plot (8x5 inches)
        for scenario in scenarios:  # Plot one line per scenario
            vals = []  # List to store path lengths for this scenario
            for size in sizes:  # Loop through each graph size
                match = [r for r in algo_entries if r['num_stocks'] == size and r.get('scenario') == scenario]  # Find matching result
                vals.append(match[0]['avg_path_length'] if match else 0)  # Extract average path length, use 0 if missing
            ax.plot(sizes, vals, marker='o', label=scenario)  # Plot line with circular markers
        ax.set_title(f'{algo} - Avg Path Length by Scenario')  # Set chart title with algorithm name
        ax.set_xlabel('Number of Stocks')  # Label x-axis
        ax.set_ylabel('Avg Path Length')  # Label y-axis
        ax.grid(True, alpha=0.3)  # Add semi-transparent grid
        ax.legend()  # Show legend with scenario names
        out = f"{output_dir}/{algo.replace(' ', '_').lower()}_path_length.png"  # Build output filename
        plt.tight_layout()  # Adjust layout spacing
        plt.savefig(out, dpi=300, bbox_inches='tight')  # Save as high-resolution PNG
        plt.close()  # Close figure to free memory
        print(f"[OK] Saved: {out}")  # Confirm successful save


def plot_graph_density(results, output_dir='results'):
    """
    Plot graph density across different scenarios.
    """
    # Get unique algorithm from results
    algorithms = set(r['algorithm'] for r in results)  # Extract all unique algorithm names
    if not algorithms:  # If no algorithms found in results
        return  # Exit early
    
    algo_filter = next(iter(algorithms))  # Pick first algorithm from set
    uf_results = [r for r in results if r['algorithm'] == algo_filter]  # Filter results for chosen algorithm
    
    # Check if num_edges exists in the data
    if not uf_results or 'num_edges' not in uf_results[0]:  # Verify edge count data is available
        print(f"[WARNING] No num_edges data found for {algo_filter}; skipping graph density chart")  # Warn user
        return  # Exit without creating chart
    
    sizes = sorted(set(r['num_stocks'] for r in uf_results))  # Get sorted list of graph sizes
    scenarios = ['stable', 'normal', 'volatile', 'crash']  # Define all four market scenarios
    
    fig, ax = plt.subplots(figsize=(10, 6))  # Create single plot (10x6 inches)
    
    # Plot density for each scenario (shows how connected the graph is)
    for scenario in scenarios:  # Loop through each market scenario
        densities = []  # List to store calculated densities
        for size in sizes:  # Go through each graph size
            matching = [r for r in uf_results  # Find results matching this size and scenario
                       if r['num_stocks'] == size and r['scenario'] == scenario]  # Filter condition
            if matching:  # If we found data for this size and scenario
                r = matching[0]  # Get the first matching result
                # Calculate graph density: ratio of actual edges to possible edges
                # Density = (# of edges) / (max possible edges)
                # For undirected graph: max edges = n(n-1)/2, so density = 2e/(n(n-1))
                n = r['num_stocks']  # Extract number of nodes (stocks) in graph
                e = r['num_edges']   # Extract number of edges (correlations) in graph
                density = (2 * e) / (n * (n - 1)) if n > 1 else 0  # Apply density formula, avoid division by zero
                densities.append(density)  # Add calculated density to list
            else:  # No data found for this size/scenario
                densities.append(0)  # Use 0 as placeholder density
        
        ax.plot(sizes, densities, marker='D',  # Plot line with diamond markers
                label=scenario.capitalize(), linewidth=2, markersize=8)  # Set scenario label and styling
    
    ax.set_xlabel('Number of Stocks', fontsize=12)  # Label x-axis
    ax.set_ylabel('Graph Density', fontsize=12)  # Label y-axis (0 to 1 scale)
    ax.set_title('Graph Density by Market Scenario',  # Set chart title
                 fontsize=14, fontweight='bold')  # Make title large and bold
    ax.legend()  # Display legend with scenario names
    ax.grid(True, alpha=0.3)  # Add semi-transparent grid lines
    
    plt.tight_layout()  # Adjust spacing to prevent label overlap
    plt.savefig(f'{output_dir}/graph_density.png', dpi=300, bbox_inches='tight')  # Save as high-res PNG
    print(f"[OK] Saved: {output_dir}/graph_density.png")  # Confirm successful save
    plt.close()  # Close figure to free memory


def plot_scalability(results, output_dir='results'):
    """
    Plot scalability: runtime vs graph size and edges for the algorithm in results.
    """
    # Get unique algorithms in results
    algorithms = sorted(set(r['algorithm'] for r in results))  # Extract and sort unique algorithm names
    
    if not results:  # If no benchmark results provided
        print('[WARNING] No data for scalability analysis')  # Warn user
        return  # Exit function early
    
    # Create legend elements
    from matplotlib.patches import Patch  # Import Patch for creating colored legend boxes
    legend_elements = [Patch(facecolor='blue', label='Stable'),  # Blue box for stable market
                      Patch(facecolor='green', label='Normal'),  # Green box for normal market
                      Patch(facecolor='orange', label='Volatile'),  # Orange box for volatile market
                      Patch(facecolor='red', label='Crash')]  # Red box for crash scenario
    
    # Plot for each algorithm
    for algo in algorithms:  # Loop through each algorithm
        algo_results = [r for r in results if r['algorithm'] == algo]  # Get this algorithm's results
        
        if not algo_results:  # If no results for this algorithm
            continue  # Skip to next algorithm
        
        # Skip if num_edges not present in data
        if 'num_edges' not in algo_results[0]:  # Check if edge count data exists
            print(f"[WARNING] No num_edges data found for {algo}; skipping scalability chart")  # Warn user
            continue  # Skip this algorithm
        
        fig, ax = plt.subplots(figsize=(10, 6))  # Create single scatter plot (10x6 inches)
        
        edges = [r['num_edges'] for r in algo_results]  # Extract edge counts as x-axis values
        runtimes = [r['runtime_seconds'] * 1000 for r in algo_results]  # Extract runtimes and convert to milliseconds
        colors = [{'stable': 'blue', 'normal': 'green',  # Map scenario names to colors
                   'volatile': 'orange', 'crash': 'red'}[r['scenario']]  # Look up color for each result's scenario
                  for r in algo_results]  # Create color list matching results order
        
        ax.scatter(edges, runtimes, c=colors, alpha=0.6, s=100)  # Create scatter plot with colored points
        ax.set_xlabel('Number of Edges', fontsize=12)  # Label x-axis
        ax.set_ylabel('Runtime (milliseconds)', fontsize=12)  # Label y-axis
        ax.set_title(f'{algo} Scalability: Runtime vs Graph Density', fontsize=14, fontweight='bold')  # Set chart title
        ax.grid(True, alpha=0.3)  # Add subtle background grid
        ax.legend(handles=legend_elements)  # Add legend showing scenario colors
        
        plt.tight_layout()  # Adjust layout to prevent overlapping elements
        out = f"{output_dir}/{algo.replace(' ', '_').lower()}_scalability.png"  # Build output filename
        plt.savefig(out, dpi=300, bbox_inches='tight')  # Save as high-resolution PNG
        print(f"[OK] Saved: {out}")  # Confirm file was saved
        plt.close()  # Close figure to free memory


def plot_component_size_distribution(results, output_dir='results'):
    """
    Plot distribution of component sizes for largest graph.
    """
    # Get unique algorithms in results
    algorithms = sorted(set(r['algorithm'] for r in results))  # Extract sorted list of unique algorithms
    
    for algo in algorithms:  # Create histogram for each algorithm
        algo_results = [r for r in results if r['algorithm'] == algo]  # Filter to this algorithm's results
        
        if not algo_results:  # If no results for this algorithm
            continue  # Skip to next algorithm
        
        # Get largest graph for each scenario (200 stocks)
        scenarios = ['stable', 'normal', 'volatile', 'crash']  # Define four market scenarios
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))  # Create 2x2 grid of histograms (14x10 inches)
        axes = axes.flatten()  # Convert 2D array to 1D for easier indexing
        
        for i, scenario in enumerate(scenarios):  # Loop through scenarios with index
            matching = [r for r in algo_results  # Find results for 200-stock graph in this scenario
                       if r['num_stocks'] == 200 and r['scenario'] == scenario]  # Filter by size and scenario
            
            if matching and matching[0].get('component_sizes'):  # If we have component size data for this scenario
                sizes = matching[0]['component_sizes']  # Extract list of individual component sizes
                
                # Create histogram
                axes[i].hist(sizes, bins=20, color='skyblue', edgecolor='black', alpha=0.7)  # Create histogram with 20 bins
                axes[i].set_xlabel('Component Size', fontsize=11)  # Label x-axis
                axes[i].set_ylabel('Frequency', fontsize=11)  # Label y-axis (how many components of each size)
                axes[i].set_title(f'{scenario.capitalize()} Market - Component Distribution',  # Set subplot title
                                fontsize=12, fontweight='bold')  # Make title bold
                axes[i].grid(True, alpha=0.3, axis='y')  # Add horizontal grid lines
                
                # Add statistics
                avg_size = np.mean(sizes)  # Calculate average component size
                max_size = max(sizes)  # Find largest component size
                axes[i].axvline(avg_size, color='red', linestyle='--',  # Draw vertical line at average
                              label=f'Avg: {avg_size:.1f}')  # Label the line with average value
                axes[i].legend()  # Show legend with average marker
            else:  # If no component size data available for this scenario
                axes[i].text(0.5, 0.5, 'No component size data',  # Display message in center of subplot
                           ha='center', va='center', transform=axes[i].transAxes)  # Center alignment using axes coordinates
                axes[i].set_title(f'{scenario.capitalize()} Market', fontweight='bold')  # Set title even for empty plot
        
        plt.suptitle(f'{algo} - Component Size Distribution (200 Stocks)',  # Set overall figure title
                    fontsize=15, fontweight='bold')  # Make title large and bold
        plt.tight_layout()  # Adjust spacing to prevent overlaps
        
        out = f"{output_dir}/{algo.replace(' ', '_').lower()}_component_distribution.png"  # Build output filename
        plt.savefig(out, dpi=300, bbox_inches='tight')  # Save as high-resolution PNG
        print(f"[OK] Saved: {out}")  # Confirm successful save
        plt.close()  # Close figure to free memory


def create_summary_table(results, output_dir='results'):
    """
    Create a summary table image for each algorithm in results.
    """
    # Get unique algorithms in results
    algorithms = sorted(set(r['algorithm'] for r in results))  # Extract sorted list of unique algorithms
    
    for algo in algorithms:  # Create one summary table per algorithm
        algo_results = [r for r in results if r['algorithm'] == algo]  # Filter to this algorithm's results
        
        if not algo_results:  # If no results for this algorithm
            continue  # Skip to next algorithm
        
        # Prepare data
        scenarios = ['stable', 'normal', 'volatile', 'crash']  # Define four market scenarios
        sizes = sorted(set(r['num_stocks'] for r in algo_results))  # Get sorted list of graph sizes tested
        
        fig, ax = plt.subplots(figsize=(12, 8))  # Create figure for table (12x8 inches)
        
        # Build table data
        table_data = []  # List to store all table rows (including header)
        # Check if this algorithm has components and edges data
        has_components = any('num_components' in r for r in algo_results)  # Check if any result has component count
        has_edges = any('num_edges' in r for r in algo_results)  # Check if any result has edge count
        
        # Determine table structure based on available data
        if has_components and has_edges:  # If we have both metrics
            table_data.append(['Scenario', 'Stocks', 'Edges', 'Components', 'Runtime (ms)'])  # 5-column header
            col_widths = [0.2, 0.15, 0.15, 0.2, 0.2]  # Relative column widths
        elif has_edges:  # If we only have edges (no components)
            table_data.append(['Scenario', 'Stocks', 'Edges', 'Runtime (ms)'])  # 4-column header
            col_widths = [0.25, 0.2, 0.2, 0.25]  # Adjusted column widths
        else:  # Minimal table (no edges or components)
            table_data.append(['Scenario', 'Stocks', 'Runtime (ms)'])  # 3-column header
            col_widths = [0.35, 0.3, 0.3]  # Wider columns for fewer items
        
        for scenario in scenarios:  # Loop through each market scenario
            for size in sizes:  # Loop through each graph size
                matching = [r for r in algo_results  # Find result matching this scenario and size
                           if r['scenario'] == scenario and r['num_stocks'] == size]  # Filter condition
                
                if matching:  # If we found matching data
                    r = matching[0]  # Use first matching result
                    row = [scenario.capitalize(), str(r['num_stocks'])]  # Start row with scenario and stock count
                    
                    if has_edges:  # If table includes edge count column
                        row.append(str(r['num_edges']))  # Add edge count to row
                    
                    if has_components:  # If table includes component count column
                        row.append(str(r.get('num_components', 'N/A')))  # Add component count or N/A if missing
                    
                    row.append(f"{r['runtime_seconds']*1000:.2f}")  # Add runtime in milliseconds formatted to 2 decimals
                    table_data.append(row)  # Append completed row to table data list
        
        # Create table
        num_cols = len(table_data[0])  # Get number of columns from header row
        
        table = ax.table(cellText=table_data, cellLoc='center', loc='center',  # Create table with centered cells
                        colWidths=col_widths)  # Apply column width proportions
        
        table.auto_set_font_size(False)  # Disable automatic font sizing
        table.set_fontsize(10)  # Set fixed font size
        table.scale(1, 2)  # Scale table: 1x width, 2x height (makes rows taller)
        
        # Style header row
        for i in range(num_cols):  # Loop through each column in header
            table[(0, i)].set_facecolor('#4472C4')  # Set blue background color
            table[(0, i)].set_text_props(weight='bold', color='white')  # Make text bold and white
        
        # Alternate row colors
        for i in range(1, len(table_data)):  # Loop through data rows (skip header)
            color = '#E7E6E6' if i % 2 == 0 else 'white'  # Alternate between light gray and white
            for j in range(num_cols):  # Loop through each cell in row
                table[(i, j)].set_facecolor(color)  # Apply background color
        
        ax.axis('off')  # Hide axes since we're only displaying a table
        ax.set_title(f'{algo} - Performance Summary',  # Set title above table with algorithm name
                    fontsize=16, fontweight='bold', pad=20)  # Large bold title with padding
        
        plt.tight_layout()  # Adjust layout to fit everything properly
        out = f"{output_dir}/{algo.replace(' ', '_').lower()}_summary_table.png"  # Build output filename
        plt.savefig(out, dpi=300, bbox_inches='tight')  # Save as high-resolution PNG (300 DPI)
        print(f"[OK] Saved: {out}")  # Confirm file was saved successfully
        plt.close()  # Close figure to free memory


def main():
    """Generate all visualizations."""
    # Parse command-line arguments
    algorithm_filter = None  # Default: no algorithm filter (show all)
    if len(sys.argv) > 1:  # If user provided command-line argument
        if sys.argv[1] in ['-h', '--help']:  # Check if user requested help
            print(__doc__)  # Print module docstring with usage information
            sys.exit(0)  # Exit successfully without generating visualizations
        algorithm_filter = sys.argv[1]  # Use first argument as algorithm filter
        # Remove .py extension if provided
        if algorithm_filter.endswith('.py'):  # If user accidentally included .py extension
            algorithm_filter = algorithm_filter[:-3]  # Strip off the .py extension
    
    print("="*70)  # Print separator line
    print("GENERATING VISUALIZATIONS")  # Print header message
    if algorithm_filter:  # If filtering by specific algorithm
        print(f"Algorithm Filter: {algorithm_filter}")  # Show which algorithm is being filtered
    print("="*70)  # Print closing separator line
    
    # Determine output directory
    if algorithm_filter:  # If filtering for specific algorithm
        output_dir = f'results/{normalize_algorithm_name(algorithm_filter)}'  # Create algorithm-specific subdirectory
    else:  # No filter, show all algorithms
        output_dir = 'results'  # Use main results directory
    
    os.makedirs(output_dir, exist_ok=True)  # Create output directory if it doesn't exist (no error if exists)
    
    # Load results
    print("\nLoading benchmark results...")  # Inform user that data loading is starting
    results = load_benchmark_results(algorithm_filter=algorithm_filter)  # Load JSON data and filter if needed
    print(f"[OK] Loaded {len(results)} benchmark results")  # Show count of loaded benchmark entries
    
    # Generate all plots
    print("\nGenerating charts...")  # Inform user that visualization generation is starting
    # runtime plots for detected algorithms
    plot_runtime_by_algorithm(results, output_dir)  # Create runtime comparison charts (2x2 subplots per algorithm)
    plot_memory_by_algorithm(results, output_dir)  # Create memory usage charts (2x2 subplots per algorithm)
    plot_components_analysis(results, output_dir)  # Create grouped bar chart showing component counts
    plot_path_length_analysis(results, output_dir)  # Create path length charts (only for BFS/DFS algorithms)
    plot_graph_density(results, output_dir)  # Create line chart showing graph density by scenario
    plot_scalability(results, output_dir)  # Create scatter plots showing runtime vs edge count
    create_summary_table(results, output_dir)  # Create formatted summary tables as PNG images
    plot_component_size_distribution(results, output_dir)  # Create histograms showing distribution of component sizes
    
    print("\n" + "="*70)  # Print separator with newline before
    print("VISUALIZATION COMPLETE!")  # Print completion message
    print("="*70)  # Print closing separator
    if algorithm_filter:  # If specific algorithm was filtered
        print(f"\nVisualization files saved to '{output_dir}/'")  # Show algorithm-specific output directory
    else:  # All algorithms were processed
        print(f"\nGenerated visualization files in '{output_dir}/':")  # Show output directory
        print("  - Per-algorithm runtime plots")  # List runtime plots generated
        print("  - Combined runtime comparison")  # Runtime comparison across scenarios
        print("  - Components analysis")  # Component count bar charts
        print("  - Path length analysis")  # Path length for BFS/DFS algorithms
        print("  - Graph density")  # Density line charts
        print("  - Scalability analysis")  # Runtime vs edge count scatter plots
        print("  - Component distribution")  # Component size histograms
        print("  - Summary table")  # Formatted summary tables


if __name__ == "__main__":  # Check if script is being run directly (not imported)
    main()  # Execute the main visualization generation function
