#!/usr/bin/env python3
"""
Visualization script for tariff model experiments.

This script creates focused, clean visualizations showing market value decline
and supply-demand relationships for portfolio managers.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from pathlib import Path


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Generate market value plots")
    parser.add_argument("--base_dir", type=str, default="./", 
                        help="Base directory containing run folders")
    return parser.parse_args()


# Dictionary mapping run directories to their display labels
# Only runs included in this dictionary will be plotted
labels = {
    "run_0": "Baseline Run",
    # Add more runs as needed
}


def load_results(input_path):
    """Load experiment results from a JSON file."""
    with open(input_path, 'r') as f:
        results = json.load(f)
    
    print(f"Loaded results from {input_path}")
    return results


def setup_plotting_style():
    """Set up clean, professional plotting style."""
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'figure.figsize': (10, 6),
        'figure.dpi': 300,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': False,
        'axes.edgecolor': 'black',
        'axes.linewidth': 1.0,
        'lines.linewidth': 2.5,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.2
    })


def plot_market_value_percentage_change(all_results, output_dir):
    """
    Create a focused plot comparing market value decline with tariff rate in percentage terms
    across multiple runs.
    
    Args:
        all_results: Dictionary mapping run labels to their results
        output_dir: Directory to save plots
    """
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Color palette for different runs
    colors = ['#d62728', '#1f77b4', '#2ca02c', '#ff7f0e', '#9467bd', '#8c564b']
    
    # Plot each run
    for i, (label, results) in enumerate(all_results.items()):
        # Extract data
        tariffs = [float(t) for t in results['equilibrium'].keys()]
        sorted_indices = np.argsort(tariffs)
        tariffs = [tariffs[i] for i in sorted_indices]
        
        # Get market values for each tariff rate
        market_values = [results['equilibrium'][str(t)]['value'] for t in tariffs]
        
        # Calculate percentage change from baseline (tariff=0)
        baseline_value = market_values[tariffs.index(0.0)]
        pct_change = [(v / baseline_value - 1) * 100 for v in market_values]
        
        # Plot percentage change
        color = colors[i % len(colors)]
        ax.plot(tariffs, pct_change, linewidth=2.5, color=color, label=label)
        
        # Highlight specific points with markers
        key_tariffs = [0.0, 0.1, 0.25, 0.5]
        for tau in key_tariffs:
            if tau in tariffs:
                idx = tariffs.index(tau)
                ax.scatter(tau, pct_change[idx], s=80, color=color, zorder=5)
    
    # Add reference line at 0%
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5, label='No Change (Baseline)')
    
    # Axis labels and title
    ax.set_xlabel('Tariff Rate')
    ax.set_ylabel('Market Value Change (%)')
    ax.set_title('Market Value Decline Due to Tariffs')
    
    # Format x-axis as percentage
    ax.set_xlim(left=-0.01, right=0.55)
    ax.set_xticks([0, 0.1, 0.25, 0.5])
    ax.set_xticklabels(['0%', '10%', '25%', '50%'])
    
    # Format y-axis with percentage sign
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0f}%'))
    
    # Add legend
    ax.legend(loc='best', frameon=True, framealpha=0.9)
    
    plt.savefig(os.path.join(output_dir, "market_value_decline.png"))
    plt.close()


def plot_supply_demand_curves(all_results, output_dir):
    """
    Plot supply and demand curves for specific tariff rates, comparing across runs.
    
    Args:
        all_results: Dictionary mapping run labels to their results
        output_dir: Directory to save plots
    """
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # We'll use different line styles for different runs
    line_styles = ['-', '--', '-.', ':']
    # Color palette for different tariff rates
    colors = ['#2ca02c', '#ff7f0e', '#9467bd', '#d62728']
    
    # Selected tariff rates to plot
    tariff_rates = [0.0, 0.1, 0.25, 0.5]
    
    # Plot each run
    for run_idx, (label, results) in enumerate(all_results.items()):
        params = results['parameters']
        a, b, S, c_s = params['a'], params['b'], params['S'], params['c_s']
        
        # Create price range for plotting
        p_star_range = np.linspace(c_s, a/b, 100)
        
        # Plot supply curve: Q = S*(P* - c_s)
        supply = S * (p_star_range - c_s)
        line_style = line_styles[run_idx % len(line_styles)]
        ax.plot(supply, p_star_range, linewidth=2.5, 
               linestyle=line_style,
               label=f'Supply Curve ({label})', 
               color='#1f77b4')
        
        for i, tau in enumerate(tariff_rates):
            # Demand curve with tariff: Q = a - b*((1+tau)*P*)
            demand = a - b * ((1 + tau) * p_star_range)
            
            # Only plot positive quantities
            positive_mask = demand > 0
            
            # Only plot first run's demand curves to avoid clutter
            if run_idx == 0:
                ax.plot(demand[positive_mask], p_star_range[positive_mask], 
                        linewidth=2.5, 
                        label=f'Demand (τ={tau:.0%})',
                        color=colors[i])
            
            # Mark equilibrium point if available
            if str(tau) in results['equilibrium']:
                eq = results['equilibrium'][str(tau)]
                marker_style = 'o' if run_idx == 0 else 's'
                ax.scatter([eq['Q']], [eq['P_star']], 
                          color=colors[i], 
                          s=80, 
                          marker=marker_style,
                          zorder=5)
    
    ax.set_xlabel('Quantity')
    ax.set_ylabel('Producer Price')
    ax.set_title('Supply and Demand Curves with Tariffs')
    
    # Add legend
    ax.legend(loc='best', frameon=True, framealpha=0.9)
    
    plt.savefig(os.path.join(output_dir, "supply_demand_curves.png"))
    plt.close()


def generate_plots(all_results, output_dir):
    """Generate the focused plots comparing multiple runs."""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up clean plotting style
    setup_plotting_style()
    
    # Generate the comparison plots
    plot_market_value_percentage_change(all_results, output_dir)
    plot_supply_demand_curves(all_results, output_dir)
    
    print(f"Comparison plots saved to {output_dir}/")


def main():
    """Main function to run the plotting script."""
    # Parse command line arguments
    args = parse_args()
    
    # Use run_0 as the output directory
    output_dir = os.path.join(args.base_dir, "run_0")
    os.makedirs(output_dir, exist_ok=True)
    
    # Load results from each run in the labels dictionary
    all_results = {}
    
    for run_dir, run_label in labels.items():
        results_path = os.path.join(args.base_dir, run_dir, "final_info.json")
        
        if not os.path.exists(results_path):
            print(f"Warning: Results file not found at {results_path}")
            continue
        
        # Load results and add to dictionary with the label
        results = load_results(results_path)
        all_results[run_label] = results
    
    if not all_results:
        print("Error: No valid run results found. Please check the 'labels' dictionary.")
        return
    
    # Generate the comparison plots
    generate_plots(all_results, output_dir)
    print(f"Plots saved to {output_dir}/")


if __name__ == "__main__":
    main()