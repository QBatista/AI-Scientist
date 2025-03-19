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
    parser.add_argument("--out_dir", type=str, default="run_0", 
                        help="Output directory for plots")
    return parser.parse_args()


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


def plot_market_value_percentage_change(results, output_dir):
    """
    Create a focused plot showing market value decline with tariff rate in percentage terms.
    
    Args:
        results: Dictionary containing experiment results
        output_dir: Directory to save plots
    """
    # Extract data
    tariffs = [float(t) for t in results['equilibrium'].keys()]
    sorted_indices = np.argsort(tariffs)
    tariffs = [tariffs[i] for i in sorted_indices]
    
    # Get market values for each tariff rate
    market_values = [results['equilibrium'][str(t)]['value'] for t in tariffs]
    
    # Calculate percentage change from baseline (tariff=0)
    baseline_value = market_values[tariffs.index(0.0)]
    pct_change = [(v / baseline_value - 1) * 100 for v in market_values]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot percentage change
    ax.plot(tariffs, pct_change, linewidth=2.5, color='#d62728', 
           label='Market Value % Change')
    
    # Add reference line at 0%
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5, 
              label='No Change (Baseline)')
    
    # Highlight specific points with markers
    key_tariffs = [0.0, 0.1, 0.25, 0.5]
    key_points_x = []
    key_points_y = []
    
    for tau in key_tariffs:
        if tau in tariffs:
            idx = tariffs.index(tau)
            key_points_x.append(tau)
            key_points_y.append(pct_change[idx])
    
    # Add markers for key points
    ax.scatter(key_points_x, key_points_y, s=80, color='#2ca02c', 
              zorder=5, label='Key Tariff Levels')
    
    # Axis labels and title
    ax.set_xlabel('Tariff Rate')
    ax.set_ylabel('Market Value Change (%)')
    ax.set_title('Market Value Decline Due to Tariffs')
    
    # Format x-axis as percentage
    ax.set_xlim(left=-0.01, right=max(tariffs) * 1.05)
    ax.set_xticks([0, 0.1, 0.25, 0.5])
    ax.set_xticklabels(['0%', '10%', '25%', '50%'])
    
    # Format y-axis with percentage sign
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0f}%'))
    
    # Ensure the y-axis shows the decline clearly
    y_min = min(min(pct_change) * 1.1, -5)  # At least -5% to show context
    y_max = max(max(pct_change) * 1.1, 5)   # At least 5% to show context
    ax.set_ylim(bottom=y_min, top=y_max)
    
    # Add legend
    ax.legend(loc='best', frameon=True, framealpha=0.9)
    
    plt.savefig(os.path.join(output_dir, "market_value_decline.png"))
    plt.close()


def plot_supply_demand_curves(results, output_dir):
    """
    Plot supply and demand curves for specific tariff rates.
    
    Args:
        results: Dictionary containing experiment results
        output_dir: Directory to save plots
    """
    params = results['parameters']
    a, b, S, c_s = params['a'], params['b'], params['S'], params['c_s']
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create price range for plotting
    p_star_range = np.linspace(c_s, a/b, 100)
    
    # Plot supply curve: Q = S*(P* - c_s)
    supply = S * (p_star_range - c_s)
    ax.plot(supply, p_star_range, linewidth=2.5, 
           label='Supply Curve', color='#1f77b4')
    
    # Select specific tariff rates to plot
    tariff_rates = [0.0, 0.1, 0.25, 0.5]
    colors = ['#2ca02c', '#ff7f0e', '#9467bd', '#d62728']
    
    for i, tau in enumerate(tariff_rates):
        # Demand curve with tariff: Q = a - b*((1+tau)*P*)
        demand = a - b * ((1 + tau) * p_star_range)
        
        # Only plot positive quantities
        positive_mask = demand > 0
        ax.plot(demand[positive_mask], p_star_range[positive_mask], 
                linewidth=2.5, 
                label=f'Demand (τ={tau:.0%})',
                color=colors[i])
        
        # Mark equilibrium point if available
        if str(tau) in results['equilibrium']:
            eq = results['equilibrium'][str(tau)]
            ax.scatter([eq['Q']], [eq['P_star']], color=colors[i], s=80, zorder=5)
    
    ax.set_xlabel('Quantity')
    ax.set_ylabel('Producer Price')
    ax.set_title('Supply and Demand Curves with Tariffs')
    
    # Add legend
    ax.legend(loc='best', frameon=True, framealpha=0.9)
    
    plt.savefig(os.path.join(output_dir, "supply_demand_curves.png"))
    plt.close()


def generate_plots(results, output_dir):
    """Generate the focused plots for portfolio managers."""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up clean plotting style
    setup_plotting_style()
    
    # Generate the focused plots
    plot_market_value_percentage_change(results, output_dir)
    plot_supply_demand_curves(results, output_dir)
    
    print(f"Portfolio manager plots saved to {output_dir}/")


def main():
    """Main function to run the plotting script."""
    # Parse command line arguments
    args = parse_args()
    
    # Create output directory
    output_dir = args.out_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # Load results directly from the final_info.json file
    results_path = os.path.join(output_dir, "final_info.json")
    
    if not os.path.exists(results_path):
        print(f"Error: Results file not found at {results_path}")
        return
    
    # Load results
    results = load_results(results_path)
    
    # Generate the plots
    generate_plots(results, output_dir)


if __name__ == "__main__":
    main() 