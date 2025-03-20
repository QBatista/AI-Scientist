#!/usr/bin/env python3
"""
Stylized partial-equilibrium model for Japanese car imports into the U.S.
under an ad valorem tariff.

This script computes equilibrium values for different tariff rates and saves
the results to a JSON file for further analysis and visualization.
"""

import numpy as np
import sympy
import json
import os
import argparse
from pathlib import Path


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run tariff experiment and save results")
    parser.add_argument("--out_dir", type=str, default="run_0", 
                        help="Output directory for results")
    return parser.parse_args()


def solve_equilibrium(a, b, S, c_s, tau):
    """
    Solve for the partial-equilibrium under an ad valorem tariff tau.
    
    Demand: Qd(P) = a - b*P,    where P = (1 + tau)*P*
    Supply: Qs(P*) = S*(P* - c_s)  if P* >= c_s, else 0
    
    We want Qd( (1+tau)*P* ) = Qs(P* ).
    
    Returns a dictionary with equilibrium values:
       P_star: net-of-tariff (producer) price
       P     : retail price to consumers
       Q     : equilibrium quantity
       profit: (P_star - c_s)*Q   (simple representation)
    """
    # Define the symbolic variable
    P_star = sympy.Symbol('P_star', real=True, nonnegative=True)
    
    # Demand at consumer price: Qd = a - b*((1+tau)*P_star)
    Qd_expr = a - b*((1 + tau)*P_star)
    
    # Supply at producer price: Qs = S*(P_star - c_s), but zero if P_star < c_s.
    # We'll ignore the corner solution (P_star < c_s => Q=0 => no trade)
    # and solve for P_star >= c_s. We'll check viability afterwards.
    Qs_expr = S*(P_star - c_s)
    
    # Solve Qd_expr = Qs_expr for P_star
    solution = sympy.solve(sympy.Eq(Qd_expr, Qs_expr), P_star, dict=True)
    
    # We might get multiple solutions or complex solutions; pick the physically valid one
    P_star_eq = None
    for sol in solution:
        val = sol[P_star]
        # We want real, nonnegative, and P_star >= c_s
        if val.is_real and val >= c_s:
            P_star_eq = val
            break
    
    # If no valid solution found, it means equilibrium might be at a corner or no trade
    if P_star_eq is None:
        # In a corner scenario, we might have P_star < c_s => supply=0 => Q=0,
        # but then price from demand side would be a / b => not meaningful for positive Q.
        # We'll just return zero trade scenario
        return {
            'P_star': 0.0,
            'P': 0.0,
            'Q': 0.0,
            'profit': 0.0
        }
    
    # Now compute the equilibrium values
    P_star_eq = float(P_star_eq)
    P_eq = (1 + tau) * P_star_eq
    Q_eq = float(a - b*P_eq)  # or S*(P_star_eq - c_s)
    
    # Producer profit in this simplified model
    profit_eq = (P_star_eq - c_s)*Q_eq
    
    return {
        'P_star': P_star_eq,
        'P': P_eq,
        'Q': Q_eq,
        'profit': profit_eq
    }

def run_experiment():
    """
    Run the tariff experiment and return results.
    
    Returns:
        dict: Results of the experiment
    """
    # -------------------------------
    # 1) Parameter Setup
    # -------------------------------
    a = 1000.0   # Intercept of demand
    b = 2.0      # Slope of demand (larger b => more elastic demand)
    S = 4.0      # Slope of supply above c_s
    c_s = 100.0  # Base cost (minimum price at which supply is positive)
    
    # Discount rate (for present value calculations)
    r = 0.05
    
    # More granular tariff rates for analysis (0% to 50%)
    tau_values = np.linspace(0, 0.50, 51).tolist()  # includes both 0% and 50%
    
    # Specific tariff rates of interest
    key_tau_values = [0.0, 0.10, 0.25, 0.50]
    
    # -------------------------------
    # 2) Solve for Equilibria
    # -------------------------------
    results = {}
    
    # Store parameters for reference
    results['parameters'] = {
        'a': a,
        'b': b,
        'S': S,
        'c_s': c_s,
        'r': r
    }
    
    # Compute equilibria for each tariff rate
    results['equilibrium'] = {}
    for tau in tau_values:
        eq = solve_equilibrium(a, b, S, c_s, tau)
        # Compute "value" from US operations as profit / r (very simplified)
        eq['value'] = eq['profit'] / r if r > 0 else float('nan')
        # Store tariff as a string key (JSON doesn't support float keys)
        results['equilibrium'][str(tau)] = eq
    
    # Compute percentage changes from baseline (0% tariff)
    results['comparisons'] = {}
    baseline = results['equilibrium']['0.0']
    
    for tau in key_tau_values:
        if tau == 0.0:
            continue  # Skip baseline
            
        current = results['equilibrium'][str(tau)]
        pct_changes = {}
        
        for metric in ['P_star', 'P', 'Q', 'profit', 'value']:
            if baseline[metric] != 0:
                pct_change = 100.0 * (current[metric] - baseline[metric]) / baseline[metric]
                pct_changes[metric] = pct_change
            else:
                pct_changes[metric] = float('nan')
                
        results['comparisons'][str(tau)] = pct_changes
    
    # Add final prediction for the drop in market value with a 25% tariff
    results['final_prediction'] = results['comparisons']['0.25']['value']
    
    return results

def main():
    """Main function to run the experiment."""
    # Parse command line arguments
    args = parse_args()
    
    # Create output directory
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    
    # Run the experiment
    results = run_experiment()
    
    # Save results to final_info.json in the output directory
    final_info_path = os.path.join(out_dir, "final_info.json")
    with open(final_info_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {final_info_path}")
    
    print(f"Experiment complete. Results saved to {out_dir}/")

if __name__ == "__main__":
    main()