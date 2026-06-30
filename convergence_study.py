#!/usr/bin/env python3
"""
Convergence Study for 2D Wave Equation Project

This script:
1. Runs the wave equation solver with different mesh refinement levels
2. Extracts key metrics (cells, DoFs, max amplitude over time)
3. Creates convergence tables and plots
4. Analyzes numerical dissipation
"""

import os
import sys
import subprocess
import re
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Configuration
# Use relative path to build directory
BUILD_DIR = os.path.dirname(os.path.abspath(__file__)) + "/build"
if not os.path.exists(BUILD_DIR):
    BUILD_DIR = "./build"  # Fallback if script is run from build directory

EXECUTABLE = os.path.join(BUILD_DIR, "exercise-01")
REFINEMENT_LEVELS = [3, 4, 5]  # Test n_refine = 3, 4, 5

def run_simulation(n_refine):
    """Run simulation with given refinement level."""
    print(f"\n{'='*60}")
    print(f"Running simulation with n_refine = {n_refine}")
    print(f"{'='*60}")

    # Create results directory
    results_dir = os.path.join(BUILD_DIR, f"results_n{n_refine}")
    os.makedirs(results_dir, exist_ok=True)

    # Clean old files
    os.system(f"cd {BUILD_DIR} && rm -f solution-*.vtu solution.pvd")

    # Run simulation
    try:
        result = subprocess.run(
            [EXECUTABLE, str(n_refine)],
            cwd=BUILD_DIR,
            capture_output=True,
            text=True,
            timeout=300
        )

        output = result.stdout + result.stderr

        # Save output log
        with open(os.path.join(results_dir, "output.log"), "w") as f:
            f.write(output)

        # Extract metrics from output
        metrics = extract_metrics(output, n_refine)

        # Move solution files to results directory
        os.system(f"cd {BUILD_DIR} && mv solution-*.vtu solution.pvd {results_dir}/ 2>/dev/null")

        return metrics

    except subprocess.TimeoutExpired:
        print(f"ERROR: Simulation timed out for n_refine={n_refine}")
        return None
    except Exception as e:
        print(f"ERROR: Failed to run simulation: {e}")
        return None

def extract_metrics(output, n_refine):
    """Extract key metrics from simulation output."""
    metrics = {
        'n_refine': n_refine,
        'n_cells': None,
        'n_dofs': None,
        'max_amplitude_t0': None,
        'max_amplitude_t1': None,
        'amplitude_loss': None,
        'final_time': 1.0
    }

    # Extract number of cells
    cells_match = re.search(r'Number of active cells: (\d+)', output)
    if cells_match:
        metrics['n_cells'] = int(cells_match.group(1))

    # Extract number of DoFs
    dofs_match = re.search(r'Number of DoFs: (\d+)', output)
    if dofs_match:
        metrics['n_dofs'] = int(dofs_match.group(1))

    # Extract amplitude at t=0
    first_amplitude = re.search(r'DEBUG after interpolation:.*?solution\|\|_linfty = ([\d.e+-]+)', output, re.DOTALL)
    if first_amplitude:
        metrics['max_amplitude_t0'] = float(first_amplitude.group(1))

    # Extract amplitude at t=1 (last output line with solution norm)
    all_amplitudes = re.findall(r'\|\|solution\|\|_linfty = ([\d.e+-]+)', output)
    if all_amplitudes:
        metrics['max_amplitude_t1'] = float(all_amplitudes[-1])

    # Compute amplitude loss
    if metrics['max_amplitude_t0'] and metrics['max_amplitude_t1']:
        metrics['amplitude_loss'] = (
            (metrics['max_amplitude_t0'] - metrics['max_amplitude_t1']) /
            metrics['max_amplitude_t0'] * 100
        )

    return metrics

def print_convergence_table(results):
    """Print mesh refinement study table."""
    print(f"\n{'='*80}")
    print("MESH REFINEMENT STUDY RESULTS")
    print(f"{'='*80}")

    print(f"\n{'n_refine':<10} {'Cells':<10} {'DoFs':<10} {'h':<12} {'||u(0)||∞':<14} {'||u(1)||∞':<14} {'Amplitude Loss':<12}")
    print("-" * 80)

    for r in results:
        if r is None:
            continue

        h = 10.0 / (2**r['n_refine'])

        print(
            f"{r['n_refine']:<10} "
            f"{r['n_cells']:<10} "
            f"{r['n_dofs']:<10} "
            f"{h:<12.4f} "
            f"{r['max_amplitude_t0']:<14.6f} "
            f"{r['max_amplitude_t1']:<14.6f} "
            f"{r['amplitude_loss']:<12.2f}%"
        )

    print("-" * 80)

def plot_convergence(results):
    """Create convergence plots."""
    results = [r for r in results if r is not None]

    if not results:
        print("No valid results to plot")
        return

    n_refines = [r['n_refine'] for r in results]
    n_cells = [r['n_cells'] for r in results]
    amplitude_losses = [r['amplitude_loss'] for r in results]
    h_values = [10.0 / (2**(r['n_refine']-1) * 4) for r in results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Dissipation vs refinement level
    ax1.plot(n_refines, amplitude_losses, 'o-', linewidth=2, markersize=8)
    ax1.set_xlabel('Refinement level (n_refine)', fontsize=12)
    ax1.set_ylabel('Amplitude Loss (%)', fontsize=12)
    ax1.set_title('Amplitude Decay Indicator vs Mesh Refinement', fontsize=13)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(n_refines)

    # Plot 2: Dissipation vs mesh size (log-log)
    ax2.loglog(h_values, amplitude_losses, 'o-', linewidth=2, markersize=8)
    ax2.set_xlabel('Mesh size h', fontsize=12)
    ax2.set_ylabel('Amplitude Loss (%)', fontsize=12)
    ax2.set_title('Amplitude Decay Indicator vs Mesh Size (log-log)', fontsize=13)
    ax2.grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    plt.savefig(os.path.join(BUILD_DIR, 'convergence_study.png'), dpi=300)
    print("\nPlot saved to: convergence_study.png")
    plt.close()

def main():
    """Run convergence study."""
    print("\n" + "="*80)
    print("PHASE 2: MESH REFINEMENT STUDY")
    print("="*80)
    print(f"Testing refinement levels: {REFINEMENT_LEVELS}")
    print(f"Domain: [-5, 5]²")
    print(f"Initial condition: u₀(x,y) = exp(-(x²+y²)/4)")
    print(f"Final time: 1.0")

    # Check if executable exists
    if not os.path.exists(EXECUTABLE):
        print(f"\nERROR: Executable not found at {EXECUTABLE}")
        print("Please compile the project first:")
        print(f"  cd {BUILD_DIR} && cmake .. && make -j4")
        sys.exit(1)

    # Run simulations
    results = []
    for n_refine in REFINEMENT_LEVELS:
        metrics = run_simulation(n_refine)
        results.append(metrics)

    # Print results table
    print_convergence_table(results)

    # Create plots
    plot_convergence(results)

    print(f"\n{'='*80}")
    print("CONVERGENCE STUDY COMPLETE")
    print(f"{'='*80}")
    print("\nResults saved to:")
    for n_refine in REFINEMENT_LEVELS:
        print(f"  results_n{n_refine}/")
    print("\nTo visualize results in ParaView:")
    for n_refine in REFINEMENT_LEVELS:
        print(f"  paraview results_n{n_refine}/solution.pvd &")

if __name__ == "__main__":
    main()
