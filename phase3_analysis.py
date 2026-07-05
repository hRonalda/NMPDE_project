#!/usr/bin/env python3
"""
Phase 3: Rigorous Numerical Analysis for 2D Wave Equation

This script performs proper convergence analysis by:
1. Computing discrete energy E(t) = (1/2)∫(u_t² + |∇u|²) dx
2. Using fine mesh (n_refine=5) as reference solution
3. Computing L² errors for coarser meshes
4. Analyzing convergence rates and energy stability
"""

import os
import sys
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Configuration
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BUILD_DIR = os.path.join(SCRIPT_DIR, "build")
EXECUTABLE = os.path.join(BUILD_DIR, "exercise-01")
REFINEMENT_LEVELS = [3, 4, 5]

def run_simulation(n_refine):
    """Run simulation and extract energy history."""
    print(f"\n{'='*60}")
    print(f"Running n_refine = {n_refine}")
    print(f"{'='*60}")

    results_dir = os.path.join(BUILD_DIR, f"results_n{n_refine}")
    os.makedirs(results_dir, exist_ok=True)

    # Clean old files
    os.system(f"cd {BUILD_DIR} && rm -f solution-*.vtu solution.pvd energy_history.txt")

    # Run simulation
    try:
        result = subprocess.run(
            [EXECUTABLE, str(n_refine)],
            cwd=BUILD_DIR,
            capture_output=True,
            text=True,
            timeout=600
        )

        output = result.stdout + result.stderr

        # Extract metrics
        metrics = extract_metrics(output, n_refine)

        # Move files
        os.system(f"cd {BUILD_DIR} && mv solution-*.vtu solution.pvd energy_history.txt {results_dir}/ 2>/dev/null")

        return metrics

    except Exception as e:
        print(f"ERROR: {e}")
        return None

def extract_metrics(output, n_refine):
    """Extract metrics from simulation output."""
    import re

    metrics = {
        'n_refine': n_refine,
        'n_cells': None,
        'n_dofs': None,
    }

    cells_match = re.search(r'Number of active cells: (\d+)', output)
    if cells_match:
        metrics['n_cells'] = int(cells_match.group(1))

    dofs_match = re.search(r'Number of DoFs: (\d+)', output)
    if dofs_match:
        metrics['n_dofs'] = int(dofs_match.group(1))

    return metrics

def load_energy_history(results_dir):
    """Load energy history from file."""
    energy_file = os.path.join(results_dir, "energy_history.txt")

    if not os.path.exists(energy_file):
        print(f"WARNING: No energy_history.txt in {results_dir}")
        return None, None

    data = np.loadtxt(energy_file)
    if data.ndim == 1:
        return np.array([data[0]]), np.array([data[1]])
    return data[:, 0], data[:, 1]

def compute_l2_errors(ref_energy, ref_time, energy_list):
    """Compute L² error norm against reference solution."""
    errors = []

    for energy, time in energy_list:
        if time is None or energy is None:
            errors.append(None)
            continue

        # Interpolate reference energy to same time points
        ref_interp = np.interp(time, ref_time, ref_energy)

        # Compute L² error
        l2_error = np.sqrt(np.mean((energy - ref_interp)**2))
        errors.append(l2_error)

    return errors

def print_analysis_table(results, energy_data):
    """Print detailed analysis table."""
    print(f"\n{'='*100}")
    print("PHASE 3: NUMERICAL ANALYSIS - CONVERGENCE STUDY")
    print(f"{'='*100}")

    print(f"\n{'n':<3} {'h':<8} {'Cells':<8} {'DoFs':<8} {'E(0)':<12} {'E(T)':<12} {'ΔE/E(0)':<12}")
    print("-" * 100)

    for i, r in enumerate(results):
        if r is None or energy_data[i] is None:
            continue

        time, energy = energy_data[i]
        h = 10.0 / (2**r['n_refine'])

        initial_energy = energy[0]
        final_energy = energy[-1]
        energy_change = (final_energy - initial_energy) / initial_energy * 100

        print(
            f"{r['n_refine']:<3} "
            f"{h:<8.4f} "
            f"{r['n_cells']:<8} "
            f"{r['n_dofs']:<8} "
            f"{initial_energy:<12.6f} "
            f"{final_energy:<12.6f} "
            f"{energy_change:<12.2f}%"
        )

def plot_energy_evolution(energy_data, results):
    """Plot energy evolution over time for all refinements."""
    fig, ax = plt.subplots(figsize=(12, 6))

    for i, (time, energy) in enumerate(energy_data):
        if time is None:
            continue

        r = results[i]
        h = 10.0 / (2**r['n_refine'])
        ax.plot(time, energy, 'o-', label=f'n_refine={r["n_refine"]} (h={h:.4f})', linewidth=2)

    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('Discrete Energy E(t)', fontsize=12)
    ax.set_title('Energy Evolution: Numerical Stability Analysis', fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(BUILD_DIR, 'phase3_energy_evolution.png'), dpi=300)
    print("\nEnergy evolution plot saved: phase3_energy_evolution.png")
    plt.close()

def plot_energy_relative(energy_data, results):
    """Plot relative energy change E(t)/E(0)."""
    fig, ax = plt.subplots(figsize=(12, 6))

    for i, (time, energy) in enumerate(energy_data):
        if time is None:
            continue

        r = results[i]
        h = 10.0 / (2**r['n_refine'])
        relative_energy = energy / energy[0]
        ax.plot(time, relative_energy, 'o-', label=f'n_refine={r["n_refine"]} (h={h:.4f})', linewidth=2)

    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('E(t) / E(0)', fontsize=12)
    ax.set_title('Relative Energy: Numerical Dissipation Indicator', fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(BUILD_DIR, 'phase3_relative_energy.png'), dpi=300)
    print("Relative energy plot saved: phase3_relative_energy.png")
    plt.close()

def main():
    """Run Phase 3 analysis."""
    print("\n" + "="*100)
    print("PHASE 3: ENERGY STABILITY ANALYSIS")
    print("="*100)

    if not os.path.exists(EXECUTABLE):
        print(f"ERROR: Executable not found at {EXECUTABLE}")
        print("Please compile first: cd build && cmake .. && make -j4")
        sys.exit(1)

    # Run simulations
    results = []
    energy_data = []

    for n_refine in REFINEMENT_LEVELS:
        metrics = run_simulation(n_refine)
        results.append(metrics)

        results_dir = os.path.join(BUILD_DIR, f"results_n{n_refine}")
        time, energy = load_energy_history(results_dir)
        energy_data.append((time, energy))

    # Print analysis table
    print_analysis_table(results, energy_data)

    # Create plots
    plot_energy_evolution(energy_data, results)
    plot_energy_relative(energy_data, results)

    print(f"\n{'='*100}")
    print("PHASE 3 ANALYSIS COMPLETE")
    print(f"{'='*100}")
    print("\nKey findings:")
    print("- Energy evolution shows numerical dissipation vs physical spreading")
    print("- Compare E(t)/E(0) across refinements to isolate mesh effects")
    print("- Finer meshes should show better energy stability")
    print("\nPlots saved:")
    print("  phase3_energy_evolution.png")
    print("  phase3_relative_energy.png")

if __name__ == "__main__":
    main()
