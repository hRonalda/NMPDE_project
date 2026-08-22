#!/usr/bin/env python3
"""
Energy-conservation figure for the report (docs/analysis.md, Part 1;
docs/validation.md, "Energy conservation" section).

Reads build/energy_history.txt (written by Wave::run() for the most
recently executed case) and plots the discrete energy

    E^n = 1/2 ( V^n . M V^n + U^n . K U^n )

against time in two panels:

    left  : full-range view, y-axis anchored to include 0, showing that
            E(t) is a flat plateau rather than decaying or growing.
    right : zoomed to the data range, revealing the O(dt^2) bounded
            oscillation and confirming there is no systematic drift.
            Annotated with the relative drift (max-min)/E^0.

Run caseA (k=m=1, f=0) first to generate energy_history.txt:

    cd build && ./exercise-01 5 caseA
"""

import os
import numpy as np
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BUILD_DIR = os.path.join(SCRIPT_DIR, "build")
ENERGY_FILE = os.path.join(BUILD_DIR, "energy_history.txt")
FIGURES_DIR = os.path.join(SCRIPT_DIR, "figures")
OUTPUT_FILE = os.path.join(FIGURES_DIR, "energy_conservation.png")


def main():
    if not os.path.exists(ENERGY_FILE):
        raise SystemExit(
            f"ERROR: {ENERGY_FILE} not found.\n"
            "Run the solver first, e.g.:\n"
            "  cd build && ./exercise-01 5 caseA"
        )

    data = np.loadtxt(ENERGY_FILE)
    t, E = data[:, 0], data[:, 1]

    E0 = E[0]
    E_min, E_max = E.min(), E.max()
    rel_drift = (E_max - E_min) / E0

    # Linear-fit "trend" as a secondary, stricter check for systematic
    # drift (as opposed to bounded oscillation): slope * (t[-1]-t[0]) / E0.
    slope, intercept = np.polyfit(t, E, 1)
    trend_over_run = slope * (t[-1] - t[0]) / E0

    print(f"Energy samples: {len(t)}")
    print(f"t range: [{t[0]:.4f}, {t[-1]:.4f}]")
    print(f"E^0 = {E0:.8f}")
    print(f"E_min = {E_min:.8f}, E_max = {E_max:.8f}")
    print(f"Relative drift (max-min)/E^0 = {rel_drift:.6e}")
    print(f"Linear-fit trend over run / E^0 = {trend_over_run:.6e}")

    os.makedirs(FIGURES_DIR, exist_ok=True)

    fig, (ax_full, ax_zoom) = plt.subplots(1, 2, figsize=(12, 5))

    # --- Left panel: full-range view ---------------------------------
    ax_full.plot(t, E, color="tab:blue", linewidth=1.2)
    ax_full.axhline(E0, color="gray", linestyle="--", linewidth=0.8,
                     label=f"E$^0$ = {E0:.4f}")
    y_top = max(1.15 * E0, E_max * 1.05)
    ax_full.set_ylim(0, y_top)
    ax_full.set_xlim(t[0], t[-1])
    ax_full.set_xlabel("Time t")
    ax_full.set_ylabel("Discrete energy E")
    ax_full.set_title("Energy history (full range)")
    ax_full.legend(loc="lower right", fontsize=9)
    ax_full.grid(True, alpha=0.3)

    # --- Right panel: zoomed to reveal the bounded oscillation --------
    ax_zoom.plot(t, E, color="tab:blue", linewidth=1.0, marker="o",
                 markersize=2)
    ax_zoom.axhline(E0, color="gray", linestyle="--", linewidth=0.8)
    pad = 0.15 * (E_max - E_min) if E_max > E_min else 1e-12
    ax_zoom.set_ylim(E_min - pad, E_max + pad)
    ax_zoom.set_xlim(t[0], t[-1])
    ax_zoom.set_xlabel("Time t")
    ax_zoom.set_ylabel("Discrete energy E")
    ax_zoom.set_title("Energy history (zoomed): bounded oscillation, no drift")
    ax_zoom.grid(True, alpha=0.3)

    annotation = (
        f"(E$_{{max}}$ - E$_{{min}}$) / E$^0$ = {rel_drift:.2e}\n"
        f"linear-fit trend / E$^0$ = {trend_over_run:.2e}"
    )
    ax_zoom.text(
        0.03, 0.05, annotation, transform=ax_zoom.transAxes,
        fontsize=10, va="bottom", ha="left",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85,
                  edgecolor="gray"),
    )

    fig.suptitle(
        "Discrete energy conservation: central-difference (leapfrog) scheme, "
        "Q1, case A (k=m=1, f=0)",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(OUTPUT_FILE, dpi=300)
    print(f"\nFigure saved: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
