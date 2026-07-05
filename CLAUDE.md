# CLAUDE.md

## Project
2D **finite element solver for the wave equation**, for a numerical PDE
course (Project 2, Politecnico di Milano — report goes to
michele.bucelli@polimi.it).

Problem:
- ∂²u/∂t² − Δu = f   in Ω
- u = g              on ∂Ω   (Dirichlet)
- u(t=0) = u₀        in Ω
- ∂u/∂t = u₁         in Ω

Discretization: P1 elements on a **structured** mesh, explicit/implicit
time stepping (see report for scheme choice).

## Grading focus (what the report must show)
The report is marked on the numerical understanding, not just working code:
- Choice of time and space discretization, and *why*.
- Numerical **dissipation** and **dispersion** properties of the scheme.
- **CFL / stability** condition for the chosen time integration.
- Weak formulation and its discrete form.
- Validation against a known solution (convergence study).

When touching the solver, preserve and be able to explain these properties.

## Important conventions (do not break these)
- **Keep the mesh structured**, not unstructured. This project will be
  extended into an HPC performance-engineering project later; a structured
  grid keeps that tractable. Do not switch to a general unstructured mesh.
- Keep **assembly**, **time-stepping**, and **I/O** in separate functions /
  units, so optimized versions (OpenMP, SIMD, CUDA) can be swapped in later
  without rewriting the solver.
- C++20. Build system: CMake.
- Don't rewrite working code wholesale — read and extend. Ask before large
  refactors.
- This is a **deal.II** project. Use deal.II's facilities (mesh, DoFHandler,
  FEValues, sparse matrices, solvers) rather than writing FEM machinery from
  scratch. Follow the deal.II 9.5 API.

## Environment
- Runs **inside a Docker container** (image: quay.io/pjbaioni/amsc_mk:2025),
  accessed via VS Code Dev Containers on a Mac. Toolchain lives in the
  container — build and test here, not on the host.
- Depends on **deal.II 9.5.1** and **Boost 1.83.0** (both preinstalled in
  the container). This is a deal.II-based FEM project.



## Build / Run
```bash
cd build
cmake ..          # Release build; finds deal.II at /usr, Boost 1.83
make              # builds target: exercise-01
./exercise-01     # run from the build/ directory

# Convergence / validation plots
python3 convergence_study.py
```
> NOTE: verify the exact target and binary name against CMakeLists.txt.

# Convergence / validation plots
python3 convergence_study.py
```
> NOTE: verify the exact target and binary name against CMakeLists.txt.

## Key files
- `src/exercise-01.cpp` — main entry point.
- `src/Wave.cpp`, `src/Wave.hpp` — the wave-equation solver class.
- `CMakeLists.txt` — deal.II autopilot build config.
- `convergence_study.py`, `phase3_analysis.py` — validation / analysis plots.
- `mesh/` — mesh files.
- `figures/`, `convergence_study.png` — results for the report.
- `docs/`, `text/` — report material.

## Workflow notes
- Verify correctness against a known analytical solution before optimizing.
- After changes, rebuild and run the convergence study to confirm nothing
  regressed.

## In the terminal, write formulas in plain ASCII (u_tt, a_k^2, omega) instead of LaTeX, since the terminal can't render it.