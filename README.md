# Wave Equation Solver — Project 2

2D finite element solver for the wave equation (Numerical Methods for
PDEs, Politecnico di Milano — Project 2).

## Problem Statement

```
∂²u/∂t² - Δu = f  in Ω
u = g           on ∂Ω
u(t=0) = u₀     in Ω
∂u/∂t = u₁      in Ω
```

## Method

- **Domain**: fixed square Ω = [-5, 5]² (`Wave::setup()`,
  `GridGenerator::hyper_cube`).
- **Space discretization**: Q1 (bilinear) finite elements on a
  **structured**, uniform quadrilateral mesh, refined globally
  (`n_refine` levels, h = 10 / 2^n_refine). Consistent mass matrix (no
  lumping).
- **Time discretization**: central differences (leapfrog),
  `M(U^{n+1} - 2U^n + U^{n-1})/Δt² + K U^n = F^n`, second-order accurate
  and non-dissipative within its stability limit. `U^{-1}` is
  initialized to second-order accuracy from `u0`, `u1`, and the initial
  acceleration `A^0` (solved from `M A^0 = F(0) - K U^0`), rather than
  the naive first-order `U^{-1} = U^0`.
- **Linear solver**: conjugate gradient with a Jacobi preconditioner
  (`M` is SPD, so CG applies directly; no other solver is used).
- **Boundary conditions**: Dirichlet `u = g` imposed by symmetric
  algebraic elimination (`MatrixTools::apply_boundary_values`) against
  the new time level each step. `g` is optional — `nullptr` gives the
  homogeneous case `g = 0`; a non-homogeneous, time-dependent `g` is
  supported and validated (case D, below). See `docs/analysis.md` for
  the full derivation of why this reproduces the correct
  boundary-acceleration coupling with no separate lifting function.
- Built on **deal.II 9.5.1** and **Boost 1.83.0**, using deal.II's mesh,
  `DoFHandler`, `FEValues`, and Trilinos-backed sparse matrices/vectors
  (`TrilinosWrappers`) throughout — no hand-rolled FEM machinery.
- Runs in a Docker container (image `quay.io/pjbaioni/amsc_mk:2025`);
  MPI-aware but exercised here at a single process.

## Project Structure

```
pde_project/
├── CMakeLists.txt          # Build configuration (deal.II autopilot)
├── README.md               # This file
├── projects.pdf             # Course project descriptions (Project 2, p.3)
├── src/
│   ├── Wave.hpp             # Solver class, exact-solution / IC classes
│   ├── Wave.cpp             # setup / assembly / time-stepping / output
│   └── exercise-01.cpp      # Main entry point, test-case selection
├── docs/
│   ├── validation.md        # Manufactured-solution derivations, case
│   │                         # results (caseA/B/C), energy conservation
│   └── analysis.md          # Dispersion relation and CFL derivations,
│                             # measured verification
├── convergence_study.py     # Runs the solver across refinement levels
├── phase3_analysis.py       # Energy-history analysis / plots
├── energy_plot.py           # Energy-conservation report figure
├── figures/                 # Plots for the report
├── mesh/                    # (unused — the mesh is generated in code
│                             # via GridGenerator::hyper_cube, not read
│                             # from a file)
└── text/                    # Report source (not yet written)
```

## Build Instructions

```bash
cd build
cmake ..          # Release build; finds deal.II at /usr, Boost 1.83
make              # builds target: exercise-01
```

## Running

```bash
./exercise-01 [n_refine] [test_case]
```

`n_refine` defaults to 3; `test_case` defaults to `gaussian`.

| test_case  | Validates | Description |
|------------|-----------|-------------|
| `gaussian` | — (qualitative) | Gaussian bump released from rest on Ω, f = 0. No exact solution; used to inspect wave propagation/reflection in ParaView and to illustrate dispersion (see `docs/validation.md`). |
| `caseA`    | Baseline convergence | Manufactured eigenmode, homogeneous g = 0, **f = 0** (free vibration). Runs one full period. |
| `caseB`    | Load-vector assembly | Same mode, different ω, so **f ≠ 0**. Case A alone can't catch a broken forcing term; case B can. |
| `caseC`    | Nonzero initial velocity | Same free-vibration mode as case A but as `sin(ωt)` instead of `cos(ωt)`, giving **u₀ = 0, u₁ = ω·S(x,y) ≠ 0** — exercises the nonzero-`u1` branch of the `U^{-1}` startup, which A and B never touch. |
| `caseD`    | Non-homogeneous, time-dependent g | Cosine eigenmode; f = 0 but the boundary trace **g ≠ 0 and oscillates in time** — exercises the per-step boundary elimination and the `g_tt(0) ≠ 0` startup branch, which A/B/C (all g = 0) cannot detect. |

Each of caseA–D prints the L2 and H1-seminorm errors against its exact
solution at the final time. Convergence/validation plots:

```bash
python3 convergence_study.py
python3 energy_plot.py
```

## Validation results

Full derivations, symbolic verification, and measured tables are in
`docs/validation.md` and `docs/analysis.md`. Summary:

- **Convergence** (caseA, B, D — manufactured solutions, measured at
  n_refine = 3–6): L2 error rate **2**, H1-seminorm error rate **1**, as
  expected for Q1 + central differences. (caseC measures rate 2 in both
  norms at the specific full-period sampling time used — an explained,
  verified superconvergence effect of that measurement point, not a
  different underlying accuracy; see `docs/validation.md`.)
- **Energy conservation** (caseA): discrete energy stays flat to a
  relative bound of ~2×10⁻⁴ over a full period with no systematic
  drift, confirming the leapfrog scheme is non-dissipative within its
  stability limit. (With non-homogeneous time-dependent g, as in caseD,
  the energy genuinely oscillates because the boundary does work on the
  system — this is physics, not dissipation; see `docs/analysis.md`.)
- **Dispersion**: derived in closed form for Q1 (consistent mass) +
  leapfrog; leading phase error is **O(h²)**, a phase **lead** (numerical
  waves travel slightly too fast) rather than a lag. Verified
  empirically: measured phase error matches the closed-form prediction
  to 6–7 significant digits and converges at rate 2.00.
- **CFL / stability**: derived limit **Δt < h/√6 ≈ 0.408h** for Q1 with
  a consistent mass matrix (the production runs use Δt ≈ 0.2h, about
  half the limit). Verified empirically: stable just below the limit,
  and unstable growth just above it matches the predicted amplification
  factor to 6 significant digits.

## Status

The solver is functionally complete and validated: general f, u₀, u₁,
and g (homogeneous or not, time-dependent or not) are all implemented
and each has a dedicated test case proving it. Convergence rates,
energy conservation, dispersion, and the CFL limit are all derived
*and* measured, not just asserted.

**What remains is the report itself** (`text/` is currently empty) —
the numerical content above is ready to be assembled into it.

## Authors
Salvatore Mariano Librici - salvatoremariano.librici@mail.polimi.it
Rong Huang - rong.huang@mail.polimi.it
Hirdesh Kumar - hirdesh.kumar@mail.polimi.it
Mehdi Ghiasipour - mehdi.ghiasipour@mail.polimi.it

