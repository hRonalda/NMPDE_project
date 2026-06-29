# Wave Equation Solver - Project 2

## Problem Statement

Solve the 2D wave equation:

```
∂²u/∂t² - Δu = f  in Ω
u = g           on ∂Ω
u(t=0) = u₀     in Ω
∂u/∂t = u₁      in Ω
```

## Project Structure

```
pde_project/
├── CMakeLists.txt          # Build configuration
├── README.md               # This file
├── src/
│   ├── Wave.hpp            # Main solver class definition
│   ├── Wave.cpp            # Implementation
│   └── exercise-01.cpp     # Main program entry point
├── text/
│   └── project-description.md  # Project details
└── mesh/
    └── mesh.msh            # Input mesh file
```

## Build Instructions

```bash
cd pde_project
mkdir build
cd build
cmake ..
make
```

## Key Components to Implement

### 1. **Wave.hpp**
- Class definition with:
  - Initial condition classes (`FunctionU0`, `FunctionU1`)
  - Member variables for mesh, FE space, solutions
  - Methods: `setup()`, `assemble()`, `solve_linear_system()`, `output()`

### 2. **Wave.cpp**
- **`setup()`**: Initialize mesh, DoF handler, vectors
- **`assemble()`**: Build system matrix and RHS using time discretization
- **`solve_linear_system()`**: Solve linear system at each time step
- **`output()`**: Export solution for visualization

### 3. **Time Discretization Methods**
Choose one or more:
- **Central Differences**: Simple, explicit method
- **Newmark Method**: Implicit, energy stable
- **Implicit Euler**: Backward Euler scheme

### 4. **Space Discretization**
- Finite element method with simplicial elements (P1, P2, etc.)
- Weak formulation via integration by parts

## TODO

- [ ] Create mesh file (mesh.msh)
- [ ] Implement `assemble()` method with time-stepping scheme
- [ ] Implement `solve_linear_system()` using Trilinos solvers
- [ ] Implement `output()` for ParaView visualization
- [ ] Add convergence studies
- [ ] Analyze numerical dissipation and dispersion properties
- [ ] Test parallel scalability
- [ ] Write project report

## References

- Lab-04 (Heat equation) - similar structure for time-dependent PDE
- Project specification in WaveFEM-Solver/projects.pdf

## Authors

[Group members to be filled]
