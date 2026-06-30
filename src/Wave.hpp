#ifndef WAVE_HPP
#define WAVE_HPP

// Basic deal.II utilities.
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/function.h>
#include <deal.II/base/function_lib.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/data_out_base.h>

// Parallel distributed triangulation.
#include <deal.II/distributed/fully_distributed_tria.h>

// DoF handling.
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

// Finite element and quadrature.
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>

// Mesh generation.
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria.h>

// Linear algebra.
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/vector.h>

// Numerical tools.
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>

// Standard C++ headers.
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

using namespace dealii;


/**
 * Wave equation solver for PDE Project 2.
 *
 * We solve the 2D wave equation:
 *
 *     u_tt - Δu = f    in Ω
 *     u = 0            on ∂Ω
 *     u(0) = u0        in Ω
 *     u_t(0) = u1      in Ω
 *
 * For the first working version, we use:
 *
 *     Ω = (0,1)^2
 *     f = 0
 *     u0(x,y) = x(1-x)y(1-y)
 *     u1(x,y) = 0
 *
 * The finite element semi-discretization gives:
 *
 *     M U''(t) + K U(t) = F(t)
 *
 * where:
 *
 *     M_ij = ∫_Ω φ_i φ_j dx
 *     K_ij = ∫_Ω ∇φ_i · ∇φ_j dx
 *
 * Then we use the central difference method in time:
 *
 *     U''(t_n) ≈ (U^{n+1} - 2U^n + U^{n-1}) / Δt²
 *
 * Therefore:
 *
 *     M U^{n+1}
 *       = 2 M U^n - M U^{n-1} - Δt² K U^n + Δt² F^n
 *
 * Since currently f = 0, the implemented right-hand side is:
 *
 *     RHS = 2 M U^n - M U^{n-1} - Δt² K U^n
 *
 * Important note:
 * This version is designed to first make the simulation and ParaView output correct.
 * Later, for the final report, we can improve the initialization of U^{-1}
 * using the initial velocity and initial acceleration.
 */
class Wave
{
public:
  // We solve the project in 2D.
  static constexpr unsigned int dim = 2;

  /**
   * Initial displacement:
   *
   * For domain [-5, 5]², we use a Gaussian-like bump centered at origin:
   *
   *     u0(x,y) = exp(-(x² + y²)/σ²)
   *
   * This function:
   *     - is smooth
   *     - has maximum 1.0 at (0, 0)
   *     - decays to near-zero at boundaries
   *     - is symmetric
   *
   * Parameter σ controls the width of the bump.
   * For σ = 2, the bump has noticeable width.
   */
  class FunctionU0 : public Function<dim>
  {
  public:
    FunctionU0() = default;

    virtual double
    value(const Point<dim> &p,
          const unsigned int /*component*/ = 0) const override
    {
      const double sigma = 2.0;
      const double r_squared = p[0] * p[0] + p[1] * p[1];
      return std::exp(-r_squared / (sigma * sigma));
    }
  };

  /**
   * Initial velocity:
   *
   *     u1(x,y) = 0
   *
   * This means the initial wave starts from rest.
   */
  class FunctionU1 : public Function<dim>
  {
  public:
    FunctionU1() = default;

    virtual double
    value(const Point<dim> & /*p*/,
          const unsigned int /*component*/ = 0) const override
    {
      return 0.0;
    }
  };

  /**
   * Constructor.
   *
   * r_          = polynomial degree of FE_Q elements.
   * T_          = final time.
   * delta_t_    = time step.
   * n_refine_   = global mesh refinement level.
   * f_          = forcing term f(x). For now, we use f = 0.
   */
  Wave(const unsigned int                              &r_,
       const double                                    &T_,
       const double                                    &delta_t_,
       const unsigned int                              &n_refine_,
       const std::function<double(const Point<dim> &)> &f_)
    : r(r_)
    , T(T_)
    , delta_t(delta_t_)
    , n_refine(n_refine_)
    , f(f_)
    , mpi_size(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD))
    , mpi_rank(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD))
    , mesh(MPI_COMM_WORLD)
    , dof_handler(mesh)
    , pcout(std::cout, mpi_rank == 0)
  {}

  // Main function: setup, time loop, output.
  void
  run();

protected:
  // Create mesh, FE space, DoF handler, matrices, and vectors.
  void
  setup();

  // Assemble mass matrix M and stiffness matrix K once.
  void
  assemble_matrices();

  // Assemble the linear system for one time step.
  void
  assemble();

  // Solve the linear system.
  void
  solve_linear_system();

  // Compute discrete energy: E = 0.5 * ∫(u_t² + |∇u|²) dx
  // Returns the total energy at current time step
  double
  compute_energy();

  // Write .vtu and .pvd files for ParaView.
  // Important: non-const because we update output_files.
  void
  output();

  // Polynomial degree of FE_Q.
  const unsigned int r;

  // Final time.
  const double T;

  // Time step size.
  const double delta_t;

  // Number of global mesh refinements.
  const unsigned int n_refine;

  // Current simulation time.
  double time = 0.0;

  // Current time step index.
  unsigned int timestep_number = 0;

  // Forcing term f(x).
  // In the current first test, f = 0.
  std::function<double(const Point<dim> &)> f;

  // MPI information.
  const unsigned int mpi_size;
  const unsigned int mpi_rank;

  // Distributed mesh.
  parallel::fullydistributed::Triangulation<dim> mesh;

  // Finite element.
  // We use FE_Q because GridGenerator::hyper_cube creates quadrilateral cells.
  // Do NOT use FE_SimplexP with hyper_cube unless the mesh is actually triangular.
  std::unique_ptr<FiniteElement<dim>> fe;

  // Quadrature rule.
  std::unique_ptr<Quadrature<dim>> quadrature;

  // Degree of freedom handler.
  DoFHandler<dim> dof_handler;

  // Mass matrix M.
  TrilinosWrappers::SparseMatrix mass_matrix;

  // Stiffness matrix K.
  TrilinosWrappers::SparseMatrix stiffness_matrix;

  // System matrix.
  // For central difference, system_matrix = M.
  TrilinosWrappers::SparseMatrix system_matrix;

  // System right-hand side.
  TrilinosWrappers::MPI::Vector system_rhs;

  /**
   * Current solution U^{n+1}.
   *
   * owned vector:
   *   used by solvers and matrix operations.
   *
   * ghosted vector:
   *   used for output and accessing locally relevant DoFs.
   */
  TrilinosWrappers::MPI::Vector solution_owned;
  TrilinosWrappers::MPI::Vector solution;

  /**
   * Previous solution U^n.
   */
  TrilinosWrappers::MPI::Vector solution_old_owned;
  TrilinosWrappers::MPI::Vector solution_old;

  /**
   * Older solution U^{n-1}.
   */
  TrilinosWrappers::MPI::Vector solution_old_old_owned;
  TrilinosWrappers::MPI::Vector solution_old_old;

  /**
   * ParaView time-series record.
   *
   * The .vtu files store each individual snapshot.
   * The .pvd file tells ParaView that these .vtu files belong to one animation.
   *
   * In ParaView, open solution.pvd, not all .vtu files separately.
   */
  std::vector<std::pair<double, std::string>> output_files;

  /**
   * Energy data for convergence analysis.
   *
   * Stores (time, energy) pairs for all time steps.
   * Used to analyze numerical dissipation separately from physical spreading.
   */
  std::vector<std::pair<double, double>> energy_history;

  // Only MPI rank 0 prints to terminal.
  ConditionalOStream pcout;
};

#endif
