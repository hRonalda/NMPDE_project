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
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

using namespace dealii;


/**
 * Manufactured exact solution for validation (see docs/validation.md).
 *
 * On the fixed domain Omega = [-5,5]^2 we use the Dirichlet eigenmode
 *
 *     u(x,y,t) = sin(a_k (x+5)) sin(a_m (y+5)) cos(omega t)      [use_sine = false]
 *     u(x,y,t) = sin(a_k (x+5)) sin(a_m (y+5)) sin(omega t)      [use_sine = true]
 *
 * with a_k = k pi / 10, a_m = m pi / 10, k and m positive integers.
 *
 * The sine factors vanish identically on all four edges x = +-5, y = +-5,
 * so u satisfies the homogeneous Dirichlet condition g = 0 exactly,
 * regardless of the time factor (cos or sin).
 *
 * Substituting into u_tt - Lap(u) gives the required forcing, with the
 * SAME coefficient (a_k^2 + a_m^2 - omega^2) in both cases:
 *
 *     f(x,y,t) = (a_k^2 + a_m^2 - omega^2)
 *                * sin(a_k (x+5)) sin(a_m (y+5)) {cos, sin}(omega t)
 *
 * Case A: cos(omega t), omega = sqrt(a_k^2 + a_m^2) -> f = 0 (free
 *         vibration); u0 = S, u1 = 0.
 * Case B: cos(omega t), any other omega (e.g. omega = 1) -> nonzero
 *         forcing, which validates the load-vector assembly.
 * Case C: sin(omega t), same omega as case A -> f = 0 again, but now
 *         u0 = 0, u1 = omega * S: this exercises the nonzero-u1 branch
 *         of the Taylor startup for U^{-1} in Wave::run(), which cases
 *         A and B never touch.
 *
 * The time is handled through Function::set_time() / get_time(), so the
 * same object provides the initial condition u0 (at time 0) and the
 * reference for the error computation (at the final time).
 */
class WaveExactSolution : public Function<2>
{
public:
  WaveExactSolution(const unsigned int k,
                    const unsigned int m,
                    const double       omega_,
                    const bool         use_sine_ = false)
    : a_k(k * numbers::PI / 10.0)
    , a_m(m * numbers::PI / 10.0)
    , omega(omega_)
    , use_sine(use_sine_)
  {}

  virtual double
  value(const Point<2> &p, const unsigned int /*component*/ = 0) const override
  {
    const double time_factor = use_sine ? std::sin(omega * get_time())
                                        : std::cos(omega * get_time());
    return std::sin(a_k * (p[0] + 5.0)) * std::sin(a_m * (p[1] + 5.0)) *
           time_factor;
  }

  // Gradient, needed for the H1 error computation.
  virtual Tensor<1, 2>
  gradient(const Point<2> &p,
           const unsigned int /*component*/ = 0) const override
  {
    const double time_factor = use_sine ? std::sin(omega * get_time())
                                        : std::cos(omega * get_time());

    Tensor<1, 2> grad;
    grad[0] =
      a_k * std::cos(a_k * (p[0] + 5.0)) * std::sin(a_m * (p[1] + 5.0)) *
      time_factor;
    grad[1] =
      a_m * std::sin(a_k * (p[0] + 5.0)) * std::cos(a_m * (p[1] + 5.0)) *
      time_factor;
    return grad;
  }

private:
  const double a_k;
  const double a_m;
  const double omega;
  const bool   use_sine;
};


/**
 * Initial velocity u1 = omega * S(x,y) for the sin(omega t) variant
 * (case C) of WaveExactSolution: the time derivative of
 * S(x,y) sin(omega t) at t = 0 is omega * S(x,y) * cos(0) = omega * S(x,y).
 * Time-independent, so this is a plain (non-time-dependent) Function.
 */
class WaveEigenmodeVelocity : public Function<2>
{
public:
  WaveEigenmodeVelocity(const unsigned int k,
                        const unsigned int m,
                        const double       omega_)
    : a_k(k * numbers::PI / 10.0)
    , a_m(m * numbers::PI / 10.0)
    , omega(omega_)
  {}

  virtual double
  value(const Point<2> &p, const unsigned int /*component*/ = 0) const override
  {
    return omega * std::sin(a_k * (p[0] + 5.0)) *
           std::sin(a_m * (p[1] + 5.0));
  }

private:
  const double a_k;
  const double a_m;
  const double omega;
};


/**
 * Cosine eigenmode with NON-homogeneous, time-dependent Dirichlet data
 * (case D, see docs/validation.md):
 *
 *     u(x,y,t) = cos(a_k (x+5)) cos(a_m (y+5)) cos(omega t)
 *
 * Like the sine mode, Lap(u) = -(a_k^2 + a_m^2) u, so choosing
 * omega = sqrt(a_k^2 + a_m^2) gives u_tt - Lap(u) = 0 (f = 0). Unlike
 * the sine mode, the cosine factors do NOT vanish on the edges of
 * [-5,5]^2 (cos(0) = 1 at x = -5, cos(k pi) = +-1 at x = +5), so the
 * boundary trace
 *
 *     g = u|_{boundary}  (nonzero, oscillating in time)
 *
 * is a genuine non-homogeneous, time-dependent Dirichlet datum. The
 * same object serves as u0 (value at t = 0), as the boundary datum g
 * (evaluated only at boundary nodes by interpolate_boundary_values),
 * and as the error reference at the final time. Compatibility holds by
 * construction: u0|_bd = g(0) (same function) and u1 = 0 = g_t(0)
 * (since d/dt cos(omega t) vanishes at t = 0). Verified symbolically
 * with sympy (PDE residual, boundary traces, compatibility).
 */
class WaveCosineExactSolution : public Function<2>
{
public:
  WaveCosineExactSolution(const unsigned int k,
                          const unsigned int m,
                          const double       omega_)
    : a_k(k * numbers::PI / 10.0)
    , a_m(m * numbers::PI / 10.0)
    , omega(omega_)
  {}

  virtual double
  value(const Point<2> &p, const unsigned int /*component*/ = 0) const override
  {
    return std::cos(a_k * (p[0] + 5.0)) * std::cos(a_m * (p[1] + 5.0)) *
           std::cos(omega * get_time());
  }

  // Gradient, needed for the H1 error computation.
  virtual Tensor<1, 2>
  gradient(const Point<2> &p,
           const unsigned int /*component*/ = 0) const override
  {
    const double ct = std::cos(omega * get_time());

    Tensor<1, 2> grad;
    grad[0] = -a_k * std::sin(a_k * (p[0] + 5.0)) *
              std::cos(a_m * (p[1] + 5.0)) * ct;
    grad[1] = -a_m * std::cos(a_k * (p[0] + 5.0)) *
              std::sin(a_m * (p[1] + 5.0)) * ct;
    return grad;
  }

private:
  const double a_k;
  const double a_m;
  const double omega;
};


/**
 * Wave equation solver for PDE Project 2.
 *
 * We solve the 2D wave equation:
 *
 *     u_tt - Δu = f    in Ω
 *     u = g            on ∂Ω
 *     u(0) = u0        in Ω
 *     u_t(0) = u1      in Ω
 *
 * on the fixed domain Ω = [-5, 5]^2 (see Wave::setup()). f, g, u0, u1
 * are all pluggable through the constructor: f defaults to zero
 * (nullptr), g defaults to homogeneous (nullptr -> g = 0). See
 * docs/validation.md for the manufactured-solution test cases
 * (caseA-D, covering f != 0, u1 != 0, and g != 0 respectively) and the
 * Gaussian test case (Wave::FunctionU0 / FunctionU1) in exercise-01.cpp.
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
 * i.e. the implemented right-hand side is:
 *
 *     RHS = 2 M U^n - M U^{n-1} - Δt² K U^n + Δt² F^n
 *
 * with the Δt² F^n term present only when f != 0 (see Wave::assemble()).
 * The Dirichlet datum g is imposed by algebraic elimination against the
 * new time level (Wave::assemble()); U^{-1} is initialized to
 * second-order accuracy from u0, u1, and the initial acceleration
 * (Wave::run()) -- see docs/analysis.md for the full derivation of both.
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
   * f_          = forcing term f(x, t). Pass nullptr for f = 0.
   * u0_         = initial displacement u(x, 0).
   * u1_         = initial velocity u_t(x, 0).
   * exact_      = exact solution for the error computation at the final
   *               time. Pass nullptr when no exact solution is known
   *               (e.g. the Gaussian test case).
   * g_          = Dirichlet boundary datum u = g on the boundary,
   *               possibly time-dependent (evaluated through
   *               set_time()/get_time()). Pass nullptr for the
   *               homogeneous case g = 0; all existing test cases do
   *               this and are unaffected.
   */
  Wave(const unsigned int                             &r_,
       const double                                   &T_,
       const double                                   &delta_t_,
       const unsigned int                             &n_refine_,
       const std::function<double(const Point<dim> &,
                                  const double)>      &f_,
       const std::shared_ptr<Function<dim>>           &u0_,
       const std::shared_ptr<Function<dim>>           &u1_,
       const std::shared_ptr<Function<dim>>           &exact_ = nullptr,
       const std::shared_ptr<Function<dim>>           &g_     = nullptr)
    : r(r_)
    , T(T_)
    , delta_t(delta_t_)
    , n_refine(n_refine_)
    , f(f_)
    , u0(u0_)
    , u1(u1_)
    , exact_solution(exact_)
    , boundary_g(g_)
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

  // Assemble the load vector F(t)_i = (f(., t), phi_i).
  // Sets load to zero when no forcing term was provided.
  void
  assemble_load_vector(const double t, TrilinosWrappers::MPI::Vector &load);

  // Compute L2 and H1 errors against the exact solution at the current
  // time. Does nothing if no exact solution was provided.
  void
  compute_errors();

  // Solve the linear system.
  void
  solve_linear_system();

  // Discrete energy E^n = 1/2 (V^n·M V^n + U^n·K U^n) with the centered
  // velocity V^n = (U^{n+1} - U^{n-1}) / (2 Δt). Must be called after the
  // solve and before the time levels are shifted; returns E at t_n, one
  // step behind the already-advanced "time".
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

  // Forcing term f(x, t). Empty (nullptr) means f = 0.
  std::function<double(const Point<dim> &, const double)> f;

  // Initial displacement u(x, 0).
  std::shared_ptr<Function<dim>> u0;

  // Initial velocity u_t(x, 0).
  std::shared_ptr<Function<dim>> u1;

  // Exact solution for validation (nullptr if unknown).
  std::shared_ptr<Function<dim>> exact_solution;

  /**
   * Dirichlet boundary datum g (nullptr means homogeneous, g = 0).
   *
   * Handled by symmetric algebraic elimination
   * (MatrixTools::apply_boundary_values) against the NEW time level:
   * in each step the constraint applies to the unknown U^{n+1}, so g is
   * evaluated at t_{n+1}, while the load vector is evaluated at t_n
   * (the center of the leapfrog stencil). Because the full history
   * vectors carry the correct boundary values of earlier times, the
   * column elimination automatically reproduces the boundary-
   * acceleration coupling term M_IB g_tt of the constrained
   * semi-discrete system -- see docs/analysis.md.
   */
  std::shared_ptr<Function<dim>> boundary_g;

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
   * Energy history: (t_n, E^n) pairs for all time steps.
   *
   * Central differences are non-dissipative, so E^n should stay flat
   * (bounded oscillation, no drift). A decaying center amplitude with
   * flat energy is physical spreading, not numerical dissipation.
   */
  std::vector<std::pair<double, double>> energy_history;

  // Only MPI rank 0 prints to terminal.
  ConditionalOStream pcout;
};

#endif
