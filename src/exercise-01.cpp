#include "Wave.hpp"


int
main(int argc, char *argv[])
{
  /**
   * Initialize MPI.
   *
   * The final argument 1 means that each MPI process uses one thread.
   */
  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv, 1);

  /**
   * Project 2 extended test case with convergence study support.
   *
   * PDE:
   *
   *     u_tt - Δu = f
   *
   * Domain:
   *
   *     Ω = [-5, 5]^2
   *
   * This is the domain shown in Project 2 figure.
   *
   * Boundary condition:
   *
   *     u = 0 on ∂Ω
   *
   * Initial displacement:
   *
   *     u0(x,y) = exp(-(x² + y²)/σ²)    with σ = 2
   *
   * This is a Gaussian bump centered at origin:
   *     - maximum = 1.0 at (0, 0)
   *     - smooth decay to near-zero at boundaries
   *     - symmetric in x and y
   *
   * Initial velocity:
   *
   *     u1(x,y) = 0
   *
   * Forcing:
   *
   *     f = 0
   *
   * Command-line usage:
   *
   *     ./exercise-01              # uses default n_refine = 3
   *     ./exercise-01 4            # uses n_refine = 4
   *     ./exercise-01 5            # uses n_refine = 5
   *
   * Expected ParaView result:
   *
   * At t = 0, the solution should be a smooth Gaussian bump centered at origin.
   * The boundary should be nearly zero.
   * The center should have maximum value 1.0.
   *
   * During time evolution, the solution should propagate as a wave,
   * reflecting from the boundary at x = ±5 and y = ±5.
   *
   * For convergence studies, higher n_refine gives:
   *     - finer mesh
   *     - more DoFs
   *     - less numerical dissipation
   *     - better accuracy
   *     - longer computation time
   */
  const unsigned int polynomial_degree = 1;
  const double final_time = 1.0;
  const double time_step = 0.01;

  // Read n_refine from command line, default to 3
  unsigned int n_refine = 3;
  if (argc > 1)
  {
    n_refine = std::atoi(argv[1]);
    std::cout << "Using command-line argument: n_refine = " << n_refine << std::endl;
  }
  else
  {
    std::cout << "Using default: n_refine = " << n_refine << std::endl;
    std::cout << "  (to specify n_refine, run: ./exercise-01 <n_refine>)" << std::endl;
  }

  /**
   * Forcing term.
   *
   * For the first version, we choose f = 0.
   */
  auto f = [](const Point<2> & /*p*/) {
    return 0.0;
  };

  /**
   * Create and run solver.
   */
  Wave wave_solver(polynomial_degree,
                   final_time,
                   time_step,
                   n_refine,
                   f);

  wave_solver.run();

  return 0;
}
