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
   * Project 2 first test case.
   *
   * PDE:
   *
   *     u_tt - Δu = f
   *
   * Domain:
   *
   *     Ω = (0,1)^2
   *
   * Boundary condition:
   *
   *     u = 0 on ∂Ω
   *
   * Initial displacement:
   *
   *     u0(x,y) = x(1-x)y(1-y)
   *
   * Initial velocity:
   *
   *     u1(x,y) = 0
   *
   * Forcing:
   *
   *     f = 0
   *
   * Expected ParaView result:
   *
   * At t = 0, the solution should be a smooth bump.
   * The boundary should be zero.
   * The center should be the maximum:
   *
   *     u0(0.5,0.5) = 0.0625
   *
   * During time evolution, the solution should oscillate like a wave.
   */
  const unsigned int polynomial_degree = 1;
  const double final_time = 1.0;
  const double time_step = 0.01;
  const unsigned int n_refine = 3;

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
