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
   * Project 2: 2D wave equation on the fixed domain Omega = [-5, 5]^2.
   *
   *     u_tt - Lap(u) = f    in Omega
   *     u = 0                on the boundary
   *     u(0)   = u0          in Omega
   *     u_t(0) = u1          in Omega
   *
   * Command-line usage:
   *
   *     ./exercise-01 [n_refine] [test_case]
   *
   * test_case is one of:
   *
   *     gaussian  (default)
   *         Gaussian bump initial condition, f = 0.
   *         Qualitative test: wave propagation and reflection.
   *
   *     caseA
   *         Validation against the manufactured exact solution
   *         (docs/validation.md) with k = m = 1 and
   *         omega = pi sqrt(2) / 10, so that f = 0 (free vibration).
   *         Runs for one full period T = 2 pi / omega = 10 sqrt(2),
   *         then prints L2 / H1 errors vs the exact solution.
   *
   *     caseB
   *         Same spatial mode, but omega = 1, so the forcing
   *         f = (lambda - omega^2) sin(a_k (x+5)) sin(a_m (y+5)) cos(t)
   *         is nonzero. Validates the load-vector assembly.
   *         Runs for one forcing period T = 2 pi.
   *
   *     caseC
   *         Same spatial mode and same omega as caseA (free vibration,
   *         f = 0), but the exact solution is
   *         u = sin(a_k (x+5)) sin(a_m (y+5)) sin(omega t) instead of
   *         cos(omega t), giving u0 = 0 and u1 = omega * S(x,y) instead
   *         of caseA's u0 = S, u1 = 0. Validates the nonzero-u1 branch
   *         of the Taylor startup for U^{-1} (docs/validation.md).
   *
   * For the validation cases the time step is tied to the mesh size,
   * dt ~= 0.2 h, safely below the central-difference stability limit
   * (approximately dt <= 0.41 h for Q1 with consistent mass), so that
   * the O(dt^2) and O(h^2) errors shrink together and the convergence
   * study shows a clean second-order slope.
   */
  const unsigned int polynomial_degree = 1;

  // Read n_refine from command line, default to 3.
  unsigned int n_refine = 3;
  if (argc > 1)
    n_refine = std::atoi(argv[1]);

  // Read test case from command line, default to "gaussian".
  std::string test_case = "gaussian";
  if (argc > 2)
    test_case = argv[2];

  std::cout << "Test case: " << test_case
            << ", n_refine = " << n_refine << std::endl;

  if (test_case == "gaussian")
  {
    /**
     * Gaussian bump released from rest, f = 0.
     *
     * No exact solution is available, so no error is computed.
     *
     * Note on interpreting this run: the decay of the center amplitude
     * (about 1.0 -> 0.52 by t = 1) is PHYSICAL, caused by the wave
     * spreading in 2D; it is not numerical dissipation. The central
     * difference scheme is non-dissipative (the discrete energy is
     * conserved up to bounded oscillation, see energy_history.txt and
     * docs/validation.md). What does differ between mesh levels is
     * numerical DISPERSION: mesh-dependent phase error that distorts
     * the wave shape, decreasing as O(h^2).
     */
    const double final_time = 1.0;
    const double time_step = 0.01;

    Wave wave_solver(polynomial_degree,
                     final_time,
                     time_step,
                     n_refine,
                     nullptr, // f = 0
                     std::make_shared<Wave::FunctionU0>(),
                     std::make_shared<Wave::FunctionU1>(),
                     nullptr); // no exact solution

    wave_solver.run();
  }
  else if (test_case == "caseA" || test_case == "caseB")
  {
    /**
     * Manufactured solution (derivation in docs/validation.md):
     *
     *     u(x,y,t) = sin(a_k (x+5)) sin(a_m (y+5)) cos(omega t)
     *
     * with a_k = k pi / 10, a_m = m pi / 10 and lambda = a_k^2 + a_m^2.
     *
     * Substitution into the PDE gives
     *
     *     f = (lambda - omega^2) sin(a_k (x+5)) sin(a_m (y+5)) cos(omega t)
     *
     * Case A: omega = sqrt(lambda)  ->  f = 0.
     * Case B: omega = 1             ->  f != 0.
     *
     * In both cases g = 0 exactly on the boundary of [-5,5]^2,
     * u0 = sin sin, u1 = 0.
     */
    const unsigned int k = 1;
    const unsigned int m = 1;
    const double a_k = k * numbers::PI / 10.0;
    const double a_m = m * numbers::PI / 10.0;
    const double lambda = a_k * a_k + a_m * a_m;

    double omega = 0.0;
    std::function<double(const Point<2> &, const double)> forcing = nullptr;

    if (test_case == "caseA")
    {
      omega = std::sqrt(lambda); // eigenfrequency -> f = 0
    }
    else
    {
      omega = 1.0;
      forcing = [a_k, a_m, lambda, omega = 1.0](const Point<2> &p,
                                                const double t) {
        return (lambda - omega * omega) *
               std::sin(a_k * (p[0] + 5.0)) *
               std::sin(a_m * (p[1] + 5.0)) *
               std::cos(omega * t);
      };
    }

    // One full period of the exact solution.
    const double final_time = 2.0 * numbers::PI / omega;

    // Tie dt to h: dt = T / n_steps with n_steps chosen so dt <= 0.2 h.
    const double h = 10.0 / (1 << n_refine);
    const unsigned int n_steps =
      static_cast<unsigned int>(std::ceil(final_time / (0.2 * h)));
    const double time_step = final_time / n_steps;

    std::cout << "  omega = " << omega
              << ", T = " << final_time
              << ", h = " << h
              << ", dt = " << time_step
              << " (dt/h = " << time_step / h << ")"
              << ", n_steps = " << n_steps << std::endl;

    auto exact = std::make_shared<WaveExactSolution>(k, m, omega);

    // u0 is the exact solution at t = 0 (a separate object, so that the
    // time of the error-reference object is not touched); u1 = 0.
    auto u0 = std::make_shared<WaveExactSolution>(k, m, omega);
    auto u1 = std::make_shared<Functions::ZeroFunction<2>>();

    Wave wave_solver(polynomial_degree,
                     final_time,
                     time_step,
                     n_refine,
                     forcing,
                     u0,
                     u1,
                     exact);

    wave_solver.run();
  }
  else if (test_case == "caseC")
  {
    /**
     * Sine-in-time variant of the caseA mode (derivation in
     * docs/validation.md): same spatial mode S(x,y) and same omega as
     * caseA (the eigenfrequency, so f = 0), but
     *
     *     u(x,y,t) = sin(a_k (x+5)) sin(a_m (y+5)) sin(omega t)
     *
     * instead of cos(omega t). This flips which initial datum is
     * nonzero: u0 = 0, u1 = omega * S(x,y), instead of caseA's u0 = S,
     * u1 = 0. It exercises the nonzero-u1 branch of the Taylor startup
     * for U^{-1} in Wave::run(), which caseA and caseB never touch
     * (both use u1 = 0).
     */
    const unsigned int k = 1;
    const unsigned int m = 1;
    const double a_k = k * numbers::PI / 10.0;
    const double a_m = m * numbers::PI / 10.0;
    const double lambda = a_k * a_k + a_m * a_m;
    const double omega = std::sqrt(lambda); // eigenfrequency -> f = 0

    // One full period of the exact solution.
    const double final_time = 2.0 * numbers::PI / omega;

    // Tie dt to h: dt = T / n_steps with n_steps chosen so dt <= 0.2 h.
    const double h = 10.0 / (1 << n_refine);
    const unsigned int n_steps =
      static_cast<unsigned int>(std::ceil(final_time / (0.2 * h)));
    const double time_step = final_time / n_steps;

    std::cout << "  omega = " << omega
              << ", T = " << final_time
              << ", h = " << h
              << ", dt = " << time_step
              << " (dt/h = " << time_step / h << ")"
              << ", n_steps = " << n_steps << std::endl;

    // use_sine = true selects sin(omega t) instead of cos(omega t).
    auto exact = std::make_shared<WaveExactSolution>(k, m, omega, true);

    auto u0 = std::make_shared<Functions::ZeroFunction<2>>();
    auto u1 = std::make_shared<WaveEigenmodeVelocity>(k, m, omega);

    Wave wave_solver(polynomial_degree,
                     final_time,
                     time_step,
                     n_refine,
                     nullptr, // f = 0 (same omega as caseA)
                     u0,
                     u1,
                     exact);

    wave_solver.run();
  }
  else
  {
    std::cerr << "Unknown test case '" << test_case
              << "'. Valid: gaussian, caseA, caseB, caseC." << std::endl;
    return 1;
  }

  return 0;
}
