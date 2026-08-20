#include "Wave.hpp"


void
Wave::run()
{
  setup();

  pcout << "============================================" << std::endl;
  pcout << "Wave Equation Solver (Central Differences)" << std::endl;
  pcout << "============================================" << std::endl;
  pcout << "Domain: [-5, 5]^" << dim << std::endl;
  pcout << "Final time: " << T << std::endl;
  pcout << "Time step: " << delta_t << std::endl;
  pcout << "Number of DoFs: " << dof_handler.n_dofs() << std::endl;

  /**
   * Initialize the solution at time t = 0.
   *
   * IMPORTANT FIX:
   *
   * Before, only the ghosted vector "solution" was initialized.
   * But the output used "solution_owned", so ParaView saw almost zero values.
   *
   * Now we initialize "solution_owned" first, then copy it to "solution".
   *
   * CRITICAL: After interpolation, we MUST call compress(VectorOperation::insert)
   * to finalize the vector data across MPI processes.
   *
   * At t = 0:
   *
   *     solution_owned = U^0
   *     solution       = U^0
   */
  VectorTools::interpolate(dof_handler, *u0, solution_owned);

  solution_owned.compress(VectorOperation::insert);
  solution = solution_owned;

  /**
   * Initialize previous time levels.
   *
   * Central difference needs U^n and U^{n-1}. The ghost level U^{-1} is
   * built from the Taylor expansion of the exact solution at t = 0:
   *
   *     U^{-1} = U^0 - dt U1 + 0.5 dt^2 A^0
   *
   * where A^0 is the initial acceleration, obtained from the
   * semi-discrete equation at t = 0:
   *
   *     M A^0 = F(0) - K U^0
   *
   * This keeps the scheme second-order accurate also at the first step,
   * and is what makes a nonzero initial velocity u1 work.
   */
  solution_old_owned = solution_owned;
  solution_old = solution_old_owned;

  {
    // Initial velocity U1.
    TrilinosWrappers::MPI::Vector u1_vec(solution_owned);
    VectorTools::interpolate(dof_handler, *u1, u1_vec);
    u1_vec.compress(VectorOperation::insert);

    // RHS of the acceleration system: F(0) - K U^0.
    TrilinosWrappers::MPI::Vector accel_rhs(solution_owned);
    assemble_load_vector(0.0, accel_rhs);

    TrilinosWrappers::MPI::Vector K_u0(solution_owned);
    stiffness_matrix.vmult(K_u0, solution_owned);
    accel_rhs -= K_u0;

    // Solve M A^0 = F(0) - K U^0, with A^0 = 0 on the boundary
    // (consistent with time-independent Dirichlet data g = 0).
    system_matrix.copy_from(mass_matrix);

    TrilinosWrappers::MPI::Vector accel(solution_owned);

    std::map<types::global_dof_index, double> boundary_values;
    VectorTools::interpolate_boundary_values(dof_handler,
                                             0,
                                             Functions::ZeroFunction<dim>(),
                                             boundary_values);
    MatrixTools::apply_boundary_values(boundary_values,
                                       system_matrix,
                                       accel,
                                       accel_rhs);

    SolverControl solver_control(10000, 1e-10);
    SolverCG<TrilinosWrappers::MPI::Vector> solver(solver_control);
    TrilinosWrappers::PreconditionJacobi preconditioner;
    preconditioner.initialize(system_matrix);
    solver.solve(system_matrix, accel, accel_rhs, preconditioner);

    // U^{-1} = U^0 - dt U1 + 0.5 dt^2 A^0.
    solution_old_old_owned = solution_owned;
    solution_old_old_owned.add(-delta_t, u1_vec);
    solution_old_old_owned.add(0.5 * delta_t * delta_t, accel);
    solution_old_old = solution_old_old_owned;
  }

  /**
   * Output the initial condition solution-0000.vtu.
   *
   * No energy is recorded here: the centered velocity
   * V^n = (U^{n+1} - U^{n-1}) / (2 Δt) needs U^1, which does not exist
   * yet. The first loop iteration records E^0 (at t = 0) instead.
   */
  output();

  /**
   * Time loop.
   *
   * At each step:
   *
   * 1. assemble the central-difference system
   * 2. solve for U^{n+1}
   * 3. shift old time levels:
   *
   *        U^{n-1} <- U^n
   *        U^n     <- U^{n+1}
   *
   * 4. output the new solution
   */
  // The 0.5 * delta_t guard protects against floating-point accumulation
  // in "time" adding a spurious extra step when delta_t divides T exactly.
  while (time < T - 0.5 * delta_t)
  {
    time += delta_t;
    ++timestep_number;

    pcout << "Step " << timestep_number
          << ", time = " << time << std::endl;

    assemble();
    solve_linear_system();

    /**
     * Energy at t_n = time - Δt, computed BEFORE the shift below while
     * all three time levels U^{n+1}, U^n, U^{n-1} are still available
     * (see compute_energy). The first iteration records E^0.
     */
    energy_history.push_back({time - delta_t, compute_energy()});

    /**
     * IMPORTANT FIX:
     *
     * We update both owned and ghosted vectors.
     *
     * This avoids the situation where the solver uses one vector,
     * but ParaView output writes another inconsistent vector.
     */
    solution_old_old_owned = solution_old_owned;
    solution_old_old = solution_old_old_owned;

    solution_old_owned = solution_owned;
    solution_old = solution_old_owned;

    output();
  }

  // Save energy history to file for convergence analysis
  if (mpi_rank == 0)
  {
    std::ofstream energy_file("energy_history.txt");
    energy_file << std::setprecision(15);
    energy_file << "# time energy\n";
    for (const auto &[t, e] : energy_history)
    {
      energy_file << t << " " << e << "\n";
    }
    energy_file.close();
    pcout << "Energy history saved to energy_history.txt" << std::endl;
  }

  // Validation: compare against the exact solution, if one was provided.
  compute_errors();

  pcout << "Simulation completed." << std::endl;
}


void
Wave::setup()
{
  /**
   * Mesh generation.
   *
   * We create a square domain:
   *
   *     Ω = [-5, 5] x [-5, 5]
   *
   * This is the domain shown in the Project 2 figure.
   *
   * GridGenerator::hyper_cube creates quadrilateral cells.
   *
   * Therefore, the correct finite element is FE_Q.
   * Do NOT use FE_SimplexP here, because FE_SimplexP is for simplex/triangle meshes.
   *
   * Note: For the larger domain, we may need more refinement to maintain
   * similar mesh density as [0,1]² with n_refine=3.
   */
  Triangulation<dim> temp_mesh;
  GridGenerator::hyper_cube(temp_mesh, -5.0, 5.0);
  temp_mesh.refine_global(n_refine);

  /**
   * Partition the serial mesh and copy it into a fully distributed triangulation.
   *
   * This keeps the code compatible with MPI.
   */
  GridTools::partition_triangulation(mpi_size, temp_mesh);
  mesh.copy_triangulation(temp_mesh);

  /**
   * Finite element space.
   *
   * r = polynomial degree.
   *
   * If r = 1, we use bilinear Q1 elements.
   * If r = 2, we use biquadratic Q2 elements.
   */
  fe = std::make_unique<FE_Q<dim>>(r);

  /**
   * Quadrature rule.
   *
   * QGauss is the correct quadrature for quadrilateral FE_Q cells.
   */
  quadrature = std::make_unique<QGauss<dim>>(r + 1);

  /**
   * DoF setup.
   */
  dof_handler.reinit(mesh);
  dof_handler.distribute_dofs(*fe);

  /**
   * Vector initialization.
   *
   * There are two kinds of vectors:
   *
   * 1. owned vectors:
   *      contain only locally owned DoFs.
   *      used by solvers and matrix-vector products.
   *
   * 2. ghosted vectors:
   *      contain locally owned + locally relevant DoFs.
   *      used for output and FE evaluations.
   */
  solution_owned.reinit(dof_handler.locally_owned_dofs(), MPI_COMM_WORLD);

  IndexSet locally_relevant_dofs =
    DoFTools::extract_locally_relevant_dofs(dof_handler);

  solution.reinit(dof_handler.locally_owned_dofs(),
                  locally_relevant_dofs,
                  MPI_COMM_WORLD);

  solution_old_owned.reinit(solution_owned);
  solution_old.reinit(dof_handler.locally_owned_dofs(),
                      locally_relevant_dofs,
                      MPI_COMM_WORLD);

  solution_old_old_owned.reinit(solution_owned);
  solution_old_old.reinit(dof_handler.locally_owned_dofs(),
                          locally_relevant_dofs,
                          MPI_COMM_WORLD);

  system_rhs.reinit(solution_owned);

  /**
   * Assemble mass and stiffness matrices once.
   *
   * Since the mesh and FE space do not change during time evolution,
   * M and K are constant.
   */
  assemble_matrices();

  pcout << "Mesh: [-5,5] x [-5,5], refined "
        << n_refine << " times" << std::endl;
  pcout << "Number of active cells: "
        << mesh.n_active_cells() << std::endl;
  pcout << "Number of DoFs: "
        << dof_handler.n_dofs() << std::endl;
  pcout << "Setup completed." << std::endl;
}


void
Wave::assemble_matrices()
{
  /**
   * Assemble:
   *
   *     M_ij = ∫_Ω φ_i φ_j dx
   *
   * and:
   *
   *     K_ij = ∫_Ω ∇φ_i · ∇φ_j dx
   *
   * These matrices come from the weak form:
   *
   *     (u_tt, v) + (∇u, ∇v) = (f, v)
   *
   * After FEM discretization:
   *
   *     M U'' + K U = F
   */
  const unsigned int dofs_per_cell = fe->dofs_per_cell;
  const unsigned int n_q = quadrature->size();

  FEValues<dim> fe_values(*fe,
                          *quadrature,
                          update_values |
                          update_gradients |
                          update_quadrature_points |
                          update_JxW_values);

  FullMatrix<double> cell_mass(dofs_per_cell, dofs_per_cell);
  FullMatrix<double> cell_stiffness(dofs_per_cell, dofs_per_cell);
  std::vector<types::global_dof_index> dof_indices(dofs_per_cell);

  /**
   * Build sparsity pattern.
   */
  DynamicSparsityPattern dsp(dof_handler.locally_owned_dofs());
  DoFTools::make_sparsity_pattern(dof_handler, dsp);

  mass_matrix.reinit(dof_handler.locally_owned_dofs(),
                     dof_handler.locally_owned_dofs(),
                     dsp,
                     MPI_COMM_WORLD);

  stiffness_matrix.reinit(dof_handler.locally_owned_dofs(),
                          dof_handler.locally_owned_dofs(),
                          dsp,
                          MPI_COMM_WORLD);

  system_matrix.reinit(dof_handler.locally_owned_dofs(),
                       dof_handler.locally_owned_dofs(),
                       dsp,
                       MPI_COMM_WORLD);

  mass_matrix = 0.0;
  stiffness_matrix = 0.0;

  /**
   * Cell loop.
   */
  for (const auto &cell : dof_handler.active_cell_iterators())
  {
    if (!cell->is_locally_owned())
      continue;

    fe_values.reinit(cell);

    cell_mass = 0.0;
    cell_stiffness = 0.0;

    for (unsigned int q = 0; q < n_q; ++q)
    {
      for (unsigned int i = 0; i < dofs_per_cell; ++i)
      {
        for (unsigned int j = 0; j < dofs_per_cell; ++j)
        {
          /**
           * Mass matrix contribution:
           *
           *     M_ij += φ_i(x_q) φ_j(x_q) JxW_q
           */
          cell_mass(i, j) += fe_values.shape_value(i, q) *
                             fe_values.shape_value(j, q) *
                             fe_values.JxW(q);

          /**
           * Stiffness matrix contribution:
           *
           *     K_ij += ∇φ_i(x_q) · ∇φ_j(x_q) JxW_q
           */
          cell_stiffness(i, j) += fe_values.shape_grad(i, q) *
                                  fe_values.shape_grad(j, q) *
                                  fe_values.JxW(q);
        }
      }
    }

    cell->get_dof_indices(dof_indices);

    mass_matrix.add(dof_indices, cell_mass);
    stiffness_matrix.add(dof_indices, cell_stiffness);
  }

  mass_matrix.compress(VectorOperation::add);
  stiffness_matrix.compress(VectorOperation::add);

  pcout << "Mass and stiffness matrices assembled." << std::endl;
}


void
Wave::assemble()
{
  /**
   * Correct central difference method.
   *
   * Starting point:
   *
   *     M U'' + K U = F
   *
   * Approximate:
   *
   *     U''(t_n) ≈ (U^{n+1} - 2U^n + U^{n-1}) / Δt²
   *
   * Substitute:
   *
   *     M (U^{n+1} - 2U^n + U^{n-1}) / Δt² + K U^n = F^n
   *
   * Multiply by Δt²:
   *
   *     M U^{n+1} - 2M U^n + M U^{n-1} + Δt² K U^n = Δt² F^n
   *
   * Move known terms to RHS:
   *
   *     M U^{n+1}
   *       = 2M U^n - M U^{n-1} - Δt² K U^n + Δt² F^n
   *
   * Therefore:
   *
   *     system_matrix = M
   *     system_rhs    = 2M U^n - M U^{n-1} - Δt² K U^n
   *
   * For the first test, f = 0, so Δt² F^n is not added.
   */
  const double dt2 = delta_t * delta_t;

  /**
   * IMPORTANT FIX:
   *
   * Before, the code used:
   *
   *     system_matrix = M + Δt² K
   *
   * That is not the standard explicit central difference formula.
   *
   * Here we use:
   *
   *     system_matrix = M
   */
  system_matrix.copy_from(mass_matrix);

  /**
   * Temporary vectors.
   *
   * M_un   = M U^n
   * M_unm1 = M U^{n-1}
   * K_un   = K U^n
   */
  TrilinosWrappers::MPI::Vector M_un(solution_owned);
  TrilinosWrappers::MPI::Vector M_unm1(solution_owned);
  TrilinosWrappers::MPI::Vector K_un(solution_owned);

  mass_matrix.vmult(M_un, solution_old);
  mass_matrix.vmult(M_unm1, solution_old_old);
  stiffness_matrix.vmult(K_un, solution_old);

  /**
   * RHS = 2M U^n - M U^{n-1} - Δt² K U^n
   */
  system_rhs = M_un;
  system_rhs *= 2.0;
  system_rhs -= M_unm1;

  K_un *= dt2;
  system_rhs -= K_un;

  /**
   * Forcing contribution:
   *
   *     system_rhs += Δt² F^n
   *
   * The scheme is centered at t_n; since "time" has already been advanced
   * to t_{n+1} in the time loop, the load vector is evaluated at
   * t_n = time - Δt.
   */
  if (f)
  {
    TrilinosWrappers::MPI::Vector load(solution_owned);
    assemble_load_vector(time - delta_t, load);
    system_rhs.add(dt2, load);
  }

  /**
   * Homogeneous Dirichlet boundary condition:
   *
   *     u = 0 on ∂Ω
   *
   * IMPORTANT FIX:
   *
   * Before, the code only set:
   *
   *     system_rhs[dof] = 0
   *     system_matrix.set(dof, dof, 1)
   *
   * but this does not properly eliminate boundary DoFs.
   *
   * MatrixTools::apply_boundary_values is safer.
   */
  std::map<types::global_dof_index, double> boundary_values;

  VectorTools::interpolate_boundary_values(dof_handler,
                                           0,
                                           Functions::ZeroFunction<dim>(),
                                           boundary_values);

  MatrixTools::apply_boundary_values(boundary_values,
                                     system_matrix,
                                     solution_owned,
                                     system_rhs);
}


void
Wave::assemble_load_vector(const double t, TrilinosWrappers::MPI::Vector &load)
{
  /**
   * Assemble the discrete load vector at time t:
   *
   *     F(t)_i = ∫_Ω f(x, t) φ_i(x) dx
   */
  load = 0.0;

  if (!f)
    return;

  const unsigned int dofs_per_cell = fe->dofs_per_cell;
  const unsigned int n_q = quadrature->size();

  FEValues<dim> fe_values(*fe,
                          *quadrature,
                          update_values |
                          update_quadrature_points |
                          update_JxW_values);

  Vector<double> cell_rhs(dofs_per_cell);
  std::vector<types::global_dof_index> dof_indices(dofs_per_cell);

  for (const auto &cell : dof_handler.active_cell_iterators())
  {
    if (!cell->is_locally_owned())
      continue;

    fe_values.reinit(cell);
    cell_rhs = 0.0;

    for (unsigned int q = 0; q < n_q; ++q)
    {
      const double f_q = f(fe_values.quadrature_point(q), t);

      for (unsigned int i = 0; i < dofs_per_cell; ++i)
        cell_rhs(i) += f_q *
                       fe_values.shape_value(i, q) *
                       fe_values.JxW(q);
    }

    cell->get_dof_indices(dof_indices);
    load.add(dof_indices, cell_rhs);
  }

  load.compress(VectorOperation::add);
}


void
Wave::compute_errors()
{
  if (!exact_solution)
    return;

  /**
   * Compare the discrete solution against the exact solution at the
   * current (final) time, in the L2 norm and the H1 seminorm:
   *
   *     e_L2 = ||u_h - u_ex||_L2
   *     e_H1 = ||grad(u_h - u_ex)||_L2
   *
   * A higher-order quadrature (r + 2) is used so that the quadrature
   * error does not pollute the measured discretization error.
   */
  exact_solution->set_time(time);

  const QGauss<dim> error_quadrature(r + 2);

  Vector<double> difference_per_cell(mesh.n_active_cells());

  VectorTools::integrate_difference(dof_handler,
                                    solution,
                                    *exact_solution,
                                    difference_per_cell,
                                    error_quadrature,
                                    VectorTools::L2_norm);
  const double error_L2 =
    VectorTools::compute_global_error(mesh,
                                      difference_per_cell,
                                      VectorTools::L2_norm);

  VectorTools::integrate_difference(dof_handler,
                                    solution,
                                    *exact_solution,
                                    difference_per_cell,
                                    error_quadrature,
                                    VectorTools::H1_seminorm);
  const double error_H1 =
    VectorTools::compute_global_error(mesh,
                                      difference_per_cell,
                                      VectorTools::H1_seminorm);

  pcout << "Errors vs exact solution at t = " << time << ":" << std::endl;
  pcout << "  L2 error          = " << error_L2 << std::endl;
  pcout << "  H1 seminorm error = " << error_H1 << std::endl;
}


void
Wave::solve_linear_system()
{
  /**
   * Solve:
   *
   *     system_matrix * solution_owned = system_rhs
   *
   * For central difference:
   *
   *     system_matrix = M
   *
   * M is symmetric positive definite, so CG is suitable.
   */
  SolverControl solver_control(10000, 1e-10);
  SolverCG<TrilinosWrappers::MPI::Vector> solver(solver_control);

  /**
   * Jacobi preconditioner.
   *
   * This is simple and enough for the first project version.
   * Later we can discuss better solvers/preconditioners if needed.
   */
  TrilinosWrappers::PreconditionJacobi preconditioner;
  preconditioner.initialize(system_matrix);

  solver.solve(system_matrix,
               solution_owned,
               system_rhs,
               preconditioner);

  /**
   * IMPORTANT:
   *
   * After solving, we MUST:
   * 1. Compress the owned vector to finalize across MPI
   * 2. Copy owned vector to ghosted vector for output
   *
   * The ghosted vector is used for output.
   */
  solution_owned.compress(VectorOperation::insert);
  solution = solution_owned;

  pcout << "  Solved in " << solver_control.last_step()
        << " CG iterations." << std::endl;
}


double
Wave::compute_energy()
{
  /**
   * Discrete energy of the semi-discrete wave equation:
   *
   *     E^n = 1/2 ( V^n · M V^n  +  U^n · K U^n )
   *
   * which is the FEM counterpart of E(t) = 1/2 ∫ (u_t² + |∇u|²) dx.
   *
   * The velocity is the centered difference
   *
   *     V^n = (U^{n+1} - U^{n-1}) / (2 Δt)
   *
   * so this function needs all three time levels and MUST be called
   * after solve_linear_system() but BEFORE the old solutions are
   * shifted, i.e. while:
   *
   *     solution_owned         = U^{n+1}
   *     solution_old_owned     = U^n
   *     solution_old_old_owned = U^{n-1}
   *
   * The result is the energy at t_n (one step behind the current
   * "time", which has already been advanced to t_{n+1}).
   *
   * M and K are the matrices assembled once in assemble_matrices();
   * nothing is reassembled here.
   */
  TrilinosWrappers::MPI::Vector velocity(solution_owned);
  velocity = solution_owned;
  velocity -= solution_old_old_owned;
  velocity *= 1.0 / (2.0 * delta_t);

  TrilinosWrappers::MPI::Vector tmp(solution_owned);

  // Kinetic part: 1/2 V·M V. The vector-vector product is MPI-global.
  mass_matrix.vmult(tmp, velocity);
  const double kinetic = 0.5 * (velocity * tmp);

  // Potential part: 1/2 U^n·K U^n.
  stiffness_matrix.vmult(tmp, solution_old_owned);
  const double potential = 0.5 * (solution_old_owned * tmp);

  return kinetic + potential;
}


void
Wave::output()
{
  pcout << "DEBUG output at time " << time
      << ", timestep " << timestep_number
      << ", ||solution||_linfty = "
      << solution.linfty_norm()
      << std::endl;
  /**
   * Output to ParaView.
   *
   * Each time step writes:
   *
   *     solution-0000.vtu
   *     solution-0001.vtu
   *     ...
   *
   * We also write:
   *
   *     solution.pvd
   *
   * The .pvd file stores the time-series information.
   *
   * In ParaView:
   *
   *     open solution.pvd
   *
   * not all .vtu files separately.
   */
  DataOut<dim> data_out;

  data_out.attach_dof_handler(dof_handler);

  /**
   * IMPORTANT FIX:
   *
   * Output the ghosted vector "solution", not "solution_owned".
   *
   * This avoids missing locally relevant values in parallel output.
   */
  data_out.add_data_vector(solution, "u");

  data_out.build_patches();

  const std::string filename =
    "solution-" + Utilities::int_to_string(timestep_number, 4) + ".vtu";

  std::ofstream output_file(filename);
  data_out.write_vtu(output_file);
  output_file.close();

  /**
   * Add this file to the ParaView time-series record.
   *
   * time = physical time
   * filename = corresponding VTU file
   */
  output_files.emplace_back(time, filename);

  /**
   * Rewrite solution.pvd every time.
   *
   * At the end of the simulation, solution.pvd contains all time steps.
   */
  std::ofstream pvd_file("solution.pvd");
  DataOutBase::write_pvd_record(pvd_file, output_files);
  pvd_file.close();

  pcout << "  Output: " << filename << std::endl;
}
