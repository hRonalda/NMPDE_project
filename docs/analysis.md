# Numerical analysis: dispersion and stability of the Q1 + leapfrog scheme

Rigorous derivations and measurements for the two graded discussion topics:

- Part 1: the discrete dispersion relation of the scheme (consistent mass),
  its O(h^2) phase error, and an empirical verification against the
  manufactured eigenmode of `docs/validation.md`.
- Part 2: the CFL stability condition, the origin of the constant in
  `dt <= 0.41 h`, and an empirical demonstration of the instability.

All formulas in plain ASCII. All numbers below were measured by actually
running the solver (see "How the measurements were made" at the end);
nothing is asserted from theory alone.

Scheme under analysis (as implemented in `src/Wave.cpp`):

- Space: Q1 (bilinear) elements on the uniform structured mesh of
  Omega = [-5,5]^2 with N = 2^n_refine cells per direction, h = 10/N.
  Consistent mass matrix (no lumping).
- Time: central differences (leapfrog),
  M (U^{n+1} - 2 U^n + U^{n-1}) / dt^2 + K U^n = F^n.

---

## Part 1. Discrete dispersion relation

### 1.1 Interior stencils of M and K

On a uniform 1D mesh with spacing h, the P1 mass and stiffness matrices
have the well-known rows (interior node j; neighbors j-1, j+1):

    K1 : (1/h)  [ -1   2  -1 ]
    M1 : (h/6)  [  1   4   1 ]

On the tensor-product Q1 mesh the 2D matrices are Kronecker products of
the 1D ones:

    M2 = M1 (x) M1
    K2 = K1 (x) M1  +  M1 (x) K1

(the stiffness splits because grad phi . grad psi = phi_x psi_x + phi_y psi_y,
and each term integrates as a 1D stiffness in one direction times a 1D mass
in the other). Multiplying out gives the 3x3 interior stencils:

    M2 : (h^2/36) [ 1   4   1 ]        K2 : (1/3) [ -1  -1  -1 ]
                  [ 4  16   4 ]                   [ -1   8  -1 ]
                  [ 1   4   1 ]                   [ -1  -1  -1 ]

Check of two representative K2 entries from the Kronecker form:
center = (2/h)(4h/6) + (4h/6)(2/h) = 8/6 + 8/6 = 8/3;
corner = (-1/h)(h/6) + (h/6)(-1/h) = -1/6 - 1/6 = -1/3. Both match.

### 1.2 Plane-wave substitution

Insert the discrete plane wave

    U_j^n = exp( i (kx x_j + ky y_j - omega_h t_n) ),
    theta_x = kx h,   theta_y = ky h

into the stencils. Every application of a stencil multiplies U_j^n by its
"symbol", obtained by summing entry * exp(i * phase offset) over the 9
neighbors. Using exp(i a) + exp(-i a) = 2 cos a:

Mass symbol (edge neighbors give 2 cos theta_x resp. 2 cos theta_y, the
four corners give exp(i(+-theta_x +- theta_y)) which sum to
4 cos theta_x cos theta_y):

    m(theta) = (h^2/36) [ 16 + 8 cos theta_x + 8 cos theta_y
                             + 4 cos theta_x cos theta_y ]
             = (h^2/9) (2 + cos theta_x)(2 + cos theta_y)

(the factorization is checked by expanding:
(2+cx)(2+cy) = 4 + 2cx + 2cy + cx cy, times 4 gives the bracket).

Stiffness symbol:

    s(theta) = (1/3) [ 8 - 2 cos theta_x - 2 cos theta_y
                         - 4 cos theta_x cos theta_y ]
             = (2/3) [ 4 - cos theta_x - cos theta_y
                         - 2 cos theta_x cos theta_y ]

which factors into the tensor form (expand to verify):

    s(theta) = (2/3) [ (1 - cos theta_x)(2 + cos theta_y)
                     + (2 + cos theta_x)(1 - cos theta_y) ]

Expansion check: (1-cx)(2+cy) + (2+cx)(1-cy)
= 2 + cy - 2cx - cx cy + 2 + cx - 2cy - cx cy
= 4 - cx - cy - 2 cx cy. Matches.

Time part. With U^n proportional to exp(-i omega_h t_n), t_n = n dt:

    (U^{n+1} - 2 U^n + U^{n-1}) / dt^2
      = U^n (exp(-i omega_h dt) - 2 + exp(+i omega_h dt)) / dt^2
      = U^n (2 cos(omega_h dt) - 2) / dt^2
      = -U^n (4/dt^2) sin^2(omega_h dt / 2)

using 1 - cos a = 2 sin^2(a/2).

### 1.3 The dispersion relation

Substituting into M (U'' discrete) + K U = 0:

    -(4/dt^2) sin^2(omega_h dt/2) m(theta) + s(theta) = 0

    sin^2(omega_h dt / 2) = (dt^2 / 4) * lambda_h(theta)          ... (D)

with the semidiscrete symbol (the generalized eigenvalue of (K,M) at
wavenumber k):

    lambda_h(theta) = s(theta) / m(theta)
                    = (6/h^2) [ G(theta_x) + G(theta_y) ],

    G(theta) = (1 - cos theta) / (2 + cos theta).

(The division separates because both s and m factor over x and y:
s/m = [(1-cx)(2+cy) + (2+cx)(1-cy)] * (2/3) / [(h^2/9)(2+cx)(2+cy)]
    = (6/h^2) [ (1-cx)/(2+cx) + (1-cy)/(2+cy) ].)

Solving (D) for the numerical frequency:

    omega_h = (2/dt) * arcsin( (dt/2) * sqrt(lambda_h(theta)) )     ... (D')

valid while dt^2 lambda_h <= 4 (otherwise no real omega_h exists; that is
exactly the instability of Part 2).

This closed form was cross-checked numerically: the symbol values
G(j pi / N) reproduce the eigenvalues of the assembled 1D generalized
problem K1 v = mu M1 v (computed with a dense eigensolver at n_refine = 4)
to 3e-14.

### 1.4 Small-(kh) expansion: the phase error is O(h^2)

Expand G. With 1 - cos theta = theta^2/2 - theta^4/24 + O(theta^6) and
2 + cos theta = 3 (1 - theta^2/6 + theta^4/72 + ...):

    G(theta) = (theta^2/2)(1 - theta^2/12 + ...)
             / (3 (1 - theta^2/6 + ...))
             = (theta^2/6) (1 - theta^2/12)(1 + theta^2/6) + O(theta^6)
             = (theta^2/6) (1 + theta^2/12) + O(theta^6)

Therefore per direction:

    (6/h^2) G(kx h) = kx^2 (1 + (kx h)^2 / 12) + O(h^4)

and, writing |k|^2 = kx^2 + ky^2:

    lambda_h = |k|^2 + (h^2/12)(kx^4 + ky^4) + O(h^4)               ... (S)

Note the sign: the consistent mass matrix OVERestimates the eigenvalue
(for the lumped/FD stencil the same computation gives
lambda_lump = |k|^2 - (h^2/12)(kx^4+ky^4) + ..., i.e. the opposite sign).

Square root of (S):

    sqrt(lambda_h) = |k| sqrt(1 + h^2 (kx^4+ky^4) / (12 |k|^2))
                   = |k| (1 + h^2 (kx^4+ky^4) / (24 |k|^2)) + O(h^4)

Time contribution. With arcsin(x) = x + x^3/6 + O(x^5), (D') gives

    omega_h = sqrt(lambda_h) (1 + dt^2 lambda_h / 24) + O(dt^4)

(also a positive correction: leapfrog raises the frequency of every mode
it resolves). Combining both, and writing the exact frequency
omega = |k| (wave speed c = 1):

    omega_h = |k| [ 1 + (h^2/24) (kx^4 + ky^4)/|k|^2
                      + (dt^2/24) |k|^2 ]  +  O(h^4, dt^4)

With the time step tied to the mesh, dt = gamma h (the solver uses
gamma ~= 0.2), the numerical phase speed c_h = omega_h / |k| is

    c_h / c = 1 + (h^2/24) [ (kx^4 + ky^4)/|k|^2 + gamma^2 |k|^2 ]
                + O(h^4)                                            ... (P)

Conclusions from (P):

1. The leading phase error is O(h^2): the scheme is second-order
   accurate in phase, consistent with the observed L2 convergence rate 2.
2. The error is a phase LEAD (c_h > c): with the consistent mass matrix
   both the spatial and the temporal contributions are positive, so
   numerical waves travel slightly TOO FAST. (Contrast with lumped
   mass / finite differences, where the spatial term is negative and can
   cancel the temporal one; no such cancellation exists here for any
   dt within the stability window.)
3. Wavelength dependence: the relative phase-speed error scales as
   (kh)^2 = (2 pi h / L)^2 for wavelength L. Short waves (few points per
   wavelength) travel disproportionately too fast; long, well-resolved
   waves are nearly exact. This is what visually distorts the wavefront
   in the Gaussian test case, and it is dispersion, not dissipation:
   part 2 of `docs/validation.md` shows the discrete energy is conserved.
4. Anisotropy: for fixed |k| and propagation angle phi,
   (kx^4 + ky^4)/|k|^4 = 1 - sin^2(2 phi)/2. The spatial error is largest
   for axis-aligned propagation (factor 1) and smallest at 45 degrees
   (factor 1/2). The grid is not isotropic to the wave.

### 1.5 Why the eigenmode measures exactly this omega_h

The validation mode S(x,y) = sin(a_k (x+5)) sin(a_m (y+5)), a_k = a_m =
pi/10, is a superposition of four plane waves (+-a_k, +-a_m). All four
have the same theta up to sign, and lambda_h is even in theta_x, theta_y,
so all four share one omega_h: on the uniform grid with homogeneous
Dirichlet BC the nodal vector S_j is an EXACT eigenvector of the discrete
pair (K, M), with eigenvalue

    lambda_h = (6/h^2) [ G(a_k h) + G(a_m h) ].

Moreover the solver's Taylor startup is phase-exact for eigenmodes: it
sets U^{-1} = U^0 - dt U1 + (dt^2/2) A^0 with M A^0 = -K U^0, i.e.
A^0 = -lambda_h U^0, so

    U^{-1} = (1 - dt^2 lambda_h / 2) U^0 = cos(omega_h dt) U^0

(using cos(omega_h dt) = 1 - 2 sin^2(omega_h dt/2) = 1 - dt^2 lambda_h/2
from (D)). Hence the fully discrete solution is EXACTLY

    U_j^n = S_j cos(omega_h t_n)

up to linear-solver tolerance (1e-10), and the phase of the computed
solution isolates the scheme's dispersion with no startup pollution.

Measurement device: the node (0,0) is a mesh vertex where S = 1 =
max_j |S_j|, so the infinity norm printed by the solver at every step is

    ||U^n||_inf = |cos(omega_h t_n)|.

Fitting omega to this time series (least squares on
2 y^2 - 1 = cos(2 omega t), scan plus golden-section refinement) yields
omega_h directly.

### 1.6 Empirical verification

Runs: `caseA` (k = m = 1, free vibration, one full exact period
T = 10 sqrt(2)) at n_refine = 3..6, unmodified solver, dt = T / ceil(T /
(0.2 h)). Measured omega_h from the fit above; predicted omega_h from the
closed form (D') with theta = a_k h. Phase error after one period:
dphi = (omega_h - omega) T, omega = pi sqrt(2)/10 = 0.44428829.

    n   h        dt        omega_h measured  omega_h predicted  |diff|
    3   1.25000  0.248108  0.44737738        0.44737794         5.6e-07
    4   0.62500  0.124054  0.44505873        0.44505886         1.3e-07
    5   0.31250  0.062300  0.44448210        0.44448094         1.2e-06
    6   0.15625  0.031219  0.44433642        0.44433646         4.3e-08

    n   h        dphi measured [rad]  dphi predicted [rad]  rate
    3   1.25000  4.3686e-02           4.3694e-02            --
    4   0.62500  1.0896e-02           1.0897e-02            2.003
    5   0.31250  2.7408e-03           2.7244e-03            1.991
    6   0.15625  6.8062e-04           6.8122e-04            2.010

(rate = log2 of the ratio of successive measured dphi; h halves each row.)

Findings:

- The measured frequency matches the derived dispersion relation (D') to
  6-7 significant digits at every level -- the derivation is confirmed
  including its constants, not just its order.
- The phase error after one period converges at rate 2.00: the predicted
  O(h^2) phase accuracy is observed.
- omega_h > omega at every level: the predicted phase LEAD (consistent
  mass) is confirmed in sign, not just magnitude.

---

## Part 2. CFL stability condition

### 2.1 Stability of leapfrog on one mode

Decompose the semidiscrete system M U'' + K U = 0 in the generalized
eigenbasis K v = lambda M v (lambda > 0 real, since K and M are SPD after
Dirichlet elimination). Each modal coefficient u^n obeys the scalar
recurrence

    u^{n+1} = (2 - dt^2 lambda) u^n - u^{n-1}

with characteristic polynomial

    xi^2 - (2 - dt^2 lambda) xi + 1 = 0.

The product of the roots is 1. If the roots are complex conjugates they
both lie ON the unit circle (|xi| = 1: no growth, no decay -- this is the
non-dissipativity used in the energy study). If they are real and
distinct, one root has |xi| > 1 and the mode grows exponentially. The
roots are complex iff the discriminant is negative:

    (2 - dt^2 lambda)^2 < 4   <=>   0 < dt^2 lambda < 4.

Requiring this for every eigenvalue gives the stability condition

    dt < 2 / sqrt(lambda_max(M^{-1} K)).                            ... (C)

(At exact equality the double root xi = -1 produces linear-in-n growth,
so the limit itself is excluded.)

### 2.2 lambda_max for Q1 consistent mass on the structured mesh

From Part 1, the symbol of M^{-1}K is

    lambda_h(theta) = (6/h^2) [ G(theta_x) + G(theta_y) ],
    G(theta) = (1 - cos theta)/(2 + cos theta),  theta in (0, pi].

G is increasing on (0, pi]:

    G'(theta) = [ sin theta (2 + cos theta) + sin theta (1 - cos theta) ]
                / (2 + cos theta)^2
              = 3 sin theta / (2 + cos theta)^2  >= 0.

So the supremum is at theta_x = theta_y = pi (the checkerboard mode,
one sign flip per cell in each direction):

    G(pi) = (1 - (-1)) / (2 + (-1)) = 2
    lambda_max = (6/h^2) (2 + 2) = 24 / h^2.

Substituting into (C):

    dt < 2 / sqrt(24/h^2) = 2 h / (2 sqrt(6)) = h / sqrt(6)
       = 0.40825 h.                                                 ... (C')

This is the origin of the "approximately dt <= 0.41 h" used throughout
the project. The constant is C = 1/sqrt(6).

Refinement for the finite Dirichlet grid: the admissible wavenumbers are
theta_j = j pi / N, j = 1..N-1 (modes sin(j pi (x+5)/10) sampled at the
nodes), so theta never reaches pi and the true largest eigenvalue is

    lambda_max,grid = 2 (6/h^2) G( (N-1) pi / N )  <  24/h^2.

For n_refine = 4 (N = 16, h = 0.625): G(15 pi/16) = 1.943467, giving
lambda_max,grid = 59.70256 (vs 61.44 = 24/h^2), verified against a dense
generalized eigensolve of the assembled 1D matrices (agreement 3e-14 via
the tensor sum lambda_2D = mu_i + mu_j). The grid-exact threshold is

    dt_crit,grid = 2 / sqrt(59.70256) = 0.258841 = 0.41415 h,

slightly larger than h/sqrt(6) = 0.255155; the gap closes as N grows.
Design rule: use dt <= C h with C = 1/sqrt(6), which is safe on every
refinement level; the production choice dt = 0.2 h sits at 49% of the
limit.

Remark (cost of the consistent mass): mass lumping (row-summing M2 to
h^2 I) changes the symbol to lambda_lump = s(theta)/h^2, whose maximum is
s(pi,pi)/h^2 = (8/3)/h^2, allowing dt < 2 sqrt(3/8) h = 1.2247 h. The
consistent mass matrix raises lambda_max by a factor 9 (3 per direction)
and therefore tightens the CFL limit by a factor 3. That is the price
paid for the consistent-mass accuracy (and for needing a linear solve per
step at all).

### 2.3 Predicted growth rate above the limit

Just above the limit, dt = (1 + eps) dt_crit gives dt^2 lambda_max =
4 (1+eps)^2 > 4, real roots, and the growing root has magnitude

    |xi| = [ |2 - dt^2 lambda_max| + sqrt( (2 - dt^2 lambda_max)^2 - 4 ) ] / 2.

For eps = 0.02: dt^2 lambda_max = 4.1616, 2 - 4.1616 = -2.1616,
|xi| = (2.1616 + sqrt(2.1616^2 - 4))/2 = 1.490835 per step.

### 2.4 Empirical demonstration

Setup: n_refine = 4 (h = 0.625, dt_crit,grid = 0.258841), Gaussian
initial bump released from rest, f = 0, 400 time steps each run. The
unstable checkerboard mode is not present in the smooth initial data; it
is seeded at the numerical noise floor (linear-solver tolerance ~1e-10 /
roundoff) and must grow out of it, which is exactly what the theory says
happens for dt above the limit.

    run                    dt         dt/h     max ||U^n||_inf   outcome
    baseline               0.125000   0.2000   1.0000            stable
    0.98 * dt_crit,grid    0.253664   0.4059   1.0000            stable
    sharpness check        0.256875   0.4110   1.0000            stable
    1.02 * dt_crit,grid    0.264018   0.4224   2.03e+61          UNSTABLE

Notes on the runs:

- All three stable runs keep ||U^n||_inf <= 1 for all 400 steps (the
  Gaussian's maximum only decreases by physical spreading).
- The sharpness check uses dt = 0.4110 h, which is ABOVE the asymptotic
  limit h/sqrt(6) = 0.40825 h but BELOW the grid-exact threshold
  0.41415 h -- and it is stable, confirming that the operative bound is
  dt_crit = 2/sqrt(lambda_max) with the actual grid spectrum, and that
  h/sqrt(6) is its safe mesh-independent envelope.
- The unstable run blows up through 60 orders of magnitude:

      step      ||U^n||_inf
         0      1.000e+00
        50      4.608e+00
       100      1.903e+09
       150      8.933e+17
       200      4.192e+26
       250      1.968e+35
       300      9.235e+43
       350      4.334e+52
       400      2.034e+61

  A least-squares fit of log ||U^n||_inf over the clean exponential window
  (1e5 < ||U|| < 1e55) gives a growth factor per step of

      exp(0.399336) = 1.490835,

  matching the predicted |xi| = 1.490835 (Section 2.3) to six significant
  digits. The instability is not generic "blow-up": it is the single
  checkerboard mode amplified by exactly the root of the characteristic
  polynomial, which is as sharp a confirmation of (C) as the scheme allows.

### 2.5 Summary of Part 2

    stability condition:   dt < 2 / sqrt(lambda_max(M^{-1}K))
    Q1 consistent mass:    lambda_max -> 24/h^2   (checkerboard mode)
    explicit form:         dt < h / sqrt(6) ~= 0.408 h    [C = 1/sqrt(6)]
    grid-exact (N cells):  dt < 0.41415 h at N = 16, -> h/sqrt(6) as N grows
    production choice:     dt = 0.2 h  (49% of the limit)
    above the limit:       exponential growth |xi|^n, verified to 6 digits

---

## Note: energy with non-homogeneous boundary data (g != 0)

The non-dissipativity result above (flat discrete energy, Part 2's
unit-circle amplification factors) is a statement about the HOMOGENEOUS
problem. With a non-homogeneous Dirichlet datum u = g on the boundary
(case D), the boundary does work on the system:

    dE/dt = integral over boundary of ( u_t * du/dn ) ds   (for f = 0)

which vanishes only when u_t = 0 on the boundary -- i.e. for g = 0 or
static-in-time g. For time-dependent g (case D: g oscillating at the
mode frequency) this power input/output is nonzero, so the discrete
energy E^n computed by `Wave::compute_energy()` genuinely oscillates
over the period. THIS IS PHYSICS, NOT DISSIPATION: a non-flat energy
trace in case D is the correct behavior of the exact solution, not a
numerical artifact, and must not be read as a regression of the scheme.

Consequently:

- The flat-energy non-dissipativity check (the energy table and figure
  in `docs/validation.md`) is meaningful exactly for the g = 0 cases
  (A, B, C, Gaussian) and would also hold for a static nonzero g.
- `compute_energy()` itself is unchanged -- it computes the same
  quadratic form regardless of the boundary data; only the
  interpretation of its output changes when g is time-dependent.

## How the measurements were made

No solver source file was modified for any measurement.

- Part 1 runs: the existing `exercise-01` binary, test case `caseA`, at
  n_refine = 3, 4, 5, 6 (run in a scratch directory). The frequency was
  extracted from the per-step `||solution||_linfty` values the solver
  already prints, which for the eigenmode equal |cos(omega_h t_n)|
  (Section 1.5). Fit: linear least squares in (A, B) for
  2 y^2 - 1 ~ A cos(2 w t) + B sin(2 w t) over a scanned w, refined by
  golden-section on the residual.
- Part 2 runs: `exercise-01` hard-codes dt (tied to 0.2 h in the
  validation cases), so a 12-line throwaway driver was compiled in /tmp
  linking the UNMODIFIED `src/Wave.cpp`, exposing (n_refine, dt, T) on
  the command line with the same Gaussian initial condition as the
  `gaussian` test case. The repository build is untouched.
- Eigenvalue cross-checks (Sections 1.3, 2.2): dense generalized
  eigensolve of the assembled 1D P1 matrices (numpy), compared with the
  symbol G at theta_j = j pi / N; 2D values via the tensor sum.
- Convergence rates quoted here are log2 ratios of successive errors
  under mesh halving.

Related evidence elsewhere in the repo:

- `docs/validation.md`: L2/H1 convergence (rate 2 / rate 1) and the
  energy-conservation study (non-dissipativity), which complements
  Part 1: the scheme's error budget is phase (dispersion), not amplitude.
