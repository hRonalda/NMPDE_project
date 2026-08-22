# Validation: manufactured exact solution

Reference for the validation test cases `caseA`, `caseB`, and `caseC` in
`src/exercise-01.cpp`. The exact solution is implemented in
`WaveExactSolution` (`src/Wave.hpp`).

## Problem

We validate the solver for the 2D wave equation on the fixed domain
`Omega = [-5, 5] x [-5, 5]`:

```
u_tt - Lap(u) = f      in Omega
u = g                  on the boundary of Omega
u(x, y, 0)   = u0      in Omega
u_t(x, y, 0) = u1      in Omega
```

## Candidate solution

Take a Dirichlet eigenmode of the Laplacian on the square, with
integers k, m >= 1:

```
u(x, y, t) = sin(a_k (x + 5)) * sin(a_m (y + 5)) * cos(omega t)

a_k = k pi / 10,    a_m = m pi / 10
```

The map `x -> (x + 5) / 10` sends `[-5, 5]` to `[0, 1]`, so the sine
factors are the standard unit-square eigenfunctions stretched to the
project domain. Write for brevity

```
S(x, y) = sin(a_k (x + 5)) * sin(a_m (y + 5)),      u = S(x, y) cos(omega t)
```

## Boundary condition: g = 0 exactly

On each edge of the square one sine factor vanishes identically:

- `x = -5`: `sin(a_k * 0) = sin(0) = 0`
- `x = +5`: `sin(a_k * 10) = sin(k pi) = 0`   (k integer)
- `y = -5` and `y = +5`: same argument with m.

Hence `u = 0` on the whole boundary for all t, i.e. the Dirichlet datum
is homogeneous: `g = 0`.

## PDE residual: what f must be

Time derivative (S does not depend on t):

```
u_tt = -omega^2 * S * cos(omega t)
```

Space derivatives (each sine differentiates twice into itself times
`-a^2`):

```
u_xx = -a_k^2 * S * cos(omega t)
u_yy = -a_m^2 * S * cos(omega t)

Lap(u) = -(a_k^2 + a_m^2) * S * cos(omega t)
```

Substituting into the PDE, with `lambda = a_k^2 + a_m^2
= (k^2 + m^2) pi^2 / 100`:

```
u_tt - Lap(u) = (lambda - omega^2) * S(x, y) * cos(omega t) =: f(x, y, t)
```

(Verified symbolically with sympy in addition to the hand derivation.)

## Test cases

### Case A: free vibration, f = 0

Choose omega equal to the eigenfrequency:

```
omega = sqrt(lambda) = pi sqrt(k^2 + m^2) / 10
```

Then `lambda - omega^2 = 0`, so `f = 0`: u solves the homogeneous wave
equation exactly. For k = m = 1:

```
omega = pi sqrt(2) / 10 ~= 0.4443
period = 2 pi / omega = 10 sqrt(2) ~= 14.142
```

The solver runs for one full period, so at the final time the exact
solution has returned to the initial condition.

### Case B: nonzero forcing

Choose any other omega; we use `omega = 1`. Then, explicitly:

```
f(x, y, t) = ((k^2 + m^2) pi^2 / 100 - 1)
             * sin(k pi (x + 5) / 10) * sin(m pi (y + 5) / 10) * cos(t)
```

with g = 0 still. This case exercises the load-vector assembly, which
Case A cannot detect (it passes even if the forcing term is ignored).
The solver runs for one forcing period `T = 2 pi`.

### Case C: nonzero initial velocity

Same spatial mode S(x, y) and the same omega as Case A (the
eigenfrequency, so `f = 0` again), but the time factor is `sin(omega t)`
instead of `cos(omega t)`:

```
u(x, y, t) = S(x, y) * sin(omega t)
```

g = 0 exactly, by the same edge argument as before (it only depends on
S, not on the time factor). But the initial data are now the OPPOSITE
of Case A's:

```
u0(x, y) = 0                    (sin(0) = 0)
u1(x, y) = omega * S(x, y)      (d/dt sin(omega t) at t = 0 is omega)
```

Case A and Case B both use `u1 = 0`, so neither ever exercises the
nonzero-`u1` branch of the Taylor startup for `U^{-1}` in `Wave::run()`
(`U^{-1} = U^0 - dt * u1 + 0.5 dt^2 A^0`). Case C is the case designed
specifically to validate that branch: implemented via
`WaveExactSolution(k, m, omega, /*use_sine=*/true)` for the exact
solution/error reference, `Functions::ZeroFunction` for u0, and the new
`WaveEigenmodeVelocity` class (`src/Wave.hpp`) for u1 = omega * S. Runs
for one full period, same as Case A.

## Initial conditions

From u at t = 0:

```
u0(x, y) = S(x, y)      (cos(0) = 1)
u1(x, y) = 0            (d/dt cos(omega t) at t = 0 is 0)
```

The `sin(omega t)` variant (Case C, above) gives `u0 = 0, u1 = omega
S(x, y)` with the same f (up to the cos -> sin factor) and the same
g = 0; it is what tests the nonzero-initial-velocity path.

## Convergence protocol

- Time step tied to mesh size: `dt ~= 0.2 h`, safely below the
  central-difference stability limit (approximately `dt <= 0.41 h` for
  Q1 elements with consistent mass matrix, wave speed c = 1), so the
  `O(dt^2)` and `O(h^2)` errors shrink together.
- Errors measured at the final time in the L2 norm and H1 seminorm via
  `VectorTools::integrate_difference` with quadrature of order r + 2.
- Expected rates for Q1 elements + central differences:
  L2 error = O(h^2), H1 error = O(h).

## Case C results: validating the nonzero-u1 startup

Measured at n_refine = 3, 4, 5, 6, errors at t = T (one full period):

```
n_refine    h          L2 error     H1 error
   3      1.25000     0.211825     0.094717
   4      0.62500     0.0540706    0.0240615
   5      0.31250     0.0135961    0.00604303
   6      0.15625     0.0034045    0.00151273
```

```
rates (log2 of successive error ratios):
   L2:  1.97   1.99   2.00
   H1:  1.98   1.99   2.00
```

L2 converges at the expected rate 2. H1 also converges at rate 2, not
the "generic" rate 1 seen in Cases A and B -- this is a real,
explainable effect, not noise, and worth a paragraph in the report:

**Why H1 is O(h^2) here.** T is defined as one full period of the
*continuous* exact solution, so `sin(omega T) = sin(2 pi) = 0` exactly.
Because u0 = 0 forces the cosine component of the discrete modal
solution to vanish identically (not just approximately), the fully
discrete solution is exactly `U_j^n = S_j * B * sin(omega_h t_n)` for
all n, for a constant B ~= 1 (omega_h is the discrete/dispersive
frequency from `docs/analysis.md`, Part 1). At t = T:

```
u_h(T)    = B * sin(omega_h T) * I_h[S]           (I_h[S] = Q1 nodal interpolant of S)
u_exact(T) = S(x,y) * sin(omega T) = 0             (exactly, continuous solution)
```

so the entire error reduces to the single scalar factor
`B * sin(omega_h T)`. Since `omega_h T = 2 pi * (omega_h/omega) = 2 pi +
O(h^2)` (the O(h^2) phase error derived in `docs/analysis.md`),
`sin(omega_h T) = O(h^2)`. This factor multiplies the FEM function
`I_h[S]` in BOTH the L2 and H1 norms, so both errors are governed by the
same O(h^2) term -- the "generic" O(h) gradient-interpolation error that
normally dominates H1 gets multiplied by this same near-zero amplitude
and drops to O(h^3), negligible.

**Control check (not part of the committed test case).** Re-running the
identical Case C setup but stopping at a quarter period instead (where
`sin(omega t) = 1`, i.e. the sine envelope is at max amplitude, not
zero) reverts the H1 rate to ~1 and reproduces Case A's H1 error
magnitudes almost exactly at matching n_refine:

```
n_refine    h          L2 error     H1 error     (Case A H1 for comparison)
   3      1.25000     0.156971     0.25455       0.253323
   4      0.62500     0.0395189    0.126259      0.126082
   5      0.31250     0.00988856   0.0630001     0.0629774
```

H1 rates there: 1.01, 1.00 -- confirming this is a measurement-time
artifact of hitting the sine envelope's zero-crossing at t = T, not a
flaw in the scheme, and that the underlying spatial accuracy is still
the expected O(h) in H1 as in Cases A and B.

**Conclusion:** both the L2 rate-2 and (once explained) the H1 rate-2
results confirm the Taylor startup's nonzero-u1 branch is implemented
correctly -- the errors converge cleanly at (at least) the expected
rates, with the H1 "over-performance" fully accounted for by the
t = T zero-crossing rather than being an unexplained anomaly.

## Energy conservation (non-dissipativity of central differences)

The discrete energy of the semi-discrete system is

```
E^n = 1/2 ( V^n . M V^n + U^n . K U^n ),
V^n = (U^{n+1} - U^{n-1}) / (2 dt)
```

the FEM counterpart of `E(t) = 1/2 integral(u_t^2 + |grad u|^2) dx`.
It is computed in `Wave::compute_energy()` reusing the assembled M and
K, and written to `energy_history.txt`.

The central difference (leapfrog) scheme is non-dissipative: within the
stability limit its amplification factors lie exactly on the unit
circle, so E^n must stay flat -- a bounded oscillation of size O(dt^2)
around the exact energy, with no drift. Measured on Case A over one
full period (T = 10 sqrt(2), dt ~= 0.2 h):

```
n_refine       dt       E^0 = E^end     (max-min)/E^0
   4        0.12405     2.44373         7.6e-04
   5        0.06230     2.46146         1.9e-04
   6        0.03122     2.46592         4.8e-05
```

**Figure:** `figures/energy_conservation.png` (generated by `energy_plot.py`
from `build/energy_history.txt`; run `./exercise-01 5 caseA` first to
produce the energy history).

> Figure X: Discrete energy E(t) for case A (k=m=1, f=0) over one full
> period on the n_refine=5 mesh. Left: full range -- the energy is
> visually constant at E^0 ~= 2.4615. Right: zoomed y-axis -- the energy
> exhibits only a bounded oscillation of relative amplitude 1.9e-04 with
> a negligible linear trend of -2.3e-06, confirming the leapfrog scheme
> is non-dissipative. The observed amplitude decay in the Gaussian case
> is therefore dispersion, not dissipation.

Observations for the report:

- E at the end of the period returns to its initial value to ~7
  significant digits at every level; the linear-fit drift over the
  period is < 2e-05 of E^0 and shrinks under refinement -> no
  dissipation.
- The oscillation amplitude decreases by a factor ~4 per halving of dt:
  the expected O(dt^2) bounded oscillation of leapfrog.
- E^0 converges to the exact continuous energy of the mode,
  E = pi^2 / 4 ~= 2.46740 (for k = m = 1: E = lambda * 25 / 2 with
  lambda = 2 pi^2 / 100, since integral(S^2) = 25 over [-5,5]^2).

Consequence: in the Gaussian test case the decay of the center
amplitude is physical 2D spreading (energy leaves the center, not the
domain); the scheme's actual numerical error there is DISPERSION
(mesh-dependent phase error), not dissipation.
