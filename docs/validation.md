# Validation: manufactured exact solution

Reference for the validation test cases `caseA` and `caseB` in
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

## Initial conditions

From u at t = 0:

```
u0(x, y) = S(x, y)      (cos(0) = 1)
u1(x, y) = 0            (d/dt cos(omega t) at t = 0 is 0)
```

A variant with `sin(omega t)` instead of `cos(omega t)` gives
`u0 = 0, u1 = omega S(x, y)` with the same f (up to the cos -> sin
factor) and the same g = 0; it can be used to test a nonzero initial
velocity.

## Convergence protocol

- Time step tied to mesh size: `dt ~= 0.2 h`, safely below the
  central-difference stability limit (approximately `dt <= 0.41 h` for
  Q1 elements with consistent mass matrix, wave speed c = 1), so the
  `O(dt^2)` and `O(h^2)` errors shrink together.
- Errors measured at the final time in the L2 norm and H1 seminorm via
  `VectorTools::integrate_difference` with quadrature of order r + 2.
- Expected rates for Q1 elements + central differences:
  L2 error = O(h^2), H1 error = O(h).
