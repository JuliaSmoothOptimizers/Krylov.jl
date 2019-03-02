# [Krylov.jl documentation](@id Home)

This package implements iterative methods for the solution of linear systems of equations
```math
  Ax = b,
```
linear least-squares problems
```math
  \min \|b - Ax\|,
```
and linear least-norm problems
```math
  \min \|x\| \quad \text{subject to} \ Ax = b.
```

It is appropriate, in particular, in situations where such a problem must be solved but a factorization is not possible, either because:
* the operator is not available explicitly,
* the operator is dense, or
* factors would consume an excessive amount of memory and/or disk space.

Iterative methods are particularly appropriate in either of the following situations:
* the problem is sufficiently large that a factorization is not feasible or would be slower,
* an effective preconditioner is known in cases where the problem has unfavorable spectral structure,
* the operator can be represented efficiently as a sparse matrix,
* the operator is *fast*, i.e., can be applied with far better complexity than if it were materialized as a matrix. Often, fast operators would materialize as *dense* matrices.

## Objective: solve ``Ax \approx b``

Given a linear operator ``A`` and a right-hand side ``b``, solve ``Ax = b``, which means:

1. when ``A`` has full column rank and ``b`` lies in the range space of ``A``, find the unique ``x`` such that ``Ax = b``; this situation occurs when
   * ``A`` is square and nonsingular, or
   * ``A`` is tall and has full column rank and ``b`` lies in the range of ``A``,
2. when ``A`` is column-rank deficient but ``b`` is in the range of ``A``, find ``x`` with minimum norm such that ``Ax = b``; this situation occurs when ``b`` is in the range of ``A`` and
   * ``A`` is square but singular, or
   * ``A`` is short and wide,
3. when ``b`` is not in the range of ``A``, regardless of the shape and rank of ``A``, find ``x`` that minimizes the residual ``\|b - Ax\|``. If there are infinitely many such ``x`` (because ``A`` is rank deficient), identify the one with minimum norm.

## How to Install

Krylov can be installed and tested through the Julia package manager:

```julia
julia> Pkg.add("Krylov")
julia> Pkg.test("Krylov")
```

## Long-Term Goals

* provide implementations of certain of the most useful Krylov method for
  linear systems with special emphasis on methods for linear least-squares
  problems and saddle-point linear system (including symmetric quasi-definite
  systems)
* provide state-of-the-art implementations alongside simple implementations of
  equivalent methods in exact artithmetic (e.g., LSQR vs. CGLS, MINRES vs. CR,
  LSMR vs. CRLS, etc.)
* provide simple, consistent calling signatures and avoid over-typing
* ensure those implementations are fast and stable.
