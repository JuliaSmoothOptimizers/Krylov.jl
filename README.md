# Krylov.jl: A Julia basket of hand-picked Krylov methods

[![DOI](https://zenodo.org/badge/31977760.svg)](https://zenodo.org/badge/latestdoi/31977760)
[![Build Status](https://travis-ci.org/JuliaSmoothOptimizers/Krylov.jl.svg?branch=master)](https://travis-ci.org/JuliaSmoothOptimizers/Krylov.jl)
[![Build status](https://ci.appveyor.com/api/projects/status/3xt558lune9f5r2v?svg=true)](https://ci.appveyor.com/project/dpo/krylov-jl)
[![Coverage Status](https://coveralls.io/repos/github/JuliaSmoothOptimizers/Krylov.jl/badge.svg?branch=master)](https://coveralls.io/github/JuliaSmoothOptimizers/Krylov.jl?branch=master)
[![codecov.io](https://codecov.io/github/JuliaSmoothOptimizers/Krylov.jl/coverage.svg?branch=master)](https://codecov.io/github/JuliaSmoothOptimizers/Krylov.jl?branch=master)

## Purpose

This package implements iterative methods for the solution of linear systems of equations
<p align="center">
  <b><i>Ax = b</i></b>,
</p>
and linear least-squares problems
<p align="center">
  minimize ‖<b><i>b</i></b> - <b><i>Ax</i></b>‖.
</p>

It is appropriate, in particular, in situations where such a problem must be solved but a factorization is not possible, either because:
* the operator is not available explicitly,
* the operator is dense, or
* factors would consume an excessive amount of memory and/or disk space.

Iterative methods are particularly appropriate in either of the following situations:
* the problem is sufficiently large that a factorization is not feasible or would be slower,
* an effective preconditioner is known in cases where the problem has unfavorable spectral structure,
* the operator can be represented efficiently as a sparse matrix,
* the operator is *fast*, i.e., can be applied with far better complexity than if it were materialized as a matrix. Often, fast operators would materialize as *dense* matrices.

## Objective: solve *Ax ≈ b*

Given a linear operator **_A_** and a right-hand side **_b_**, solve **_Ax ≈ b_**, which means:

1. when **_A_** has full column rank and **_b_** lies in the range space of **_A_**, find the unique **_x_** such that **_Ax = b_**; this situation occurs when
   * **_A_** is square and nonsingular, or
   * **_A_** is tall and has full column rank and **_b_** lies in the range of **_A_**,
2. when **_A_** is column-rank deficient but **_b_** is in the range of **_A_**, find **_x_** with minimum norm such that **_Ax = b_**; this situation occurs when **_b_** is in the range of **_A_** and
   * **_A_** is square but singular, or
   * **_A_** is short and wide,
3. when **_b_** is not in the range of **_A_**, regardless of the shape and rank of **_A_**, find **_x_** that minimizes the residual ‖**_b_** - **_Ax_**‖. If there are infinitely many such **_x_** (because **_A_** is rank deficient), identify the one with minimum norm.

## How to Install

At the Julia prompt, type

````JULIA
julia> Pkg.clone("https://github.com/JuliaSmoothOptimizers/Krylov.jl.git")
julia> Pkg.build("Krylov")
julia> Pkg.test("Krylov")
````

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

This content is released under the [MIT](http://opensource.org/licenses/MIT) License.
<a rel="license" href="http://opensource.org/licenses/MIT">
<img alt="MIT license" height="40" src="http://upload.wikimedia.org/wikipedia/commons/c/c3/License_icon-mit.svg" /></a>
