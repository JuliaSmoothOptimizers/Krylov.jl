# Krylov.jl: A Julia basket of hand-picked Krylov methods

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3548984.svg)](https://doi.org/10.5281/zenodo.3548984)

| **Documentation** | **Travis, AppVeyor and Cirrus build statuses** | **Coverage** |
|:-----------------:|:----------------------------------------------:|:------------:|
| [![](https://img.shields.io/badge/docs-dev-blue.svg)](https://JuliaSmoothOptimizers.github.io/Krylov.jl/dev) | [![Build Status](https://img.shields.io/travis/JuliaSmoothOptimizers/Krylov.jl?logo=travis)](https://travis-ci.org/JuliaSmoothOptimizers/Krylov.jl) [![Build status](https://img.shields.io/cirrus/github/JuliaSmoothOptimizers/Krylov.jl?logo=appveyor)](https://ci.appveyor.com/project/dpo/krylov-jl) [![Build Status](https://img.shields.io/cirrus/github/JuliaSmoothOptimizers/Krylov.jl?logo=Cirrus%20CI)](https://cirrus-ci.com/github/JuliaSmoothOptimizers/Krylov.jl) | [![Coverage Status](https://coveralls.io/repos/github/JuliaSmoothOptimizers/Krylov.jl/badge.svg?branch=master)](https://coveralls.io/github/JuliaSmoothOptimizers/Krylov.jl?branch=master) [![codecov.io](https://codecov.io/github/JuliaSmoothOptimizers/Krylov.jl/coverage.svg?branch=master)](https://codecov.io/github/JuliaSmoothOptimizers/Krylov.jl?branch=master) |

## Content

This package provides implementations of certain of the most useful Krylov method for a variety of problems:

1. Square or rectangular full-rank systems

<p align="center">
  <b><i>Ax = b</i></b>
</p>

should be solved when **_b_** lies in the range space of **_A_**. This situation occurs when
  * **_A_** is square and nonsingular,
  * **_A_** is tall and has full column rank and **_b_** lies in the range of **_A_**.

2. Linear least-squares problems

<p align="center">
  minimize ‖<b><i>b</i></b> - <b><i>Ax</i></b>‖
</p>

should be solved when **_b_** is not in the range of **_A_** (inconsistent systems), regardless of the shape and rank of **_A_**. This situation mainly occurs when
  * **_A_** is square and singular,
  * **_A_** is tall and thin.

Underdetermined sytems are less common but also occur.

If there are infinitely many such **_x_** (because **_A_** is column rank-deficient), one with minimum norm is identified

<p align="center">
  minimize ‖<b><i>x</i></b>‖ &nbsp; subject to &nbsp; <b><i>x</i></b> ∈ argmin ‖<b><i>b</i></b> - <b><i>Ax</i></b>‖.
</p>

3. Linear least-norm problems

<p align="center">
  minimize ‖<b><i>x</i></b>‖ &nbsp; subject to &nbsp; <b><i>Ax = b</i></b>
</p>

sould be solved when **_A_** is column rank-deficient but **_b_** is in the range of **_A_** (consistent systems), regardless of the shape of **_A_**.
This situation mainly occurs when
  * **_A_** is square and singular,
  * **_A_** is short and wide.

Overdetermined sytems are less common but also occur.

4. Adjoint systems

<p align="center">
  <b><i>Ax = b</i></b> &nbsp; and &nbsp; <b><i>Aᵀy = c</i></b>
</p>

where **_A_** can have any shape.

5. Saddle-point or symmetric quasi-definite (SQD) systems

<p align="center">
  [<b><i>M </i></b>&nbsp;&nbsp;&nbsp;<b><i> A</i></b>]&nbsp; [<b><i>x</i></b>]            =           [<b><i>b</i></b>]
  <br>
  [<b><i>Aᵀ</i></b>&nbsp;&nbsp;      <b><i>-N</i></b>]&nbsp; [<b><i>y</i></b>]&nbsp;&nbsp;&nbsp;&nbsp;[<b><i>c</i></b>]
</p>

where **_A_** can have any shape.

Krylov solvers are particularly appropriate in situations where such problems must be solved but a factorization is not possible, either because:
* **_A_** is not available explicitly,
* **_A_** would be dense or would consume an excessive amount of memory if it were materialized,
* factors would consume an excessive amount of memory.

Iterative methods are recommended in either of the following situations:
* the problem is sufficiently large that a factorization is not feasible or would be slow,
* an effective preconditioner is known in cases where the problem has unfavorable spectral structure,
* the operator can be represented efficiently as a sparse matrix,
* the operator is *fast*, i.e., can be applied with better complexity than if it were materialized as a matrix. Certain fast operators would materialize as *dense* matrices.

## Features

All solvers in Krylov.jl are compatible with **GPU** and work in any floating-point data type.

## How to Install

Krylov can be installed and tested through the Julia package manager:

```julia
julia> ]
pkg> add Krylov
pkg> test Krylov
```
