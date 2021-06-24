# Krylov.jl: A Julia basket of hand-picked Krylov methods

| **Documentation** | **Linux/macOS/Windows/FreeBSD** | **Coverage** | **DOI** |
|:-----------------:|:-------------------------------:|:------------:|:-------:|
| [![docs-stable][docs-stable-img]][docs-stable-url] [![docs-dev][docs-dev-img]][docs-dev-url] | [![build-gh][build-gh-img]][build-gh-url] [![build-cirrus][build-cirrus-img]][build-cirrus-url] [![build-drone][build-drone-img]][build-drone-url] | [![codecov][codecov-img]][codecov-url] | [![doi][doi-img]][doi-url] |

[docs-stable-img]: https://img.shields.io/badge/docs-stable-blue.svg
[docs-stable-url]: https://JuliaSmoothOptimizers.github.io/Krylov.jl/stable
[docs-dev-img]: https://img.shields.io/badge/docs-dev-purple.svg
[docs-dev-url]: https://JuliaSmoothOptimizers.github.io/Krylov.jl/dev
[build-gh-img]: https://github.com/JuliaSmoothOptimizers/Krylov.jl/workflows/CI/badge.svg?branch=master
[build-gh-url]: https://github.com/JuliaSmoothOptimizers/Krylov.jl/actions
[build-cirrus-img]: https://img.shields.io/cirrus/github/JuliaSmoothOptimizers/Krylov.jl?logo=Cirrus%20CI
[build-cirrus-url]: https://cirrus-ci.com/github/JuliaSmoothOptimizers/Krylov.jl
[build-drone-img]: https://cloud.drone.io/api/badges/JuliaSmoothOptimizers/Krylov.jl/status.svg
[build-drone-url]: https://cloud.drone.io/JuliaSmoothOptimizers/Krylov.jl
[codecov-img]: https://codecov.io/gh/JuliaSmoothOptimizers/Krylov.jl/branch/master/graph/badge.svg
[codecov-url]: https://app.codecov.io/gh/JuliaSmoothOptimizers/Krylov.jl
[doi-img]: https://img.shields.io/badge/DOI-10.5281%2Fzenodo.822073-blue.svg
[doi-url]: https://doi.org/10.5281/zenodo.822073

## How to Cite

If you use Krylov.jl in your work, please cite using the format given in [`CITATION.bib`](https://github.com/JuliaSmoothOptimizers/Krylov.jl/blob/master/CITATION.bib).

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

All solvers in Krylov.jl have in-place version, are compatible with **GPU** and work in any floating-point data type.

## How to Install

Krylov can be installed and tested through the Julia package manager:

```julia
julia> ]
pkg> add Krylov
pkg> test Krylov
```
