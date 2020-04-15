# [Krylov.jl documentation](@id Home)

This package provides implementations of certain of the most useful Krylov method for a variety of problems:

1 - Square or rectangular full-rank systems

```math
  Ax = b
```

should be solved when **_b_** lies in the range space of **_A_**. This situation occurs when
  * **_A_** is square and nonsingular,
  * **_A_** is tall and has full column rank and **_b_** lies in the range of **_A_**.

2 - Linear least-squares problems

```math
  \min \|b - Ax\|
```

should be solved when **_b_** is not in the range of **_A_** (inconsistent systems), regardless of the shape and rank of **_A_**. This situation mainly occurs when
  * **_A_** is square and singular,
  * **_A_** is tall and thin.

Underdetermined sytems are less common but also occur.

If there are infinitely many such **_x_** (because **_A_** is column rank-deficient), one with minimum norm is identified

```math
  \min \|x\| \quad \text{subject to} \quad x \in \argmin \|b - Ax\|.
```

3 - Linear least-norm problems

```math
  \min \|x\| \quad \text{subject to} \quad Ax = b
```

sould be solved when **_A_** is column rank-deficient but **_b_** is in the range of **_A_** (consistent systems), regardless of the shape of **_A_**.
This situation mainly occurs when
  * **_A_** is square and singular,
  * **_A_** is short and wide.

Overdetermined sytems are less common but also occur.

4 - Adjoint systems

```math
  Ax = b \quad \text{and} \quad A^T y = c
```

where **_A_** can have any shape.

5 - Saddle-point or symmetric quasi-definite (SQD) systems

```math
  \begin{bmatrix} M & \phantom{-}A \\ A^T & -N \end{bmatrix} \begin{bmatrix} x \\ y \end{bmatrix} = \left(\begin{bmatrix} b \\ 0 \end{bmatrix},\begin{bmatrix} 0 \\ c \end{bmatrix},\begin{bmatrix} b \\ c \end{bmatrix}\right)
```

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
