# [Preconditioners](@id preconditioners)

The solvers in Krylov.jl support preconditioners, i.e., transformations that modify a linear systems $Ax = b$ into an equivalent form that may yield faster convergence in finite-precision arithmetic.
Preconditioning can be used to reduce the condition number of the problem or clusterize its eigenvalues for instance.

The design of preconditioners is highly dependent on the origin of the problem and most preconditioners need to take application dependent information and structures into account.
Specialized preconditioners generally outperform generic preconditioners such as incomplete factorizations.

The construction of a preconditioner also necessitates a trade-off because we need to apply it at least once per iteration within a Krylov method.
Hence, a preconditioner must be constructed such that it is cheap to apply, while also capturing the characteristics of the original system in some sense.

There exist three variants of preconditioning:

| Left preconditioning               | Two-sided preconditioning                                              | Right preconditioning                        |
|:----------------------------------:|:----------------------------------------------------------------------:|:--------------------------------------------:|
| $P_{\ell}^{-1}Ax = P_{\ell}^{-1}b$ | $P_{\ell}^{-1}AP_r^{-1}y = P_{\ell}^{-1}b~~\text{with}~~x = P_r^{-1}y$ | $AP_r^{-1}y = b~~\text{with}~~x = P_r^{-1}y$ |

where $P_{\ell}$ and $P_r$ are square and nonsingular.

We consider that $P_{\ell}^{-1}$ and $P_r^{-1}$ are the default preconditioners in Krylov.jl and that we can apply them with the operation $y \leftarrow P^{-1} * x$.
It is also common to call $P_{\ell}$ and $P_r$ the preconditioners if the equivalent operation $y \leftarrow P~\backslash~x$ is available.
Krylov.jl supports both approach thanks to the argument `ldiv` of the Krylov solvers.

## How to use preconditioners in Krylov.jl?

!!! tip
    A preconditioner only needs to support the operation `mul!(y, P⁻¹, x)` when `ldiv=false` or `ldiv!(y, P, x)` when `ldiv=true` to be used in Krylov.jl.

### Square non-Hermitian linear systems

Methods concerned: [`CGS`](@ref cgs), [`BiCGSTAB`](@ref bicgstab), [`DQGMRES`](@ref dqgmres), [`GMRES`](@ref gmres), [`DIOM`](@ref diom) and [`FOM`](@ref fom).

A Krylov method dedicated to non-Hermitian linear systems allows the three variants of preconditioning.

| Preconditioners | $P_{\ell}^{-1}$       | $P_{\ell}$           | $P_r^{-1}$            | $P_r$                |
|:---------------:|:---------------------:|:--------------------:|:---------------------:|:--------------------:|
| Arguments       | `M` with `ldiv=false` | `M` with `ldiv=true` | `N` with `ldiv=false` | `N` with `ldiv=true` |

### Hermitian linear systems

Methods concerned: [`SYMMLQ`](@ref symmlq), [`CG`](@ref cg), [`CG-LANCZOS`](@ref cg_lanczos), [`CG-LANCZOS-SHIFT`](@ref cg_lanczos_shift), [`CR`](@ref cr), [`MINRES`](@ref minres) and [`MINRES-QLP`](@ref minres_qlp).

When $A$ is Hermitian, we can only use the centered preconditioning $L^{-1}AL^{-T}y = L^{-1}b$ with $x = L^{-T}y$.
This split preconditioning is a special case of two-sided preconditioning $P_{\ell} = L = P_r^T$ that maintains the hermicity of the linear systems.

| Preconditioners | $P^{-1} = L^{-T}L^{-1}$ | $P = LL^{T}$         |
|:---------------:|:-----------------------:|:--------------------:|
| Arguments       | `M` with `ldiv=false`   | `M` with `ldiv=true` |

!!! warning
    The preconditioner `M` must be hermitian and positive definite.

### Linear least-squares problems

Methods concerned: [`CGLS`](@ref cgls), [`CRLS`](@ref crls), [`LSLQ`](@ref lslq), [`LSQR`](@ref lsqr) and [`LSMR`](@ref lsmr).

| Formulation           | Without preconditioning | With preconditioning           |
|:---------------------:|:-----------------------:|:------------------------------:|
| least-squares problem | $\min \\|b - Ax\\|^2_2$ | $\min \\|b - Ax\\|^2_{E^{-1}}$ |
| Normal equation       | $A^TAx = A^Tb$          | $A^TE^{-1}Ax = A^TE^{-1}b$     |
| Augmented system      | $\begin{bmatrix} I & A \\ A^T & 0 \end{bmatrix} \begin{bmatrix} r \\ x \end{bmatrix} = \begin{bmatrix} b \\ 0 \end{bmatrix}$ | $\begin{bmatrix} E & A \\ A^T & 0 \end{bmatrix} \begin{bmatrix} r \\ x \end{bmatrix} = \begin{bmatrix} b \\ 0 \end{bmatrix}$ |

[`LSLQ`](@ref lslq), [`LSQR`](@ref lsqr) and [`LSMR`](@ref lsmr) also handle regularized least-squares problems.

| Formulation           | Without preconditioning                         | With preconditioning                                   |
|:---------------------:|:-----------------------------------------------:|:------------------------------------------------------:|
| least-squares problem | $\min \\|b - Ax\\|^2_2 + \lambda^2 \\|x\\|^2_2$ | $\min \\|b - Ax\\|^2_{E^{-1}} + \lambda^2 \\|x\\|^2_F$ |
| Normal equation       | $(A^TA + \lambda^2 I)x = A^Tb$              | $(A^TE^{-1}A + \lambda^2 F)x = A^TE^{-1}b$                      |
| Augmented system      | $\begin{bmatrix} I & A \\ A^T & -\lambda^2 I \end{bmatrix} \begin{bmatrix} r \\ x \end{bmatrix} = \begin{bmatrix} b \\ 0 \end{bmatrix}$ | $\begin{bmatrix} E & A \\ A^T & -\lambda^2 F \end{bmatrix} \begin{bmatrix} r \\ x \end{bmatrix} = \begin{bmatrix} b \\ 0 \end{bmatrix}$ |

| Preconditioners | $E^{-1}$                | $E$                  | $F^{-1}$                | $F$                  |
|:---------------:|:-----------------------:|:--------------------:|:-----------------------:|:--------------------:|
| Arguments       | `M` with `ldiv=false`   | `M` with `ldiv=true` | `N` with `ldiv=false`   | `N` with `ldiv=true` |

!!! warning
    The preconditioners `M` and `N` must be hermitian and positive definite.

### Linear least-norm problems

Methods concerned: [`CGNE`](@ref cgne), [`CRMR`](@ref crmr), [`LNLQ`](@ref lnlq), [`CRAIG`](@ref craig) and [`CRAIGMR`](@ref craigmr).

| Formulation          | Without preconditioning                 | With preconditioning                           |
|:--------------------:|:---------------------------------------:|:----------------------------------------------:|
| minimum-norm problem | $\min \\|x\\|^2_2~~\text{s.t.}~~Ax = b$ | $\min \\|x\\|^2_F~~\text{s.t.}~~Ax = b$ |
| Normal equation      | $AA^Ty = b~~\text{with}~~x = A^Ty$      | $AF^{-1}A^Ty = b~~\text{with}~~x = F^{-1}A^Ty$           |
| Augmented system     | $\begin{bmatrix} -I & A^T \\ \phantom{-}A & 0 \end{bmatrix} \begin{bmatrix} x \\ y \end{bmatrix} = \begin{bmatrix} 0 \\ b \end{bmatrix}$ | $\begin{bmatrix} -F & A^T \\ \phantom{-}A & 0 \end{bmatrix} \begin{bmatrix} x \\ y \end{bmatrix} = \begin{bmatrix} 0 \\ b \end{bmatrix}$ |

[`LNLQ`](@ref lslq), [`CRAIG`](@ref lsqr) and [`CRAIGMR`](@ref lsmr) also handle penalized minimum-norm problems.

| Formulation          | Without preconditioning                                             | With preconditioning                                                                    |
|:--------------------:|:-------------------------------------------------------------------:|:---------------------------------------------------------------------------------------:|
| minimum-norm problem | $\min \\|x\\|^2_2 + \\|y\\|^2_2~~\text{s.t.}~~Ax + \lambda^2 y = b$ | $\min \\|x\\|^2_F + \\|y\\|^2_E~~\text{s.t.}~~Ax + \lambda^2 Ey = b$ |
| Normal equation      | $(AA^T + \lambda^2 I)y = b~~\text{with}~~x = A^Ty$                  | $(AF^{-1}A^T + \lambda^2 E)y = b~~\text{with}~~x = F^{-1}A^Ty$                               |
| Augmented system     | $\begin{bmatrix} -I & A^T \\ \phantom{-}A & \lambda^2 I \end{bmatrix} \begin{bmatrix} x \\ y \end{bmatrix} = \begin{bmatrix} 0 \\ b \end{bmatrix}$ | $\begin{bmatrix} -F & A^T \\ \phantom{-}A & \lambda^2 E \end{bmatrix} \begin{bmatrix} x \\ y \end{bmatrix} = \begin{bmatrix} 0 \\ b \end{bmatrix}$ |

| Preconditioners | $E^{-1}$                | $E$                  | $F^{-1}$                | $F$                  |
|:---------------:|:-----------------------:|:--------------------:|:-----------------------:|:--------------------:|
| Arguments       | `M` with `ldiv=false`   | `M` with `ldiv=true` | `N` with `ldiv=false`   | `N` with `ldiv=true` |

!!! warning
    The preconditioners `M` and `N` must be hermitian and positive definite.

### Saddle-point and symmetric quasi-definite systems

When a Hermitian system $Kz = d$ has the 2x2 block structure
```math
  \begin{bmatrix} \tau E & \phantom{-}A \\ A^T & \nu F \end{bmatrix} \begin{bmatrix} x \\ y \end{bmatrix} = \begin{bmatrix} b \\ c \end{bmatrix},
```
where $E$ and $F$ are Hermitian and positive definite, [`TriCG`](@ref tricg) and [`TriMR`](@ref trimr) can take advantage of this form if preconditioners `M` and `N` that model the inverse of $E$ and $F$ are available.

| Preconditioners | $E^{-1}$              | $E$                  | $F^{-1}$              | $F$                  |
|:---------------:|:---------------------:|:--------------------:|:---------------------:|:--------------------:|
| Arguments       | `M` with `ldiv=false` | `M` with `ldiv=true` | `N` with `ldiv=false` | `N` with `ldiv=true` |

!!! warning
    The preconditioners `M` and `N` must be hermitian and positive definite.

### Generalized saddle-point and unsymmetric partitioned systems

When an non-Hermitian system $Kz = d$ has the 2x2 block structure
```math
  \begin{bmatrix} \lambda M & A \\ B & \mu N \end{bmatrix} \begin{bmatrix} x \\ y \end{bmatrix} = \begin{bmatrix} b \\ c \end{bmatrix},
```
[`GPMR`](@ref gpmr) can take advantage of this structure if we model the inverse of $M$ and $N$ with the help of preconditioners `C`, `D`, `E` and `F`.

| Relations       | $CE = M^{-1}$                 | $EC = M$                     | $DF = N^{-1}$                 | $FD = N$                     |
|:---------------:|:-----------------------------:|:----------------------------:|:-----------------------------:|:----------------------------:|
| Arguments       | `C` and `E` with `ldiv=false` | `C` and `E` with `ldiv=true` | `D` and `F` with `ldiv=false` | `D` and `F` with `ldiv=true` |

!!! note
    Our implementations of [`BiLQ`](@ref bilq), [`QMR`](@ref qmr), [`BiLQR`](@ref bilqr), [`USYMLQ`](@ref usymlq), [`USYMQR`](@ref usymqr) and [`TriLQR`](@ref trilqr) don't support preconditioning.

!!! info
    The default value of a preconditioner in Krylov.jl is the identity operator `I`.

## Packages that provide preconditioners

- [IncompleteLU.jl](https://github.com/haampie/IncompleteLU.jl) implements the left-looking or Crout version of ILU decompositions.
- [ILUZero.jl](https://github.com/mcovalt/ILUZero.jl)  is a Julia implementation of incomplete LU factorization with zero level of fill-in. 
- [LimitedLDLFactorizations.jl](https://github.com/JuliaSmoothOptimizers/LimitedLDLFactorizations.jl) for limited-memory LDLᵀ factorization of symmetric matrices.
- [AlgebraicMultigrid.jl](https://github.com/JuliaLinearAlgebra/AlgebraicMultigrid.jl) provides two algebraic multigrid (AMG) preconditioners.
- [RandomizedPreconditioners.jl](https://github.com/tjdiamandis/RandomizedPreconditioners.jl) uses randomized numerical linear algebra to construct approximate inverses of matrices.

## Examples

```julia
using Krylov
n, m = size(A)
d = [A[i,i] ≠ 0 ? 1 / abs(A[i,i]) : 1 for i=1:n]  # Jacobi preconditioner
P⁻¹ = diagm(d)
x, stats = symmlq(A, b, M=P⁻¹)
```

```julia
using Krylov
n, m = size(A)
d = [1 / norm(A[:,i]) for i=1:m]  # diagonal preconditioner
P⁻¹ = diagm(d)
x, stats = minres(A, b, M=P⁻¹)
```

```julia
using IncompleteLU, Krylov
Pℓ = ilu(A)
x, stats = gmres(A, b, M=Pℓ, ldiv=true)  # left preconditioning
```

```julia
using LimitedLDLFactorizations, Krylov
P = lldl(A)
P.D .= abs.(P.D)
x, stats = cg(A, b, M=P, ldiv=true)  # centered preconditioning
```

```julia
using ILUZero, Krylov
Pᵣ = ilu0(A)
x, stats = bicgstab(A, b, N=Pᵣ, ldiv=true)  # right preconditioning
```
