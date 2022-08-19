## [Preconditioners](@id preconditioners)

The solvers in Krylov.jl support preconditioners that modify a given linear systems $Ax = b$ into a form that allows a faster convergence.

It exists three variants of preconditioning:

| Left preconditioning | Two-sided preconditioning       | Right preconditioning          |
|:--------------------:|:-------------------------------:|:------------------------------:|
| $MAx = Mb$           | $MANy = Mb~~\text{with}~~x = Ny$| $ANy = b~~\text{with}~~x = Ny$ |

#### Unsymmetric linear systems

A Krylov method dedicated to unsymmetric systems allows the three variants.
We provide these preconditioners with the arguments `M` and `N`.
It concerns the methods [`CGS`](@ref cgs), [`BiCGSTAB`](@ref bicgstab), [`DQGMRES`](@ref dqgmres), [`GMRES`](@ref gmres), [`DIOM`](@ref diom) and [`FOM`](@ref fom).

#### Symmetric linear systems

When $A$ is symmetric, we can only use the centered / split preconditioning $LAL^Tx = Lb$.
It is a special case of two-sided preconditioning $M=L=N^T$ that maintains the symmetry of the linear systems.
Krylov methods dedicated to symmetric systems take directly as input a symmetric positive preconditioner $P=LL^T$.
We provide this preconditioner with the argument `M` in [`SYMMLQ`](@ref symmlq), [`CG`](@ref cg), [`CG-LANCZOS`](@ref cg_lanczos), [`CG-LANCZOS-SHIFT`](@ref cg_lanczos_shift), [`CR`](@ref cr), [`MINRES`](@ref minres) and [`MINRES-QLP`](@ref minres_qlp).

#### Least-squares problems

| Formulation           | Without preconditioning | With preconditioning    |
|:---------------------:|:-----------------------:|:-----------------------:|
| least-squares problem | $\min \\|b - Ax\\|^2_2$ | $\min \\|b - Ax\\|^2_M$ |
| Normal equation       | $A^TAx = A^Tb$          | $A^TMAx = A^TMb$        |
| Augmented system      | $\begin{bmatrix} I & A \\ A^T & 0 \end{bmatrix} \begin{bmatrix} r \\ x \end{bmatrix} = \begin{bmatrix} b \\ 0 \end{bmatrix}$ | $\begin{bmatrix} M & A \\ A^T & 0 \end{bmatrix} \begin{bmatrix} r \\ x \end{bmatrix} = \begin{bmatrix} b \\ 0 \end{bmatrix}$ |

We provide a symmetric positive definite preconditioner with the argument `M` in [`CGLS`](@ref cgls), [`CRLS`](@ref crls), [`LSLQ`](@ref lslq), [`LSQR`](@ref lsqr) and [`LSMR`](@ref lsmr).

A second positive definite preconditioner `N` is supported by [`LSLQ`](@ref lslq), [`LSQR`](@ref lsqr) and [`LSMR`](@ref lsmr).
It is dedicated to regularized least-squares problems.

| Formulation           | Without preconditioning                         | With preconditioning                                   |
|:---------------------:|:-----------------------------------------------:|:------------------------------------------------------:|
| least-squares problem | $\min \\|b - Ax\\|^2_2 + \lambda^2 \\|x\\|^2_2$ | $\min \\|b - Ax\\|^2_M + \lambda^2 \\|x\\|^2_{N^{-1}}$ |
| Normal equation       | $(A^TA + \lambda^2 I)x = A^Tb$              | $(A^TMA + \lambda^2 N^{-1})x = A^TMb$                      |
| Augmented system      | $\begin{bmatrix} I & A \\ A^T & -\lambda^2 I \end{bmatrix} \begin{bmatrix} r \\ x \end{bmatrix} = \begin{bmatrix} b \\ 0 \end{bmatrix}$ | $\begin{bmatrix} M & A \\ A^T & -\lambda^2 N \end{bmatrix} \begin{bmatrix} r \\ x \end{bmatrix} = \begin{bmatrix} b \\ 0 \end{bmatrix}$ |

#### Minimum-norm problems

| Formulation          | Without preconditioning                 | With preconditioning                           |
|:--------------------:|:---------------------------------------:|:----------------------------------------------:|
| minimum-norm problem | $\min \\|x\\|^2_2~~\text{s.t.}~~Ax = b$ | $\min \\|x\\|^2_{N^{-1}}~~\text{s.t.}~~Ax = b$ |
| Normal equation      | $AA^Ty = b~~\text{with}~~x = A^Ty$      | $ANA^Ty = b~~\text{with}~~x = NA^Ty$           |
| Augmented system     | $\begin{bmatrix} -I & A^T \\ \phantom{-}A & 0 \end{bmatrix} \begin{bmatrix} x \\ y \end{bmatrix} = \begin{bmatrix} 0 \\ b \end{bmatrix}$ | $\begin{bmatrix} -N & A^T \\ \phantom{-}A & 0 \end{bmatrix} \begin{bmatrix} x \\ y \end{bmatrix} = \begin{bmatrix} 0 \\ b \end{bmatrix}$ |

We provide a symmetric positive definite preconditioner with the argument `N` in [`CGNE`](@ref cgne), [`CRMR`](@ref crmr), [`LNLQ`](@ref lnlq), [`CRAIG`](@ref craig) and [`CRAIGMR`](@ref craigmr).
A second positive definite preconditioner `M` is supported by [`LNLQ`](@ref lslq), [`CRAIG`](@ref lsqr) and [`CRAIGMR`](@ref lsmr).
It is dedicated to penalized minimum-norm problems.

| Formulation          | Without preconditioning                                             | With preconditioning                                                                    |
|:--------------------:|:-------------------------------------------------------------------:|:---------------------------------------------------------------------------------------:|
| minimum-norm problem | $\min \\|x\\|^2_2 + \\|y\\|^2_2~~\text{s.t.}~~Ax + \lambda^2 y = b$ | $\min \\|x\\|^2_{N^{-1}} + \\|y\\|^2_{M^{-1}}~~\text{s.t.}~~Ax + \lambda^2 M^{-1}y = b$ |
| Normal equation      | $(AA^T + \lambda^2 I)y = b~~\text{with}~~x = A^Ty$                  | $(ANA^T + \lambda^2 M^{-1})y = b~~\text{with}~~x = NA^Ty$                               |
| Augmented system     | $\begin{bmatrix} -I & A^T \\ \phantom{-}A & \lambda^2 I \end{bmatrix} \begin{bmatrix} x \\ y \end{bmatrix} = \begin{bmatrix} 0 \\ b \end{bmatrix}$ | $\begin{bmatrix} -N^{-1} & A^T \\ \phantom{-}A & \lambda^2 M^{-1} \end{bmatrix} \begin{bmatrix} x \\ y \end{bmatrix} = \begin{bmatrix} 0 \\ b \end{bmatrix}$ |

#### Saddle-point and symmetric quasi-definite systems

When a symmetric system $Kz = d$ has the 2x2 block structure
```math
  \begin{bmatrix} \tau M^{-1} & \phantom{-}A \\ A^T & \nu N^{-1} \end{bmatrix} \begin{bmatrix} x \\ y \end{bmatrix} = \begin{bmatrix} b \\ c \end{bmatrix},
```
where $M^{-1}$ and $N^{-1}$ are symmetric positive definite, [`TriCG`](@ref tricg) and [`TriMR`](@ref trimr) can take advantage of this structure if preconditioners `M` and `N` that model $M$ and $N$ are available.

#### Generalized saddle-point and unsymmetric partitioned systems

When an unsymmetric system $Kz = d$ has the 2x2 block structure
```math
  \begin{bmatrix} \lambda M^{-1} & A \\ B & \mu N^{-1} \end{bmatrix} \begin{bmatrix} x \\ y \end{bmatrix} = \begin{bmatrix} b \\ c \end{bmatrix},
```
[`GPMR`](@ref gpmr) can take advantage of this structure if preconditioners `C`, `D`, `E` and `F` such that $CE = M$ and $DF = N$ are available.

!!! tip
	A preconditioner `P` only needs to support the operation `mul!(y, P, x)` to be used in Krylov.jl.

!!! note
    Our implementations of [`BiLQ`](@ref bilq), [`QMR`](@ref qmr), [`BiLQR`](@ref bilqr), [`USYMLQ`](@ref usymlq), [`USYMQR`](@ref usymqr) and [`TriLQR`](@ref trilqr) don't support preconditioning.

## Packages that provide preconditioners

- [IncompleteLU.jl](https://github.com/haampie/IncompleteLU.jl) implements the left-looking or Crout version of ILU decompositions.
- [ILUZero.jl](https://github.com/mcovalt/ILUZero.jl)  is a Julia implementation of incomplete LU factorization with zero level of fill-in. 
- [LimitedLDLFactorizations.jl](https://github.com/JuliaSmoothOptimizers/LimitedLDLFactorizations.jl) for limited-memory LDLᵀ factorization of symmetric matrices.
- [AlgebraicMultigrid.jl](https://github.com/JuliaLinearAlgebra/AlgebraicMultigrid.jl) provides two algebraic multigrid (AMG) preconditioners.
