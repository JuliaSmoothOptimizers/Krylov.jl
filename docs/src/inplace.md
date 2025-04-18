## [In-place methods](@id in-place)

All solvers in Krylov.jl have an in-place variant implemented in a method whose name ends with `!`.
A workspace (`KrylovWorkspace` or `BlockKrylovWorkspace`), which contains the storage needed by a Krylov method, can be used to solve multiple linear systems with the same dimensions and the same floating-point precision.
It should also be the same number of right-hand sides in block Krylov methods.
The section [storage requirements](@ref storage-requirements) specifies the memory needed for each Krylov method.

Each `KrylovWorkspace` has three constructors with consistent argument patterns:

```@constructors
XyzWorkspace(A, b)
XyzWorkspace(m, n, S)
XyzWorkspace(kc::KrylovConstructor)
```
The only exceptions are `CgLanczosShiftWorkspace` and `CglsLanczosShiftWorkspace`, which require an additional argument `nshifts`.
Additionally, some constructors accept keyword arguments.

`Xyz` represents the name of the Krylov method, written in lowercase except for its first letter (such as `Cg`, `Minres`, `Lsmr` or `Bicgstab`).
If the name of the Krylov method contains an underscore (e.g., `minres_qlp` or `cgls_lanczos_shift`), the workspace constructor transforms it by capitalizing each word and removing underscores, resulting in names like `MinresQlpWorkspace` or `CglsLanczosShiftWorkspace`.

Given an operator `A` and a right-hand side `b`, you can create a `KrylovWorkspace` based on the size of `A` and the type of `b`, or explicitly provide the dimensions `(m, n)` and the storage type `S`.

We assume that `S(undef, 0)`, `S(undef, n)`, and `S(undef, m)` are well-defined for the storage type `S`.
For more advanced vector types, workspaces can also be created with the help of a `KrylovConstructor`.
```@docs
Krylov.KrylovConstructor
```
See the section [custom workspaces](@ref custom_workspaces) for an example where this constructor is the only applicable option.

For example, use `S = Vector{Float64}` if you want to solve linear systems in double precision on the CPU and `S = CuVector{Float32}` if you want to solve linear systems in single precision on an Nvidia GPU.

The workspace is always the first argument of the in-place methods:

```@solvers
minres_workspace = MinresWorkspace(m, n, Vector{Float64})
minres!(minres_workspace, A1, b1)

bicgstab_workspace = BicgstabWorkspace(m, n, Vector{ComplexF64})
bicgstab!(bicgstab_workspace, A2, b2)

gmres_workspace = GmresWorkspace(m, n, Vector{BigFloat})
gmres!(gmres_workspace, A3, b3)

lsqr_workspace = LsqrWorkspace(m, n, CuVector{Float32})
lsqr!(lsqr_workspace, A4, b4)
```

## [Workspace accessors](@id workspace-accessors)

In-place solvers update the workspace, from which solutions and statistics can be retrieved.
The following functions are available for post-solve analysis.

These functions are not exported and must be accessed using the prefix `Krylov.`, e.g. `Krylov.solution(workspace)`.

```@docs
Krylov.results
Krylov.solution
Krylov.statistics
Krylov.elapsed_time
Krylov.solution_count
Krylov.iteration_count
Krylov.Aprod_count
Krylov.Atprod_count
Krylov.issolved
```

## Examples

We illustrate the use of in-place Krylov solvers with two well-known optimization methods.
The details of the optimization methods are described in the section about [Factorization-free operators](@ref matrix-free).

### Example 1: Newton's method for convex optimization without linesearch

```@newton
using Krylov
import Krylov: solution

function newton(∇f, ∇²f, x₀; itmax = 200, tol = 1e-8)

    n = length(x₀)
    x = copy(x₀)
    gx = ∇f(x)
    
    iter = 0
    S = typeof(x)
    workspace = CgWorkspace(n, n, S)

    solved = false
    tired = false

    while !(solved || tired)
 
        Hx = ∇²f(x)              # Compute ∇²f(xₖ)
        cg!(workspace, Hx, -gx)  # Solve ∇²f(xₖ)Δx = -∇f(xₖ)
        Δx = solution(workspace) # Recover Δx from the workspace
        x = x + Δx               # Update xₖ₊₁ = xₖ + Δx
        gx = ∇f(x)               # ∇f(xₖ₊₁)
        
        iter += 1
        solved = norm(gx) ≤ tol
        tired = iter ≥ itmax
    end
    return x
end
```

### Example 2: The Gauss-Newton method for nonlinear least squares without linesearch

```@gauss_newton
using Krylov
import Krylov: solution

function gauss_newton(F, JF, x₀; itmax = 200, tol = 1e-8)

    n = length(x₀)
    x = copy(x₀)
    Fx = F(x)
    m = length(Fx)
    
    iter = 0
    S = typeof(x)
    workspace = LsmrWorkspace(m, n, S)

    solved = false
    tired = false

    while !(solved || tired)
 
        Jx = JF(x)                 # Compute J(xₖ)
        lsmr!(workspace, Jx, -Fx)  # Minimize ‖J(xₖ)Δx + F(xₖ)‖
        Δx = solution(workspace)   # Recover Δx from the workspace
        x = x + Δx                 # Update xₖ₊₁ = xₖ + Δx
        Fx_old = Fx                # F(xₖ)
        Fx = F(x)                  # F(xₖ₊₁)
        
        iter += 1
        solved = norm(Fx - Fx_old) / norm(Fx) ≤ tol
        tired = iter ≥ itmax
    end
    return x
end
```
