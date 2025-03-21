## [In-place methods](@id in-place)

All solvers in Krylov.jl have an in-place variant implemented in a method whose name ends with `!`.
A workspace (`KrylovSolver`), which contains the storage needed by a Krylov method, can be used to solve multiple linear systems with the same dimensions and the same floating-point precision.
The section [storage requirements](@ref storage-requirements) specifies the memory needed for each Krylov method.
Each `KrylovSolver` has three constructors:

```@constructors
XyzSolver(A, b)
XyzSolver(m, n, S)
XyzSolver(kc::KrylovConstructor)
```

`Xyz` represents the name of the Krylov method, written in lowercase except for its first letter (e.g., `Cg`, `Minres`, `Lsmr`, `Bicgstab`, etc.).
If the name of the Krylov method contains an underscore (e.g., `minres_qlp` or `cgls_lanczos_shift`), the workspace constructor transforms it by capitalizing each word and removing underscores, resulting in names like `MinresQlpSolver` or `CglsLanczosShiftSolver`.

!!! note
    The constructors of `CgLanczosShiftSolver` and `CglsLanczosShiftSolver` require an additional argument `nshifts`.

Given an operator `A` and a right-hand side `b`, you can create a `KrylovSolver` based on the size of `A` and the type of `b`, or explicitly provide the dimensions `(m, n)` and the storage type `S`.
We assume that `S(undef, 0)`, `S(undef, n)`, and `S(undef, m)` are well-defined for the storage type `S`.
For more advanced vector types, workspaces can also be created with the help of a `KrylovConstructor`.
```@docs
Krylov.KrylovConstructor
```
See the section [custom workspaces](@ref custom_workspaces) for an example where this constructor is the only applicable option.

For example, use `S = Vector{Float64}` if you want to solve linear systems in double precision on the CPU and `S = CuVector{Float32}` if you want to solve linear systems in single precision on an Nvidia GPU.

The workspace is always the first argument of the in-place methods:

```@solvers
minres_solver = MinresSolver(m, n, Vector{Float64})
minres!(minres_solver, A1, b1)

bicgstab_solver = BicgstabSolver(m, n, Vector{ComplexF64})
bicgstab!(bicgstab_solver, A2, b2)

gmres_solver = GmresSolver(m, n, Vector{BigFloat})
gmres!(gmres_solver, A3, b3)

lsqr_solver = LsqrSolver(m, n, CuVector{Float32})
lsqr!(lsqr_solver, A4, b4)
```

A generic function `solve!` is also available and dispatches to the appropriate Krylov method.

```@docs
Krylov.solve!
```

!!! note
    The function `solve!` is not exported to prevent potential conflicts with other Julia packages.

In-place methods return an updated `solver` workspace.
Solutions and statistics can be recovered via `solver.x`, `solver.y` and `solver.stats`.
Functions `solution`, `statistics` and `results` can be also used.

```@docs
Krylov.nsolution
Krylov.issolved
Krylov.solution
Krylov.statistics
Krylov.results
```

## Examples

We illustrate the use of in-place Krylov solvers with two well-known optimization methods.
The details of the optimization methods are described in the section about [Factorization-free operators](@ref matrix-free).

### Example 1: Newton's method for convex optimization without linesearch

```@newton
using Krylov

function newton(∇f, ∇²f, x₀; itmax = 200, tol = 1e-8)

    n = length(x₀)
    x = copy(x₀)
    gx = ∇f(x)
    
    iter = 0
    S = typeof(x)
    solver = CgSolver(n, n, S)
    Δx = solver.x

    solved = false
    tired = false

    while !(solved || tired)
 
        Hx = ∇²f(x)           # Compute ∇²f(xₖ)
        cg!(solver, Hx, -gx)  # Solve ∇²f(xₖ)Δx = -∇f(xₖ)
        x = x + Δx            # Update xₖ₊₁ = xₖ + Δx
        gx = ∇f(x)            # ∇f(xₖ₊₁)
        
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

function gauss_newton(F, JF, x₀; itmax = 200, tol = 1e-8)

    n = length(x₀)
    x = copy(x₀)
    Fx = F(x)
    m = length(Fx)
    
    iter = 0
    S = typeof(x)
    solver = LsmrSolver(m, n, S)
    Δx = solver.x

    solved = false
    tired = false

    while !(solved || tired)
 
        Jx = JF(x)              # Compute J(xₖ)
        lsmr!(solver, Jx, -Fx)  # Minimize ‖J(xₖ)Δx + F(xₖ)‖
        x = x + Δx              # Update xₖ₊₁ = xₖ + Δx
        Fx_old = Fx             # F(xₖ)
        Fx = F(x)               # F(xₖ₊₁)
        
        iter += 1
        solved = norm(Fx - Fx_old) / norm(Fx) ≤ tol
        tired = iter ≥ itmax
    end
    return x
end
```
