## In-place methods

All solvers in Krylov.jl have an in-place variant implemented in a method whose name ends with `!`.
A workspace (`KrylovSolver`) that contains the storage needed by a Krylov method can be used to solve multiple linear systems that have the same dimensions in the same floating-point precision.
Each `KrylovSolver` has two constructors:

```@constructors
XyzSolver(A, b)
XyzSolver(m, n, S)
```

`Xyz` is the name of the Krylov method with lowercase letters except its first one (`Cg`, `Minres`, `Lsmr`, `Bicgstab`, ...).
Given an operator `A` and a right-hand side `b`, you can create a `KrylovSolver` based on the size of `A` and the type of `b` or explicitly give the dimensions `(m, n)` and the storage type `S`.

For example, use `S = Vector{Float64}` if you want to solve linear systems in double precision on the CPU and `S = CuVector{Float32}` if you want to solve linear systems in single precision on an Nvidia GPU.

!!! note
    `DiomSolver`, `DqgmresSolver` and `CgLanczosShiftSolver` require an additional argument (`memory` or `nshifts`).

The workspace is always the first argument of the in-place methods:

```@solvers
minres_solver = MinresSolver(n, n, Vector{Float64})
x, stats = minres!(minres_solver, A1, b1)

dqgmres_solver = DqgmresSolver(n, n, memory, Vector{BigFloat})
x, stats = dqgmres!(dqgmres_solver, A2, b2)

lsqr_solver = LsqrSolver(m, n, CuVector{Float32})
x, stats = lsqr!(lsqr_solver, A3, b3)
```
  
## Examples

We illustrate the use of in-place Krylov solvers with two well-known optimization methods.
The details of the optimization methods are described in the section about [Factorization-free operators](@ref factorization-free).

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

    solved = false
    tired = false

    while !(solved || tired)
 
        Hx = ∇²f(x)                       # Compute ∇²f(xₖ)
        Δx, stats = cg!(solver, Hx, -gx)  # Solve ∇²f(xₖ)Δx = -∇f(xₖ)
        x = x + Δx                        # Update xₖ₊₁ = xₖ + Δx
        gx = ∇f(x)                        # ∇f(xₖ₊₁)
        
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

    solved = false
    tired = false

    while !(solved || tired)
 
        Jx = JF(x)                          # Compute J(xₖ)
        Δx, stats = lsmr!(solver, Jx, -Fx)  # Minimize ‖J(xₖ)Δx + F(xₖ)‖
        x = x + Δx                          # Update xₖ₊₁ = xₖ + Δx
        Fx_old = Fx                         # F(xₖ)
        Fx = F(x)                           # F(xₖ₊₁)
        
        iter += 1
        solved = norm(Fx - Fx_old) / norm(Fx) ≤ tol
        tired = iter ≥ itmax
    end
    return x
end
```
