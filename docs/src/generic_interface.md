## [Generic interface](@id generic-interface)

Krylov.jl provides a generic interface for solving linear systems using (block) Krylov methods.
The interface is designed to be common for all methods and contains three routines [`krylov_workspace`](@ref krylov_workspace), [`krylov_solve`](@ref krylov_solve) and [`krylov_solve!`](@ref krylov_solve!).
They allow to build Krylov workspaces and call both in-place and out-of-place variants of the solvers with a unified API.

```@docs
krylov_workspace
krylov_solve!
krylov_solve
```

In-place solvers update the workspace, from which solutions and statistics can be retrieved.
The following functions are available for post-solve analysis:

```@docs
Krylov.results
Krylov.solution
Krylov.statistics
Krylov.nsolution
Krylov.issolved
```

## Examples

```julia
using Krylov, Random

# Define a symmetric positive definite matrix A and a right-hand side vector b
n = 1000
A = sprandn(n, n, 0.005)
A = A * A' + I
b = randn(n)

# Out-of-place interface
for method in (:cg, :cr, :car)
    x, stats = krylov_solve(Val{method}(), A, b)
    r = b - A * x
    println("Residual norm for $(method): ", norm(r))
end
```

```julia
using Krylov, Random

# Define a square nonsymmetric matrix A and a right-hand side vector b
n = 100
A = sprand(n, n, 0.05) + I
b = rand(n)

# In-place interface
for method in (:bicgstab, :gmres)
    # Create a workspace for the Krylov method
    solver = krylov_workspace(Val(method), A, b)

    # Solve the system in-place
    krylov_solve!(solver, A, b)

    # Get the statistics
    stats = statistics(solver)

    # Retrieve the solution
    x = solution(solver)

    # Check if the solver converged
    solved = issolved(solver)
    println("Converged $method: ", solved)

    # Display the number of iterations
    niter = niterations(solver)
    println("Number of iterations for $method: ", niter)
end
```
