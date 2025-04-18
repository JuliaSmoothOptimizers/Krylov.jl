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
krylov_results
krylov_solution
krylov_nsolution
krylov_statistics
krylov_elapsed_time
krylov_niterations
krylov_issolved
krylov_Aprod
krylov_Atprod
krylov_Bprod
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
    x, stats = krylov_solve(Val(method), A, b)
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
    workspace = krylov_workspace(Val(method), A, b)

    # Solve the system in-place
    krylov_solve!(workspace, A, b)

    # Get the statistics
    stats = krylov_statistics(workspace)

    # Retrieve the solution
    x = krylov_solution(workspace)

    # Check if the solver converged
    solved = krylov_issolved(workspace)
    println("Convergence of $method: ", solved)

    # Display the number of iterations
    niter = krylov_niterations(workspace)
    println("Number of iterations for $method: ", niter)

    # Display the elapsed timer
    timer = krylov_elapsed_time(workspace)
    println("Elapsed time for $method: ", timer, " seconds")

    println()
end
```
