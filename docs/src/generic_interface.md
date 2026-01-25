## [Generic interface](@id generic-interface)

Krylov.jl provides a generic interface for solving linear systems using (block) Krylov methods.
The interface is designed to be common for all methods and contains three routines [`krylov_workspace`](@ref krylov_workspace), [`krylov_solve`](@ref krylov_solve) and [`krylov_solve!`](@ref krylov_solve!).
They allow to build Krylov workspaces and call both in-place and out-of-place variants of the solvers with a unified API.

```@docs
krylov_workspace
krylov_solve
krylov_solve!
```

The section on [workspace accessors](@ref workspace-accessors) describes how to retrieve the solution, statistics, and other results from a workspace after calling `krylov_solve!`.

## Examples

```@example op_interface
using Krylov, SparseArrays, LinearAlgebra

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

```@example ip_interface
using Krylov, SparseArrays, LinearAlgebra

# Define a square nonsymmetric matrix A and a right-hand side vector b
n = 100
A = sprand(n, n, 0.05) + I
b = rand(n)

# In-place interface
for method in (:bicgstab, :gmres, :qmr)
    # Create a workspace for the Krylov method
    workspace = krylov_workspace(Val(method), A, b)

    # Solve the system in-place
    krylov_solve!(workspace, A, b)

    # Get the statistics
    stats = Krylov.statistics(workspace)

    # Retrieve the solution
    x = Krylov.solution(workspace)

    # Check if the solver converged
    solved = Krylov.issolved(workspace)
    println("Convergence of $method: ", solved)

    # Display the number of iterations
    niter = Krylov.iteration_count(workspace)
    println("Number of iterations for $method: ", niter)

    # Display the allocation timer
    allocation_timer = Krylov.elapsed_allocation_time(workspace)
    println("Elapsed allocation time for $method: ", allocation_timer, " seconds")

    # Display the computation timer
    computation_timer = Krylov.elapsed_time(workspace)
    println("Elapsed time for $method: ", computation_timer, " seconds")

    # Display the number of operator-vector products with A and A'
    nAprod = Krylov.Aprod_count(workspace)
    nAtprod = Krylov.Atprod_count(workspace)
    println("Number of operator-vector products with A and A' for $method: ", (nAprod, nAtprod))

    println()
end
```
