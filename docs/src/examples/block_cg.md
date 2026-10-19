```@example block_cg
using Krylov, SparseArrays, LinearAlgebra, Printf

# Weighted graph Laplacian of an m × m grid: a pure Neumann problem, positive semidefinite
# with the constants as null space.
m = 40
n = m^2
idx(i, j) = (j - 1) * m + i
I, J, V = Int[], Int[], Float64[]
for j in 1:m, i in 1:m, (di, dj) in ((1, 0), (0, 1))
    (i + di > m || j + dj > m) && continue
    a, b = idx(i, j), idx(i + di, j + dj)
    w = 1 + 9 * (sin(a) + 1) / 2                 # conductivities between 1 and 10
    append!(I, (a, b, a, b)); append!(J, (a, b, b, a)); append!(V, (w, w, -w, -w))
end
A = sparse(I, J, V, n, n)

# Four right-hand sides: sources on the boundary, with zero sum (compatible data)
B = zeros(n, 4)
for (k, (a, b)) in enumerate(((1, n), (m, n - m + 1), (idx(1, m ÷ 2), idx(m, m ÷ 2)), (idx(m ÷ 2, 1), idx(m ÷ 2, m))))
    B[a, k], B[b, k] = 1.0, -1.0
end

# Solve AX = B on the range of A, with the solution orthogonal to the constants
X, stats = block_cg(A, B; nullspace = ones(n))
show(stats)
@printf("Relative residual: %8.1e\n", norm(A * X - B) / norm(B))
@printf("Mean of the solutions: %8.1e\n", maximum(abs, sum(X; dims = 1)) / n)

# The same with a Jacobi preconditioner (M ≈ A⁻¹, applied with mul!)
X, stats = block_cg(A, B; nullspace = ones(n), M = inv(Diagonal(diag(A))))
@printf("Jacobi preconditioner: %d iterations\n", stats.niter)
```
