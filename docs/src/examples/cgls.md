```@example cgls
using MatrixMarket, SuiteSparseMatrixCollection
using Krylov, LinearOperators
using LinearAlgebra, Printf

ssmc = ssmc_db(verbose=false)
matrix = ssmc_matrices(ssmc, "HB", "well1033")
path = fetch_ssmc(matrix, format="MM")

A = MatrixMarket.mmread(joinpath(path[1], "$(matrix.name[1]).mtx"))
b = MatrixMarket.mmread(joinpath(path[1], "$(matrix.name[1])_b.mtx"))[:]
(m, n) = size(A)
@printf("System size: %d rows and %d columns\n", m, n)

# Define a regularization parameter.
λ = 1.0e-3

(x, stats) = cgls(A, b, λ=λ)
show(stats)
resid = norm(A' * (A * x - b) + λ * x) / norm(b)
@printf("CGLS: Relative residual: %8.1e\n", resid)
@printf("CGLS: ‖x‖: %8.1e\n", norm(x))
```
