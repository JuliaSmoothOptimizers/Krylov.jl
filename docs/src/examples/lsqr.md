## LSQR

```@example lsqr
using MatrixMarket, SuiteSparseMatrixCollection
using Krylov, LinearOperators
using LinearAlgebra, Printf

ssmc = ssmc_db(verbose=false)
matrix = ssmc_matrices(ssmc, "HB", "illc1033")
path = fetch_ssmc(matrix, format="MM")

A = MatrixMarket.mmread(joinpath(path[1], "$(matrix.name[1]).mtx"))
b = MatrixMarket.mmread(joinpath(path[1], "$(matrix.name[1])_b.mtx"))[:]
(m, n) = size(A)
@printf("System size: %d rows and %d columns\n", m, n)

# Define a regularization parameter.
λ = 1.0e-3

(x, stats) = lsqr(A, b, λ=λ, atol=0.0, btol=0.0)
show(stats)
resid = norm(A' * (A * x - b) + λ * x) / norm(b)
@printf("LSQR: Relative residual: %8.1e\n", resid)
@printf("LSQR: ‖x‖: %8.1e\n", norm(x))
```
