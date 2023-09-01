```@example minares
using Krylov, MatrixMarket, SuiteSparseMatrixCollection
using LinearAlgebra, Printf

ssmc = ssmc_db(verbose=false)
matrix = ssmc_matrices(ssmc, "GHS_indef", "laser")
path = fetch_ssmc(matrix, format="MM")

n = matrix.nrows[1]
A = MatrixMarket.mmread(joinpath(path[1], "$(matrix.name[1]).mtx"))
b = ones(n)

# Solve Ax = b.
x, stats = minares(A, b)
show(stats)
r = b - A * x
Ar = A * r
@printf("Relative A-residual: %8.1e\n", norm(A * r) / norm(A * b))
```
