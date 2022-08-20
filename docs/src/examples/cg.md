## CG

```@example cg
using Krylov, MatrixMarket, SuiteSparseMatrixCollection
using LinearAlgebra, Printf

ssmc = ssmc_db(verbose=false)
matrix = ssmc_matrices(ssmc, "HB", "bcsstk09")
path = fetch_ssmc(matrix, format="MM")

n = matrix.nrows[1]
A = MatrixMarket.mmread(joinpath(path[1], "$(matrix.name[1]).mtx"))
b = ones(n)
b_norm = norm(b)

# Solve Ax = b.
(x, stats) = cg(A, b)
show(stats)
r = b - A * x
@printf("Relative residual: %8.1e\n", norm(r) / b_norm)
```
