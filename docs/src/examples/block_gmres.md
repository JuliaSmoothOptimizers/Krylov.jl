```@example block_gmres
using Krylov, MatrixMarket, SuiteSparseMatrixCollection
using LinearAlgebra, Printf

ssmc = ssmc_db(verbose=false)
matrix = ssmc_matrices(ssmc, "HB", "sherman2")
path = fetch_ssmc(matrix, format="MM")

n = matrix.nrows[1]
A = MatrixMarket.mmread(joinpath(path[1], "$(matrix.name[1]).mtx"))
B = Matrix{Float64}(I, n, 5)
B_norm = norm(B)

# Solve Ax = B
X, stats = block_gmres(A, B)
show(stats)
R = B - A * X
@printf("Relative residual: %8.1e\n", norm(R) / B_norm)
```
