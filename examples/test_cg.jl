using Krylov, MatrixMarket
using LinearAlgebra, Printf

# mtx = "data/bcsstk09.mtx"
mtx = "data/1138bus.mtx"

A = MatrixMarket.mmread(mtx)
n = size(A, 1)
b = ones(n)
b_norm = norm(b)

# Solve Ax = b.
(x, stats) = cg(A, b)
show(stats)
r = b - A * x
@printf("Relative residual: %8.1e\n", norm(r) / b_norm)
