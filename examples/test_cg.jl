using MatrixMarket
using Krylov, LinearOperators
using LinearAlgebra, Printf

# mtx = "data/bcsstk09.mtx"
mtx = "data/1138bus.mtx"

A = MatrixMarket.mmread(mtx)
n = size(A, 1)
b = ones(n)
b_norm = norm(b)

# Define a linear operator with preallocation.
Ap = zeros(n)
op = LinearOperator(n, n, true, true, p -> mul!(Ap, A, p))

# Solve Ax = b.
(x, stats) = cg(op, b)
show(stats)
r = b - A * x
@printf("Relative residual: %8.1e\n", norm(r) / b_norm)
