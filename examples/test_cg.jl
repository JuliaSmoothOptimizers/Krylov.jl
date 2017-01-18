using Krylov
using LinearOperators
using MatrixMarket
# using ProfileView

mtx = "data/1138bus.mtx";
# mtx = "data/bcsstk09.mtx";
# mtx = "data/bcsstk18.mtx";

A = MatrixMarket.mmread(mtx);
n = size(A, 1);
b = ones(n); b_norm = norm(b);

# Define a linear operator with preallocation.
Ap = zeros(n);
op = LinearOperator(n, n, true, true, p -> A_mul_B!(1.0,  A, p, 0.0, Ap))

# Solve Ax=b.
(x, stats) = cg(op, b);
# @profile x = cg(A, b);
@time (x, stats) = cg(op, b);
show(stats);
r = b - A * x;
@printf("Relative residual: %8.1e\n", norm(r)/b_norm);

# ProfileView.view()
