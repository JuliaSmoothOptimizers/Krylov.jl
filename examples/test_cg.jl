using Krylov
using MatrixMarket
using ProfileView

mtx = "data/1138bus.mtx";
# mtx = "data/bcsstk09.mtx";
# mtx = "data/bcsstk18.mtx";

A = MatrixMarket.mmread(mtx); A = A + tril(A, -1)';
n = size(A, 1);
b = ones(n); b_norm = norm(b);

# Solve Ax=b.
(x, rNorms) = cg(A, b);
# @profile x = cg(A, b);
@time (x, rNorms) = cg(A, b);
r = b - A * x;
@printf("Relative residual: %8.1e\n", norm(r)/b_norm);

# ProfileView.view()
