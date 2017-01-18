using Krylov
using LinearOperators
using MatrixMarket
# using ProfileView

function residuals(A, b, shifts, x)
  nshifts = size(shifts, 1);
  r = [ (b - A * x[:,i] - shifts[i] * x[:,i]) for i = 1 : nshifts ];
  return r;
end

# mtx = "data/1138bus.mtx";
mtx = "data/bcsstk09.mtx";

A = MatrixMarket.mmread(mtx);
n = size(A, 1);
b = ones(n); b_norm = norm(b);

# Define a linear operator with preallocation.
Ap = zeros(n);
op = LinearOperator(n, n, true, true, p -> A_mul_B!(1.0,  A, p, 0.0, Ap))

# Solve Ax=b.
(x, stats) = cg_lanczos(op, b);
@time (x, stats) = cg_lanczos(op, b);
show(stats);
r = b - A * x;
@printf("Relative residual without shift: %8.1e\n", norm(r)/norm(b));

# Solve (A+αI)x = b sequentially.
shifts = [1, 2, 3, 4];
(x, stats) = cg_lanczos_shift_seq(op, b, shifts, verbose=false);
# @profile (x, stats) = cg_lanczos_shift_seq(op, b, shifts);
@time (x, stats) = cg_lanczos_shift_seq(op, b, shifts, verbose=false);
show(stats);
r = residuals(A, b, shifts, x);
resids = map(norm, r) / b_norm;
@printf("Relative residuals with shifts:\n");
for resid in resids
  @printf(" %8.1e", resid);
end
@printf("\n");

# ProfileView.view()
