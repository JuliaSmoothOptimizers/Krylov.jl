using Krylov
using MatrixMarket
# using ProfileView

function residuals(A, b, shifts, x)
  nshifts = size(shifts, 1);
  r = { (b - A * x[:,i] - shifts[i] * x[:,i]) for i = 1 : nshifts };
  return r;
end

# mtx = "data/1138bus.mtx";
mtx = "data/bcsstk09.mtx";

A = MatrixMarket.mmread(mtx); A = A + tril(A, -1)';
n = size(A, 1);
b = ones(n); b_norm = norm(b);

# Solve Ax=b.
# x = cg_lanczos(A, b);
# @time x = cg_lanczos(A, b);
# r = b - A * x;
# @printf("Relative residual without shift: %8.1e\n", norm(r)/norm(b));

# Solve (A+αI)x = b.
shifts = [1, 2, 3, 4];
x = cg_lanczos_shift_seq(A, b, shifts, verbose=true);
# Profile.init(delay=0.1)
# @profile x = cg_lanczos_shift_seq(A, b, shifts);
@time x = cg_lanczos_shift_seq(A, b, shifts, verbose=true);
r = residuals(A, b, shifts, x);
resids = map(norm, r) / b_norm;
@printf("Relative residuals with shifts:\n");
for resid in resids
  @printf(" %8.1e", resid);
end
@printf("\n");

# ProfileView.view()
