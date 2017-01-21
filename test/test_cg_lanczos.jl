function residuals(A, b, shifts, x)
  nshifts = size(shifts, 1);
  r = [ (b - A * x[:,i] - shifts[i] * x[:,i]) for i = 1 : nshifts ];
  return r;
end

cg_tol = 1.0e-6;

# Cubic splines matrix.
n = 10;
A = spdiagm((ones(n-1), 4*ones(n), ones(n-1)), (-1, 0, 1))
b = A * [1:n;];
b_norm = norm(b);

(x, stats) = cg_lanczos(A, b, itmax=n);
resid = norm(b - A * x) / b_norm;
@printf("CG_Lanczos: Relative residual: %8.1e\n", resid);
@test(resid <= cg_tol);
@test(stats.solved);

# Test negative curvature detection.
A[n-1,n-1] = -4.0
(x, stats) = cg_lanczos(A, b, check_curvature=true)
@test(stats.status == "negative curvature")
A[n-1,n-1] = 4.0

shifts=[1:6;];

(x, stats) = cg_lanczos_shift_seq(A, b, shifts, itmax=n);
r = residuals(A, b, shifts, x);
resids = map(norm, r) / b_norm;
@printf("CG_Lanczos: Relative residuals with shifts:");
for resid in resids
  @printf(" %8.1e", resid);
end
@printf("\n");
@test(all(resids .<= cg_tol));
@test(stats.solved);

# Test negative curvature detection.
shifts = [-4, -3, 2;]
(x, stats) = cg_lanczos_shift_seq(A, b, shifts, check_curvature=true, itmax=n);
@test(stats.flagged == [true, true, false])

# Code coverage.
(x, stats) = cg_lanczos(full(A), b);
(x, stats) = cg_lanczos_shift_seq(full(A), b, [1:6;], verbose=true);
show(stats);

# Test b == 0
(x, stats) = cg_lanczos(A, zeros(size(A,1)))
@test x == zeros(size(A,1))
@test stats.status == "x = 0 is a zero-residual solution"
(x, stats) = cg_lanczos_shift_seq(A, zeros(size(A,1)), [1:6;])
@test x == zeros(size(A,1), 6)
@test stats.status == "x = 0 is a zero-residual solution"

# Test integer values
A = ceil(Int, A) + eye(size(A,1))
b = A * ones(Int, size(A,1))
(x, stats) = cg_lanczos(A, b)
@test stats.solved
