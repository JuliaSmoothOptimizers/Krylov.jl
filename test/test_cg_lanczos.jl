function residuals(A, b, shifts, x)
  nshifts = size(shifts, 1);
  r = { (b - A * x[:,i] - shifts[i] * x[:,i]) for i = 1 : nshifts };
  return r;
end

cg_tol = 1.0e-6;

# Cubic splines matrix.
n = 10;
A = spdiagm((ones(n-1), 4*ones(n), ones(n-1)), (-1, 0, 1))
b = A * [1:n];
b_norm = norm(b);

for mat in {A, full(A), LinearOperator(A)}
  (x, stats) = cg_lanczos(mat, b, itmax=n);
  show(stats);
  resid = norm(b - A * x) / b_norm;
  @printf("CG_Lanczos: Relative residual: %8.1e\n", resid);
  @test(resid <= cg_tol);
  @test(stats.solved);
end

shifts=[1:6];

for mat in {A, full(A), LinearOperator(A)}
  (x, stats) = cg_lanczos_shift_seq(mat, b, shifts, itmax=n);
  show(stats);
  r = residuals(A, b, shifts, x);
  resids = map(norm, r) / b_norm;
  @printf("CG_Lanczos: Relative residuals with shifts:");
  for resid in resids
    @printf(" %8.1e", resid);
  end
  @printf("\n");
  @test(all(resids .<= cg_tol));
  @test(stats.solved);

  (x, stats) = cg_lanczos_shift_par(mat, b, shifts, itmax=n);
  show(stats);
  r = residuals(A, b, shifts, convert(Array, x));
  resids = map(norm, r) / b_norm;
  @printf("CG_Lanczos: Relative residuals with shifts:");
  for resid in resids
    @printf(" %8.1e", resid);
  end
  @printf("\n");
  @test(all(resids .<= cg_tol));
  @test(stats.solved);
end
