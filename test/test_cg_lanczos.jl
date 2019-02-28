function residuals(A, b, shifts, x)
  nshifts = length(shifts)
  r = [ (b - A * x[i] - shifts[i] * x[i]) for i = 1 : nshifts ];
  return r;
end

function test_cg_lanczos()
  cg_tol = 1.0e-6;
  n = 10

  # Cubic splines matrix.
  A, b = symmetric_definite()
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
  shifts = [-4; -3; 2]
  (x, stats) = cg_lanczos_shift_seq(A, b, shifts, check_curvature=true, itmax=n);
  @test(stats.flagged == [true, true, false])

  # Code coverage.
  (x, stats) = cg_lanczos(Matrix(A), b);
  (x, stats) = cg_lanczos_shift_seq(Matrix(A), b, [1:6;], verbose=true);
  show(stats);

  # Test b == 0
  A, b = zero_rhs()
  (x, stats) = cg_lanczos(A, b)
  @test x == zeros(size(A,1))
  @test stats.status == "x = 0 is a zero-residual solution"
  (x, stats) = cg_lanczos_shift_seq(A, zeros(size(A,1)), [1:6;])
  @test x == [zeros(size(A,1)) for i=1:6]
  @test stats.status == "x = 0 is a zero-residual solution"

  # Test integer values
  A, b = square_int()
  (x, stats) = cg_lanczos(A, b)
  @test stats.solved

  # Test with Jacobi (or diagonal) preconditioner
  A, b, M = square_preconditioned()
  (x, stats) = cg_lanczos(A, b, M=M)
  show(stats)
  r = b - A * x
  resid = norm(r) / norm(b)
  @printf("CG_Lanczos: Relative residual: %8.1e\n", resid)
  @test(resid ≤ cg_tol)
  @test(stats.solved)

  shifts = [1:10;]
  (x, stats) = cg_lanczos_shift_seq(A, b, shifts, M=M)
  r = residuals(A, b, shifts, x)
  resids = map(norm, r) / norm(b)
  @printf("CG_Lanczos: Relative residuals with shifts:")
  for resid in resids
    @printf(" %8.1e", resid)
  end
  @printf("\n")
  @test(all(resids .≤ cg_tol))
  @test(stats.solved)
end

test_cg_lanczos()
