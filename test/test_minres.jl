function test_minres()
  minres_tol = 1.0e-5

  # Cubic spline matrix.
  A, b = symmetric_definite()
  (x, stats) = minres(A, b)
  r = b - A * x
  resid = norm(r) / norm(b)
  @printf("MINRES: Relative residual: %8.1e\n", resid)
  @test(resid ≤ minres_tol)
  @test(stats.solved)

  # Symmetric indefinite variant.
  A, b = symmetric_indefinite()
  (x, stats) = minres(A, b)
  r = b - A * x
  resid = norm(r) / norm(b)
  @printf("MINRES: Relative residual: %8.1e\n", resid)
  @test(resid ≤ minres_tol)
  @test(stats.solved)

  # Code coverage.
  (x, stats) = minres(Matrix(A), b)
  show(stats)

  # Sparse Laplacian.
  A, b = sparse_laplacian()
  (x, stats) = minres(A, b)
  r = b - A * x
  resid = norm(r) / norm(b)
  @printf("MINRES: Relative residual: %8.1e\n", resid)
  @test(resid ≤ minres_tol)
  @test(stats.solved)

  # Symmetric indefinite variant, almost singular.
  A, b = almost_singular()
  (x, stats) = minres(A, b)
  r = b - A * x
  resid = norm(r) / norm(b)
  @printf("MINRES: Relative residual: %8.1e\n", resid)
  @test(resid ≤ 100 * minres_tol)
  @test(stats.solved)

  # Test b == 0
  A, b = zero_rhs()
  (x, stats) = minres(A, b)
  @test x == zeros(size(A,1))
  @test stats.status == "x = 0 is a zero-residual solution"

  # Shifted system
  A, b = symmetric_indefinite()
  λ = 2.0
  (x, stats) = minres(A, b, λ=λ)
  r = b - (A + λ*I) * x
  resid = norm(r) / norm(b)
  @printf("MINRES: Relative residual: %8.1e\n", resid)
  @test(resid ≤ minres_tol)
  @test(stats.solved)

  # test with Jacobi (or diagonal) preconditioner and history = true
  A, b, M = square_preconditioned()
  (x, stats) = minres(A, b, M=M, history=true)
  show(stats)
  r = b - A * x
  resid = norm(r) / norm(b)
  @printf("MINRES: Relative residual: %8.1e\n", resid)
  @test(resid ≤ minres_tol)
  @test(stats.solved)
  @test(length(stats.residuals) > 0)

  # in-place minres (minres!) with Jacobi (or diagonal) preconditioner
  A, b, M = square_preconditioned()
  y = zeros(size(A, 2))
  opA = PreallocatedLinearOperator(y, Symmetric(A))
  solver = MinresSolver(opA, b)
  x, stats = minres!(solver, opA, b, M=M)
  show(stats)
  r = b - A * x
  resid = norm(r) / norm(b)
  @printf("MINRES: Relative residual: %8.1e\n", resid)
  @test(resid ≤ minres_tol)
  @test(stats.solved)
end

test_minres()
