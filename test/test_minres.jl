function test_minres()
  minres_tol = 1.0e-6

  # Cubic spline matrix.
  A, b = symmetric_definite()
  (x, stats) = minres(A, b)
  r = b - A * x
  resid = norm(r) / norm(b)
  @printf("MINRES: Relative residual: %8.1e\n", resid)
  @test(resid <= minres_tol)
  @test(stats.solved)

  # radius = 0.75 * norm(x)
  # (x, stats) = minres(A, b, radius=radius, itmax=10)
  # show(stats)
  # @test(stats.solved)
  # @test(abs(radius - norm(x)) <= minres_tol * radius)

  # Symmetric indefinite variant.
  A, b = symmetric_indefinite()
  (x, stats) = minres(A, b)
  r = b - A * x
  resid = norm(r) / norm(b)
  @printf("MINRES: Relative residual: %8.1e\n", resid)
  @test(resid <= minres_tol)
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
  @test(resid <= minres_tol)
  @test(stats.solved)

  # radius = 0.75 * norm(x)
  # (x, stats) = minres(A, b, radius=radius, itmax=10)
  # show(stats)
  # @test(stats.solved)
  # @test(abs(radius - norm(x)) <= minres_tol * radius)

  # Symmetric indefinite variant, almost singular.
  A, b = almost_singular()
  (x, stats) = minres(A, b)
  r = b - A * x
  resid = norm(r) / norm(b)
  @printf("MINRES: Relative residual: %8.1e\n", resid)
  @test(resid <= 100 * minres_tol)
  @test(stats.solved)

  # Test b == 0
  A, b = zero_rhs()
  (x, stats) = minres(A, b)
  @test x == zeros(size(A,1))
  @test stats.status == "x = 0 is a zero-residual solution"

  # Test integer values
  A, b = square_int()
  (x, stats) = minres(A, b)
  @test stats.solved

  # Test with Jacobi (or diagonal) preconditioner
  A, b, M = square_preconditioned()
  (x, stats) = minres(A, b, M=M)
  show(stats)
  r = b - A * x
  resid = norm(r) / norm(b)
  @printf("MINRES: Relative residual: %8.1e\n", resid)
  @test(resid ≤ minres_tol)
  @test(stats.solved)
end

test_minres()
