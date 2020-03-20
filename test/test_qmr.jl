function test_qmr()
  qmr_tol = 1.0e-6

  # Symmetric and positive definite system.
  A, b = symmetric_definite()
  (x, stats) = qmr(A, b)
  r = b - A * x
  resid = norm(r) / norm(b)
  @printf("QMR: Relative residual: %8.1e\n", resid)
  @test(resid ≤ qmr_tol)
  @test(stats.solved)

  # Symmetric indefinite variant.
  A, b = symmetric_indefinite()
  (x, stats) = qmr(A, b)
  r = b - A * x
  resid = norm(r) / norm(b)
  @printf("QMR: Relative residual: %8.1e\n", resid)
  @test(resid ≤ qmr_tol)
  @test(stats.solved)

  # Nonsymmetric and positive definite systems.
  A, b = nonsymmetric_definite()
  (x, stats) = qmr(A, b)
  r = b - A * x
  resid = norm(r) / norm(b)
  @printf("QMR: Relative residual: %8.1e\n", resid)
  @test(resid ≤ qmr_tol)
  @test(stats.solved)

  # Nonsymmetric indefinite variant.
  A, b = nonsymmetric_indefinite()
  (x, stats) = qmr(A, b)
  r = b - A * x
  resid = norm(r) / norm(b)
  @printf("QMR: Relative residual: %8.1e\n", resid)
  @test(resid ≤ qmr_tol)
  @test(stats.solved)

  # Code coverage.
  (x, stats) = qmr(Matrix(A), b)
  show(stats)

  # Sparse Laplacian.
  A, b = sparse_laplacian()
  (x, stats) = qmr(A, b)
  r = b - A * x
  resid = norm(r) / norm(b)
  @printf("QMR: Relative residual: %8.1e\n", resid)
  @test(resid ≤ qmr_tol)
  @test(stats.solved)

  # Test b == 0
  A, b = zero_rhs()
  (x, stats) = qmr(A, b)
  @test x == zeros(size(A,1))
  @test stats.status == "x = 0 is a zero-residual solution"

  # Poisson equation in polar coordinates.
  A, b = polar_poisson()
  (x, stats) = qmr(A, b)
  r = b - A * x
  resid = norm(r) / norm(b)
  @printf("QMR: Relative residual: %8.1e\n", resid)
  @test(resid ≤ qmr_tol)
  @test(stats.solved)
end

test_qmr()
