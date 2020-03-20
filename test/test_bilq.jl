function test_bilq()
  bilq_tol = 1.0e-6

  # Symmetric and positive definite system.
  A, b = symmetric_definite()
  (x, stats) = bilq(A, b)
  r = b - A * x
  resid = norm(r) / norm(b)
  @printf("BILQ: Relative residual: %8.1e\n", resid)
  @test(resid ≤ bilq_tol)
  @test(stats.solved)

  # Symmetric indefinite variant.
  A, b = symmetric_indefinite()
  (x, stats) = bilq(A, b)
  r = b - A * x
  resid = norm(r) / norm(b)
  @printf("BILQ: Relative residual: %8.1e\n", resid)
  @test(resid ≤ bilq_tol)
  @test(stats.solved)

  # Nonsymmetric and positive definite systems.
  A, b = nonsymmetric_definite()
  (x, stats) = bilq(A, b)
  r = b - A * x
  resid = norm(r) / norm(b)
  @printf("BILQ: Relative residual: %8.1e\n", resid)
  @test(resid ≤ bilq_tol)
  @test(stats.solved)

  # Nonsymmetric indefinite variant.
  A, b = nonsymmetric_indefinite()
  (x, stats) = bilq(A, b)
  r = b - A * x
  resid = norm(r) / norm(b)
  @printf("BILQ: Relative residual: %8.1e\n", resid)
  @test(resid ≤ bilq_tol)
  @test(stats.solved)

  # Code coverage.
  (x, stats) = bilq(Matrix(A), b)
  show(stats)

  # Sparse Laplacian.
  A, b = sparse_laplacian()
  (x, stats) = bilq(A, b)
  r = b - A * x
  resid = norm(r) / norm(b)
  @printf("BILQ: Relative residual: %8.1e\n", resid)
  @test(resid ≤ bilq_tol)
  @test(stats.solved)

  # Test b == 0
  A, b = zero_rhs()
  (x, stats) = bilq(A, b)
  @test x == zeros(size(A,1))
  @test stats.status == "x = 0 is a zero-residual solution"

  # System that cause a breakdown with the Lanczos biorthogonalization process.
  A, b, c = unsymmetric_breakdown()
  (x, stats) = bilq(A, b, c=c)
  r = b - A * x
  resid = norm(r) / norm(b)
  @printf("BILQ: Relative residual: %8.1e\n", resid)
  @test(resid ≤ bilq_tol)
  @test(stats.solved)

  # Poisson equation in polar coordinates.
  A, b = polar_poisson()
  (x, stats) = bilq(A, b)
  r = b - A * x
  resid = norm(r) / norm(b)
  @printf("BiLQ: Relative residual: %8.1e\n", resid)
  @test(resid ≤ bilq_tol)
  @test(stats.solved)
end

test_bilq()
