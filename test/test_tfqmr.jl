function test_tfqmr()
  tfqmr_tol = 1.0e-6

  # Symmetric and positive definite variant.
  A, b = symmetric_definite()
  (x, stats) = tfqmr(A, b)
  r = b - A * x
  resid = norm(r) / norm(b)
  @printf("TFQMR: Relative residual: %8.1e\n", resid)
  @test(resid ≤ tfqmr_tol)
  @test(stats.solved)

  # Symmetric indefinite variant.
  A, b = symmetric_indefinite()
  (x, stats) = tfqmr(A, b)
  r = b - A * x
  resid = norm(r) / norm(b)
  @printf("TFQMR: Relative residual: %8.1e\n", resid)
  @test(resid ≤ tfqmr_tol)
  @test(stats.solved)

  # Nonsymmetric and positive definite variant.
  A, b = nonsymmetric_definite()
  (x, stats) = tfqmr(A, b)
  r = b - A * x
  resid = norm(r) / norm(b)
  @printf("TFQMR: Relative residual: %8.1e\n", resid)
  @test(resid ≤ tfqmr_tol)
  @test(stats.solved)

  # Nonsymmetric indefinite variant.
  A, b = nonsymmetric_indefinite()
  (x, stats) = tfqmr(A, b)
  r = b - A * x
  resid = norm(r) / norm(b)
  @printf("TFQMR: Relative residual: %8.1e\n", resid)
  @test(resid ≤ tfqmr_tol)
  @test(stats.solved)

  # Code coverage.
  (x, stats) = tfqmr(Matrix(A), b)
  show(stats)

  # Sparse Laplacian.
  A, b = sparse_laplacian()
  (x, stats) = tfqmr(A, b)
  r = b - A * x
  resid = norm(r) / norm(b)
  @printf("TFQMR: Relative residual: %8.1e\n", resid)
  @test(resid ≤ tfqmr_tol)
  @test(stats.solved)

  # Test b == 0
  A, b = zero_rhs()
  (x, stats) = diom(A, b)
  @test x == zeros(size(A,1))
  @test stats.status == "x = 0 is a zero-residual solution"

  # Test integer values
  A, b = square_int()
  (x, stats) = tfqmr(A, b)
  @test stats.solved

  # Test with Jacobi (or diagonal) preconditioner
  A, b, M = square_preconditioned()
  (x, stats) = tfqmr(A, b, M=M)
  show(stats)
  r = b - A * x
  resid = norm(r) / norm(b)
  @printf("TFQMR: Relative residual: %8.1e\n", resid)
  @test(resid ≤ tfqmr_tol)
  @test(stats.solved)
end

test_tfqmr()
