function test_diom()
  diom_tol = 1.0e-6

  # Symmetric and positive definite system.
  A, b = symmetric_definite()
  (x, stats) = diom(A, b)
  r = b - A * x
  resid = norm(r) / norm(b)
  @printf("DIOM: Relative residual: %8.1e\n", resid)
  @test(resid ≤ diom_tol)
  @test(stats.solved)

  # Symmetric indefinite variant.
  A, b = symmetric_indefinite()
  (x, stats) = diom(A, b)
  r = b - A * x
  resid = norm(r) / norm(b)
  @printf("DIOM: Relative residual: %8.1e\n", resid)
  @test(resid ≤ diom_tol)
  @test(stats.solved)

  # Nonsymmetric and positive definite systems.
  A, b = nonsymmetric_definite()
  (x, stats) = diom(A, b)
  r = b - A * x
  resid = norm(r) / norm(b)
  @printf("DIOM: Relative residual: %8.1e\n", resid)
  @test(resid ≤ diom_tol)
  @test(stats.solved)

  # Nonsymmetric indefinite variant.
  A, b = nonsymmetric_indefinite()
  (x, stats) = diom(A, b)
  r = b - A * x
  resid = norm(r) / norm(b)
  @printf("DIOM: Relative residual: %8.1e\n", resid)
  @test(resid ≤ diom_tol)
  @test(stats.solved)

  # Code coverage.
  (x, stats) = diom(sparse(A), b)
  show(stats)

  # Sparse Laplacian.
  A, b = sparse_laplacian()
  (x, stats) = diom(A, b)
  r = b - A * x
  resid = norm(r) / norm(b)
  @printf("DIOM: Relative residual: %8.1e\n", resid)
  @test(resid ≤ diom_tol)
  @test(stats.solved)

  # Symmetric indefinite variant, almost singular.
  A, b = almost_singular()
  (x, stats) = diom(A, b)
  r = b - A * x
  resid = norm(r) / norm(b)
  @printf("DIOM: Relative residual: %8.1e\n", resid)
  @test(resid ≤ diom_tol)
  @test(stats.solved)

  # Test b == 0
  A, b = zero_rhs()
  (x, stats) = diom(A, b)
  @test x == zeros(size(A,1))
  @test stats.status == "x = 0 is a zero-residual solution"

  # Test integer values
  A, b = square_int()
  (x, stats) = diom(A, b)
  @test stats.solved

  # Test with Jacobi (or diagonal) preconditioner
  A, b, M = square_preconditioned()
  (x, stats) = diom(A, b, M=M)
  show(stats)
  r = b - A * x
  resid = norm(r) / norm(b)
  @printf("DIOM: Relative residual: %8.1e\n", resid)
  @test(resid ≤ diom_tol)
  @test(stats.solved)
end

test_diom()
