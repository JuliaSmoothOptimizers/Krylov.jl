@testset "cgs" begin
  cgs_tol = 1.0e-6

  # Symmetric and positive definite system.
  A, b = symmetric_definite()
  (x, stats) = cgs(A, b)
  r = b - A * x
  resid = norm(r) / norm(b)
  @test(resid ≤ cgs_tol)
  @test(stats.solved)

  # Symmetric indefinite variant.
  A, b = symmetric_indefinite()
  (x, stats) = cgs(A, b)
  r = b - A * x
  resid = norm(r) / norm(b)
  @test(resid ≤ cgs_tol)
  @test(stats.solved)

  # Nonsymmetric and positive definite systems.
  A, b = nonsymmetric_definite()
  (x, stats) = cgs(A, b)
  r = b - A * x
  resid = norm(r) / norm(b)
  @test(resid ≤ cgs_tol)
  @test(stats.solved)

  # Nonsymmetric indefinite variant.
  A, b = nonsymmetric_indefinite()
  (x, stats) = cgs(A, b)
  r = b - A * x
  resid = norm(r) / norm(b)
  @test(resid ≤ cgs_tol)
  @test(stats.solved)

  # Code coverage.
  (x, stats) = cgs(sparse(A), b)

  # Sparse Laplacian.
  A, b = sparse_laplacian()
  (x, stats) = cgs(A, b)
  r = b - A * x
  resid = norm(r) / norm(b)
  @test(resid ≤ cgs_tol)
  @test(stats.solved)

  # Test b == 0
  A, b = zero_rhs()
  (x, stats) = cgs(A, b)
  @test x == zeros(size(A,1))
  @test stats.status == "x = 0 is a zero-residual solution"

  # Left preconditioning
  A, b, M = square_preconditioned()
  (x, stats) = cgs(A, b, M=M)
  r = b - A * x
  resid = norm(r) / norm(b)
  @test(resid ≤ cgs_tol)
  @test(stats.solved)

  # Right preconditioning
  A, b, N = square_preconditioned()
  (x, stats) = cgs(A, b, N=N)
  r = b - A * x
  resid = norm(r) / norm(b)
  @test(resid ≤ cgs_tol)
  @test(stats.solved)

  # Split preconditioning
  A, b, M, N = two_preconditioners()
  (x, stats) = cgs(A, b, M=M, N=N)
  r = b - A * x
  resid = norm(r) / norm(b)
  @test(resid ≤ cgs_tol)
  @test(stats.solved)
end
