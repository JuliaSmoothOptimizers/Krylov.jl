function test_bicgstab()
  bicgstab_tol = 1.0e-6

  # Symmetric and positive definite system.
  A, b = symmetric_definite()
  (x, stats) = bicgstab(A, b)
  r = b - A * x
  resid = norm(r) / norm(b)
  @test(resid ≤ bicgstab_tol)
  @test(stats.solved)

  # Symmetric indefinite variant.
  A, b = symmetric_indefinite()
  (x, stats) = bicgstab(A, b)
  r = b - A * x
  resid = norm(r) / norm(b)
  @test(resid ≤ bicgstab_tol)
  @test(stats.solved)

  # Nonsymmetric and positive definite systems.
  A, b = nonsymmetric_definite()
  (x, stats) = bicgstab(A, b)
  r = b - A * x
  resid = norm(r) / norm(b)
  @test(resid ≤ bicgstab_tol)
  @test(stats.solved)

  # Nonsymmetric indefinite variant.
  A, b = nonsymmetric_indefinite()
  (x, stats) = bicgstab(A, b)
  r = b - A * x
  resid = norm(r) / norm(b)
  @test(resid ≤ bicgstab_tol)
  @test(stats.solved)

  # Code coverage.
  (x, stats) = bicgstab(Matrix(A), b)

  # Sparse Laplacian.
  A, b = sparse_laplacian()
  (x, stats) = bicgstab(A, b)
  r = b - A * x
  resid = norm(r) / norm(b)
  @test(resid ≤ bicgstab_tol)
  @test(stats.solved)

  # Symmetric indefinite variant, almost singular.
  A, b = almost_singular()
  (x, stats) = bicgstab(A, b)
  r = b - A * x
  resid = norm(r) / norm(b)
  @test(resid ≤ bicgstab_tol)
  @test(stats.solved)

  # Test b == 0
  A, b = zero_rhs()
  (x, stats) = bicgstab(A, b)
  @test x == zeros(size(A,1))
  @test stats.status == "x = 0 is a zero-residual solution"

  # Left preconditioning
  A, b, M = square_preconditioned()
  (x, stats) = bicgstab(A, b, M=M)
  r = b - A * x
  resid = norm(r) / norm(b)
  @test(resid ≤ bicgstab_tol)
  @test(stats.solved)

  # Right preconditioning
  A, b, N = square_preconditioned()
  (x, stats) = bicgstab(A, b, N=N)
  r = b - A * x
  resid = norm(r) / norm(b)
  @test(resid ≤ bicgstab_tol)
  @test(stats.solved)

  # Split preconditioning
  A, b, M, N = two_preconditioners(500, 32)
  (x, stats) = bicgstab(A, b, M=M, N=N)
  r = b - A * x
  resid = norm(r) / norm(b)
  @test(resid ≤ bicgstab_tol)
  @test(stats.solved)
end

@testset "bicgstab" begin
  test_bicgstab()
end
