@testset "minres_qlp" begin
  minres_qlp_tol = 1.0e-6

  # Cubic spline matrix.
  A, b = symmetric_definite()
  (x, stats) = minres_qlp(A, b)
  r = b - A * x
  resid = norm(r) / norm(b)
  @test(resid ≤ minres_qlp_tol)
  @test(stats.solved)

  # Symmetric indefinite variant.
  A, b = symmetric_indefinite()
  (x, stats) = minres_qlp(A, b)
  r = b - A * x
  resid = norm(r) / norm(b)
  @test(resid ≤ minres_qlp_tol)
  @test(stats.solved)

  # Code coverage.
  (x, stats) = minres_qlp(Matrix(A), b)

  # Sparse Laplacian.
  A, b = sparse_laplacian()
  (x, stats) = minres_qlp(A, b)
  r = b - A * x
  resid = norm(r) / norm(b)
  @test(resid ≤ minres_qlp_tol)
  @test(stats.solved)

  # Symmetric indefinite variant, almost singular.
  A, b = almost_singular()
  (x, stats) = minres_qlp(A, b)
  r = b - A * x
  resid = norm(r) / norm(b)
  @test(resid ≤ minres_qlp_tol)
  @test(stats.solved)

  # Test b == 0
  A, b = zero_rhs()
  (x, stats) = minres_qlp(A, b)
  @test x == zeros(size(A,1))
  @test stats.status == "x = 0 is a zero-residual solution"

  # Shifted system
  A, b = symmetric_indefinite()
  λ = 2.0
  (x, stats) = minres_qlp(A, b, λ=λ)
  r = b - (A + λ*I) * x
  resid = norm(r) / norm(b)
  @test(resid ≤ minres_qlp_tol)
  @test(stats.solved)

  # Singular inconsistent system
  A, b = square_inconsistent()
  (x, stats) = minres_qlp(A, b)
  @test stats.inconsistent

  # Test with Jacobi (or diagonal) preconditioner
  A, b, M = square_preconditioned()
  (x, stats) = minres_qlp(A, b, M=M)
  r = b - A * x
  resid = sqrt(dot(r, M * r)) / norm(b)
  @test(resid ≤ minres_qlp_tol)
  @test(stats.solved)
end
