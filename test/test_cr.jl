@testset "cr" begin
  cr_tol = 1.0e-6

  # Symmetric and positive definite system.
  A, b = symmetric_definite()
  (x, stats) = cr(A, b)
  r = b - A * x
  resid = norm(r) / norm(b)
  @test(resid ≤ cr_tol)
  @test(stats.solved)

  # Code coverage
  (x, stats) = cr(Matrix(A), b)

  radius = 0.75 * norm(x)
  (x, stats) = cr(A, b, radius=radius)
  @test(stats.solved)
  @test abs(norm(x) - radius) ≤ cr_tol * radius

  # Sparse Laplacian
  A, _ = sparse_laplacian()
  Random.seed!(0)
  b = randn(size(A, 1))
  itmax = 0
  # case: ‖x*‖ > Δ
  radius = 10.
  (x, stats) = cr(A, b, radius=radius)
  xNorm = norm(x)
  r = b - A * x
  resid = norm(r) / norm(b)
  @test abs(xNorm - radius) ≤ cr_tol * radius
  @test(stats.solved)
  # case: ‖x*‖ < Δ
  radius = 30.
  (x, stats) = cr(A, b, radius=radius)
  xNorm = norm(x)
  r = b - A * x
  resid = norm(r) / norm(b)
  @test(resid ≤ cr_tol)
  @test(stats.solved)

  radius = 0.75 * xNorm
  (x, stats) = cr(A, b, radius=radius)
  @test(stats.solved)
  @test(abs(radius - norm(x)) ≤ cr_tol * radius)

  # Test b == 0
  A, b = zero_rhs()
  (x, stats) = cr(A, b)
  @test x == zeros(size(A,1))
  @test stats.status == "x = 0 is a zero-residual solution"

  # Test with Jacobi (or diagonal) preconditioner
  A, b, M = square_preconditioned()
  (x, stats) = cr(A, b, M=M)
  r = b - A * x
  resid = sqrt(dot(r, M * r)) / norm(b)
  @test(resid ≤ cr_tol)
  @test(stats.solved)
end
