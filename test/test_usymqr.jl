@testset "usymqr" begin
  usymqr_tol = 1.0e-6

  # Symmetric and positive definite system.
  A, b = symmetric_definite()
  c = copy(b)
  (x, stats) = usymqr(A, b, c)
  r = b - A * x
  resid = norm(r) / norm(b)
  @test(resid ≤ usymqr_tol)
  @test(stats.solved)

  # Symmetric indefinite variant.
  A, b = symmetric_indefinite()
  c = copy(b)
  (x, stats) = usymqr(A, b, c)
  r = b - A * x
  resid = norm(r) / norm(b)
  @test(resid ≤ usymqr_tol)
  @test(stats.solved)

  # Nonsymmetric and positive definite systems.
  A, b = nonsymmetric_definite()
  c = copy(b)
  (x, stats) = usymqr(A, b, c)
  r = b - A * x
  resid = norm(r) / norm(b)
  @test(resid ≤ usymqr_tol)
  @test(stats.solved)

  # Nonsymmetric indefinite variant.
  A, b = nonsymmetric_indefinite()
  c = copy(b)
  (x, stats) = usymqr(A, b, c)
  r = b - A * x
  resid = norm(r) / norm(b)
  @test(resid ≤ usymqr_tol)
  @test(stats.solved)

  # Code coverage.
  (x, stats) = usymqr(Matrix(A), b, c)

  # Sparse Laplacian.
  A, b = sparse_laplacian()
  c = copy(b)
  (x, stats) = usymqr(A, b, c)
  r = b - A * x
  resid = norm(r) / norm(b)
  @test(resid ≤ usymqr_tol)
  @test(stats.solved)

  # Symmetric indefinite variant, almost singular.
  A, b = almost_singular()
  c = copy(b)
  (x, stats) = usymqr(A, b, c)
  r = b - A * x
  resid = norm(r) / norm(b)
  @test(resid ≤ usymqr_tol)
  @test(stats.solved)

  # Test b == 0
  A, b = zero_rhs()
  c = copy(b)
  (x, stats) = usymqr(A, b, c)
  @test x == zeros(size(A,1))
  @test stats.status == "x = 0 is a zero-residual solution"

  # Underdetermined and consistent systems.
  A, b = under_consistent()
  c = ones(25)
  (x, stats) = usymqr(A, b, c)
  r = b - A * x
  resid = norm(r) / norm(b)
  @test(resid ≤ usymqr_tol)

  # Underdetermined and inconsistent systems.
  A, b = under_inconsistent()
  c = [(-1.0)^i for i=1:25]
  (x, stats) = usymqr(A, b, c)
  @test stats.inconsistent

  # Square and consistent systems.
  A, b = square_consistent()
  c = ones(10)
  (x, stats) = usymqr(A, b, c)
  r = b - A * x
  resid = norm(r) / norm(b)
  @test(resid ≤ usymqr_tol)

  # Square and inconsistent systems.
  A, b = square_inconsistent()
  c = ones(10)
  (x, stats) = usymqr(A, b, c)
  @test stats.inconsistent

  # Overdetermined and consistent systems.
  A, b = over_consistent()
  c = ones(10)
  (x, stats) = usymqr(A, b, c)
  r = b - A * x
  resid = norm(r) / norm(b)
  @test(resid ≤ usymqr_tol)

  # Overdetermined and inconsistent systems.
  A, b = over_inconsistent()
  c = [(-2.0)^i for i=1:10]
  (x, stats) = usymqr(A, b, c)
  @test stats.inconsistent

  # Poisson equation in polar coordinates.
  A, b = polar_poisson()
  (x, stats) = usymqr(A, b, b)
  r = b - A * x
  resid = norm(r) / norm(b)
  @test(resid ≤ usymqr_tol)
  @test(stats.solved)
end
