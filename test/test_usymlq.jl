function test_usymlq()
  usymlq_tol = 1.0e-6

  # Symmetric and positive definite system.
  A, b = symmetric_definite()
  c = copy(b)
  (x, stats) = usymlq(A, b, c)
  r = b - A * x
  resid = norm(r) / norm(b)
  @printf("USYMLQ: Relative residual: %8.1e\n", resid)
  @test(resid ≤ usymlq_tol)
  @test(stats.solved)

  # Symmetric indefinite variant.
  A, b = symmetric_indefinite()
  c = copy(b)
  (x, stats) = usymlq(A, b, c)
  r = b - A * x
  resid = norm(r) / norm(b)
  @printf("USYMLQ: Relative residual: %8.1e\n", resid)
  @test(resid ≤ usymlq_tol)
  @test(stats.solved)

  # Nonsymmetric and positive definite systems.
  A, b = nonsymmetric_definite()
  c = copy(b)
  (x, stats) = usymlq(A, b, c)
  r = b - A * x
  resid = norm(r) / norm(b)
  @printf("USYMLQ: Relative residual: %8.1e\n", resid)
  @test(resid ≤ usymlq_tol)
  @test(stats.solved)

  # Nonsymmetric indefinite variant.
  A, b = nonsymmetric_indefinite()
  c = copy(b)
  (x, stats) = usymlq(A, b, c)
  r = b - A * x
  resid = norm(r) / norm(b)
  @printf("USYMLQ: Relative residual: %8.1e\n", resid)
  @test(resid ≤ usymlq_tol)
  @test(stats.solved)

  # Code coverage.
  (x, stats) = usymlq(Matrix(A), b, c)
  show(stats)

  # Sparse Laplacian.
  A, b = sparse_laplacian()
  c = copy(b)
  (x, stats) = usymlq(A, b, c)
  r = b - A * x
  resid = norm(r) / norm(b)
  @printf("USYMLQ: Relative residual: %8.1e\n", resid)
  @test(resid ≤ usymlq_tol)
  @test(stats.solved)

  # Symmetric indefinite variant, almost singular.
  A, b = almost_singular()
  c = copy(b)
  (x, stats) = usymlq(A, b, c)
  r = b - A * x
  resid = norm(r) / norm(b)
  @printf("USYMLQ: Relative residual: %8.1e\n", resid)
  @test(resid ≤ usymlq_tol)
  @test(stats.solved)

  # Test b == 0
  A, b = zero_rhs()
  c = copy(b)
  (x, stats) = usymlq(A, b, c)
  @test x == zeros(size(A,1))
  @test stats.status == "x = 0 is a zero-residual solution"

  # Test integer values
  A, b = square_int()
  c = copy(b)
  (x, stats) = usymlq(A, b, c)
  @test stats.solved

  # Underdetermined and consistent systems.
  A, b = under_consistent()
  c = ones(25)
  (x, stats) = usymlq(A, b, c)
  r = b - A * x
  resid = norm(r) / norm(b)
  @printf("USYMLQ: Relative residual: %8.1e\n", resid)
  @test(resid ≤ usymlq_tol)

  # Square and consistent systems.
  A, b = square_consistent()
  c = ones(10)
  (x, stats) = usymlq(A, b, c)
  r = b - A * x
  resid = norm(r) / norm(b)
  @printf("USYMLQ: Relative residual: %8.1e\n", resid)
  @test(resid ≤ usymlq_tol)

  # Overdetermined and consistent systems.
  A, b = over_consistent()
  c = ones(10)
  (x, stats) = usymlq(A, b, c)
  r = b - A * x
  resid = norm(r) / norm(b)
  @printf("USYMLQ: Relative residual: %8.1e\n", resid)
  @test(resid ≤ usymlq_tol)
end

test_usymlq()
