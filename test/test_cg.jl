function test_cg()
  cg_tol = 1.0e-6

  # Cubic spline matrix.
  A, b = symmetric_definite()
  (x, stats) = cg(A, b, itmax=10)
  r = b - A * x
  resid = norm(r) / norm(b)
  @printf("CG: Relative residual: %8.1e\n", resid)
  @test(resid ≤ cg_tol)
  @test(stats.solved)

  # Code coverage.
  (x, stats) = cg(Matrix(A), b)
  show(stats)

  radius = 0.75 * norm(x)
  (x, stats) = cg(A, b, radius=radius, itmax=10)
  show(stats)
  @test(stats.solved)
  @test(abs(radius - norm(x)) ≤ cg_tol * radius)

  # Sparse Laplacian.
  A, b = sparse_laplacian()
  (x, stats) = cg(A, b)
  r = b - A * x
  resid = norm(r) / norm(b)
  @printf("CG: Relative residual: %8.1e\n", resid)
  @test(resid ≤ cg_tol)
  @test(stats.solved)

  radius = 0.75 * norm(x)
  (x, stats) = cg(A, b, radius=radius, itmax=10)
  show(stats)
  @test(stats.solved)
  @test(abs(radius - norm(x)) ≤ cg_tol * radius)

  opA = LinearOperator(A)
  (xop, statsop) = cg(opA, b, radius=radius, itmax=10)
  @test(abs(radius - norm(xop)) ≤ cg_tol * radius)

  n = 100
  B = LBFGSOperator(n)
  Random.seed!(0)
  for i = 1:5
    push!(B, rand(n), rand(n))
  end
  b = B * ones(n)
  (x, stats) = cg(B, b, itmax=2n)
  @test norm(x - ones(n)) ≤ cg_tol * norm(x)
  @test stats.solved

  # Test b == 0
  A, b = zero_rhs()
  (x, stats) = cg(A, b)
  @test x == zeros(size(A,1))
  @test stats.status == "x = 0 is a zero-residual solution"

  # Test with Jacobi (or diagonal) preconditioner
  A, b, M = square_preconditioned()
  (x, stats) = cg(A, b, M=M)
  show(stats)
  r = b - A * x
  resid = sqrt(dot(r, M * r)) / norm(b)
  @printf("CG: Relative residual: %8.1e\n", resid)
  @test(resid ≤ cg_tol)
  @test(stats.solved)

  # Test that precision is not lost (#126)
  A = rand(BigFloat, 3, 3)
  A = A * A'
  b = rand(BigFloat, 3)
  x = cg(A, b)[1]
  @test eltype(x) == BigFloat

  # Test linesearch
  A, b = symmetric_indefinite()
  x, stats = cg(A, b, linesearch=true)
  @test stats.status == "nonpositive curvature detected"
  @test !stats.inconsistent

  # Test singular and consistent system
  A, b = singular_consistent()
  x, stats = cg(A, b)
  r = b - A * x
  resid = norm(r) / norm(b)
  @printf("CG: Relative residual: %8.1e\n", resid)
  @test(resid ≤ cg_tol)
  @test !stats.inconsistent

  # Test inconsistent system
  A, b = square_inconsistent()
  x, stats = cg(A, b)
  @test stats.inconsistent
end

test_cg()
