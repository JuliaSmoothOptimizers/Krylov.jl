function test_cr()
  cr_tol = 1.0e-6

  # Symmetric and positive definite system.
  A, b = symmetric_definite()
  (x, stats) = cr(A, b)
  r = b - A * x
  resid = norm(r) / norm(b)
  @printf("CR: Relative residual: %8.1e\n", resid)
  @test(resid <= cr_tol)
  @test(stats.solved)

  # Code coverage
  (x, stats) = cr(Matrix(A), b)
  show(stats)

  radius = 0.75 * norm(x)
  (x, stats) = cr(A, b, radius=radius)
  show(stats)
  @test(stats.solved)
  @test abs(norm(x) - radius) ≤ cr_tol * radius

  # Sparse Laplacian
  A, _ = sparse_laplacian()
  b = randn(size(A, 1))
  itmax = 0
  # case: ‖x*‖ > Δ
  radius = 10.
  (x, stats) = cr(A, b, radius=radius)
  xNorm = norm(x)
  r = b - A * x
  resid = norm(r) / norm(b)
  @printf("CR: Relative residual: %8.1e\n", resid)
  @test abs(xNorm - radius) ≤ cr_tol * radius
  @test(stats.solved)
  # case: ‖x*‖ < Δ
  radius = 30.
  (x, stats) = cr(A, b, radius=radius)
  xNorm = norm(x)
  r = b - A * x
  resid = norm(r) / norm(b)
  @printf("CR: Relative residual: %8.1e\n", resid)
  @test(resid <= cr_tol)
  @test(stats.solved)

  radius = 0.75 * xNorm
  (x, stats) = cr(A, b, radius=radius)
  show(stats)
  @test(stats.solved)
  @test(abs(radius - norm(x)) <= cr_tol * radius)

  opA = LinearOperator(A)
  (xop, statsop) = cr(opA, b, radius=radius)
  @test(abs(radius - norm(xop)) <= cr_tol * radius)

  n = 100
  itmax = 2 * n
  B = LBFGSOperator(n)
  Random.seed!(0)
  for i = 1:5
    push!(B, rand(n), rand(n))
  end
  b = B * ones(n)
  (x, stats) = cr(B, b)
  @test norm(x - ones(n)) ≤ cr_tol * norm(x)
  @test stats.solved

  # Test b == 0
  A, b = zero_rhs()
  (x, stats) = cr(A, b)
  @test x == zeros(size(A,1))
  @test stats.status == "x = 0 is a zero-residual solution"

  # Test integer values
  A, b = square_int()
  (x, stats) = cr(A, b)
  @test stats.solved

  # Test with Jacobi (or diagonal) preconditioner
  A, b, M = square_preconditioned()
  (x, stats) = cr(A, b, M=M, itmax=10);
  show(stats)
  r = b - A * x;
  resid = norm(r) / norm(b);
  @printf("CR: Relative residual: %8.1e\n", resid);
  @test(resid <= cr_tol);
  # @test(stats.solved);
end

test_cr()
