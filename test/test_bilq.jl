function test_bilq()
  bilq_tol = 1.0e-6

  # Symmetric and positive definite systems (cubic spline matrix).
  n = 10
  A = spdiagm(-1 => ones(n-1), 0 => 4*ones(n), 1 => ones(n-1))
  b = A * [1:n;]
  (x_lq, x_cg, stats) = bilq(A, b)
  r = b - A * x_cg
  resid = norm(r) / norm(b)
  @printf("BILQ: Relative residual: %8.1e\n", resid)
  @test(resid ≤ bilq_tol)
  @test(stats.solved)

  # Symmetric indefinite variant.
  A = A - 3 * I
  b = A * [1:n;]
  (x_lq, x_cg, stats) = bilq(A, b)
  r = b - A * x_cg
  resid = norm(r) / norm(b)
  @printf("BILQ: Relative residual: %8.1e\n", resid)
  @test(resid ≤ bilq_tol)
  @test(stats.solved)

  # Nonsymmetric and positive definite systems.
  A = [i == j ? 10.0 : i < j ? 1.0 : -1.0 for i=1:n, j=1:n]
  b = A * [1:n;]
  (x_lq, x_cg, stats) = bilq(A, b)
  r = b - A * x_cg
  resid = norm(r) / norm(b)
  @printf("BILQ: Relative residual: %8.1e\n", resid)
  @test(resid ≤ bilq_tol)
  @test(stats.solved)

  # Nonsymmetric indefinite variant.
  A = [i == j ? 10*(-1.0)^(i*j) : i < j ? 1.0 : -1.0 for i=1:n, j=1:n]
  b = A * [1:n;]
  (x_lq, x_cg, stats) = bilq(A, b)
  r = b - A * x_cg
  resid = norm(r) / norm(b)
  @printf("BILQ: Relative residual: %8.1e\n", resid)
  @test(resid ≤ bilq_tol)
  @test(stats.solved)

  # Code coverage.
  (x_lq, x_cg, stats) = bilq(sparse(A), b)
  show(stats)

  # Sparse Laplacian.
  A = get_div_grad(16, 16, 16)
  b = ones(size(A, 1))
  (x_lq, x_cg, stats) = bilq(A, b)
  r = b - A * x_cg
  resid = norm(r) / norm(b)
  @printf("BILQ: Relative residual: %8.1e\n", resid)
  @test(resid ≤ bilq_tol)
  @test(stats.solved)

  # Test b == 0
  (x_lq, x_cg, stats) = bilq(A, zeros(size(A,1)))
  @test x_cg == zeros(size(A,1))
  @test stats.status == "x = 0 is a zero-residual solution"

  # Test integer values
  A = spdiagm(-1 => ones(Int, n-1), 0 => 4*ones(Int, n), 1 => ones(Int, n-1))
  b = A * [1:n;]
  (x_lq, x_cg, stats) = bilq(A, b)
  @test stats.solved
end

test_bilq()
