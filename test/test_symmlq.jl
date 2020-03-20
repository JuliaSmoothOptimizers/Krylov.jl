function test_symmlq()
  symmlq_tol = 1.0e-6

  # Symmetric and positive definite system.
  n = 10
  A, b = symmetric_definite()
  (x, stats) = symmlq(A, b)
  r = b - A * x
  resid = norm(r) / norm(b)
  @printf("SYMMLQ: Relative residual: %8.1e\n", resid)
  @test(resid ≤ symmlq_tol)
  @test(stats.solved)

  # Symmetric indefinite variant.
  A, b = symmetric_indefinite()
  (x, stats) = symmlq(A, b)
  r = b - A * x
  resid = norm(r) / norm(b)
  @printf("SYMMLQ: Relative residual: %8.1e\n", resid)
  @test(resid ≤ symmlq_tol)
  @test(stats.solved)

  # Code coverage.
  (x, stats) = symmlq(Matrix(A), b)
  show(stats)

  # Sparse Laplacian.
  A, b = sparse_laplacian()
  (x, stats) = symmlq(A, b, atol=1e-12, rtol=1e-12)
  r = b - A * x
  resid = norm(r) / norm(b)
  @show stats
  @printf("SYMMLQ: Relative residual: %8.1e\n", resid)
  @test(resid ≤ 100 * symmlq_tol)
  @test(stats.solved)

  # System that cause a breakdown with the symmetric Lanczos process.
  A, b = symmetric_breakdown()
  (x, stats) = symmlq(A, b)
  r = b - A * x
  resid = norm(r) / norm(b)
  @printf("SYMMLQ: Relative residual: %8.1e\n", resid)
  @test(resid ≤ symmlq_tol)
  @test(stats.solved)
  show(stats)

  # Test b == 0
  A, b = zero_rhs()
  (x, stats) = symmlq(A, b)
  @test x == b
  @test stats.status == "x = 0 is a zero-residual solution"

  # Test error estimate
  A = Matrix(get_div_grad(8, 8, 8))
  b = ones(size(A, 1))
  λest = (1 - 1e-10) * eigmin(A)
  x_exact = A \ b
  (xlq, stats) = symmlq(A, b, λest=λest, transfer_to_cg = false)
  xcg = cg(A, b)[1]
  err = norm(x_exact - xlq)
  errcg = norm(x_exact - xcg)
  @printf("SYMMLQ    : true error: %8.1e\n", err)
  @printf("SYMMLQ-CG : true error: %8.1e\n", errcg)
  @printf("SYMMLQ    : err up-bnd : %8.1e\n", stats.errors[end])
  @printf("SYMMLQ-CG : err up-bnd : %8.1e\n", stats.errorscg[end])
  @test( err ≤ stats.errors[end] )
  @test( errcg ≤ stats.errorscg[end])
  (x, stats) = symmlq(A, b, λest=λest, window=1)
  @printf("SYMMLQ    : err up-bnd : %8.1e\n", stats.errors[end])
  @printf("SYMMLQ-CG : err up-bnd : %8.1e\n", stats.errorscg[end])
  @test( err ≤ stats.errors[end] )
  @test( errcg ≤ stats.errorscg[end])
  (x, stats) = symmlq(A, b, λest=λest, window=5)
  @printf("SYMMLQ    : err up-bnd : %8.1e\n", stats.errors[end])
  @printf("SYMMLQ-CG : err up-bnd : %8.1e\n", stats.errorscg[end])
  @test( err ≤ stats.errors[end] )
  @test( errcg ≤ stats.errorscg[end])

  # Test with Jacobi (or diagonal) preconditioner
  A, b, M = square_preconditioned()
  (x, stats) = symmlq(A, b, M=M)
  show(stats)
  r = b - A * x
  resid = norm(r) / norm(b)
  @printf("SYMMLQ: Relative residual: %8.1e\n", resid)
  @test(resid ≤ symmlq_tol)
  @test(stats.solved)
end

test_symmlq()
