function test_crls()
  ⪅(x,y) = (x ≈ y) || (x < y)
  crls_tol = 1.0e-5;

  for npower = 1 : 4
    (b, A, D, HY, HZ, Acond, rnorm) = test(40, 40, 4, npower, 0);  # No regularization.

    (x, stats) = crls(A, b);
    resid = norm(A' * (A*x - b)) / norm(b)
    @printf("CRLS: Relative residual: %8.1e\n", resid);
    @test(resid <= crls_tol);
    @test(stats.solved);

    λ = 1.0e-3;
    (x, stats) = crls(A, b, λ=λ);
    resid = norm(A' * (A*x - b) + λ * x) / norm(b)
    @printf("CRLS: Relative residual: %8.1e\n", resid);
    @test(resid <= crls_tol);
    @test(stats.solved);
  end

  # Test with preconditioning.
  Random.seed!(0)
  A = rand(10, 6); b = rand(10);
  M = InverseLBFGSOperator(10, 4);
  for _ = 1 : 6
    s = rand(10);
    y = rand(10);
    push!(M, s, y);
  end

  (x, stats) = crls(A, b, M=M);
  resid = norm(A' * M * (A * x - b)) / sqrt(dot(b, M * b));
  @printf("CRLS: Preconditioned residual: %8.1e\n", resid);
  @test resid <= crls_tol;

  # test trust-region constraint
  (x, stats) = crls(A, b)

  radius = 0.75 * norm(x)
  (x, stats) = crls(A, b, radius=radius)
  @test(stats.solved)
  @test(abs(radius - norm(x)) <= crls_tol * radius)

  opA = LinearOperator(A)
  (xop, statsop) = crls(opA, b, radius=radius)
  @test(abs(radius - norm(xop)) <= crls_tol * radius)

  # Code coverage.
  (b, A, D, HY, HZ, Acond, rnorm) = test(40, 40, 4, 3, 0);
  (x, stats) = crls(Matrix(A), b);
  (x, stats) = crls(sparse(Matrix(A)), b);
  show(stats);

  # Test b == 0
  A, b = zero_rhs()
  (x, stats) = crls(A, b)
  @test x == zeros(size(A,1))
  @test stats.status == "x = 0 is a zero-residual solution"

  # Test integer values
  A, b = over_int()
  (x, stats) = crls(A, b)
  @test stats.solved

  # Test A positive semi-definite
  radius = 10.
  m,n = 10,7
  U = qr(rand(m,m)).Q
  V = qr(rand(n,n)).Q
  S = [diagm(0 => [0, 1.0e-6, 1, 4, 20, 15, 1.0e5]) ; zeros(3,7)]
  A = U * S * V
  p = V[:,1]; b = A'\p;
  (x, stats) = crls(A, b, radius=radius)
  @test stats.solved
  @test stats.status == "zero-curvature encountered"
  @test norm(x) ⪅ radius
end

test_crls()
