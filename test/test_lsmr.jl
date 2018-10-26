function test_lsmr()
  lsmr_tol = 1.0e-5;

  for npower = 1 : 4
    (b, A, D, HY, HZ, Acond, rnorm) = test(40, 40, 4, npower, 0);  # No regularization.

    (x, stats) = lsmr(A, b);
    resid = norm(A' * (A*x - b)) / norm(b)
    @printf("LSMR: Relative residual: %8.1e\n", resid);
    @test(resid <= lsmr_tol);
    @test(stats.solved);

    λ = 1.0e-3;
    (x, stats) = lsmr(A, b, λ=λ);
    resid = norm(A' * (A*x - b) + λ * λ * x) / norm(b)
    @printf("LSMR: Relative residual: %8.1e\n", resid);
    @test(resid <= lsmr_tol);
    @test(stats.solved);
  end

  A = rand(10, 6); b = rand(10)

  # test trust-region constraint
  (x, stats) = lsmr(A, b)

  radius = 0.75 * norm(x)
  (x, stats) = lsmr(A, b, radius=radius)
  @test(stats.solved)
  @test(abs(radius - norm(x)) <= lsmr_tol * radius)

  opA = LinearOperator(A)
  (xop, statsop) = lsmr(opA, b, radius=radius)
  @test(abs(radius - norm(xop)) <= lsmr_tol * radius)

  # Code coverage.
  (b, A, D, HY, HZ, Acond, rnorm) = test(40, 40, 4, 3, 0);
  (x, stats) = lsmr(Matrix(A), b);
  (x, stats) = lsmr(sparse(Matrix(A)), b);
  show(stats);

  # Test b == 0
  (x, stats) = lsmr(A, zeros(size(A,1)))
  @test x == zeros(size(A,1))
  @test stats.status == "x = 0 is a zero-residual solution"

  # Test integer values
  A = [I; rand(1:10, 2, 3)]
  b = A * ones(Int, 3)
  (x, stats) = lsmr(A, b)
  @test stats.solved
end

test_lsmr()
