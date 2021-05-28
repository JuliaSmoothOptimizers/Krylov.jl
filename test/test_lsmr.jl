@testset "lsmr" begin
  lsmr_tol = 1.0e-5

  for npower = 1 : 4
    (b, A, D, HY, HZ, Acond, rnorm) = test(40, 40, 4, npower, 0)  # No regularization.

    (x, stats) = lsmr(A, b)
    r = b - A * x
    resid = norm(A' * r) / norm(b)
    @test(resid ≤ lsmr_tol)
    @test(stats.solved)

    λ = 1.0e-3
    (x, stats) = lsmr(A, b, λ=λ)
    r = b - A * x
    resid = norm(A' * r - λ * λ * x) / norm(b)
    @test(resid ≤ lsmr_tol)
    @test(stats.solved)
  end

  A = [i/j - j/i for i=1:10, j=1:6]
  b = A * ones(6)

  # test trust-region constraint
  (x, stats) = lsmr(A, b)

  radius = 0.75 * norm(x)
  (x, stats) = lsmr(A, b, radius=radius)
  @test(stats.solved)
  @test(abs(radius - norm(x)) ≤ lsmr_tol * radius)

  # Code coverage.
  (b, A, D, HY, HZ, Acond, rnorm) = test(40, 40, 4, 3, 0)
  (x, stats) = lsmr(Matrix(A), b)
  (x, stats) = lsmr(sparse(Matrix(A)), b)

  # Test b == 0
  A, b = zero_rhs()
  (x, stats) = lsmr(A, b)
  @test x == zeros(size(A,1))
  @test stats.status == "x = 0 is a zero-residual solution"

  # Test with preconditioners
  A, b, M, N = two_preconditioners()
  (x, stats) = lsmr(A, b, M=M, N=N)
  r = b - A * x
  resid = sqrt(dot(r, M * r)) / norm(b)
  @test(resid ≤ lsmr_tol)
  @test(stats.solved)

  # Test regularization
  A, b, λ = regularization()
  (x, stats) = lsmr(A, b, λ=λ)
  r = b - A * x
  resid = norm(A' * r - λ^2 * x) / norm(b)
  @test(resid ≤ lsmr_tol)

  # Test saddle-point systems
  A, b, D = saddle_point()
  D⁻¹ = inv(D)
  (x, stats) = lsmr(A, b, M=D⁻¹)
  r = D⁻¹ * (b - A * x)
  resid = norm(A' * r) / norm(b)
  @test(resid ≤ lsmr_tol)

  # Test symmetric and quasi-definite systems
  A, b, M, N = sqd()
  M⁻¹ = inv(M)
  N⁻¹ = inv(N)
  (x, stats) = lsmr(A, b, M=M⁻¹, N=N⁻¹, sqd=true)
  r = M⁻¹ * (b - A * x)
  resid = norm(A' * r - N * x) / norm(b)
  @test(resid ≤ lsmr_tol)

  # Test dimension of additional vectors
  for transpose ∈ (false, true)
    A, b, c, D = small_sp(transpose)
    D⁻¹ = inv(D)
    (x, stats) = lsmr(A, b, M=D⁻¹)

    A, b, c, M, N = small_sqd(transpose)
    M⁻¹ = inv(M)
    N⁻¹ = inv(N)
    (x, stats) = lsmr(A, b, M=M⁻¹, N=N⁻¹, sqd=true)
  end
end
