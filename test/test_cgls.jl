@testset "cgls" begin
  cgls_tol = 1.0e-5

  for npower = 1 : 4
    (b, A, D, HY, HZ, Acond, rnorm) = test(40, 40, 4, npower, 0)  # No regularization.

    (x, stats) = cgls(A, b)
    resid = norm(A' * (A*x - b)) / norm(b)
    @test(resid ≤ cgls_tol)
    @test(stats.solved)

    λ = 1.0e-3
    (x, stats) = cgls(A, b, λ=λ)
    resid = norm(A' * (A*x - b) + λ * x) / norm(b)
    @test(resid ≤ cgls_tol)
    @test(stats.solved)
  end

  # Test with preconditioning.
  Random.seed!(0)
  A, b = over_inconsistent(10, 6)
  M = InverseLBFGSOperator(10, mem=4)
  for _ = 1 : 6
    s = rand(10)
    y = rand(10)
    push!(M, s, y)
  end

  (x, stats) = cgls(A, b, M=M)
  resid = norm(A' * M * (A * x - b)) / sqrt(dot(b, M * b))
  @test resid ≤ cgls_tol

  # test trust-region constraint
  (x, stats) = cgls(A, b)

  radius = 0.75 * norm(x)
  (x, stats) = cgls(A, b, radius=radius)
  @test(stats.solved)
  @test(abs(radius - norm(x)) ≤ cgls_tol * radius)

  opA = LinearOperator(A)
  (xop, statsop) = cgls(opA, b, radius=radius)
  @test(abs(radius - norm(xop)) ≤ cgls_tol * radius)

  # Code coverage.
  (b, A, D, HY, HZ, Acond, rnorm) = test(40, 40, 4, 3, 0)
  (x, stats) = cgls(Matrix(A), b)
  (x, stats) = cgls(sparse(Matrix(A)), b)

  # Test b == 0
  (x, stats) = cgls(A, zeros(size(A,1)))
  @test x == zeros(size(A,1))
  @test stats.status == "x = 0 is a zero-residual solution"

  # Test dimension of additional vectors
  for transpose ∈ (false, true)
    A, b, c, D = small_sp(transpose)
    D⁻¹ = inv(D)
    (x, stats) = cgls(A, b, M=D⁻¹, λ=1.0)
  end
end
