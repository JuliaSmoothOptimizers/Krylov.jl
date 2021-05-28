@testset "crls" begin
  ⪅(x,y) = (x ≈ y) || (x < y)
  crls_tol = 1.0e-5

  for npower = 1 : 4
    (b, A, D, HY, HZ, Acond, rnorm) = test(40, 40, 4, npower, 0)  # No regularization.

    (x, stats) = crls(A, b)
    resid = norm(A' * (A*x - b)) / norm(b)
    @test(resid ≤ crls_tol)
    @test(stats.solved)

    λ = 1.0e-3
    (x, stats) = crls(A, b, λ=λ)
    resid = norm(A' * (A*x - b) + λ * x) / norm(b)
    @test(resid ≤ crls_tol)
    @test(stats.solved)
  end

  # Test with preconditioning.
  A, b, M = saddle_point()
  M⁻¹ = inv(M)
  (x, stats) = crls(A, b, M=M⁻¹)
  resid = norm(A' * M⁻¹ * (A * x - b)) / sqrt(dot(b, M⁻¹ * b))
  @test resid ≤ crls_tol

  # test trust-region constraint
  (x, stats) = crls(A, b)

  radius = 0.75 * norm(x)
  (x, stats) = crls(A, b, radius=radius)
  @test(stats.solved)
  @test(abs(radius - norm(x)) ≤ crls_tol * radius)

  # Code coverage.
  (b, A, D, HY, HZ, Acond, rnorm) = test(40, 40, 4, 3, 0)
  (x, stats) = crls(Matrix(A), b)
  (x, stats) = crls(sparse(Matrix(A)), b)

  # Test b == 0
  A, b = zero_rhs()
  (x, stats) = crls(A, b)
  @test x == zeros(size(A,1))
  @test stats.status == "x = 0 is a zero-residual solution"

  # Test A positive semi-definite
  radius = 10.
  m,n = 10,7
  U = qr(rand(m,m)).Q
  V = qr(rand(n,n)).Q
  S = [diagm(0 => [0, 1.0e-6, 1, 4, 20, 15, 1.0e5]) ; zeros(3,7)]
  A = U * S * V
  p = V[:,1]; b = A'\p
  (x, stats) = crls(A, b, radius=radius)
  @test stats.solved
  @test (stats.status == "zero-curvature encountered") || (stats.status == "on trust-region boundary")
  @test norm(x) ⪅ radius

  # Test dimension of additional vectors
  for transpose ∈ (false, true)
    A, b, c, D = small_sp(transpose)
    D⁻¹ = inv(D)
    (x, stats) = crls(A, b, M=D⁻¹, λ=1.0)
  end
end
