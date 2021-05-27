@testset "lslq" begin
  lslq_tol = 1.0e-5

  for npower = 1 : 4
    (b, A, D, HY, HZ, Acond, rnorm) = test(40, 40, 4, npower, 0)  # No regularization.

    (x, x_cg, err_lbnds, err_ubnds_lq, err_ubnds_cg, stats) = lslq(A, b)
    r = b - A * x
    resid = norm(A' * r) / norm(b)
    @test(resid ≤ lslq_tol)
    @test(stats.solved)

    λ = 1.0e-3
    (x, x_cg, err_lbnds, err_ubnds_lq, err_ubnds_cg, stats) = lslq(A, b, λ=λ)
    r = b - A * x
    resid = norm(A' * r - λ * λ * x) / norm(b)
    @test(resid ≤ lslq_tol)
    @test(stats.solved)
  end

  # Test with smallest singular value estimate
  Σ = diagm(0 => 1:4)
  U, _ = qr(rand(6, 6))
  V, _ = qr(rand(4, 4))
  A = U * [Σ ; zeros(2, 4)] * V'
  b = ones(6)
  (x, x_cg, err_lbnds, err_ubnds_lq, err_ubnds_cg, stats) = lslq(A, b, σ=1 - 1.0e-10, history=true)
  @test isapprox(err_ubnds_lq[end], 0.0, atol=sqrt(eps(Float64)))
  @test isapprox(err_ubnds_cg[end], 0.0, atol=sqrt(eps(Float64)))
  x_exact = A \ b
  @test norm(x - x_exact) ≤ sqrt(eps(Float64)) * norm(x_exact)
  @test norm(x_cg - x_exact) ≤ sqrt(eps(Float64)) * norm(x_exact)

  # Code coverage.
  (b, A, D, HY, HZ, Acond, rnorm) = test(40, 40, 4, 3, 0)
  (x, x_cg, err_lbnds, err_ubnds_lq, err_ubnds_cg, stats) = lslq(Matrix(A), b)
  (x, x_cg, err_lbnds, err_ubnds_lq, err_ubnds_cg, stats) = lslq(sparse(Matrix(A)), b)

  # Test b == 0
  (x, x_cg, err_lbnds, err_ubnds_lq, err_ubnds_cg, stats) = lslq(A, zeros(size(A,1)))
  @test x == zeros(size(A,1))
  @test stats.status == "x = 0 is a zero-residual solution"

  # Test with preconditioners
  A, b, M, N = two_preconditioners()
  (x_lq, x_cg, err_lbnds, err_ubnds_lq, err_ubnds_cg, stats) = lslq(A, b, M=M, N=N)
  r = b - A * x_lq
  resid = sqrt(dot(r, M * r)) / norm(b)
  @test(resid ≤ lslq_tol)
  @test(stats.solved)

  # Test regularization
  A, b, λ = regularization()
  (x_lq, x_cg, err_lbnds, err_ubnds_lq, err_ubnds_cg, stats) = lslq(A, b, λ=λ)
  r = b - A * x_lq
  resid = norm(A' * r - λ^2 * x_lq) / norm(b)
  @test(resid ≤ lslq_tol)

  # Test saddle-point systems
  A, b, D = saddle_point()
  D⁻¹ = inv(D)
  (x_lq, x_cg, err_lbnds, err_ubnds_lq, err_ubnds_cg, stats) = lslq(A, b, M=D⁻¹)
  r = D⁻¹ * (b - A * x_lq)
  resid = norm(A' * r) / norm(b)
  @test(resid ≤ lslq_tol)

  # Test symmetric and quasi-definite systems
  A, b, M, N = sqd()
  M⁻¹ = inv(M)
  N⁻¹ = inv(N)
  (x_lq, x_cg, err_lbnds, err_ubnds_lq, err_ubnds_cg, stats) = lslq(A, b, M=M⁻¹, N=N⁻¹, sqd=true)
  r = M⁻¹ * (b - A * x_lq)
  resid = norm(A' * r - N * x_lq) / norm(b)
  @test(resid ≤ lslq_tol)

  # Test dimension of additional vectors
  for transpose ∈ (false, true)
    A, b, c, D = small_sp(transpose)
    D⁻¹ = inv(D)
    (x_lq, x_cg, err_lbnds, err_ubnds_lq, err_ubnds_cg, stats) = lslq(A, b, M=D⁻¹)

    A, b, c, M, N = small_sqd(transpose)
    M⁻¹ = inv(M)
    N⁻¹ = inv(N)
    (x_lq, x_cg, err_lbnds, err_ubnds_lq, err_ubnds_cg, stats) = lslq(A, b, M=M⁻¹, N=N⁻¹, sqd=true)
  end
end
