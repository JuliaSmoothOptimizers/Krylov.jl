@testset "craigmr" begin
  craigmr_tol = 1.0e-6

  function test_craigmr(A, b; λ=0.0, history=false)
    (nrow, ncol) = size(A)
    (x, y, stats) = craigmr(A, b, λ=λ, history=history)
    r = b - A * x
    Ar = A' * r
    # if λ > 0
    #   s = r / sqrt(λ)
    #   r = r - sqrt(λ) * s
    # end
    resid = norm(r) / norm(b)
    Aresid = norm(Ar) / (norm(A) * norm(b))
    return (x, y, stats, resid, Aresid)
  end

  # Underdetermined consistent.
  A, b = under_consistent()
  (x, y, stats, resid, Aresid) = test_craigmr(A, b)
  @test(norm(x - A' * y) ≤ craigmr_tol * norm(x))
  @test(resid ≤ craigmr_tol)
  @test(stats.solved)
  (xI, xmin, xmin_norm) = check_min_norm(A, b, x)
  @test(norm(xI - xmin) ≤ cond(A) * craigmr_tol * xmin_norm)

  # Underdetermined inconsistent.
  A, b = under_inconsistent()
  (x, y, stats, resid) = test_craigmr(A, b, history=true)
  @test(norm(x - A' * y) ≤ craigmr_tol * norm(x))
  @test(stats.inconsistent)
  @test(stats.Aresiduals[end] ≤ craigmr_tol)

  # Square consistent.
  A, b = square_consistent()
  (x, y, stats, resid) = test_craigmr(A, b)
  @test(norm(x - A' * y) ≤ craigmr_tol * norm(x))
  @test(resid ≤ craigmr_tol)
  @test(stats.solved)
  (xI, xmin, xmin_norm) = check_min_norm(A, b, x)
  @test(norm(xI - xmin) ≤ cond(A) * craigmr_tol * xmin_norm)

  # Square inconsistent.
  A, b = square_inconsistent()
  (x, y, stats, resid) = test_craigmr(A, b, history=true)
  @test(norm(x - A' * y) ≤ craigmr_tol * norm(x))
  @test(stats.inconsistent)
  @test(stats.Aresiduals[end] ≤ craigmr_tol)

  # Overdetermined consistent.
  A, b = over_consistent()
  (x, y, stats, resid) = test_craigmr(A, b)
  @test(norm(x - A' * y) ≤ craigmr_tol * norm(x))
  @test(resid ≤ craigmr_tol)
  @test(stats.solved)
  (xI, xmin, xmin_norm) = check_min_norm(A, b, x)
  @test(norm(xI - xmin) ≤ cond(A) * craigmr_tol * xmin_norm)

  # Overdetermined inconsistent.
  A, b = over_inconsistent()
  (x, y, stats, resid) = test_craigmr(A, b, history=true)
  @test(norm(x - A' * y) ≤ craigmr_tol * norm(x))
  @test(stats.inconsistent)
  @test(stats.Aresiduals[end] ≤ craigmr_tol)

  # With regularization, all systems are underdetermined and consistent.
  # (x, y, stats, resid) = test_craigmr(A, b, λ=1.0e-3)
  # @test(norm(x - A' * y) ≤ craigmr_tol * norm(x))
  # @test(resid ≤ craigmr_tol)
  # @test(stats.solved)
  # (xI, xmin, xmin_norm) = check_min_norm(A, b, x, λ=1.0e-3)
  # @test(norm(xI - xmin) ≤ cond(A) * craigmr_tol * xmin_norm)

  # Code coverage.
  (x, y, stats) = craigmr(sparse(A), b, λ=1.0e-3)

  # Test b == 0
  A, b = zero_rhs()
  (x, y, stats) = craigmr(A, b, λ=1.0e-3)
  @test x == zeros(size(A,2))
  @test y == zeros(size(A,1))
  @test stats.status == "x = 0 is a zero-residual solution"

  # Test regularization
  A, b, λ = regularization()
  (x, y, stats) = craigmr(A, b, λ=λ)
  s = λ * y
  r = b - (A * x + λ * s)
  resid = norm(r) / norm(b)
  @test(resid ≤ craigmr_tol)
  r2 = b - (A * A' + λ^2 * I) * y
  resid2 = norm(r2) / norm(b)
  @test(resid2 ≤ craigmr_tol)

  # Test saddle-point systems
  A, b, D = saddle_point()
  D⁻¹ = inv(D)
  (x, y, stats) = craigmr(A, b, N=D⁻¹)
  r = b - A * x
  resid = norm(r) / norm(b)
  @test(resid ≤ craigmr_tol)
  r2 = b - (A * D⁻¹ * A') * y
  resid2 = norm(r2) / norm(b)
  @test(resid2 ≤ craigmr_tol)

  # Test with preconditioners
  A, b, M⁻¹, N⁻¹ = two_preconditioners()
  (x, y, stats) = craigmr(A, b, M=M⁻¹, N=N⁻¹, sqd=false)
  r = b - A * x
  resid = sqrt(dot(r, M⁻¹ * r)) / norm(b)
  @test(resid ≤ craigmr_tol)
  @test(norm(x - N⁻¹ * A' * y) ≤ craigmr_tol * norm(x))

  # Test symmetric and quasi-definite systems
  A, b, M, N = sqd()
  M⁻¹ = inv(M)
  N⁻¹ = inv(N)
  (x, y, stats) = craigmr(A, b, M=M⁻¹, N=N⁻¹, sqd=true)
  r = b - (A * x + M * y)
  resid = norm(r) / norm(b)
  @test(resid ≤ craigmr_tol)
  r2 = b - (A * N⁻¹ * A' + M) * y
  resid2 = norm(r2) / norm(b)
  @test(resid2 ≤ craigmr_tol)

  # Test dimension of additional vectors
  for transpose ∈ (false, true)
    A, b, c, D = small_sp(transpose)
    D⁻¹ = inv(D)
    (x, y, stats) = craigmr(A', c, N=D⁻¹)

    A, b, c, M, N = small_sqd(transpose)
    M⁻¹ = inv(M)
    N⁻¹ = inv(N)
    (x, y, stats) = craigmr(A, b, M=M⁻¹, N=N⁻¹, sqd=true)
  end
end
