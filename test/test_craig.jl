@testset "craig" begin
  craig_tol = 1.0e-6

  function test_craig(A, b; λ=0.0)
    (x, y, stats) = craig(A, b, λ=λ)
    r = b - A * x
    # if λ > 0
    #   s = r / sqrt(λ)
    #   r = r - sqrt(λ) * s
    # end
    resid = norm(r) / norm(b)
    return (x, y, stats, resid)
  end

  # Underdetermined consistent.
  A, b = under_consistent()
  (x, y, stats, resid) = test_craig(A, b)
  @test(norm(x - A' * y) ≤ craig_tol * norm(x))
  @test(resid ≤ craig_tol)
  @test(stats.solved)
  (xI, xmin, xmin_norm) = check_min_norm(A, b, x)
  @test(norm(xI - xmin) ≤ cond(A) * craig_tol * xmin_norm)

  # Underdetermined inconsistent.
  A, b = under_inconsistent()
  (x, y, stats, resid) = test_craig(A, b)
  # @test(norm(x - A' * y) ≤ craig_tol * norm(x))
  @test(stats.inconsistent || stats.status == "condition number exceeds tolerance")

  # Square consistent.
  A, b = square_consistent()
  (x, y, stats, resid) = test_craig(A, b)
  @test(norm(x - A' * y) ≤ craig_tol * norm(x))
  @test(resid ≤ craig_tol)
  @test(stats.solved)
  (xI, xmin, xmin_norm) = check_min_norm(A, b, x)
  @test(norm(xI - xmin) ≤ cond(A) * craig_tol * xmin_norm)

  # Square inconsistent.
  A, b = square_inconsistent()
  (x, y, stats, resid) = test_craig(A, b)
  # @test(norm(x - A' * y) ≤ craig_tol * norm(x))
  @test(stats.inconsistent || stats.status == "condition number exceeds tolerance")

  # Overdetermined consistent.
  A, b = over_consistent()
  (x, y, stats, resid) = test_craig(A, b)
  @test(norm(x - A' * y) ≤ craig_tol * norm(x))
  @test(resid ≤ craig_tol)
  @test(stats.solved)
  (xI, xmin, xmin_norm) = check_min_norm(A, b, x)
  @test(norm(xI - xmin) ≤ cond(A) * craig_tol * xmin_norm)

  # Overdetermined inconsistent.
  A, b = over_inconsistent()
  (x, y, stats, resid) = test_craig(A, b)
  # @test(norm(x - A' * y) ≤ craig_tol * norm(x))
  @test(stats.inconsistent || stats.status == "condition number exceeds tolerance")

  # With regularization, all systems are underdetermined and consistent.
  # (x, y, stats, resid) = test_craig(A, b, λ=1.0e-3)
  # @test(norm(x - A' * y) ≤ craig_tol * norm(x))
  # @test(resid ≤ craig_tol)
  # @test(stats.solved)
  # (xI, xmin, xmin_norm) = check_min_norm(A, b, x, λ=1.0e-3)
  # @test(norm(xI - xmin) ≤ cond(A) * craig_tol * xmin_norm)

  # Code coverage.
  (x, y, stats) = craig(sparse(A), b, λ=1.0e-3)

  # Test b == 0
  A, b = zero_rhs()
  (x, y, stats) = craig(A, b, λ=1.0e-3)
  @test x == zeros(size(A,2))
  @test y == zeros(size(A,1))
  @test stats.status == "x = 0 is a zero-residual solution"

  # Test regularization
  A, b, λ = regularization()
  (x, y, stats) = craig(A, b, λ=λ)
  s = λ * y
  r = b - (A * x + λ * s)
  resid = norm(r) / norm(b)
  @test(resid ≤ craig_tol)
  r2 = b - (A * A' + λ^2 * I) * y
  resid2 = norm(r2) / norm(b)
  @test(resid2 ≤ craig_tol)

  # Test saddle-point systems
  A, b, D = saddle_point()
  D⁻¹ = inv(D)
  (x, y, stats) = craig(A, b, N=D⁻¹)
  r = b - A * x
  resid = norm(r) / norm(b)
  @test(resid ≤ craig_tol)
  r2 = b - (A * D⁻¹ * A') * y
  resid2 = norm(r2) / norm(b)
  @test(resid2 ≤ craig_tol)

  # Test with preconditioners
  A, b, M⁻¹, N⁻¹ = two_preconditioners()
  (x, y, stats) = craig(A, b, M=M⁻¹, N=N⁻¹, sqd=false)
  r = b - A * x
  resid = sqrt(dot(r, M⁻¹ * r)) / norm(b)
  @test(resid ≤ craig_tol)
  @test(norm(x - N⁻¹ * A' * y) ≤ craig_tol * norm(x))

  # Test symmetric and quasi-definite systems
  A, b, M, N = sqd()
  M⁻¹ = inv(M)
  N⁻¹ = inv(N)
  (x, y, stats) = craig(A, b, M=M⁻¹, N=N⁻¹, sqd=true)
  r = b - (A * x + M * y)
  resid = norm(r) / norm(b)
  @test(resid ≤ craig_tol)
  r2 = b - (A * N⁻¹ * A' + M) * y
  resid2 = norm(r2) / norm(b)
  @test(resid2 ≤ craig_tol)

  # Test dimension of additional vectors
  for transpose ∈ (false, true)
    A, b, c, D = small_sp(transpose)
    D⁻¹ = inv(D)
    (x, y, stats) = craig(A', c, N=D⁻¹)

    A, b, c, M, N = small_sqd(transpose)
    M⁻¹ = inv(M)
    N⁻¹ = inv(N)
    (x, y, stats) = craig(A, b, M=M⁻¹, N=N⁻¹, sqd=true)
  end
end
