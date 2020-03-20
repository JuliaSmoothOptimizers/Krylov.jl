function test_craigmr()
  craigmr_tol = 1.0e-6

  function test_craigmr(A, b; λ=0.0)
    (nrow, ncol) = size(A)
    (x, y, stats) = craigmr(A, b, λ=λ)
    r = b - A * x
    Ar = A' * r
    # if λ > 0
    #   s = r / sqrt(λ)
    #   r = r - sqrt(λ) * s
    # end
    resid = norm(r) / norm(b)
    Aresid = norm(Ar) / (norm(A) * norm(b))
    @printf("CRAIGMR: residual: %7.1e  least-squares: %7.1e\n", resid, Aresid)
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
  (x, y, stats, resid) = test_craigmr(A, b)
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
  (x, y, stats, resid) = test_craigmr(A, b)
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
  (x, y, stats, resid) = test_craigmr(A, b)
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
  show(stats)

  # Test b == 0
  A, b = zero_rhs()
  (x, y, stats) = craigmr(A, b, λ=1.0e-3)
  @test x == zeros(size(A,2))
  @test y == zeros(size(A,1))
  @test stats.status == "x = 0 is a zero-residual solution"

  # Test with preconditioners
  A, b, M, N = two_preconditioners()
  (x, y, stats) = craigmr(A, b, M=M, N=N)
  show(stats)
  r = b - A * x
  resid = sqrt(dot(r, M * r)) / norm(b)
  @printf("CRAIGMR: Relative residual: %8.1e\n", resid)
  @test(norm(x - N * A' * y) ≤ craigmr_tol * norm(x))
  @test(resid ≤ craigmr_tol)
  @test(stats.solved)
end

test_craigmr()
