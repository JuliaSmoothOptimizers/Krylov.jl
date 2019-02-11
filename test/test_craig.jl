function test_craig()
  craig_tol = 1.0e-6

  function test_craig(A, b; λ=0.0)
    (x, y, stats) = craig(A, b, λ=λ)
    r = b - A * x
    # if λ > 0
    #   s = r / sqrt(λ);
    #   r = r - sqrt(λ) * s;
    # end
    resid = norm(r) / norm(b)
    @printf("CRAIG: residual: %7.1e\n", resid)
    return (x, y, stats, resid)
  end

  # Underdetermined consistent.
  A, b = under_consistent()
  (x, y, stats, resid) = test_craig(A, b)
  @test(norm(x - A' * y) <= craig_tol * norm(x))
  @test(resid <= craig_tol)
  @test(stats.solved)
  (xI, xmin, xmin_norm) = check_min_norm(A, b, x)
  @test(norm(xI - xmin) <= cond(A) * craig_tol * xmin_norm)

  # Underdetermined inconsistent.
  A, b = under_inconsistent()
  (x, y, stats, resid) = test_craig(A, b)
  # @test(norm(x - A' * y) <= craig_tol * norm(x))
  show(stats)
  @test(stats.inconsistent)

  # Square consistent.
  A, b = square_consistent()
  (x, y, stats, resid) = test_craig(A, b)
  @test(norm(x - A' * y) <= craig_tol * norm(x))
  @test(resid <= craig_tol)
  @test(stats.solved)
  (xI, xmin, xmin_norm) = check_min_norm(A, b, x)
  @test(norm(xI - xmin) <= cond(A) * craig_tol * xmin_norm)

  # Square inconsistent.
  A, b = square_inconsistent()
  (x, y, stats, resid) = test_craig(A, b)
  # @test(norm(x - A' * y) <= craig_tol * norm(x))
  @test(stats.inconsistent)

  # Overdetermined consistent.
  A, b = over_consistent()
  (x, y, stats, resid) = test_craig(A, b)
  @test(norm(x - A' * y) <= craig_tol * norm(x))
  @test(resid <= craig_tol)
  @test(stats.solved)
  (xI, xmin, xmin_norm) = check_min_norm(A, b, x)
  @test(norm(xI - xmin) <= cond(A) * craig_tol * xmin_norm)

  # Overdetermined inconsistent.
  A, b = over_inconsistent()
  (x, y, stats, resid) = test_craig(A, b)
  # @test(norm(x - A' * y) <= craig_tol * norm(x))
  @test(stats.inconsistent)

  # With regularization, all systems are underdetermined and consistent.
  # (x, y, stats, resid) = test_craig(A, b, λ=1.0e-3)
  # @test(norm(x - A' * y) <= craig_tol * norm(x))
  # @test(resid <= craig_tol)
  # @test(stats.solved)
  # (xI, xmin, xmin_norm) = check_min_norm(A, b, x, λ=1.0e-3)
  # @test(norm(xI - xmin) <= cond(A) * craig_tol * xmin_norm)

  # Code coverage.
  (x, y, stats) = craig(sparse(A), b, λ=1.0e-3)
  show(stats)

  # Test b == 0
  A, b = zero_rhs()
  (x, y, stats) = craig(A, b, λ=1.0e-3)
  @test x == zeros(size(A,2))
  @test y == zeros(size(A,1))
  @test stats.status == "x = 0 is a zero-residual solution"

  # Test integer values
  A, b = over_int()
  (x, y, stats) = craig(A, b)
  @test stats.solved
end

test_craig()
