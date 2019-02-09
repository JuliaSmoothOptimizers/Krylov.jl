function test_crmr()
  crmr_tol = 1.0e-6;

  function test_crmr(A, b; λ=0.0, M=opEye())
    (nrow, ncol) = size(A);
    (x, stats) = crmr(A, b, λ=λ, M=M);
    r = b - A * x;
    if λ > 0
      s = r / sqrt(λ);
      r = r - sqrt(λ) * s;
    end
    resid = norm(r) / norm(b);
    @printf("CRMR: residual: %7.1e\n", resid);
    return (x, stats, resid)
  end

  # Underdetermined consistent.
  A, b = under_consistent()
  (x, stats, resid) = test_crmr(A, b);
  @test(resid <= crmr_tol);
  @test(stats.solved);
  (xI, xmin, xmin_norm) = check_min_norm(A, b, x);
  @test(norm(xI - xmin) <= cond(A) * crmr_tol * xmin_norm);

  # Underdetermined inconsistent.
  A, b = under_inconsistent()
  (x, stats, resid) = test_crmr(A, b);
  @test(stats.inconsistent);
  @test(stats.Aresiduals[end] <= crmr_tol);

  # Square consistent.
  A, b = square_consistent()
  (x, stats, resid) = test_crmr(A, b);
  @test(resid <= crmr_tol);
  @test(stats.solved);
  (xI, xmin, xmin_norm) = check_min_norm(A, b, x);
  @test(norm(xI - xmin) <= cond(A) * crmr_tol * xmin_norm);

  # Square inconsistent.
  A, b = square_inconsistent()
  (x, stats, resid) = test_crmr(A, b);
  @test(stats.inconsistent);
  @test(stats.Aresiduals[end] <= crmr_tol);

  # Overdetermined consistent.
  A, b = over_consistent()
  (x, stats, resid) = test_crmr(A, b);
  @test(resid <= crmr_tol);
  @test(stats.solved);
  (xI, xmin, xmin_norm) = check_min_norm(A, b, x);
  @test(norm(xI - xmin) <= cond(A) * crmr_tol * xmin_norm);

  # Overdetermined inconsistent.
  A, b = over_inconsistent()
  (x, stats, resid) = test_crmr(A, b);
  @test(stats.inconsistent);
  @test(stats.Aresiduals[end] <= crmr_tol);

  # With regularization, all systems are underdetermined and consistent.
  (x, stats, resid) = test_crmr(A, b, λ=1.0e-3);
  @test(resid <= crmr_tol);
  @test(stats.solved);
  (xI, xmin, xmin_norm) = check_min_norm(A, b, x, λ=1.0e-3);
  @test(norm(xI - xmin) <= cond(A) * crmr_tol * xmin_norm);

  # Code coverage.
  (x, stats, resid) = test_crmr(sparse(A), b, λ=1.0e-3);
  show(stats);

  # Test b == 0
  A, b = zero_rhs()
  (x, stats) = crmr(A, b)
  @test x == zeros(size(A,2))
  @test stats.status == "x = 0 is a zero-residual solution"

  # Test integer values
  A, b = over_int()
  (x, stats) = crmr(A, b)
  @test stats.solved

  # Test preconditioner with an under-determined problem:
  # Find the least norm force that transfers mass unit distance with zero final velocity
  A = 0.5 * [19.0 17.0 15.0 13.0 11.0 9.0 7.0 5.0 3.0 1.0;
              2.0  2.0  2.0  2.0  2.0 2.0 2.0 2.0 2.0 2.0]
  b = [1.0; 0.0]
  M = LinearOperator(Diagonal(1 ./ (A * A')))
  (x, stats, resid) = test_crmr(A, b, M=M)
  @test(resid ≤ crmr_tol)
  @test(stats.solved)
  (xI, xmin, xmin_norm) = check_min_norm(A, b, x)
  @test(norm(xI - xmin) ≤ cond(A) * crmr_tol * xmin_norm)
end

test_crmr()
