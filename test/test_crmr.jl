function test_crmr(A, b; λ=0.0, N=I, history=false)
  (nrow, ncol) = size(A)
  (x, stats) = crmr(A, b, λ=λ, N=N, history=history)
  r = b - A * x
  if λ > 0
    s = r / sqrt(λ)
    r = r - sqrt(λ) * s
  end
  resid = norm(r) / norm(b)
  return (x, stats, resid)
end

@testset "crmr" begin
  crmr_tol = 1.0e-6

  for FC in (Float64, ComplexF64)
    @testset "Data Type: $FC" begin

      # Underdetermined consistent.
      A, b = under_consistent(FC=FC)
      (x, stats, resid) = test_crmr(A, b)
      @test(resid ≤ crmr_tol)
      @test(stats.solved)
      (xI, xmin, xmin_norm) = check_min_norm(A, b, x)
      @test(norm(xI - xmin) ≤ cond(A) * crmr_tol * xmin_norm)

      # Underdetermined inconsistent.
      A, b = under_inconsistent(FC=FC)
      (x, stats, resid) = test_crmr(A, b, history=true)
      @test(stats.inconsistent)
      @test(stats.Aresiduals[end] ≤ crmr_tol)

      # Square consistent.
      A, b = square_consistent(FC=FC)
      (x, stats, resid) = test_crmr(A, b)
      @test(resid ≤ crmr_tol)
      @test(stats.solved)
      (xI, xmin, xmin_norm) = check_min_norm(A, b, x)
      @test(norm(xI - xmin) ≤ cond(A) * crmr_tol * xmin_norm)

      # Square inconsistent.
      A, b = square_inconsistent(FC=FC)
      (x, stats, resid) = test_crmr(A, b, history=true)
      @test(stats.inconsistent)
      @test(stats.Aresiduals[end] ≤ crmr_tol)

      # Overdetermined consistent.
      A, b = over_consistent(FC=FC)
      (x, stats, resid) = test_crmr(A, b)
      @test(resid ≤ crmr_tol)
      @test(stats.solved)
      (xI, xmin, xmin_norm) = check_min_norm(A, b, x)
      @test(norm(xI - xmin) ≤ cond(A) * crmr_tol * xmin_norm)

      # Overdetermined inconsistent.
      A, b = over_inconsistent(FC=FC)
      (x, stats, resid) = test_crmr(A, b, history=true)
      @test(stats.inconsistent)
      @test(stats.Aresiduals[end] ≤ crmr_tol)

      # With regularization, all systems are underdetermined and consistent.
      (x, stats, resid) = test_crmr(A, b, λ=1.0e-3)
      @test(resid ≤ crmr_tol)
      @test(stats.solved)
      (xI, xmin, xmin_norm) = check_min_norm(A, b, x, λ=1.0e-3)
      @test(norm(xI - xmin) ≤ cond(A) * crmr_tol * xmin_norm)

      # Test b == 0
      A, b = zero_rhs(FC=FC)
      (x, stats) = crmr(A, b)
      @test norm(x) == 0
      @test stats.status == "x is a zero-residual solution"

      # Test preconditioner with an under-determined problem:
      # Find the least norm force that transfers mass unit distance with zero final velocity
      A = 0.5 * [19.0 17.0 15.0 13.0 11.0 9.0 7.0 5.0 3.0 1.0;
                  2.0  2.0  2.0  2.0  2.0 2.0 2.0 2.0 2.0 2.0]
      b = [1.0; 0.0]
      N = Diagonal(1 ./ (A * A'))
      (x, stats, resid) = test_crmr(A, b, N=N)
      @test(resid ≤ crmr_tol)
      @test(stats.solved)
      (xI, xmin, xmin_norm) = check_min_norm(A, b, x)
      @test(norm(xI - xmin) ≤ cond(A) * crmr_tol * xmin_norm)

      # Test dimension of additional vectors
      for transpose ∈ (false, true)
        A, b, c, D = small_sp(transpose, FC=FC)
        D⁻¹ = inv(D)
        (x, stats) = crmr(A, b, N=D⁻¹, λ=1.0)
      end

      # test callback function
      A, b = over_consistent(FC=FC)
      solver = CrmrSolver(A, b)
      tol = 1.0e-1
      cb_n2 = TestCallbackN2LN(A, b, real(zero(eltype(b))), tol = tol)
      crmr!(solver, A, b, callback = cb_n2)
      @test solver.stats.status == "user-requested exit"
      @test cb_n2(solver)

      @test_throws TypeError crmr(A, b, callback = solver -> "string", history = true)
    end
  end
end
