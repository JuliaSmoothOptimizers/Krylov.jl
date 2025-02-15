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

@testset "craigmr" begin
  craigmr_tol = 1.0e-6

  for FC in (Float64, ComplexF64)
    @testset "Data Type: $FC" begin

      # Underdetermined consistent.
      A, b = under_consistent(FC=FC)
      (x, y, stats, resid, Aresid) = test_craigmr(A, b)
      @test(norm(x - A' * y) ≤ craigmr_tol * norm(x))
      @test(resid ≤ craigmr_tol)
      @test(stats.solved)
      (xI, xmin, xmin_norm) = check_min_norm(A, b, x)
      @test(norm(xI - xmin) ≤ cond(A) * craigmr_tol * xmin_norm)

      # Underdetermined inconsistent.
      A, b = under_inconsistent(FC=FC)
      (x, y, stats, resid) = test_craigmr(A, b, history=true)
      @test(norm(x - A' * y) ≤ craigmr_tol * norm(x))
      @test(stats.inconsistent)
      @test(stats.Aresiduals[end] ≤ craigmr_tol)

      # Square consistent.
      A, b = square_consistent(FC=FC)
      (x, y, stats, resid) = test_craigmr(A, b)
      @test(norm(x - A' * y) ≤ craigmr_tol * norm(x))
      @test(resid ≤ craigmr_tol)
      @test(stats.solved)
      (xI, xmin, xmin_norm) = check_min_norm(A, b, x)
      @test(norm(xI - xmin) ≤ cond(A) * craigmr_tol * xmin_norm)

      # Square inconsistent.
      A, b = square_inconsistent(FC=FC)
      (x, y, stats, resid) = test_craigmr(A, b, history=true)
      @test(norm(x - A' * y) ≤ craigmr_tol * norm(x))
      @test(stats.inconsistent)
      @test(stats.Aresiduals[end] ≤ craigmr_tol)

      # Overdetermined consistent.
      A, b = over_consistent(FC=FC)
      (x, y, stats, resid) = test_craigmr(A, b)
      @test(norm(x - A' * y) ≤ craigmr_tol * norm(x))
      @test(resid ≤ craigmr_tol)
      @test(stats.solved)
      (xI, xmin, xmin_norm) = check_min_norm(A, b, x)
      @test(norm(xI - xmin) ≤ cond(A) * craigmr_tol * xmin_norm)

      # Overdetermined inconsistent.
      A, b = over_inconsistent(FC=FC)
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
      A, b = zero_rhs(FC=FC)
      (x, y, stats) = craigmr(A, b, λ=1.0e-3)
      @test norm(x) == 0
      @test norm(y) == 0
      @test stats.status == "x is a zero-residual solution"

      # Test regularization
      A, b, λ = regularization(FC=FC)
      (x, y, stats) = craigmr(A, b, λ=λ)
      s = λ * y
      r = b - (A * x + λ * s)
      resid = norm(r) / norm(b)
      @test(resid ≤ craigmr_tol)
      r2 = b - (A * A' + λ^2 * I) * y
      resid2 = norm(r2) / norm(b)
      @test(resid2 ≤ craigmr_tol)

      # Test saddle-point systems
      A, b, D = saddle_point(FC=FC)
      D⁻¹ = inv(D)
      (x, y, stats) = craigmr(A, b, N=D⁻¹)
      r = b - A * x
      resid = norm(r) / norm(b)
      @test(resid ≤ craigmr_tol)
      r2 = b - (A * D⁻¹ * A') * y
      resid2 = norm(r2) / norm(b)
      @test(resid2 ≤ craigmr_tol)

      # Test with preconditioners
      A, b, M⁻¹, N⁻¹ = two_preconditioners(FC=FC)
      (x, y, stats) = craigmr(A, b, M=M⁻¹, N=N⁻¹, sqd=false)
      r = b - A * x
      resid = sqrt(real(dot(r, M⁻¹ * r))) / norm(b)
      @test(resid ≤ craigmr_tol)
      @test(norm(x - N⁻¹ * A' * y) ≤ craigmr_tol * norm(x))

      # Test symmetric and quasi-definite systems
      A, b, M, N = sqd(FC=FC)
      M⁻¹ = inv(M)
      N⁻¹ = inv(N)
      (x, y, stats) = craigmr(A, b, M=M⁻¹, N=N⁻¹, sqd=true)
      r = b - (A * x + M * y)
      resid = norm(r) / norm(b)
      @test(resid ≤ craigmr_tol)
      r2 = b - (A * N⁻¹ * A' + M) * y
      resid2 = norm(r2) / norm(b)
      @test(resid2 ≤ craigmr_tol)

      λ = 4.0
      (x, y, stats) = craigmr(A, b, M=M⁻¹, N=N⁻¹, λ=λ)
      r = b - (A * x + λ^2 * M * y)
      resid = norm(r) / norm(b)
      @test(resid ≤ craigmr_tol)
      r2 = b - (A * N⁻¹ * A' + λ^2 * M) * y
      resid2 = norm(r2) / norm(b)
      @test(resid2 ≤ craigmr_tol)

      # Test dimension of additional vectors
      for transpose ∈ (false, true)
        A, b, c, D = small_sp(transpose)
        D⁻¹ = inv(D)
        (x, y, stats) = craigmr(A', c, N=D⁻¹)

        A, b, c, M, N = small_sqd(transpose, FC=FC)
        M⁻¹ = inv(M)
        N⁻¹ = inv(N)
        (x, y, stats) = craigmr(A, b, M=M⁻¹, N=N⁻¹, sqd=true)
      end

      # test callback function
      A, b = over_consistent(FC=FC)
      solver = CraigmrSolver(A, b)
      tol = 1.0e-1
      cb_n2 = TestCallbackN2LN(A, b, real(zero(eltype(b))), tol = tol)
      craigmr!(solver, A, b, callback = cb_n2)
      @test solver.stats.status == "user-requested exit"
      @test cb_n2(solver)

      @test_throws TypeError craigmr(A, b, callback = solver -> "string", history = true)
    end
  end
end
