function test_lnlq(A, b,transfer_to_craig)
  (x, y, stats) = lnlq(A, b, transfer_to_craig=transfer_to_craig, utolx=0.0, utoly=0.0)
  r = b - A * x
  resid = norm(r) / norm(b)
  return (x, y, stats, resid)
end

@testset "lnlq" begin
  lnlq_tol = 1.0e-6

  for FC in (Float64, ComplexF64)
    @testset "Data Type: $FC" begin

      # Code coverage.
      A, b = kron_unsymmetric(4)
      (x, y, stats) = lnlq(A, b, transfer_to_craig=false)

      # Test b == 0
      A, b = zero_rhs(FC=FC)
      (x, y, stats) = lnlq(A, b)
      @test norm(x) == 0
      @test norm(y) == 0
      @test stats.status == "x is a zero-residual solution"

      for transfer_to_craig ∈ (false, true)
        # Underdetermined consistent.
        A, b = under_consistent(FC=FC)
        (x, y, stats, resid) = test_lnlq(A, b, transfer_to_craig)
        @test(norm(x - A' * y) ≤ lnlq_tol * norm(x))
        @test(resid ≤ lnlq_tol)
        @test(stats.solved)
        (xI, xmin, xmin_norm) = check_min_norm(A, b, x)
        @test(norm(xI - xmin) ≤ cond(A) * lnlq_tol * xmin_norm)
        (xₛ, yₛ, stats) = lnlq(A, b, transfer_to_craig=transfer_to_craig, atol=0.0, rtol=0.0, σ=0.5)
        (xI, xmin, xmin_norm) = check_min_norm(A, b, xₛ)
        @test(norm(xI - xmin) ≤ cond(A) * lnlq_tol * xmin_norm)

        # Square consistent.
        A, b = square_consistent(FC=FC)
        (x, y, stats, resid) = test_lnlq(A, b, transfer_to_craig)
        @test(norm(x - A' * y) ≤ lnlq_tol * norm(x))
        @test(resid ≤ lnlq_tol)
        @test(stats.solved)
        (xI, xmin, xmin_norm) = check_min_norm(A, b, x)
        @test(norm(xI - xmin) ≤ cond(A) * lnlq_tol * xmin_norm)
        (xₛ, yₛ, stats) = lnlq(A, b, transfer_to_craig=transfer_to_craig, atol=0.0, rtol=0.0, σ=0.5)
        (xI, xmin, xmin_norm) = check_min_norm(A, b, xₛ)
        @test(norm(xI - xmin) ≤ cond(A) * lnlq_tol * xmin_norm)

        # Overdetermined consistent.
        A, b = over_consistent(FC=FC)
        (x, y, stats, resid) = test_lnlq(A, b, transfer_to_craig)
        @test(norm(x - A' * y) ≤ lnlq_tol * norm(x))
        @test(resid ≤ lnlq_tol)
        @test(stats.solved)
        (xI, xmin, xmin_norm) = check_min_norm(A, b, x)
        @test(norm(xI - xmin) ≤ cond(A) * lnlq_tol * xmin_norm)
        (xₛ, yₛ, stats) = lnlq(A, b, transfer_to_craig=transfer_to_craig, atol=0.0, rtol=0.0, σ=0.5)
        (xI, xmin, xmin_norm) = check_min_norm(A, b, xₛ)
        @test(norm(xI - xmin) ≤ cond(A) * lnlq_tol * xmin_norm)

        # Test regularization
        A, b, λ = regularization(FC=FC)
        (x, y, stats) = lnlq(A, b, λ=λ, transfer_to_craig=transfer_to_craig, utolx=0.0, utoly=0.0)
        (xₛ, yₛ, stats) = lnlq(A, b, transfer_to_craig=transfer_to_craig, atol=0.0, rtol=0.0, utolx=1e-10, utoly=1e-10, λ=λ)
        for (x, y) in ((x, y), (xₛ, yₛ))
          s = λ * y
          r = b - (A * x + λ * s)
          resid = norm(r) / norm(b)
          @test(resid ≤ lnlq_tol)
          r2 = b - (A * A' + λ^2 * I) * y
          resid2 = norm(r2) / norm(b)
          @test(resid2 ≤ lnlq_tol)
        end

        # Test saddle-point systems
        A, b, D = saddle_point(FC=FC)
        D⁻¹ = inv(D)
        (x, y, stats) = lnlq(A, b, N=D⁻¹, transfer_to_craig=transfer_to_craig)
        (xₛ, yₛ, stats) = lnlq(A, b, N=D⁻¹, transfer_to_craig=transfer_to_craig, atol=0.0, rtol=0.0, σ=0.001)
        for (x, y) in ((x, y), (xₛ, yₛ))
          r = b - A * x
          resid = norm(r) / norm(b)
          @test(resid ≤ lnlq_tol)
          r2 = b - (A * D⁻¹ * A') * y
          resid2 = norm(r2) / norm(b)
          @test(resid2 ≤ lnlq_tol)
        end

        # Test with preconditioners
        A, b, M⁻¹, N⁻¹ = two_preconditioners(FC=FC)
        (x, y, stats) = lnlq(A, b, M=M⁻¹, N=N⁻¹, sqd=false, transfer_to_craig=transfer_to_craig)
        (xₛ, yₛ, stats) = lnlq(A, b, M=M⁻¹, N=N⁻¹, sqd=false, transfer_to_craig=transfer_to_craig, atol=0.0, rtol=0.0, σ=0.5)
        for (x, y) in ((x, y), (xₛ, yₛ))
          r = b - A * x
          resid = sqrt(real(dot(r, M⁻¹ * r))) / norm(b)
          @test(resid ≤ lnlq_tol)
          @test(norm(x - N⁻¹ * A' * y) ≤ lnlq_tol * norm(x))
        end

        # Test symmetric and quasi-definite systems
        A, b, M, N = sqd(FC=FC)
        M⁻¹ = inv(M)
        N⁻¹ = inv(N)
        (x, y, stats) = lnlq(A, b, M=M⁻¹, N=N⁻¹, sqd=true, transfer_to_craig=transfer_to_craig)
        (xₛ, yₛ, stats) = lnlq(A, b, M=M⁻¹, N=N⁻¹, sqd=true, transfer_to_craig=transfer_to_craig, atol=0.0, rtol=0.0, σ=0.5)
        for (x, y) in ((x, y), (xₛ, yₛ))
          r = b - (A * x + M * y)
          resid = norm(r) / norm(b)
          @test(resid ≤ lnlq_tol)
          r2 = b - (A * N⁻¹ * A' + M) * y
          resid2 = norm(r2) / norm(b)
          @test(resid2 ≤ lnlq_tol)
        end

        λ = 4.0
        (x, y, stats) = lnlq(A, b, M=M⁻¹, N=N⁻¹, λ=λ, transfer_to_craig=transfer_to_craig)
        (xₛ, yₛ, stats) = lnlq(A, b, M=M⁻¹, N=N⁻¹, λ=λ, transfer_to_craig=transfer_to_craig, atol=0.0, rtol=0.0, σ=0.5)
        for (x, y) in ((x, y), (xₛ, yₛ))
          r = b - (A * x + λ^2 * M * y)
          resid = norm(r) / norm(b)
          @test(resid ≤ lnlq_tol)
          r2 = b - (A * N⁻¹ * A' + λ^2 * M) * y
          resid2 = norm(r2) / norm(b)
          @test(resid2 ≤ lnlq_tol)
        end

        # Test dimension of additional vectors
        for transpose ∈ (false, true)
          A, b, c, D = small_sp(transpose, FC=FC)
          D⁻¹ = inv(D)
          (x, y, stats) = lnlq(A', c, N=D⁻¹, transfer_to_craig=transfer_to_craig)

          A, b, c, M, N = small_sqd(transpose, FC=FC)
          M⁻¹ = inv(M)
          N⁻¹ = inv(N)
          (x, y, stats) = lnlq(A, b, M=M⁻¹, N=N⁻¹, sqd=true, transfer_to_craig=transfer_to_craig)
        end

        # test callback function
        A, b = over_consistent(FC=FC)
        solver = LnlqSolver(A, b)
        tol = 1.0e-1
        cb_n2 = TestCallbackN2LN(A, b, real(zero(eltype(b))), tol = tol)
        lnlq!(solver, A, b, callback = cb_n2)
        @test solver.stats.status == "user-requested exit"
        @test cb_n2(solver)

        @test_throws TypeError lnlq(A, b, callback = solver -> "string", history = true)
      end
    end
  end
end
