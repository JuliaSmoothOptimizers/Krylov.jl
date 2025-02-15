@testset "lslq" begin
  lslq_tol = 1.0e-5

  for FC in (Float64, ComplexF64)
    @testset "Data Type: $FC" begin

      for npower = 1 : 4
        (b, A, D, HY, HZ, Acond, rnorm) = test(40, 40, 4, npower, 0)  # No regularization.

        (x, stats) = lslq(A, b)
        r = b - A * x
        resid = norm(A' * r) / norm(b)
        @test(resid ≤ lslq_tol)
        @test(stats.solved)

        λ = 1.0e-3
        (x, stats) = lslq(A, b, λ=λ)
        r = b - A * x
        resid = norm(A' * r - λ * λ * x) / norm(b)
        @test(resid ≤ lslq_tol)
        @test(stats.solved)
      end

      # Test error in upper bound computation
      (x, stats) = lslq(A, b, σ=1.0)
      @test stats.error_with_bnd

      # Test b == 0
      A, b = zero_rhs(FC=FC)
      (x, stats) = lslq(A, b)
      @test norm(x) == 0
      @test stats.status == "x is a zero-residual solution"

      for transfer_to_lsqr ∈ (false, true)
        # Test with smallest singular value estimate
        Σ = diagm(0 => 1:4)
        U, _ = qr(rand(6, 6))
        V, _ = qr(rand(4, 4))
        A = U * [Σ ; zeros(2, 4)] * V'
        b = ones(6)
        (x, stats) = lslq(A, b, σ=1.0 - 1.0e-10, history=true, transfer_to_lsqr=transfer_to_lsqr)
        @test isapprox(stats.err_ubnds_lq[end], 0.0, atol=sqrt(eps(Float64)))
        @test isapprox(stats.err_ubnds_cg[end], 0.0, atol=sqrt(eps(Float64)))
        x_exact = A \ b
        @test norm(x - x_exact) ≤ sqrt(eps(Float64)) * norm(x_exact)

        # Test with preconditioners
        A, b, M, N = two_preconditioners(FC=FC)
        (x_lq, stats) = lslq(A, b, M=M, N=N, transfer_to_lsqr=transfer_to_lsqr)
        r = b - A * x_lq
        resid = sqrt(real(dot(r, M * r))) / norm(b)
        @test(resid ≤ lslq_tol)
        @test(stats.solved)

        # Test regularization
        A, b, λ = regularization(FC=FC)
        (x_lq, stats) = lslq(A, b, λ=λ, transfer_to_lsqr=transfer_to_lsqr)
        r = b - A * x_lq
        resid = norm(A' * r - λ^2 * x_lq) / norm(b)
        @test(resid ≤ lslq_tol)

        # Test saddle-point systems
        A, b, D = saddle_point(FC=FC)
        D⁻¹ = inv(D)
        (x_lq, stats) = lslq(A, b, M=D⁻¹, transfer_to_lsqr=transfer_to_lsqr)
        r = D⁻¹ * (b - A * x_lq)
        resid = norm(A' * r) / norm(b)
        @test(resid ≤ lslq_tol)

        # Test symmetric and quasi-definite systems
        A, b, M, N = sqd(FC=FC)
        M⁻¹ = inv(M)
        N⁻¹ = inv(N)
        (x_lq, stats) = lslq(A, b, M=M⁻¹, N=N⁻¹, sqd=true, transfer_to_lsqr=transfer_to_lsqr)
        r = M⁻¹ * (b - A * x_lq)
        resid = norm(A' * r - N * x_lq) / norm(b)
        @test(resid ≤ lslq_tol)

        λ = 4.0
        (x_lq, stats) = lslq(A, b, M=M⁻¹, N=N⁻¹, λ=λ, transfer_to_lsqr=transfer_to_lsqr)
        r = M⁻¹ * (b - A * x_lq)
        resid = norm(A' * r - λ^2 * N * x_lq) / norm(b)
        @test(resid ≤ lslq_tol)

        # Test dimension of additional vectors
        for transpose ∈ (false, true)
          A, b, c, D = small_sp(transpose, FC=FC)
          D⁻¹ = inv(D)
          (x_lq, stats) = lslq(A, b, M=D⁻¹, transfer_to_lsqr=transfer_to_lsqr)

          A, b, c, M, N = small_sqd(transpose, FC=FC)
          M⁻¹ = inv(M)
          N⁻¹ = inv(N)
          (x_lq, stats) = lslq(A, b, M=M⁻¹, N=N⁻¹, sqd=true, transfer_to_lsqr=transfer_to_lsqr)
        end
        # test callback function
        A, b, M = saddle_point(FC=FC)
        M⁻¹ = inv(M)
        solver = LslqSolver(A, b)
        tol = 1.0e-1
        cb_n2 = TestCallbackN2LS(A, b, zero(eltype(b)), tol = tol)
        lslq!(solver, A, b, M=M⁻¹, callback = cb_n2)
        @test solver.stats.status == "user-requested exit"
        @test cb_n2(solver)

        @test_throws TypeError lslq(A, b, M=M⁻¹, callback = solver -> "string", history = true)
      end
    end
  end
end
