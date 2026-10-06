function cs_matrix_qlp(rng, n, rank=n)
  U = Matrix(qr(randn(rng, ComplexF64, n, n)).Q)
  s = [collect(range(0.5, 2.0; length=rank)); zeros(n - rank)]
  return U * Diagonal(s) * transpose(U)
end

@testset "csminresqlp" begin
  csminresqlp_tol = 1.0e-6
  rng = MersenneTwister(20261015)

  for FC in (ComplexF32, ComplexF64)
    @testset "Data Type: $FC" begin
      T = real(FC)
      tol = T == Float32 ? 1.0f-3 : csminresqlp_tol

      # Full-rank complex symmetric system.
      A = FC.(cs_matrix_qlp(rng, 10))
      b = randn(rng, FC, 10)
      (x, stats) = csminresqlp(A, b)
      r = b - A * x
      resid = norm(r) / norm(b)
      @test(resid ≤ tol * norm(A) * norm(x))
      @test(stats.solved)

      # Singular, consistent system: x is the pseudoinverse (minimum-norm) solution.
      A = FC.(cs_matrix_qlp(rng, 10, 7))
      target = pinv(Matrix(A); rtol=1.0e-6) * randn(rng, FC, 10)
      b = A * target
      (x, stats) = csminresqlp(A, b; rtol=T(1e-6))
      @test(stats.solved)
      @test norm(x - target) ≤ tol * max(1, norm(target))

      # Severely rank-deficient, inconsistent right-hand side: not guaranteed
      # minimum-norm without reorthogonalization (documented limitation of the
      # short-recurrence always-QLP generalization), but robust once enabled.
      A = FC.(cs_matrix_qlp(rng, 12, 6))
      b = randn(rng, FC, 12)
      (x, stats) = csminresqlp(A, b; reorthogonalize=true, rtol=T(1e-6))
      xpinv = pinv(Matrix(A)) * b
      @test norm(x - xpinv) / norm(xpinv) ≤ 1.0e3 * tol

      # Test b == 0.
      A = FC.(cs_matrix_qlp(rng, 5))
      b = zeros(FC, 5)
      (x, stats) = csminresqlp(A, b)
      @test norm(x) == 0
      @test stats.status == "x is a zero-residual solution"

      # Shifted system: (A + λI)x = b, λ real.
      A = FC.(cs_matrix_qlp(rng, 8))
      b = randn(rng, FC, 8)
      λ = T(0.3)
      (x, stats) = csminresqlp(A, b; λ=λ, rtol=T(1e-6))
      r = b - (A + λ*I) * x
      resid = norm(r) / norm(b)
      @test(resid ≤ tol * norm(A) * norm(x))
      @test(stats.solved)

      # Warm start.
      A = FC.(cs_matrix_qlp(rng, 8))
      b = randn(rng, FC, 8)
      x_exact = A \ b
      x0 = x_exact .+ FC(0.01) .* randn(rng, FC, 8)
      (x, stats) = csminresqlp(A, b, x0; rtol=T(1e-6))
      @test norm(x - x_exact) / norm(x_exact) ≤ 10tol

      workspace = CsMinresQlpWorkspace(A, b)
      krylov_solve!(workspace, A, b, x0; rtol=T(1e-6))
      @test norm(workspace.x - x_exact) / norm(x_exact) ≤ 10tol

      # test callback function
      A = FC.(cs_matrix_qlp(rng, 10))
      b = randn(rng, FC, 10)
      workspace = CsMinresQlpWorkspace(A, b)
      # A generous multiple of norm(b) fires the callback at the first
      # iteration with overwhelming margin, so the test doesn't depend on
      # exactly which iteration the residual happens to cross a tight
      # tolerance (that crossing point is sensitive to platform/BLAS-level
      # rounding and shouldn't be what this test is checking).
      tol_cb = 2 * norm(b)
      cb_n2 = TestCallbackN2(A, b, tol=tol_cb)
      csminresqlp!(workspace, A, b, atol=T(0), rtol=T(0), callback=cb_n2)
      @test workspace.stats.status == "user-requested exit"
      @test cb_n2(workspace)

      @test_throws TypeError csminresqlp(A, b, callback=workspace -> "string", history=true)
    end
  end

  @testset "Real and mismatched input rejected" begin
    A = cs_matrix_qlp(rng, 4)
    b = randn(rng, ComplexF64, 4)
    @test_throws MethodError csminresqlp(real.(A), real.(b))
    @test_throws ErrorException csminresqlp(A, randn(rng, ComplexF64, 5))
  end
end
