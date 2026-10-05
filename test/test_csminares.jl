function cs_matrix(rng, n, rank=n)
  U = Matrix(qr(randn(rng, ComplexF64, n, n)).Q)
  s = [collect(range(0.5, 2.0; length=rank)); zeros(n - rank)]
  return U * Diagonal(s) * transpose(U)
end

@testset "csminares" begin
  csminares_tol = 1.0e-6
  rng = MersenneTwister(20261015)

  for FC in (ComplexF32, ComplexF64)
    @testset "Data Type: $FC" begin
      T = real(FC)
      tol = T == Float32 ? 1.0f-3 : csminares_tol

      # Full-rank complex symmetric system.
      A = FC.(cs_matrix(rng, 10))
      b = randn(rng, FC, 10)
      (x, stats) = csminares(A, b)
      r = b - A * x
      resid = norm(r) / norm(b)
      @test(resid ≤ tol * norm(A) * norm(x))
      @test(stats.solved)

      # Singular, consistent system: x is the pseudoinverse solution.
      A = FC.(cs_matrix(rng, 10, 7))
      target = pinv(Matrix(A); rtol=1.0e-6) * randn(rng, FC, 10)
      b = A * target
      (x, stats) = csminares(A, b; rtol=T(1e-6), Artol=T(1e-6))
      @test(stats.solved)
      @test norm(x - target) ≤ tol * max(1, norm(target))

      # Test b == 0.
      A = FC.(cs_matrix(rng, 5))
      b = zeros(FC, 5)
      (x, stats) = csminares(A, b)
      @test norm(x) == 0
      @test stats.status == "x is a zero-residual solution"

      # A*b == 0 but A'*b != 0: the adjoint, not A itself, drives the objective.
      A2 = FC[1 im; im -1]
      b2 = FC[1, im]
      @test norm(A2 * b2) == 0
      @test norm(A2' * b2) > 0
      (x, stats) = csminares(A2, b2)
      @test x ≈ pinv(Matrix(A2)) * b2 atol=tol
      @test stats.niter == 1

      # Shifted system: (A + λI)x = b.
      A = FC.(cs_matrix(rng, 8))
      b = randn(rng, FC, 8)
      λ = FC(0.3 + 0.1im)
      (x, stats) = csminares(A, b; λ=λ, rtol=T(1e-6))
      r = b - (A + λ*I) * x
      resid = norm(r) / norm(b)
      @test(resid ≤ tol * norm(A) * norm(x))
      @test(stats.solved)

      # Warm start.
      A = FC.(cs_matrix(rng, 8))
      b = randn(rng, FC, 8)
      x_exact = A \ b
      x0 = x_exact .+ FC(0.01) .* randn(rng, FC, 8)
      (x, stats) = csminares(A, b, x0; rtol=T(1e-6))
      @test norm(x - x_exact) / norm(x_exact) ≤ 10tol

      workspace = CsMinaresWorkspace(A, b)
      krylov_solve!(workspace, A, b, x0; rtol=T(1e-6))
      @test norm(workspace.x - x_exact) / norm(x_exact) ≤ 10tol

      # Preconditioning is not yet supported: M must be the identity.
      @test_throws ErrorException csminares(A, b, M=2I)

      # test callback function
      A = FC.(cs_matrix(rng, 10))
      b = randn(rng, FC, 10)
      workspace = CsMinaresWorkspace(A, b)
      tol_cb = T(1.0)
      cb_n2 = TestCallbackN2(A, b, tol=tol_cb)
      csminares!(workspace, A, b, atol=T(0), rtol=T(0), Artol=T(0), callback=cb_n2)
      @test workspace.stats.status == "user-requested exit"
      @test cb_n2(workspace)

      @test_throws TypeError csminares(A, b, callback=workspace -> "string", history=true)
    end
  end

  @testset "Real and mismatched input rejected" begin
    A = cs_matrix(rng, 4)
    b = randn(rng, ComplexF64, 4)
    @test_throws MethodError csminares(real.(A), real.(b))
    @test_throws ErrorException csminares(A, randn(rng, ComplexF64, 5))
  end
end
