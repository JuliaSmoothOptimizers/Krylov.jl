@testset "minares" begin
  minares_tol = 1.0e-6

  for FC in (Float64, ComplexF64)
    @testset "Data Type: $FC" begin

      # Cubic spline matrix.
      A, b = symmetric_definite(FC=FC)
      (x, stats) = minares(A, b)
      r = b - A * x
      resid = norm(r) / norm(b)
      @test(resid ≤ minares_tol * norm(A) * norm(x))
      @test(stats.solved)

      # Symmetric indefinite variant.
      A, b = symmetric_indefinite(FC=FC)
      (x, stats) = minares(A, b)
      r = b - A * x
      resid = norm(r) / norm(b)
      @test(resid ≤ minares_tol * norm(A) * norm(x))
      @test(stats.solved)

      # Sparse Laplacian.
      A, b = sparse_laplacian(FC=FC)
      (x, stats) = minares(A, b)
      r = b - A * x
      resid = norm(r) / norm(b)
      @test(resid ≤ minares_tol * norm(A) * norm(x))
      @test(stats.solved)

      # Symmetric indefinite variant, almost singular.
      A, b = almost_singular(FC=FC)
      (x, stats) = minares(A, b)
      r = b - A * x
      resid = norm(r) / norm(b)
      @test(resid ≤ minares_tol * norm(A) * norm(x))
      @test(stats.solved)

      # Test b == 0
      A, b = zero_rhs(FC=FC)
      (x, stats) = minares(A, b)
      @test norm(x) == 0
      @test stats.status == "x is a zero-residual solution"

      # Singular inconsistent system
      A, b = square_inconsistent(FC=FC)
      (x, stats) = minares(A, b)
      r = b - A * x
      Aresid = norm(A*r) / norm(A*b)
      @test(Aresid ≤ minares_tol)
      # @test stats.inconsistent

      # Symmetric inconsistent system
      A, b = symmetric_inconsistent()
      (x, stats) = minares(A, b)
      r = b - A * x
      Aresid = norm(A*r) / norm(A*b)
      @test(Aresid ≤ minares_tol)
      # @test stats.inconsistent

      # Shifted system
      A, b = symmetric_indefinite(FC=FC)
      λ = 2.0
      (x, stats) = minares(A, b, λ=λ)
      r = b - (A + λ*I) * x
      resid = norm(r) / norm(b)
      @test(resid ≤ minares_tol * norm(A) * norm(x))
      @test(stats.solved)

      # Test with Jacobi (or diagonal) preconditioner
      # A, b, M = square_preconditioned(FC=FC)
      # (x, stats) = minares(A, b, M=M)
      # r = b - A * x
      # resid = sqrt(real(dot(r, M * r))) / sqrt(real(dot(b, M * b)))
      # @test(resid ≤ minares_tol * norm(A) * norm(x))
      # @test(stats.solved)

      # test callback function
      A, b = sparse_laplacian(FC=FC)
      workspace = MinaresWorkspace(A, b)
      tol = 1.0
      cb_n2 = TestCallbackN2(A, b, tol = tol)
      minares!(workspace, A, b, atol = 0.0, rtol = 0.0, Artol = 0.0, callback = cb_n2)
      @test workspace.stats.status == "user-requested exit"
      @test cb_n2(workspace)

      @test_throws TypeError minares(A, b, callback = workspace -> "string", history = true)
    end
  end

  @testset "complex_symmetric" begin
    for FC in (ComplexF64, ComplexF32)
      T = real(FC)
      tol = T === Float32 ? T(1.0e-3) : T(1.0e-6)

      # Nonsingular complex symmetric system.
      A, b = complex_symmetric_definite(FC=FC)
      (x, stats) = minares(A, b; complex_symmetric=true)
      @test norm(x - A \ b) / norm(A \ b) ≤ tol
      @test stats.solved

      # Warm start.
      x0 = (A \ b) .+ FC(0.01) .* ones(FC, size(A, 1))
      (x, stats) = minares(A, b, x0; complex_symmetric=true)
      @test norm(x - A \ b) / norm(A \ b) ≤ tol

      # Real shift.
      λ = T(0.3)
      (x, stats) = minares(A, b; complex_symmetric=true, λ=λ)
      r = b - (A + λ * I) * x
      @test norm(r) / norm(b) ≤ tol * norm(A) * norm(x)
      @test stats.solved

      # Singular, inconsistent complex symmetric system.
      As, bs = complex_symmetric_inconsistent(FC=FC)
      (x, stats) = minares(As, bs; complex_symmetric=true)
      r = bs - As * x
      @test norm(As' * r) / norm(As' * bs) ≤ tol

      # krylov_solve generic interface.
      workspace = krylov_workspace(Val(:minares), A, b)
      krylov_solve!(workspace, A, b; complex_symmetric=true)
      @test norm(workspace.x - A \ b) / norm(A \ b) ≤ tol

      # Verbose output.
      io = IOBuffer()
      minares(A, b; complex_symmetric=true, verbose=1, iostream=io)
      @test occursin("MINARES: system of size", String(take!(io)))

      # Allocation-free in-place path, in its own `let` block for the reason
      # given in test_minres_qlp.jl.
      nbytes = let ws = MinaresWorkspace(A, b)
        minares!(ws, A, b; complex_symmetric=true)
        @allocated minares!(ws, A, b; complex_symmetric=true)
      end
      @test nbytes == 0

      # complex_symmetric=true requires a complex element type.
      Ar, br = symmetric_indefinite(FC=T)
      @test_throws ErrorException minares(Ar, br; complex_symmetric=true)

      # complex_symmetric=true does not support preconditioning.
      @test_throws ArgumentError minares(A, b; complex_symmetric=true, M=Diagonal(ones(FC, size(A, 1))))
    end

    # Singular, consistent complex symmetric system: the minimum-norm solution.
    A, b = complex_symmetric_singular()
    (x, stats) = minares(A, b; complex_symmetric=true)
    @test norm(x - pinv(A) * b) / norm(pinv(A) * b) ≤ 1.0e-6
    @test stats.solved

    # Iterates and both estimates against a dense oracle at every k. Only the
    # ‖rₖ‖ estimate check sees the LQ factorization of Uₖ.
    A, b = complex_symmetric_dense()
    for k = 1:8
      workspace = MinaresWorkspace(A, b)
      minares!(workspace, A, b; complex_symmetric=true, itmax=k, atol=0.0, rtol=0.0, Artol=0.0, history=true)
      x, stats = workspace.x, workspace.stats
      r = b - A * x
      @test norm(x - cs_minares_oracle(A, b, k)) ≤ 1.0e-8 * norm(x)
      @test abs(stats.residuals[end] - norm(r)) ≤ 1.0e-8 * norm(b)
      @test abs(stats.Aresiduals[end] - norm(A' * r)) ≤ 1.0e-8 * norm(A' * b)
    end

    # Complex{BigFloat}.
    A, b = complex_symmetric_definite(FC=Complex{BigFloat})
    (x, stats) = minares(A, b; complex_symmetric=true)
    @test norm(x - Matrix(A) \ b) / norm(Matrix(A) \ b) ≤ sqrt(eps(BigFloat))
    @test stats.solved
  end
end
