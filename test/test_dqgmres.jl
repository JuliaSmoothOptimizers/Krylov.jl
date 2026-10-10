@testset "dqgmres" begin
  dqgmres_tol = 1.0e-6

  for FC in (Float64, ComplexF64)
    @testset "Data Type: $FC" begin

      # Symmetric and positive definite system.
      A, b = symmetric_definite(FC=FC)
      (x, stats) = dqgmres(A, b)
      r = b - A * x
      resid = norm(r) / norm(b)
      @test(resid ≤ dqgmres_tol)
      @test(stats.solved)

      # Symmetric indefinite variant.
      A, b = symmetric_indefinite(FC=FC)
      (x, stats) = dqgmres(A, b)
      r = b - A * x
      resid = norm(r) / norm(b)
      @test(resid ≤ dqgmres_tol)
      @test(stats.solved)

      # Nonsymmetric and positive definite systems.
      A, b = nonsymmetric_definite(FC=FC)
      (x, stats) = dqgmres(A, b)
      r = b - A * x
      resid = norm(r) / norm(b)
      @test(resid ≤ dqgmres_tol)
      @test(stats.solved)

      # Nonsymmetric indefinite variant.
      A, b = nonsymmetric_indefinite(FC=FC)
      (x, stats) = dqgmres(A, b)
      r = b - A * x
      resid = norm(r) / norm(b)
      @test(resid ≤ dqgmres_tol)
      @test(stats.solved)

      # Sparse Laplacian.
      A, b = sparse_laplacian(FC=FC)
      (x, stats) = dqgmres(A, b)
      r = b - A * x
      resid = norm(r) / norm(b)
      @test(resid ≤ dqgmres_tol)
      @test(stats.solved)

      # Symmetric indefinite variant, almost singular.
      A, b = almost_singular(FC=FC)
      (x, stats) = dqgmres(A, b, reorthogonalization=true)
      r = b - A * x
      resid = norm(r) / norm(b)
      @test(resid ≤ dqgmres_tol)
      @test(stats.solved)

      # Test b == 0
      A, b = zero_rhs(FC=FC)
      (x, stats) = dqgmres(A, b)
      @test norm(x) == 0
      @test stats.status == "x is a zero-residual solution"

      # Poisson equation in polar coordinates.
      A, b = polar_poisson(FC=FC)
      (x, stats) = dqgmres(A, b, memory=200)
      r = b - A * x
      resid = norm(r) / norm(b)
      @test(resid ≤ dqgmres_tol)
      @test(stats.solved)

      # Test with Jacobi (or diagonal) preconditioner
      A, b, M = square_preconditioned(FC=FC)
      (x, stats) = dqgmres(A, b, M=M)
      r = b - A * x
      resid = norm(M * r) / norm(M * b)
      @test(resid ≤ dqgmres_tol)
      @test(stats.solved)

      # Right preconditioning
      A, b, N = square_preconditioned(FC=FC)
      (x, stats) = dqgmres(A, b, N=N)
      r = b - A * x
      resid = norm(r) / norm(b)
      @test(resid ≤ dqgmres_tol)
      @test(stats.solved)

      # Split preconditioning
      A, b, M, N = two_preconditioners(FC=FC)
      (x, stats) = dqgmres(A, b, M=M, N=N)
      r = b - A * x
      resid = norm(M * r) / norm(M * b)
      @test(resid ≤ dqgmres_tol)
      @test(stats.solved)

      # Inner product ⟨x, y⟩_W = xᴴWy
      A, b = nonsymmetric_indefinite(FC=FC)
      n = length(b)
      s = [10.0^((-1)^i * (i % 7)) for i = 1:n]
      W = Diagonal(1 ./ s.^2)
      Wnorm(r) = sqrt(real(dot(r, W * r)))
      # (k ≤ memory: beyond, the incomplete orthogonalization amplifies rounding errors)
      for k in (1, 3, 5)
        (x, stats) = dqgmres(A, b, W=W, itmax=k, memory=5, atol=0.0, rtol=0.0, history=true)
        # The Krylov basis in the inner product of W = S⁻² is the one of S⁻¹AS y = S⁻¹b with x = Sy
        (y, _) = dqgmres(A, b, M=Diagonal(1 ./ s), N=Diagonal(s), itmax=k, memory=5, atol=0.0, rtol=0.0)
        @test norm((x - y) ./ s) ≤ 1.0e-10 * norm(y ./ s)
        @test stats.residuals[1] ≈ Wnorm(b)
      end

      # Dense inner product with and without left preconditioning, warm start and reorthogonalization
      A, b, M = square_preconditioned(FC=FC)
      n = length(b)
      B = FC[sin(i + 2j) for i = 1:n, j = 1:n]
      W = B' * B + I
      Wnorm_dense(r) = sqrt(real(dot(r, W * r)))
      x0 = FC[cos(i) for i = 1:n]
      for MM in (I, M), reorthogonalization in (false, true)
        (x, stats) = dqgmres(A, b, x0, M=MM, W=W, reorthogonalization=reorthogonalization, history=true)
        @test stats.residuals[1] ≈ Wnorm_dense(MM * (b - A * x0))
        @test Wnorm_dense(MM * (b - A * x)) ≤ dqgmres_tol * Wnorm_dense(MM * b)
        @test(stats.solved)
      end

      # test callback function
      A, b = sparse_laplacian(FC=FC)
      workspace = DqgmresWorkspace(A, b)
      tol = 1.0e-1
      cb_n2 = TestCallbackN2(A, b, tol = tol)
      dqgmres!(workspace, A, b, atol = 0.0, rtol = 0.0, callback = cb_n2)
      @test workspace.stats.status == "user-requested exit"
      @test cb_n2(workspace)

      @test_throws TypeError dqgmres(A, b, callback = workspace -> "string", history = true)
    end
  end
end
