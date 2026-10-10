@testset "fom" begin
  fom_tol = 1.0e-6

  for FC in (Float64, ComplexF64)
    @testset "Data Type: $FC" begin

      # Symmetric and positive definite system.
      A, b = symmetric_definite(FC=FC)
      (x, stats) = fom(A, b)
      r = b - A * x
      resid = norm(r) / norm(b)
      @test(resid ≤ fom_tol)
      @test(stats.solved)

      # Symmetric indefinite variant.
      A, b = symmetric_indefinite(FC=FC)
      (x, stats) = fom(A, b)
      r = b - A * x
      resid = norm(r) / norm(b)
      @test(resid ≤ fom_tol)
      @test(stats.solved)

      # Nonsymmetric and positive definite systems.
      A, b = nonsymmetric_definite(FC=FC)
      (x, stats) = fom(A, b)
      r = b - A * x
      resid = norm(r) / norm(b)
      @test(resid ≤ fom_tol)
      @test(stats.solved)

      # Nonsymmetric indefinite variant.
      A, b = nonsymmetric_indefinite(FC=FC)
      (x, stats) = fom(A, b)
      r = b - A * x
      resid = norm(r) / norm(b)
      @test(resid ≤ fom_tol)
      @test(stats.solved)

      # Symmetric indefinite variant, almost singular.
      A, b = almost_singular(FC=FC)
      (x, stats) = fom(A, b)
      r = b - A * x
      resid = norm(r) / norm(b)
      @test(resid ≤ 100 * fom_tol)
      @test(stats.solved)

      # Singular system.
      A, b = square_inconsistent(FC=FC)
      (x, stats) = fom(A, b)
      @test(stats.inconsistent)

      # Test b == 0
      A, b = zero_rhs(FC=FC)
      (x, stats) = fom(A, b)
      @test norm(x) == 0
      @test stats.status == "x is a zero-residual solution"

      # Poisson equation in polar coordinates.
      A, b = polar_poisson(FC=FC)
      (x, stats) = fom(A, b, reorthogonalization=true)
      r = b - A * x
      resid = norm(r) / norm(b)
      @test(resid ≤ fom_tol)
      @test(stats.solved)

      # Left preconditioning
      A, b, M = square_preconditioned(FC=FC)
      (x, stats) = fom(A, b, M=M)
      r = b - A * x
      resid = norm(M * r) / norm(M * b)
      @test(resid ≤ fom_tol)
      @test(stats.solved)

      # Right preconditioning
      A, b, N = square_preconditioned(FC=FC)
      (x, stats) = fom(A, b, N=N)
      r = b - A * x
      resid = norm(r) / norm(b)
      @test(resid ≤ fom_tol)
      @test(stats.solved)

      # Split preconditioning
      A, b, M, N = two_preconditioners(FC=FC)
      (x, stats) = fom(A, b, M=M, N=N)
      r = b - A * x
      resid = norm(M * r) / norm(M * b)
      @test(resid ≤ fom_tol)
      @test(stats.solved)

      # Restart
      for restart ∈ (false, true)
        memory = 10

        A, b = sparse_laplacian(FC=FC)
        (x, stats) = fom(A, b, restart=restart, memory=memory)
        r = b - A * x
        resid = norm(r) / norm(b)
        @test(resid ≤ fom_tol)
        @test(stats.niter > memory)
        @test(stats.solved)

        M = Diagonal(1 ./ diag(A))
        (x, stats) = fom(A, b, M=M, restart=restart, memory=memory)
        r = b - A * x
        resid = norm(M * r) / norm(M * b)
        @test(resid ≤ fom_tol)
        @test(stats.niter > memory)
        @test(stats.solved)

        N = Diagonal(1 ./ diag(A))
        (x, stats) = fom(A, b, N=N, restart=restart, memory=memory)
        r = b - A * x
        resid = norm(r) / norm(b)
        @test(resid ≤ fom_tol)
        @test(stats.niter > memory)
        @test(stats.solved)

        N = Diagonal(1 ./ sqrt.(diag(A)))
        N = Diagonal(1 ./ sqrt.(diag(A)))
        (x, stats) = fom(A, b, M=M, N=N, restart=restart, memory=memory)
        r = b - A * x
        resid = norm(M * r) / norm(M * b)
        @test(resid ≤ fom_tol)
        @test(stats.niter > memory)
        @test(stats.solved)
      end
      
      # Inner product ⟨x, y⟩_W = xᴴWy
      A, b = nonsymmetric_indefinite(FC=FC)
      n = length(b)
      s = [10.0^((-1)^i * (i % 7)) for i = 1:n]
      W = Diagonal(1 ./ s.^2)
      Wnorm(r) = sqrt(real(dot(r, W * r)))
      # (k < n: at k = n, hₙ₊₁.ₙ ≈ 0 and the FOM iterate is dominated by rounding errors)
      for k in (1, 3, 5)
        (x, stats) = fom(A, b, W=W, itmax=k, memory=k, atol=0.0, rtol=0.0, history=true)
        # The Krylov basis in the inner product of W = S⁻² is the one of S⁻¹AS y = S⁻¹b with x = Sy
        (y, _) = fom(A, b, M=Diagonal(1 ./ s), N=Diagonal(s), itmax=k, memory=k, atol=0.0, rtol=0.0)
        @test norm((x - y) ./ s) ≤ 1.0e-10 * norm(y ./ s)
        @test stats.residuals[1] ≈ Wnorm(b)
        @test stats.residuals[end] ≈ Wnorm(b - A * x) atol=1.0e-8 * Wnorm(b)
      end

      # Dense inner product with and without left preconditioning, warm start and reorthogonalization
      A, b, M = square_preconditioned(FC=FC)
      n = length(b)
      B = FC[sin(i + 2j) for i = 1:n, j = 1:n]
      W = B' * B + I
      Wnorm_dense(r) = sqrt(real(dot(r, W * r)))
      x0 = FC[cos(i) for i = 1:n]
      for MM in (I, M), restart in (false, true), reorthogonalization in (false, true)
        (x, stats) = fom(A, b, x0, M=MM, W=W, restart=restart, memory=5, reorthogonalization=reorthogonalization, history=true)
        @test stats.residuals[1] ≈ Wnorm_dense(MM * (b - A * x0))
        @test Wnorm_dense(MM * (b - A * x)) ≤ fom_tol * Wnorm_dense(MM * b)
        @test(stats.solved)
      end

      # test callback function
      @test_throws TypeError fom(A, b, restart = true, callback = workspace -> "string", history = true)
    end
  end
end
