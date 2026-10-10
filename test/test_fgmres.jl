import LinearAlgebra.mul!

mutable struct FlexiblePreconditioner{T,S}
  D::Diagonal{T, S}
  ω::T
end

function mul!(y::Vector, P::FlexiblePreconditioner, x::Vector)
  P.ω = -P.ω
  mul!(y, P.D, x)
  y .*= P.ω
end

@testset "fgmres" begin
  fgmres_tol = 1.0e-6

  for FC in (Float64, ComplexF64)
    @testset "Data Type: $FC" begin

      # Symmetric and positive definite system.
      A, b = symmetric_definite(FC=FC)
      (x, stats) = fgmres(A, b)
      r = b - A * x
      resid = norm(r) / norm(b)
      @test(resid ≤ fgmres_tol)
      @test(stats.solved)

      # Symmetric indefinite variant.
      A, b = symmetric_indefinite(FC=FC)
      (x, stats) = fgmres(A, b)
      r = b - A * x
      resid = norm(r) / norm(b)
      @test(resid ≤ fgmres_tol)
      @test(stats.solved)

      # Nonsymmetric and positive definite systems.
      A, b = nonsymmetric_definite(FC=FC)
      (x, stats) = fgmres(A, b)
      r = b - A * x
      resid = norm(r) / norm(b)
      @test(resid ≤ fgmres_tol)
      @test(stats.solved)

      # Nonsymmetric indefinite variant.
      A, b = nonsymmetric_indefinite(FC=FC)
      (x, stats) = fgmres(A, b)
      r = b - A * x
      resid = norm(r) / norm(b)
      @test(resid ≤ fgmres_tol)
      @test(stats.solved)

      # Symmetric indefinite variant, almost singular.
      A, b = almost_singular(FC=FC)
      (x, stats) = fgmres(A, b)
      r = b - A * x
      resid = norm(r) / norm(b)
      @test(resid ≤ 100 * fgmres_tol)
      @test(stats.solved)

      # Singular system.
      A, b = square_inconsistent(FC=FC)
      (x, stats) = fgmres(A, b)
      r = b - A * x
      Aresid = norm(A' * r) / norm(A' * b)
      @test(Aresid ≤ fgmres_tol)
      @test(stats.inconsistent)

      # Test b == 0
      A, b = zero_rhs(FC=FC)
      (x, stats) = fgmres(A, b)
      @test norm(x) == 0
      @test stats.status == "x is a zero-residual solution"

      # Poisson equation in polar coordinates.
      A, b = polar_poisson(FC=FC)
      (x, stats) = fgmres(A, b, reorthogonalization=true)
      r = b - A * x
      resid = norm(r) / norm(b)
      @test(resid ≤ fgmres_tol)
      @test(stats.solved)

      # Left preconditioning
      A, b, M = square_preconditioned(FC=FC)
      (x, stats) = fgmres(A, b, M=M)
      r = b - A * x
      resid = norm(M * r) / norm(M * b)
      @test(resid ≤ fgmres_tol)
      @test(stats.solved)

      # Right preconditioning
      A, b, N = square_preconditioned(FC=FC)
      (x, stats) = fgmres(A, b, N=N)
      r = b - A * x
      resid = norm(r) / norm(b)
      @test(resid ≤ fgmres_tol)
      @test(stats.solved)

      # Split preconditioning
      A, b, M, N = two_preconditioners(FC=FC)
      (x, stats) = fgmres(A, b, M=M, N=N)
      r = b - A * x
      resid = norm(M * r) / norm(M * b)
      @test(resid ≤ fgmres_tol)
      @test(stats.solved)

      # Inner product ⟨x, y⟩_W = xᴴWy
      A, b = nonsymmetric_indefinite(FC=FC)
      n = length(b)
      s = [10.0^((-1)^i * (i % 7)) for i = 1:n]
      W = Diagonal(1 ./ s.^2)
      Wnorm(r) = sqrt(real(dot(r, W * r)))
      for k in (1, 5, n)
        (x, stats) = fgmres(A, b, W=W, itmax=k, memory=k, atol=0.0, rtol=0.0, history=true)
        # FGMRES in the inner product of W = S⁻² is FGMRES on S⁻¹AS y = S⁻¹b with x = Sy
        (y, _) = fgmres(A, b, M=Diagonal(1 ./ s), N=Diagonal(s), itmax=k, memory=k, atol=0.0, rtol=0.0)
        @test norm((x - y) ./ s) ≤ 1.0e-10 * norm(y ./ s)
        @test stats.residuals[1] ≈ Wnorm(b)
        @test stats.residuals[end] ≈ Wnorm(b - A * x) atol=1.0e-8 * Wnorm(b)
      end

      # After a restart, v₁ is normalized by the recomputed ‖r₀‖_W, not by the estimate
      n = 40
      A = FC.(I + 2 * [sin(i * j + 1) for i = 1:n, j = 1:n] / sqrt(n))
      s = [10.0^((i % 9) - 4) for i = 1:n]
      W = Diagonal(1 ./ s.^2)
      b = FC.(s .* [cos(3i) for i = 1:n])
      workspace = FgmresWorkspace(A, b; memory=3)
      deviation = Ref(0.0)
      callback = workspace -> (workspace.inner_iter == 1 && (deviation[] = max(deviation[], abs(sqrt(real(dot(workspace.V[1], W * workspace.V[1]))) - 1))); false)
      fgmres!(workspace, A, b, W=W, restart=true, itmax=60, atol=0.0, rtol=1.0e-12, callback=callback)
      @test workspace.stats.niter > 3
      @test deviation[] ≤ 1.0e-12

      # Dense inner product with and without left preconditioning, warm start, restart and reorthogonalization
      A, b, M = square_preconditioned(FC=FC)
      n = length(b)
      B = FC[sin(i + 2j) for i = 1:n, j = 1:n]
      W = B' * B + I
      Wnorm_dense(r) = sqrt(real(dot(r, W * r)))
      x0 = FC[cos(i) for i = 1:n]
      for MM in (I, M), restart in (false, true), reorthogonalization in (false, true)
        (x, stats) = fgmres(A, b, x0, M=MM, W=W, restart=restart, reorthogonalization=reorthogonalization, memory=5, history=true)
        @test stats.residuals[1] ≈ Wnorm_dense(MM * (b - A * x0))
        @test Wnorm_dense(MM * (b - A * x)) ≤ fgmres_tol * Wnorm_dense(MM * b)
        @test(stats.solved)
      end

      # Restart
      for restart ∈ (false, true)
        memory = 10

        A, b = sparse_laplacian(FC=FC)
        (x, stats) = fgmres(A, b, restart=restart, memory=memory)
        r = b - A * x
        resid = norm(r) / norm(b)
        @test(resid ≤ fgmres_tol)
        @test(stats.niter > memory)
        @test(stats.solved)

        M = Diagonal(1 ./ diag(A))
        (x, stats) = fgmres(A, b, M=M, restart=restart, memory=memory)
        r = b - A * x
        resid = norm(M * r) / norm(M * b)
        @test(resid ≤ fgmres_tol)
        @test(stats.niter > memory)
        @test(stats.solved)

        N = Diagonal(1 ./ diag(A))
        (x, stats) = fgmres(A, b, N=N, restart=restart, memory=memory)
        r = b - A * x
        resid = norm(r) / norm(b)
        @test(resid ≤ fgmres_tol)
        @test(stats.niter > memory)
        @test(stats.solved)

        N = Diagonal(1 ./ sqrt.(diag(A)))
        N = Diagonal(1 ./ sqrt.(diag(A)))
        (x, stats) = fgmres(A, b, M=M, N=N, restart=restart, memory=memory)
        r = b - A * x
        resid = norm(M * r) / norm(M * b)
        @test(resid ≤ fgmres_tol)
        @test(stats.niter > memory)
        @test(stats.solved)
      end

      A, b = polar_poisson(FC=FC)
      J = inv(Diagonal(A))  # Jacobi preconditioner
      N = FlexiblePreconditioner(J, 1.0)
      (x, stats) = fgmres(A, b, N=N)
      r = b - A * x
      resid = norm(r) / norm(b)
      @test(resid ≤ fgmres_tol)
      @test(stats.solved)

      # Flexible preconditioning with an inner product
      W = Diagonal(FC[1 + i % 3 for i = 1:length(b)])
      N = FlexiblePreconditioner(J, 1.0)
      (x, stats) = fgmres(A, b, N=N, W=W)
      r = b - A * x
      @test sqrt(real(dot(r, W * r))) ≤ fgmres_tol * sqrt(real(dot(b, W * b)))
      @test(stats.solved)
    end
  end
end
