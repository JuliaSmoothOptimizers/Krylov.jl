@testset "diom" begin
  diom_tol = 1.0e-6

  for FC in (Float64, ComplexF64)
    @testset "Data Type: $FC" begin

      # Symmetric and positive definite system.
      A, b = symmetric_definite(FC=FC)
      (x, stats) = diom(A, b)
      r = b - A * x
      resid = norm(r) / norm(b)
      @test(resid ≤ diom_tol)
      @test(stats.solved)

      # Symmetric indefinite variant.
      A, b = symmetric_indefinite(FC=FC)
      (x, stats) = diom(A, b)
      r = b - A * x
      resid = norm(r) / norm(b)
      @test(resid ≤ diom_tol)
      @test(stats.solved)

      # Nonsymmetric and positive definite systems.
      A, b = nonsymmetric_definite(FC=FC)
      (x, stats) = diom(A, b)
      r = b - A * x
      resid = norm(r) / norm(b)
      @test(resid ≤ diom_tol)
      @test(stats.solved)

      # Nonsymmetric indefinite variant.
      A, b = nonsymmetric_indefinite(FC=FC)
      (x, stats) = diom(A, b)
      r = b - A * x
      resid = norm(r) / norm(b)
      @test(resid ≤ diom_tol)
      @test(stats.solved)

      # Sparse Laplacian.
      A, b = sparse_laplacian(FC=FC)
      (x, stats) = diom(A, b)
      r = b - A * x
      resid = norm(r) / norm(b)
      @test(resid ≤ diom_tol)
      @test(stats.solved)

      # Symmetric indefinite variant, almost singular.
      A, b = almost_singular(FC=FC)
      (x, stats) = diom(A, b, reorthogonalization=true)
      r = b - A * x
      resid = norm(r) / norm(b)
      @test(resid ≤ diom_tol)
      @test(stats.solved)

      # Test b == 0
      A, b = zero_rhs(FC=FC)
      (x, stats) = diom(A, b)
      @test norm(x) == 0
      @test stats.status == "x is a zero-residual solution"

      # Poisson equation in polar coordinates.
      A, b = polar_poisson(FC=FC)
      (x, stats) = diom(A, b, memory=150)
      r = b - A * x
      resid = norm(r) / norm(b)
      @test(resid ≤ diom_tol)
      @test(stats.solved)

      # Test with Jacobi (or diagonal) preconditioner
      A, b, M = square_preconditioned(FC=FC)
      (x, stats) = diom(A, b, M=M)
      r = b - A * x
      resid = norm(M * r) / norm(M * b)
      @test(resid ≤ diom_tol)
      @test(stats.solved)

      # Right preconditioning
      A, b, N = square_preconditioned(FC=FC)
      (x, stats) = diom(A, b, N=N)
      r = b - A * x
      resid = norm(r) / norm(b)
      @test(resid ≤ diom_tol)
      @test(stats.solved)

      # Split preconditioning
      A, b, M, N = two_preconditioners(FC=FC)
      (x, stats) = diom(A, b, M=M, N=N)
      r = b - A * x
      resid = norm(M * r) / norm(M * b)
      @test(resid ≤ diom_tol)
      @test(stats.solved)

      # Inner product ⟨x, y⟩_W = xᴴWy
      A, b = nonsymmetric_indefinite(FC=FC)
      n = length(b)
      s = [10.0^((-1)^i * (i % 7)) for i = 1:n]
      W = Diagonal(1 ./ s.^2)
      Wnorm(r) = sqrt(real(dot(r, W * r)))
      # (k ≤ memory: beyond, the incomplete orthogonalization amplifies rounding errors)
      for k in (1, 3, 5)
        (x, stats) = diom(A, b, W=W, itmax=k, memory=5, atol=0.0, rtol=0.0, history=true)
        # The Krylov basis in the inner product of W = S⁻² is the one of S⁻¹AS y = S⁻¹b with x = Sy
        (y, _) = diom(A, b, M=Diagonal(1 ./ s), N=Diagonal(s), itmax=k, memory=5, atol=0.0, rtol=0.0)
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
      for MM in (I, M), reorthogonalization in (false, true)
        (x, stats) = diom(A, b, x0, M=MM, W=W, reorthogonalization=reorthogonalization, history=true)
        @test stats.residuals[1] ≈ Wnorm_dense(MM * (b - A * x0))
        @test Wnorm_dense(MM * (b - A * x)) ≤ diom_tol * Wnorm_dense(MM * b)
        @test(stats.solved)
      end
      if FC == Float64
        @test_throws ErrorException diom(A, b, W=W, radius=1.0)
      end

      # test callback function
      workspace = DiomWorkspace(A, b)
      tol = 1.0e-1
      cb_n2 = TestCallbackN2(A, b, tol = tol)
      diom!(workspace, A, b, callback = cb_n2)
      @test workspace.stats.status == "user-requested exit"
      @test cb_n2(workspace)
      
      if FC == Float64
        # trust region tests, same as in test_cg
        # Test radius > 0  and b^T * A * b = 0
        A, b = zero_rhs(FC=FC)
        solver = DiomWorkspace(A, b)
        diom!(solver, A, b,radius = 10 * real(one(FC)))
        x, stats = solver.x, solver.stats
        @test stats.status == "x is a zero-residual solution"
        @test norm(x) == zero(FC)
        @test stats.niter == 0

        # Test radius > 0 and pᵀAp < 0
        A = FC[
          10.0 0.0 0.0 0.0;
          0.0 8.0 0.0 0.0;
          0.0 0.0 5.0 0.0;
          0.0 0.0 0.0 -1.0
        ]
        b = FC[1.0, 1.0, 1.0, 0.1]
        solver = DiomWorkspace(A, b)
        diom!(solver, A, b; radius = 10 * real(one(FC)))
        x, stats, = solver.x, solver.stats
        @test stats.indefinite == true
        
        # Test residual of the solution with trust region
        A = FC[
          10.0 0.0 0.0 0.0;
          0.0 8.0 0.0 0.0;
          0.0 0.0 5.0 0.0;
          0.0 0.0 0.0 -1.0
        ]
        b = FC[1.0, 1.0, 1.0, 0.1]
        solver = DiomWorkspace(A, b)
        diom!(solver, A, b; radius = 0.5 * real(one(FC)), history = true)
        x, stats, = solver.x, solver.stats
        r = b - A * x
        normr = norm(r)
        @test isapprox(normr, stats.residuals[end], atol=1.0e-8)
        @test stats.status == "on trust-region boundary"

        # test quadratic function values are computed correctly
        A = FC[10.0 0.0 0.0 0.0;
          0.0 8.0 0.0 0.0;
          0.0 0.0 5.0 0.0;
          0.0 0.0 0.0 1.0
        ]
        b = FC[1.0, 1.0, 1.0, 0.1]
        solver = DiomWorkspace(A, b)
        diom!(solver, A, b; radius = 10 * real(one(FC)), history = true)
        x, stats, = solver.x, solver.stats
        qxs = stats.qvals
        q = -dot(b, x) + dot(x, A * x) / 2
        @test length(qxs) == stats.niter + 1
        @test abs(qxs[end] - q) ≤ 1.0e-10
        @test abs(qxs[1]) ≤ 1.0e-10  # q(0) = 0
        # test that q is decreasing
        @test all(diff(qxs) .<= 1.0e-10)

        # test quadratic function with trust-region
        A = FC[
          10.0 0.0 0.0 0.0;
          0.0 8.0 0.0 0.0;
          0.0 0.0 5.0 0.0;
          0.0 0.0 0.0 -1.0
        ]
        b = FC[1.0, 1.0, 1.0, 0.1]
        solver = DiomWorkspace(A, b)
        diom!(solver, A, b; radius = 0.5 * real(one(FC)), history = true)
        x, stats, = solver.x, solver.stats
        q = -dot(b, x) + dot(x, A * x) / 2
        qxs = stats.qvals
        @test abs(q - qxs[end]) ≤ 1.0e-10
        @test stats.status == "on trust-region boundary"

        # test trust-region with warm start
        A = FC[
          10.0 0.0 0.0 0.0;
          0.0 8.0 0.0 0.0;
          0.0 0.0 5.0 0.0;
          0.0 0.0 0.0 -1.0
        ]
        b = FC[1.0, 1.0, 1.0, 0.1]
        x0 = FC[0.5, 0.5, 0.5, 0.05]
        solver = DiomWorkspace(A, b)
        diom!(solver, A, b, x0; radius = 0.5 * real(one(FC)), history = true)
        x, stats, = solver.x, solver.stats
        q = -dot(b, x0) + dot(x0, A * x0) / 2
        qxs = stats.qvals
        @test abs(q - qxs[1]) ≤ 1.0e-10
        r = b - A * x
        normr = norm(r)
        @test isapprox(normr, stats.residuals[end], atol=1.0e-8)
        @test stats.status == "on trust-region boundary"
      end
    
      @test_throws TypeError diom(A, b, callback = workspace -> "string", history = true)
    end
  end
end
