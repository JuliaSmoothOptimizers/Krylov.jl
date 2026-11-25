@testset "minres_qlp" begin
  minres_qlp_tol = 1.0e-6

  for FC in (Float64, ComplexF64)
    @testset "Data Type: $FC" begin

      # Cubic spline matrix.
      A, b = symmetric_definite(FC=FC)
      (x, stats) = minres_qlp(A, b)
      r = b - A * x
      resid = norm(r) / norm(b)
      @test(resid ≤ minres_qlp_tol * norm(A) * norm(x))
      @test(stats.solved)

      # Symmetric indefinite variant.
      A, b = symmetric_indefinite(FC=FC)
      (x, stats) = minres_qlp(A, b)
      r = b - A * x
      resid = norm(r) / norm(b)
      @test(resid ≤ minres_qlp_tol * norm(A) * norm(x))
      @test(stats.solved)

      # Sparse Laplacian.
      A, b = sparse_laplacian(FC=FC)
      (x, stats) = minres_qlp(A, b)
      r = b - A * x
      resid = norm(r) / norm(b)
      @test(resid ≤ minres_qlp_tol * norm(A) * norm(x))
      @test(stats.solved)

      # Symmetric indefinite variant, almost singular.
      A, b = almost_singular(FC=FC)
      (x, stats) = minres_qlp(A, b)
      r = b - A * x
      resid = norm(r) / norm(b)
      @test(resid ≤ minres_qlp_tol * norm(A) * norm(x))
      @test(stats.solved)

      # Test b == 0
      A, b = zero_rhs(FC=FC)
      (x, stats) = minres_qlp(A, b)
      @test norm(x) == 0
      @test stats.status == "x is a zero-residual solution"

      # Singular inconsistent system
      A, b = square_inconsistent(FC=FC)
      (x, stats) = minres_qlp(A, b)
      r = b - A * x
      Aresid = norm(A*r) / norm(A*b)
      @test(Aresid ≤ minres_qlp_tol)
      @test stats.inconsistent

      # Symmetric inconsistent system
      A, b = symmetric_inconsistent()
      (x, stats) = minres_qlp(A, b)
      r = b - A * x
      Aresid = norm(A*r) / norm(A*b)
      @test(Aresid ≤ minres_qlp_tol)
      @test stats.inconsistent

      # Shifted system
      A, b = symmetric_indefinite(FC=FC)
      λ = 2.0
      (x, stats) = minres_qlp(A, b, λ=λ)
      r = b - (A + λ*I) * x
      resid = norm(r) / norm(b)
      @test(resid ≤ minres_qlp_tol * norm(A) * norm(x))
      @test(stats.solved)

      # Test with Jacobi (or diagonal) preconditioner
      A, b, M = square_preconditioned(FC=FC)
      (x, stats) = minres_qlp(A, b, M=M)
      r = b - A * x
      resid = sqrt(real(dot(r, M * r))) / norm(b)
      @test(resid ≤ minres_qlp_tol * norm(A) * norm(x))
      @test(stats.solved)

      # test callback function
      A, b = sparse_laplacian(FC=FC)
      workspace = MinresQlpWorkspace(A, b)
      tol = 1.0
      cb_n2 = TestCallbackN2(A, b, tol = tol)
      minres_qlp!(workspace, A, b, atol = 0.0, rtol = 0.0, Artol = 0.0, callback = cb_n2)
      @test workspace.stats.status == "user-requested exit"
      @test cb_n2(workspace)

      # Test linesearch
      A, b = symmetric_indefinite(FC=FC)
      workspace = MinresQlpWorkspace(A, b)
      minres_qlp!(workspace, A, b, linesearch=true)
      x, stats, npc_dir = workspace.x, workspace.stats, workspace.npc_dir
      @test stats.status == "nonpositive curvature"
      @test stats.indefinite == true
      # Verify that the returned direction indeed exhibits nonpositive curvature.
      # For both real and complex cases, ensure to take the real part.
      @test real(dot(npc_dir, A * npc_dir)) <= 0

      # Test linesearch: when nonpositive curvature is detected at iteration k > 1,
      # x must be the iterate xₖ₋₁ and npc_dir the residual rₖ₋₁ = b - Axₖ₋₁.
      for (λneg, bneg) in ((-1.0, 1.0), (-1.0e-2, 1.0e-3))  # NPC detected early / late
        A = diagm(FC[λneg; collect(1.0:19.0)])
        b = FC[bneg; ones(19)]
        workspace = MinresQlpWorkspace(A, b)
        minres_qlp!(workspace, A, b, linesearch=true, atol=0.0, rtol=0.0, Artol=0.0)
        x, stats, npc_dir = workspace.x, workspace.stats, workspace.npc_dir
        @test stats.status == "nonpositive curvature"
        k = stats.niter
        @test k > 1
        xₖ₋₁, _ = minres_qlp(A, b, itmax=k-1, atol=0.0, rtol=0.0, Artol=0.0)
        @test x ≈ xₖ₋₁
        @test npc_dir ≈ b - A * xₖ₋₁
      end

      # Test Linesearch which would stop on the first call since A is negative definite
      A, b = symmetric_indefinite(FC=FC; shift = 5)
      workspace = MinresQlpWorkspace(A, b)
      minres_qlp!(workspace, A, b, linesearch=true)
      x, stats, npc_dir = workspace.x, workspace.stats, workspace.npc_dir
      @test stats.status == "nonpositive curvature"
      @test stats.niter == 1 
      @test all(x .== b)
      @test stats.solved == true
      @test stats.indefinite == true
      @test stats.npcCount == 1
      @test real(dot(npc_dir, A * npc_dir)) <= 0      

      # Test when b^TAb=0 and linesearch is true
      A, b = system_zero_quad(FC=FC)
      workspace = MinresQlpWorkspace(A, b)
      minres_qlp!(workspace, A, b, linesearch=true)
      x, stats, npc_dir = workspace.x, workspace.stats, workspace.npc_dir
      @test stats.status == "nonpositive curvature"
      @test all(x .== b)
      @test stats.solved == true
      @test stats.indefinite == true
      @test real(dot(npc_dir, A * npc_dir)) ≈ 0.0

      # Test if warm_start and linesearch are both true, it should throw an error
      A, b = symmetric_indefinite(FC=FC)
      @test_throws MethodError minres_qlp(A, b, warm_start = true, linesearch = true)

      @test_throws TypeError minres_qlp(A, b, callback = workspace -> "string", history = true)

      # ---------------------------------------------------------------------------
      # Trust-region constraint ‖x‖ ≤ radius
      # ---------------------------------------------------------------------------
      # xⱼ: iterate j of MINRES-QLP without trust region (x₀ = 0)
      xⱼ(A, b, j) = j == 0 ? zero(b) : minres_qlp(A, b, itmax=j, atol=0.0, rtol=0.0, Artol=0.0)[1]
      q(A, b, x) = real(dot(x, A * x)) / 2 - real(dot(b, x))

      # Invalid combinations
      A, b = symmetric_definite(FC=FC)
      @test_throws ErrorException minres_qlp(A, b, radius=one(Float64), linesearch=true)
      @test_throws ErrorException minres_qlp(A, b, zero(b), radius=one(Float64))  # warm start
      A, b, M = square_preconditioned(FC=FC)
      @test_throws ErrorException minres_qlp(A, b, M=M, radius=one(Float64))

      # radius ≥ ‖x*‖: same result as without the constraint
      A, b = symmetric_definite(FC=FC)
      x_free, stats_free = minres_qlp(A, b)
      x, stats = minres_qlp(A, b, radius=2 * norm(x_free))
      @test x ≈ x_free
      @test stats.status == stats_free.status
      @test stats.npcCount == 0

      # Boundary crossed at iteration 1: x = radius * b / ‖b‖
      A, b = symmetric_definite(FC=FC)
      radius = 1.0e-3 * norm(b)
      x, stats = minres_qlp(A, b, radius=radius)
      @test stats.niter == 1
      @test stats.status == "on trust-region boundary"
      @test stats.solved
      @test x ≈ radius * b / norm(b)

      # Boundary crossed at iteration k > 1: x lies on the segment [xₖ₋₁, xₖ] and on the boundary.
      # The radii are chosen so that k ≥ 5, i.e. part of xₖ is already accumulated in x when the boundary is crossed.
      A, b = sparse_laplacian(FC=FC)
      x_free, _ = minres_qlp(A, b)
      for rel in (0.5, 0.9, 0.99)
        radius = rel * norm(x_free)
        x, stats = minres_qlp(A, b, radius=radius, atol=0.0, rtol=0.0)
        k = stats.niter
        @test k ≥ 5
        @test stats.status == "on trust-region boundary"
        @test stats.solved
        @test stats.npcCount == 0
        @test norm(x) ≈ radius
        xₖ₋₁, xₖ = xⱼ(A, b, k-1), xⱼ(A, b, k)
        @test norm(xₖ₋₁) ≤ radius < norm(xₖ)
        d = xₖ - xₖ₋₁
        σ = real(dot(d, x - xₖ₋₁)) / real(dot(d, d))
        @test 0 < σ ≤ 1
        @test x ≈ xₖ₋₁ + σ * d
        @test q(A, b, x) < q(A, b, xₖ₋₁)

        # SPD: MINRES and CR generate the same iterates in exact arithmetic,
        # so they must return the same trust-region step.
        x_cr, stats_cr = cr(A, b, radius=radius, atol=0.0, rtol=0.0)
        @test x ≈ x_cr
        @test stats.niter == stats_cr.niter
      end

      # Nonpositive curvature at iteration 1: x = radius * b / ‖b‖
      A, b = symmetric_indefinite(FC=FC; shift=5)
      radius = 0.1 * norm(b)
      workspace = MinresQlpWorkspace(A, b)
      minres_qlp!(workspace, A, b, radius=radius)
      x, stats, npc_dir = workspace.x, workspace.stats, workspace.npc_dir
      @test stats.niter == 1
      @test stats.npcCount == 1
      @test stats.indefinite
      @test stats.status == "on trust-region boundary"
      @test x ≈ radius * b / norm(b)
      @test real(dot(npc_dir, A * npc_dir)) ≤ 0

      # Nonpositive curvature at iteration k > 1: x = xₖ₋₁ + σ rₖ₋₁ with σ > 0, on the boundary
      for (λneg, bneg) in ((-1.0, 1.0), (-1.0e-2, 1.0e-3))  # NPC detected early / late
        A = diagm(FC[λneg; collect(1.0:19.0)])
        b = FC[bneg; ones(19)]
        radius = 10 * norm(b)
        workspace = MinresQlpWorkspace(A, b)
        minres_qlp!(workspace, A, b, radius=radius, atol=0.0, rtol=0.0, Artol=0.0)
        x, stats, npc_dir = workspace.x, workspace.stats, workspace.npc_dir
        k = stats.niter
        @test k > 1
        @test stats.npcCount == 1
        @test stats.indefinite
        @test stats.status == "on trust-region boundary"
        @test norm(x) ≈ radius
        xₖ₋₁ = xⱼ(A, b, k-1)
        rₖ₋₁ = b - A * xₖ₋₁
        @test npc_dir ≈ rₖ₋₁
        @test real(dot(npc_dir, A * npc_dir)) ≤ 0
        σ = real(dot(rₖ₋₁, x - xₖ₋₁)) / real(dot(rₖ₋₁, rₖ₋₁))
        @test σ > 0
        @test x ≈ xₖ₋₁ + σ * rₖ₋₁
        @test q(A, b, x) < q(A, b, xₖ₋₁)
      end

      # Workspace reuse: trust region, then no trust region, then linesearch
      A, b = symmetric_indefinite(FC=FC; shift=5)
      workspace = MinresQlpWorkspace(A, b)
      minres_qlp!(workspace, A, b, radius=0.1 * norm(b))
      @test workspace.stats.status == "on trust-region boundary"
      A, b = symmetric_definite(FC=FC)
      minres_qlp!(workspace, A, b)
      @test workspace.stats.status == "solution good enough given atol and rtol"
      @test workspace.stats.npcCount == 0
      @test workspace.x ≈ minres_qlp(A, b)[1]

      # Generic interface
      A, b = symmetric_definite(FC=FC)
      radius = 1.0e-3 * norm(b)
      workspace = krylov_workspace(Val(:minres_qlp), A, b)
      krylov_solve!(workspace, A, b, radius=radius)
      @test norm(solution(workspace)) ≈ radius

      # Test: Ensure stats are reset when reusing workspace
      # (pᵀAp < 0)
      A = FC[
        10.0 0.0 0.0 0.0;
        0.0 8.0 0.0 0.0;
        0.0 0.0 5.0 0.0;
        0.0 0.0 0.0 -1.0
      ]
      b = FC[1.0, 1.0, 1.0, 0.1]
      
      # Initialize workspace and solve
      solver = MinresQlpWorkspace(A, b)
      minres_qlp!(solver, A, b; linesearch=true)
      
      # Verify the "npc" state was recorded
      @test solver.stats.npcCount == 1
      @test solver.stats.indefinite == true
      @test solver.stats.status == "nonpositive curvature"

      # Reuse the SAME solver on a Positive Definite System

      A = FC[
        10.0 0.0 0.0 0.0;
        0.0 8.0 0.0 0.0;
        0.0 0.0 5.0 0.0;
        0.0 0.0 0.0 1.0
      ]
      b = FC[1.0, 1.0, 1.0, 1.0]

      # Run the solver again on the same workspace
      minres_qlp!(solver, A, b; linesearch=true)

      # Verify the RESET works
      @test solver.stats.npcCount == 0
      @test solver.stats.indefinite == false
      @test solver.stats.solved == true
    end
  end
end
