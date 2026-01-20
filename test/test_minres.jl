@testset "minres" begin
  minres_tol = 1.0e-6

  for FC in (Float64, ComplexF64)
    @testset "Data Type: $FC" begin

      # Cubic spline matrix.
      A, b = symmetric_definite(FC=FC)
      (x, stats) = minres(A, b)
      r = b - A * x
      resid = norm(r) / norm(b)
      @test(resid ≤ minres_tol * norm(A) * norm(x))
      @test(stats.solved)

      # Symmetric indefinite variant.
      A, b = symmetric_indefinite(FC=FC)
      (x, stats) = minres(A, b)
      r = b - A * x
      resid = norm(r) / norm(b)
      @test(resid ≤ minres_tol * norm(A) * norm(x))
      @test(stats.solved)

      # Sparse Laplacian.
      A, b = sparse_laplacian(FC=FC)
      (x, stats) = minres(A, b)
      r = b - A * x
      resid = norm(r) / norm(b)
      @test(resid ≤ minres_tol * norm(A) * norm(x))
      @test(stats.solved)

      # Symmetric indefinite variant, almost singular.
      A, b = almost_singular(FC=FC)
      (x, stats) = minres(A, b)
      r = b - A * x
      resid = norm(r) / norm(b)
      @test(resid ≤ minres_tol * norm(A) * norm(x))
      @test(stats.solved)

      # Test b == 0
      A, b = zero_rhs(FC=FC)
      (x, stats) = minres(A, b)
      @test norm(x) == 0
      @test stats.status == "x is a zero-residual solution"

      # Shifted system
      A, b = symmetric_indefinite(FC=FC)
      λ = 2.0
      (x, stats) = minres(A, b, λ=λ)
      r = b - (A + λ*I) * x
      resid = norm(r) / norm(b)
      @test(resid ≤ minres_tol * norm(A) * norm(x))
      @test(stats.solved)

      # test with Jacobi (or diagonal) preconditioner and history = true
      A, b, M = square_preconditioned(FC=FC)
      (x, stats) = minres(A, b, M=M, history=true)
      r = b - A * x
      resid = norm(r) / norm(b)
      @test(resid ≤ minres_tol * norm(A) * norm(x))
      @test(stats.solved)
      @test(length(stats.residuals) > 0)

      # in-place minres (minres!) with Jacobi (or diagonal) preconditioner
      A, b, M = square_preconditioned(FC=FC)
      workspace = MinresWorkspace(A, b)
      minres!(workspace, A, b, M=M)
      r = b - A * workspace.x
      resid = norm(r) / norm(b)
      @test(resid ≤ minres_tol * norm(A) * norm(x))
      @test(stats.solved)
      @test stats.indefinite == false

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
      solver = MinresWorkspace(A, b)
      minres!(solver, A, b; linesearch=true)
      
      # Verify the "npc" state was recorded
      @test solver.stats.npcCount == 2
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
      minres!(solver, A, b; linesearch=true)

      # Verify the RESET works
      @test solver.stats.npcCount == 0
      @test solver.stats.indefinite == false
      @test solver.stats.solved == true

      # Test linesearch
      A, b = symmetric_indefinite(FC=FC)
      workspace = MinresWorkspace(A, b)
      minres!(workspace, A, b, linesearch=true)
      x, stats, npc_dir = workspace.x, workspace.stats, workspace.npc_dir
      @test stats.status == "nonpositive curvature"
      @test stats.indefinite == true
      # Verify that the returned direction indeed exhibits nonpositive curvature.
      # For both real and complex cases, ensure to take the real part.
      @test real(dot(npc_dir, A * npc_dir)) <= 0

      # Test Linesearch which would stop on the first call since A is negative definite
      A, b = symmetric_indefinite(FC=FC; shift = 5)
      workspace = MinresWorkspace(A, b)
      minres!(workspace, A, b, linesearch=true)
      x, stats, npc_dir = workspace.x, workspace.stats, workspace.npc_dir
      @test stats.status == "nonpositive curvature"
      @test stats.niter == 1 # in Minres they add 1 to the number of iterations first step
      @test all(x .== b)
      @test stats.solved == true
      @test stats.indefinite == true
      @test stats.npcCount == 1
      @test real(dot(npc_dir, A * npc_dir)) <= 0

      # Test when b^TAb=0 and linesearch is true
      A, b = system_zero_quad(FC=FC)
      workspace = MinresWorkspace(A, b)
      minres!(workspace, A, b, linesearch=true)
      x, stats, npc_dir = workspace.x, workspace.stats, workspace.npc_dir
      @test stats.status == "nonpositive curvature"
      @test all(x .== b)
      @test stats.solved == true
      @test stats.indefinite == true
      @test real(dot(npc_dir, A * npc_dir)) ≈ 0.0

      # negative curvature
      A = FC[
        10.0 0.0 0.0 0.0;
        0.0 8.0 0.0 0.0;
        0.0 0.0 5.0 0.0;
        0.0 0.0 0.0 -1.0
      ]
      b = FC[1.0, 1.0, 1.0, 0.1]
      solver = MinresWorkspace(A, b)
      minres!(solver, A, b; linesearch = true)
      x, stats, npc_dir, w1 = solver.x, solver.stats, solver.npc_dir, solver.w1
      @test stats.npcCount == 2
      @test real(dot(npc_dir, A*npc_dir)) ≤ norm(npc_dir)^2 + minres_tol
      @test real(dot(w1, A*w1)) < minres_tol

      # Test if warm_start and linesearch are both true, it should throw an error
      A, b = symmetric_indefinite(FC=FC)
      @test_throws MethodError minres(A, b, warm_start = true, linesearch = true)          

      # test callback function
      workspace = MinresWorkspace(A, b)
      storage_vec = similar(b, size(A, 1))
      tol = 1.0e-1
      cb_n2 = TestCallbackN2(A, b, tol = tol)
      x, stats = minres(A, b, callback = workspace -> cb_n2(workspace)) # n = 10
      minres!(workspace, A, b, callback = cb_n2)
      @test workspace.stats.status == "user-requested exit"
      @test cb_n2(workspace)

      @test_throws TypeError minres(A, b, callback = workspace -> "string", history = true)
    end
  end
end
