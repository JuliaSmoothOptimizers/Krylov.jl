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
      # A = FC[
      #   10.0 0.0 0.0 0.0;
      #   0.0 8.0 0.0 0.0;
      #   0.0 0.0 5.0 0.0;
      #   0.0 0.0 0.0 -1.0
      # ]
      # b = FC[1.0, 1.0, 1.0, 0.1]
      workspace = MinresQlpWorkspace(A, b)
      minres_qlp!(workspace, A, b, linesearch=true)
      x, stats, npc_dir = workspace.x, workspace.stats, workspace.npc_dir
      print(stats.niter, " ", stats.status, "\n")
      @test stats.status == "nonpositive curvature"
      @test stats.indefinite == true
      # Verify that the returned direction indeed exhibits nonpositive curvature.
      # For both real and complex cases, ensure to take the real part.
      @test real(dot(npc_dir, A * npc_dir)) <= 0

      # # Test Linesearch which would stop on the first call since A is negative definite
      # A, b = symmetric_indefinite(FC=FC; shift = 5)
      # workspace = MinresQlpWorkspace(A, b)
      # minres_qlp!(workspace, A, b, linesearch=true)
      # x, stats, npc_dir = workspace.x, workspace.stats, workspace.npc_dir
      # @test stats.status == "nonpositive curvature"
      # @test stats.niter == 0
      # @test all(x .== b)
      # @test stats.solved == true
      # @test stats.indefinite == true
      # @test stats.npcCount == 1
      # @test real(dot(npc_dir, A * npc_dir)) <= 0
      

      # # Test when b^TAb=0 and linesearch is true
      # A, b = system_zero_quad(FC=FC)
      # workspace = MinresQlpWorkspace(A, b)
      # minres_qlp!(workspace, A, b, linesearch=true)
      # x, stats, npc_dir = workspace.x, workspace.stats, workspace.npc_dir
      # @test stats.status == "nonpositive curvature"
      # @test all(x .== b)
      # @test stats.solved == true
      # @test stats.indefinite == true
      # @test real(dot(npc_dir, A * npc_dir)) ≈ 0.0

      # Test if warm_start and linesearch are both true, it should throw an error
      A, b = symmetric_indefinite(FC=FC)
      @test_throws MethodError minres_qlp(A, b, warm_start = true, linesearch = true)

      @test_throws TypeError minres_qlp(A, b, callback = workspace -> "string", history = true)
    end
  end
end
