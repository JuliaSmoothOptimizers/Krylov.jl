@testset "cg" begin
  cg_tol = 1.0e-6

  for FC in (Float64, ComplexF64)
    @testset "Data Type: $FC" begin

      # Cubic spline matrix.
      A, b = symmetric_definite(FC=FC)
      (x, stats) = cg(A, b, itmax=10)
      r = b - A * x
      resid = norm(r) / norm(b)
      @test(resid ≤ cg_tol)
      @test(stats.solved)

      if FC == Float64
        radius = 0.75 * norm(x)
        (x, stats) = cg(A, b, radius=radius, itmax=10)
        @test(stats.solved)
        @test(abs(radius - norm(x)) ≤ cg_tol * radius)
      end

      # Sparse Laplacian.
      A, b = sparse_laplacian(FC=FC)
      (x, stats) = cg(A, b)
      r = b - A * x
      resid = norm(r) / norm(b)
      @test(resid ≤ cg_tol)
      @test(stats.solved)

      if FC == Float64
        radius = 0.75 * norm(x)
        (x, stats) = cg(A, b, radius=radius, itmax=10)
        @test(stats.solved)
        @test(abs(radius - norm(x)) ≤ cg_tol * radius)
      end

      # Test b == 0
      A, b = zero_rhs(FC=FC)
      (x, stats) = cg(A, b)
      @test norm(x) == 0
      @test stats.status == "x is a zero-residual solution"

      # Test with Jacobi (or diagonal) preconditioner
      A, b, M = square_preconditioned(FC=FC)
      (x, stats) = cg(A, b, M=M)
      r = b - A * x
      resid = sqrt(real(dot(r, M * r))) / sqrt(real(dot(b, M * b)))
      @test(resid ≤ cg_tol)
      @test(stats.solved)

      # Test line search with A indefinite; in this example, CG should stop at the first iteration
      A, b = symmetric_indefinite(FC = FC; shift = 10)
      solver = CgWorkspace(A, b)
      cg!(solver,A, b, linesearch=true)
      x, stats, npc_dir = solver.x, solver.stats, solver.npc_dir
      @test stats.status == "nonpositive curvature detected"
      @test !stats.inconsistent
      @test stats.niter == 0
      @test stats.indefinite == true
      @test stats.npcCount == 1
      @test real(dot(npc_dir, A * npc_dir)) <= 0
      @test all(npc_dir .== b)

      # Test when b^TAb=0 and linesearch is true
      A, b = zero_rhs(FC=FC)
      solver = CgWorkspace(A, b)
      cg!(solver, A, b, linesearch=true)
      x, stats, npc_dir = solver.x, solver.stats, solver.npc_dir
      @test stats.status == "x is a zero-residual solution"
      @test norm(x) == zero(FC)
      @test stats.niter == 0

      # Test radius > 0  and b^TAb=0
      A, b = zero_rhs(FC=FC)
      solver = CgWorkspace(A, b)
      cg!(solver, A, b,radius = 10 * real(one(FC)))
      x, stats, npc_dir = solver.x, solver.stats, solver.npc_dir
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
      solver = CgWorkspace(A, b)
      cg!(solver, A, b; radius = 10 * real(one(FC)))
      x, stats, npc_dir = solver.x, solver.stats, solver.npc_dir
      @test stats.npcCount == 1
      @test stats.status == "nonpositive curvature detected"
      @test stats.indefinite == true
      @test real(dot(npc_dir, A * npc_dir)) <= 0.01 

      # Test singular and consistent system
      A, b = singular_consistent(FC=FC)
      x, stats = cg(A, b)
      r = b - A * x
      resid = norm(r) / norm(b)
      @test(resid ≤ cg_tol)
      @test !stats.inconsistent

      # Test inconsistent system
      if FC == Float64
        A, b = square_inconsistent(FC=FC)
        x, stats = cg(A, b)
        @test stats.inconsistent
      end

      # Poisson equation in cartesian coordinates.
      A, b = cartesian_poisson(FC=FC)
      (x, stats) = cg(A, b)
      r = b - A * x
      resid = norm(r) / norm(b)
      @test(resid ≤ cg_tol)
      @test(stats.solved)

      # test callback function
      A, b = cartesian_poisson(FC=FC)
      workspace = CgWorkspace(A, b)
      tol = 1.0e-1
      cb_n2 = TestCallbackN2(A, b, tol = tol)
      cg!(workspace, A, b, callback = cb_n2)
      @test workspace.stats.status == "user-requested exit"
      @test cb_n2(workspace)

      @test_throws TypeError cg(A, b, callback = workspace -> "string", history = true)
      
      # Test that the cg workspace throws an error when radius > 0 and linesearch is true
      A, b = symmetric_indefinite(FC = FC, shift = 5)
      @test_throws ErrorException cg(A, b, radius = real(one(FC)), linesearch = true)
    end
  end
end
