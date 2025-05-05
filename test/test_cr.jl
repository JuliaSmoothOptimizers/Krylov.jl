@testset "cr" begin
  cr_tol = 1.0e-6

  for FC in (Float64, ComplexF64)
    @testset "Data Type: $FC" begin

      # Symmetric and positive definite system.
      A, b = symmetric_definite(FC=FC)
      (x, stats) = cr(A, b)
      r = b - A * x
      resid = norm(r) / norm(b)
      @test(resid ≤ cr_tol)
      @test(stats.solved)
      @test stats.indefinite == false

      # Code coverage
      (x, stats) = cr(Matrix(A), b)

      if FC == Float64
        radius = 0.75 * norm(x)
        (x, stats) = cr(A, b, radius=radius)
        @test(stats.solved)
        @test abs(norm(x) - radius) ≤ cr_tol * radius

        # Sparse Laplacian
        A, _ = sparse_laplacian(FC=FC)
        b = randn(size(A, 1))
        itmax = 0
        # case: ‖x*‖ > Δ
        radius = 10.
        (x, stats) = cr(A, b, radius=radius)
        xNorm = norm(x)
        r = b - A * x
        resid = norm(r) / norm(b)
        @test abs(xNorm - radius) ≤ cr_tol * radius
        @test(stats.solved)
        # case: ‖x*‖ < Δ
        radius = 30.
        (x, stats) = cr(A, b, radius=radius)
        xNorm = norm(x)
        r = b - A * x
        resid = norm(r) / norm(b)
        @test(resid ≤ cr_tol)
        @test(stats.solved)

        radius = 0.75 * xNorm
        (x, stats) = cr(A, b, radius=radius)
        @test(stats.solved)
        @test(abs(radius - norm(x)) ≤ cr_tol * radius)
      end

      # Test b == 0
      A, b = zero_rhs(FC=FC)
      (x, stats) = cr(A, b)
      @test norm(x) == 0
      @test stats.status == "x is a zero-residual solution"

      # Test with Jacobi (or diagonal) preconditioner
      A, b, M = square_preconditioned(FC=FC)
      (x, stats) = cr(A, b, M=M, atol=1e-5, rtol=0.0)
      r = b - A * x
      resid = sqrt(real(dot(r, M * r))) / sqrt(real(dot(b, M * b)))
      @test(resid ≤ 10 * cr_tol)
      @test(stats.solved)

      # Test linesearch
      A, b = symmetric_indefinite(FC=FC)
      solver = CrSolver(A, b)
      cr!(solver, A, b, linesearch=true)
      x, stats, npc_dir = solver.x, solver.stats, solver.npc_dir
      @test stats.status == "nonpositive curvature"
      @test stats.indefinite == true
      # Verify that the returned direction indeed exhibits nonpositive curvature.
      # For both real and complex cases, ensure to take the real part.
      curvature = real(dot(npc_dir, A * npc_dir))
      @test curvature <= 0
      
      # Test Linesearch which would stop on the first call since A is negative definite
      A, b = symmetric_indefinite(FC=FC; shift = 5)
      solver = CrSolver(A, b)
      cr!(solver, A, b, linesearch=true)
      x, stats, npc_dir = solver.x, solver.stats, solver.npc_dir
      @test stats.status == "nonpositive curvature"
      @test stats.niter == 0
      @test all(x .== b)
      @test stats.solved == true
      @test stats.indefinite == true
      curvature = real(dot(npc_dir, A * npc_dir))
      @test curvature <= 0

      # Test when b^TAb=0 and linesearch is true
      A, b = system_zero_quad(FC=FC)
      solver = CrSolver(A, b)
      cr!(solver, A, b, linesearch=true)
      x, stats, npc_dir = solver.x, solver.stats, solver.npc_dir
      @test stats.status == "b is a zero-curvature direction"
      @test all(x .== b)
      @test stats.solved == true
      @test real(dot(npc_dir, A * npc_dir)) ≈ 0.0

      # Test if warm_start and linesearch are both true, it should throw an error
      A, b = symmetric_indefinite(FC=FC)
      @test_throws MethodError cr(A, b, warm_start = true, linesearch = true)     

      # Test when b^TAb=0 and linesearch is false
      A, b = system_zero_quad(FC=FC)
      x, stats = cr(A,b, linesearch=false)
      @test stats.status == "b is a zero-curvature direction"
      @test norm(x) == zero(FC)
      @test stats.solved == true
      
      # Test callback function
      A, b = symmetric_definite(FC=FC)
      workspace = CrWorkspace(A, b)
      tol = 1.0e-1
      cb_n2 = TestCallbackN2(A, b, tol = tol)
      cr!(solver, A, b, callback = cb_n2)
      @test solver.stats.status == "user-requested exit"
      @test cb_n2(solver)
      @test_throws TypeError cr(A, b, callback = solver -> "string", history = true)


      # Test on trust-region boundary when radius > 0
      A, b = symmetric_indefinite(FC=FC, shift = 5)
      solver = CrSolver(A, b)
      cr!(solver, A, b,  radius = one(Float64))
      x, stats, npc_dir = solver.x, solver.stats, solver.npc_dir
      @test stats.status == "on trust-region boundary"
      @test norm(x) ≈ 1.0
      curvature = real(dot(x, A * x))
      @test curvature <= 0

      # Test on trust-region boundary when radius = 1 and linesearch is true
      A, b = symmetric_indefinite(FC=FC, shift = 5)
      @test_throws ErrorException cr(A, b, radius = one(Float64), linesearch = true) 


    end
  end
end
