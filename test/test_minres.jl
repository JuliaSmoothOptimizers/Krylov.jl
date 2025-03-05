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
      @test stats.status == "b is a zero-curvature directions"

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
      solver = MinresSolver(A, b)
      minres!(solver, A, b, M=M)
      r = b - A * solver.x
      resid = norm(r) / norm(b)
      @test(resid ≤ minres_tol * norm(A) * norm(x))
      @test(stats.solved)
      @test stats.indefinite == false

      # Test linesearch
      A, b = symmetric_indefinite(FC=FC)
      solver = MinresSolver(A, b)
      minres!(solver, A, b, linesearch=true)
      x, stats, npc_dir = solver.x, solver.stats, solver.npc_dir
      @test stats.status == "nonpositive curvature"
      @test stats.indefinite == true
      # Verify that the returned direction indeed exhibits nonpositive curvature.
      # For both real and complex cases, ensure to take the real part.
      curvature = real(dot(npc_dir, A * npc_dir))
      @test curvature <= 0


      # Test Linesearch which would stop on the first call since A is negative definite
      A, b = symmetric_indefinite(FC=FC; shift = 5)
      x, stats = minres(A, b, linesearch=true)
      @test stats.status == "nonpositive curvature"
      @test stats.niter == 1 # in Minres they add 1 to the number of iterations first step
      @test all(x .== b)
      @test stats.solved == true
      @test stats.indefinite == true

      # Test when b^TAb=0 and linesearch is true
      A, b = system_zero_quad(FC=FC)
      x, stats = minres(A, b, linesearch=true)
      @test stats.status == "nonpositive curvature"
      @test all(x .== b)
      @test stats.solved == true
      @test stats.indefinite == true


      # Test if warm_start and linesearch are both true, it should through an error
      A, b = symmetric_indefinite(FC=FC)
      @test_throws MethodError minres(A, b, warm_start = true, linesearch = true)      
      
      # Test the constructor warm_start and linesearch when both are true, it should through an error 
       # Test linesearch
       A, b = symmetric_indefinite(FC=FC)
       solver = MinresSolver(A, b)
       @test_throws MethodError minres!(solver, A, b, linesearch=true, warm_start = true)

      # test callback function
      solver = MinresSolver(A, b)
      storage_vec = similar(b, size(A, 1))
      tol = 1.0e-1
      cb_n2 = TestCallbackN2(A, b, tol = tol)
      x, stats = minres(A, b, callback = solver -> cb_n2(solver)) # n = 10
      minres!(solver, A, b, callback = cb_n2)
      @test solver.stats.status == "user-requested exit"
      @test cb_n2(solver)

      @test_throws TypeError minres(A, b, callback = solver -> "string", history = true)
    end
  end
end
