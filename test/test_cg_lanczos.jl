@testset "cg_lanczos" begin
  cg_lanczos_tol = 1.0e-6
  n = 10

  for FC in (Float64, ComplexF64)
    @testset "Data Type: $FC" begin

      # Cubic splines matrix.
      A, b = symmetric_definite(FC=FC)
      (x, stats) = cg_lanczos(A, b, itmax=n)
      resid = norm(b - A * x) / norm(b)
      @test(resid ≤ cg_lanczos_tol)
      @test(stats.solved)

      # Test negative curvature detection.
      A[n-1,n-1] = -4.0
      (x, stats) = cg_lanczos(A, b, check_curvature=true)
      @test(stats.status == "negative curvature")

      # Test b == 0
      A, b = zero_rhs(FC=FC)
      (x, stats) = cg_lanczos(A, b)
      @test norm(x) == 0
      @test stats.status == "x is a zero-residual solution"

      # Test with Jacobi (or diagonal) preconditioner
      A, b, M = square_preconditioned(FC=FC)
      (x, stats) = cg_lanczos(A, b, M=M)
      r = b - A * x
      resid = norm(r) / norm(b)
      @test(resid ≤ cg_lanczos_tol)
      @test(stats.solved)

      # test callback function
      A, b = cartesian_poisson(FC=FC)
      solver = CgLanczosSolver(A, b)
      tol = 1.0e-1
      cb_n2 = TestCallbackN2(A, b, tol = tol)
      cg_lanczos!(solver, A, b, callback = cb_n2)
      @test solver.stats.status == "user-requested exit"
      @test cb_n2(solver)

      @test_throws TypeError cg_lanczos(A, b, callback = solver -> "string", history = true)
    end
  end
end
