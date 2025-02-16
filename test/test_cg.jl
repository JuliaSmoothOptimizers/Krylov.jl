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

      # Test linesearch
      A, b = symmetric_indefinite(FC=FC)
      x, stats = cg(A, b, linesearch=true)
      @test stats.status == "nonpositive curvature detected"
      @test !stats.inconsistent

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
      solver = CgSolver(A, b)
      tol = 1.0e-1
      cb_n2 = TestCallbackN2(A, b, tol = tol)
      cg!(solver, A, b, callback = cb_n2)
      @test solver.stats.status == "user-requested exit"
      @test cb_n2(solver)

      @test_throws TypeError cg(A, b, callback = solver -> "string", history = true)
    end
  end
end
