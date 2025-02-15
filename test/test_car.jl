@testset "car" begin
  car_tol = 1.0e-6

  for FC in (Float64, ComplexF64)
    @testset "Data Type: $FC" begin

      # Symmetric and positive definite system.
      A, b = symmetric_definite(FC=FC)
      (x, stats) = car(A, b)
      r = b - A * x
      resid = norm(r) / norm(b)
      @test(resid ≤ car_tol)
      @test(stats.solved)

      # Sparse Laplacian.
      A, b = sparse_laplacian(FC=FC)
      (x, stats) = car(A, b)
      r = b - A * x
      resid = norm(r) / norm(b)
      @test(resid ≤ car_tol)
      @test(stats.solved)

      # Test b == 0
      A, b = zero_rhs(FC=FC)
      (x, stats) = car(A, b)
      @test norm(x) == 0
      @test stats.status == "x is a zero-residual solution"

      # Test with Jacobi (or diagonal) preconditioner
      A, b, M = square_preconditioned(FC=FC)
      (x, stats) = car(A, b, M=M)
      r = b - A * x
      resid = norm(M * r) / norm(M * b)
      @test(resid ≤ car_tol)
      @test(stats.solved)

      # Test singular and consistent system
      A, b = singular_consistent(FC=FC)
      x, stats = car(A, b)
      r = b - A * x
      resid = norm(r) / norm(b)
      @test(resid ≤ car_tol)
      @test !stats.inconsistent

      # Poisson equation in cartesian coordinates.
      A, b = cartesian_poisson(FC=FC)
      (x, stats) = car(A, b)
      r = b - A * x
      resid = norm(r) / norm(b)
      @test(resid ≤ car_tol)
      @test(stats.solved)

      # test callback function
      A, b = cartesian_poisson(FC=FC)
      solver = CarSolver(A, b)
      tol = 1.0e-1
      cb_n2 = TestCallbackN2(A, b, tol = tol)
      car!(solver, A, b, callback = cb_n2)
      @test solver.stats.status == "user-requested exit"
      @test cb_n2(solver)

      @test_throws TypeError car(A, b, callback = solver -> "string", history = true)
    end
  end
end
