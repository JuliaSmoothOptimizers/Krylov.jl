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

      # The estimates are ‖Mrₖ‖₂ and ‖AMrₖ‖_M with a preconditioner
      A, b = symmetric_definite(FC=FC)
      n = length(b)
      X = rand(FC, n, n)
      M = Matrix(Hermitian(X * X' + n * I))
      for k in 1:5
        (x, stats) = car(A, b, M=M, itmax=k, atol=0.0, rtol=0.0, history=true)
        r = b - A * x
        @test stats.residuals[end] ≈ norm(M * r)
        @test stats.Aresiduals[end] ≈ sqrt(real(dot(A * M * r, M * (A * M * r))))
      end

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
      workspace = CarWorkspace(A, b)
      tol = 1.0e-1
      cb_n2 = TestCallbackN2(A, b, tol = tol)
      car!(workspace, A, b, callback = cb_n2)
      @test workspace.stats.status == "user-requested exit"
      @test cb_n2(workspace)

      @test_throws TypeError car(A, b, callback = workspace -> "string", history = true)
    end
  end
end
