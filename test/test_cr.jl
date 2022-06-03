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

      # Code coverage
      (x, stats) = cr(Matrix(A), b)

      if FC == Float64
        radius = 0.75 * norm(x)
        (x, stats) = cr(A, b, radius=radius)
        @test(stats.solved)
        @test abs(norm(x) - radius) ≤ cr_tol * radius

        # Sparse Laplacian
        A, _ = sparse_laplacian(FC=FC)
        Random.seed!(0)
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
      @test stats.status == "x = 0 is a zero-residual solution"

      # Test with Jacobi (or diagonal) preconditioner
      A, b, M = square_preconditioned(FC=FC)
      (x, stats) = cr(A, b, M=M)
      r = b - A * x
      resid = sqrt(real(dot(r, M * r))) / norm(b)
      @test(resid ≤ cr_tol)
      @test(stats.solved)

      
      # test callback function
      A, b = symmetric_definite(FC=FC)
      solver = CrSolver(A, b)
      storage_vec = similar(b, size(A, 1))
      tol = 1.0e-1
      cr!(solver, A, b, callback = (args...) -> test_callback_n2(args..., storage_vec = storage_vec, tol = tol))
      @test solver.stats.status == "user-requested exit"
      @test test_callback_n2(solver, A, b, storage_vec = storage_vec, tol = tol)

      @test_throws TypeError cr(A, b, callback = (args...) -> "string", history = true)
    end
  end
end
