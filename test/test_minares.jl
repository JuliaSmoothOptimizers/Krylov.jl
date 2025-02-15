@testset "minares" begin
  minares_tol = 1.0e-6

  for FC in (Float64, ComplexF64)
    @testset "Data Type: $FC" begin

      # Cubic spline matrix.
      A, b = symmetric_definite(FC=FC)
      (x, stats) = minares(A, b)
      r = b - A * x
      resid = norm(r) / norm(b)
      @test(resid ≤ minares_tol * norm(A) * norm(x))
      @test(stats.solved)

      # Symmetric indefinite variant.
      A, b = symmetric_indefinite(FC=FC)
      (x, stats) = minares(A, b)
      r = b - A * x
      resid = norm(r) / norm(b)
      @test(resid ≤ minares_tol * norm(A) * norm(x))
      @test(stats.solved)

      # Sparse Laplacian.
      A, b = sparse_laplacian(FC=FC)
      (x, stats) = minares(A, b)
      r = b - A * x
      resid = norm(r) / norm(b)
      @test(resid ≤ minares_tol * norm(A) * norm(x))
      @test(stats.solved)

      # Symmetric indefinite variant, almost singular.
      A, b = almost_singular(FC=FC)
      (x, stats) = minares(A, b)
      r = b - A * x
      resid = norm(r) / norm(b)
      @test(resid ≤ minares_tol * norm(A) * norm(x))
      @test(stats.solved)

      # Test b == 0
      A, b = zero_rhs(FC=FC)
      (x, stats) = minares(A, b)
      @test norm(x) == 0
      @test stats.status == "x is a zero-residual solution"

      # Singular inconsistent system
      A, b = square_inconsistent(FC=FC)
      (x, stats) = minares(A, b)
      r = b - A * x
      Aresid = norm(A*r) / norm(A*b)
      @test(Aresid ≤ minares_tol)
      # @test stats.inconsistent

      # Symmetric inconsistent system
      A, b = symmetric_inconsistent()
      (x, stats) = minares(A, b)
      r = b - A * x
      Aresid = norm(A*r) / norm(A*b)
      @test(Aresid ≤ minares_tol)
      # @test stats.inconsistent

      # Shifted system
      A, b = symmetric_indefinite(FC=FC)
      λ = 2.0
      (x, stats) = minares(A, b, λ=λ)
      r = b - (A + λ*I) * x
      resid = norm(r) / norm(b)
      @test(resid ≤ minares_tol * norm(A) * norm(x))
      @test(stats.solved)

      # Test with Jacobi (or diagonal) preconditioner
      # A, b, M = square_preconditioned(FC=FC)
      # (x, stats) = minares(A, b, M=M)
      # r = b - A * x
      # resid = sqrt(real(dot(r, M * r))) / sqrt(real(dot(b, M * b)))
      # @test(resid ≤ minares_tol * norm(A) * norm(x))
      # @test(stats.solved)

      # test callback function
      A, b = sparse_laplacian(FC=FC)
      solver = MinaresSolver(A, b)
      tol = 1.0
      cb_n2 = TestCallbackN2(A, b, tol = tol)
      minares!(solver, A, b, atol = 0.0, rtol = 0.0, Artol = 0.0, callback = cb_n2)
      @test solver.stats.status == "user-requested exit"
      @test cb_n2(solver)

      @test_throws TypeError minares(A, b, callback = solver -> "string", history = true)
    end
  end
end
