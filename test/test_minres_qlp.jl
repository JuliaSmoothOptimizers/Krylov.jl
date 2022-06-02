@testset "minres_qlp" begin
  minres_qlp_tol = 1.0e-6

  for FC in (Float64, ComplexF64)
    @testset "Data Type: $FC" begin

      # Cubic spline matrix.
      A, b = symmetric_definite(FC=FC)
      (x, stats) = minres_qlp(A, b)
      r = b - A * x
      resid = norm(r) / norm(b)
      @test(resid ≤ minres_qlp_tol)
      @test(stats.solved)

      # Symmetric indefinite variant.
      A, b = symmetric_indefinite(FC=FC)
      (x, stats) = minres_qlp(A, b)
      r = b - A * x
      resid = norm(r) / norm(b)
      @test(resid ≤ minres_qlp_tol)
      @test(stats.solved)

      # Sparse Laplacian.
      A, b = sparse_laplacian(FC=FC)
      (x, stats) = minres_qlp(A, b)
      r = b - A * x
      resid = norm(r) / norm(b)
      @test(resid ≤ minres_qlp_tol)
      @test(stats.solved)

      # Symmetric indefinite variant, almost singular.
      A, b = almost_singular(FC=FC)
      (x, stats) = minres_qlp(A, b)
      r = b - A * x
      resid = norm(r) / norm(b)
      @test(resid ≤ minres_qlp_tol)
      @test(stats.solved)

      # Test b == 0
      A, b = zero_rhs(FC=FC)
      (x, stats) = minres_qlp(A, b)
      @test norm(x) == 0
      @test stats.status == "x = 0 is a zero-residual solution"

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
      @test(resid ≤ minres_qlp_tol)
      @test(stats.solved)

      # Test with Jacobi (or diagonal) preconditioner
      A, b, M = square_preconditioned(FC=FC)
      (x, stats) = minres_qlp(A, b, M=M)
      r = b - A * x
      resid = sqrt(real(dot(r, M * r))) / norm(b)
      @test(resid ≤ minres_qlp_tol)
      @test(stats.solved)

      # test callback function
      A, b = sparse_laplacian(FC=FC)
      solver = MinresQlpSolver(A, b)
      storage_vec = similar(b, size(A, 1))
      tol = 1.0
      minres_qlp!(solver, A, b, atol = 0.0, rtol = 0.0, ctol = 0.0,
              callback = (args...) -> test_callback_n2(args..., storage_vec = storage_vec, tol = tol))
      @test solver.stats.status == "user-requested exit"
      @test test_callback_n2(solver, A, b, storage_vec = storage_vec, tol = tol)

      @test_throws TypeError minres_qlp(A, b, callback = (args...) -> "string", history = true)
    end
  end
end
