@testset "minres" begin
  minres_tol = 1.0e-5

  for FC in (Float64, ComplexF64)
    @testset "Data Type: $FC" begin

      # Cubic spline matrix.
      A, b = symmetric_definite(FC=FC)
      (x, stats) = minres(A, b)
      r = b - A * x
      resid = norm(r) / norm(b)
      @test(resid ≤ minres_tol)
      @test(stats.solved)

      # Symmetric indefinite variant.
      A, b = symmetric_indefinite(FC=FC)
      (x, stats) = minres(A, b)
      r = b - A * x
      resid = norm(r) / norm(b)
      @test(resid ≤ minres_tol)
      @test(stats.solved)

      # Sparse Laplacian.
      A, b = sparse_laplacian(FC=FC)
      (x, stats) = minres(A, b)
      r = b - A * x
      resid = norm(r) / norm(b)
      @test(resid ≤ minres_tol)
      @test(stats.solved)

      # Symmetric indefinite variant, almost singular.
      A, b = almost_singular(FC=FC)
      (x, stats) = minres(A, b)
      r = b - A * x
      resid = norm(r) / norm(b)
      @test(resid ≤ 100 * minres_tol)
      @test(stats.solved)

      # Test b == 0
      A, b = zero_rhs(FC=FC)
      (x, stats) = minres(A, b)
      @test norm(x) == 0
      @test stats.status == "x = 0 is a zero-residual solution"

      # Shifted system
      A, b = symmetric_indefinite(FC=FC)
      λ = 2.0
      (x, stats) = minres(A, b, λ=λ)
      r = b - (A + λ*I) * x
      resid = norm(r) / norm(b)
      @test(resid ≤ minres_tol)
      @test(stats.solved)

      # test with Jacobi (or diagonal) preconditioner and history = true
      A, b, M = square_preconditioned(FC=FC)
      (x, stats) = minres(A, b, M=M, history=true)
      r = b - A * x
      resid = norm(r) / norm(b)
      @test(resid ≤ minres_tol)
      @test(stats.solved)
      @test(length(stats.residuals) > 0)

      # in-place minres (minres!) with Jacobi (or diagonal) preconditioner
      A, b, M = square_preconditioned(FC=FC)
      solver = MinresSolver(A, b)
      minres!(solver, A, b, M=M)
      r = b - A * solver.x
      resid = norm(r) / norm(b)
      @test(resid ≤ minres_tol)
      @test(stats.solved)

      # Test restart
      A, b = restart(FC=FC)
      solver = MinresSolver(A, b)
      minres!(solver, A, b, itmax=50)
      @test !solver.stats.solved
      @test solver.warm_start == false
      (x, stats) = minres(A, b, solver.x)
      r1 = b - A * x
      minres!(solver, A, b, solver.x)
      @test solver.warm_start == false
      r2 = b - A * solver.x
      resid1 = norm(r1) / norm(b)
      resid2 = norm(r2) / norm(b)
      @test(resid1 ≤ minres_tol)
      @test(resid2 ≤ minres_tol)
      @test solver.stats.solved
    end
  end
end
