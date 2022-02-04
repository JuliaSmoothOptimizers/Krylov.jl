@testset "minres_qlp" begin
  minres_qlp_tol = 1.0e-6

  for FC in (Float64,)
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

      # Shifted system
      A, b = symmetric_indefinite(FC=FC)
      λ = 2.0
      (x, stats) = minres_qlp(A, b, λ=λ)
      r = b - (A + λ*I) * x
      resid = norm(r) / norm(b)
      @test(resid ≤ minres_qlp_tol)
      @test(stats.solved)

      # Singular inconsistent system
      A, b = square_inconsistent(FC=FC)
      (x, stats) = minres_qlp(A, b)
      @test stats.inconsistent

      # Test with Jacobi (or diagonal) preconditioner
      A, b, M = square_preconditioned(FC=FC)
      (x, stats) = minres_qlp(A, b, M=M)
      r = b - A * x
      resid = sqrt(dot(r, M * r)) / norm(b)
      @test(resid ≤ minres_qlp_tol)
      @test(stats.solved)

      # Test restart
      A, b = restart(FC=FC)
      solver = MinresQlpSolver(A, b)
      minres_qlp!(solver, A, b, itmax=50)
      @test !solver.stats.solved
      minres_qlp!(solver, A, b, restart=true)
      r = b - A * solver.x
      resid = norm(r) / norm(b)
      @test(resid ≤ minres_qlp_tol)
      @test solver.stats.solved
    end
  end
end
