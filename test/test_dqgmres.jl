@testset "dqgmres" begin
  dqgmres_tol = 1.0e-6

  for FC in (Float64,)
    @testset "Data Type: $FC" begin

      # Symmetric and positive definite system.
      A, b = symmetric_definite(FC=FC)
      (x, stats) = dqgmres(A, b)
      r = b - A * x
      resid = norm(r) / norm(b)
      @test(resid ≤ dqgmres_tol)
      @test(stats.solved)

      # Symmetric indefinite variant.
      A, b = symmetric_indefinite(FC=FC)
      (x, stats) = dqgmres(A, b)
      r = b - A * x
      resid = norm(r) / norm(b)
      @test(resid ≤ dqgmres_tol)
      @test(stats.solved)

      # Nonsymmetric and positive definite systems.
      A, b = nonsymmetric_definite(FC=FC)
      (x, stats) = dqgmres(A, b)
      r = b - A * x
      resid = norm(r) / norm(b)
      @test(resid ≤ dqgmres_tol)
      @test(stats.solved)

      # Nonsymmetric indefinite variant.
      A, b = nonsymmetric_indefinite(FC=FC)
      (x, stats) = dqgmres(A, b)
      r = b - A * x
      resid = norm(r) / norm(b)
      @test(resid ≤ dqgmres_tol)
      @test(stats.solved)

      # Sparse Laplacian.
      A, b = sparse_laplacian(FC=FC)
      (x, stats) = dqgmres(A, b)
      r = b - A * x
      resid = norm(r) / norm(b)
      @test(resid ≤ dqgmres_tol)
      @test(stats.solved)

      # Symmetric indefinite variant, almost singular.
      A, b = almost_singular(FC=FC)
      (x, stats) = dqgmres(A, b)
      r = b - A * x
      resid = norm(r) / norm(b)
      @test(resid ≤ 100 * dqgmres_tol)
      @test(stats.solved)

      # Test b == 0
      A, b = zero_rhs(FC=FC)
      (x, stats) = dqgmres(A, b)
      @test norm(x) == 0
      @test stats.status == "x = 0 is a zero-residual solution"

      # Poisson equation in polar coordinates.
      A, b = polar_poisson(FC=FC)
      (x, stats) = dqgmres(A, b, memory=100)
      r = b - A * x
      resid = norm(r) / norm(b)
      @test(resid ≤ dqgmres_tol)
      @test(stats.solved)

      # Test restart
      A, b = restart(FC=FC)
      solver = DqgmresSolver(A, b, 20)
      dqgmres!(solver, A, b, itmax=50)
      @test !solver.stats.solved
      dqgmres!(solver, A, b, restart=true)
      r = b - A * solver.x
      resid = norm(r) / norm(b)
      @test(resid ≤ dqgmres_tol)
      @test solver.stats.solved

      # Test with Jacobi (or diagonal) preconditioner
      A, b, M = square_preconditioned(FC=FC)
      (x, stats) = dqgmres(A, b, M=M)
      r = b - A * x
      resid = norm(M * r) / norm(M * b)
      @test(resid ≤ dqgmres_tol)
      @test(stats.solved)

      # Right preconditioning
      A, b, N = square_preconditioned(FC=FC)
      (x, stats) = dqgmres(A, b, N=N)
      r = b - A * x
      resid = norm(r) / norm(b)
      @test(resid ≤ dqgmres_tol)
      @test(stats.solved)

      # Split preconditioning
      A, b, M, N = two_preconditioners(FC=FC)
      (x, stats) = dqgmres(A, b, M=M, N=N)
      r = b - A * x
      resid = norm(M * r) / norm(M * b)
      @test(resid ≤ dqgmres_tol)
      @test(stats.solved)
    end
  end
end
