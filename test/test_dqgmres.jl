@testset "dqgmres" begin
  dqgmres_tol = 1.0e-6

  for FC in (Float64,)
    @testset "Data Type: $FC" begin

      # Symmetric and positive definite system.
      A, b = symmetric_definite()
      (x, stats) = dqgmres(A, b)
      r = b - A * x
      resid = norm(r) / norm(b)
      @test(resid ≤ dqgmres_tol)
      @test(stats.solved)

      # Symmetric indefinite variant.
      A, b = symmetric_indefinite()
      (x, stats) = dqgmres(A, b)
      r = b - A * x
      resid = norm(r) / norm(b)
      @test(resid ≤ dqgmres_tol)
      @test(stats.solved)

      # Nonsymmetric and positive definite systems.
      A, b = nonsymmetric_definite()
      (x, stats) = dqgmres(A, b)
      r = b - A * x
      resid = norm(r) / norm(b)
      @test(resid ≤ dqgmres_tol)
      @test(stats.solved)

      # Nonsymmetric indefinite variant.
      A, b = nonsymmetric_indefinite()
      (x, stats) = dqgmres(A, b)
      r = b - A * x
      resid = norm(r) / norm(b)
      @test(resid ≤ dqgmres_tol)
      @test(stats.solved)

      # Code coverage.
      (x, stats) = dqgmres(Matrix(A), b)

      # Sparse Laplacian.
      A, b = sparse_laplacian()
      (x, stats) = dqgmres(A, b)
      r = b - A * x
      resid = norm(r) / norm(b)
      @test(resid ≤ dqgmres_tol)
      @test(stats.solved)

      # Symmetric indefinite variant, almost singular.
      A, b = almost_singular()
      (x, stats) = dqgmres(A, b)
      r = b - A * x
      resid = norm(r) / norm(b)
      @test(resid ≤ 100 * dqgmres_tol)
      @test(stats.solved)

      # Test b == 0
      A, b = zero_rhs()
      (x, stats) = dqgmres(A, b)
      @test x == zeros(size(A,1))
      @test stats.status == "x = 0 is a zero-residual solution"

      # Poisson equation in polar coordinates.
      A, b = polar_poisson()
      (x, stats) = dqgmres(A, b, memory=100)
      r = b - A * x
      resid = norm(r) / norm(b)
      @test(resid ≤ dqgmres_tol)
      @test(stats.solved)

      # Test restart
      A, b = restart()
      solver = DqgmresSolver(A, b, 20)
      dqgmres!(solver, A, b, itmax=50)
      @test !solver.stats.solved
      dqgmres!(solver, A, b, restart=true)
      r = b - A * solver.x
      resid = norm(r) / norm(b)
      @test(resid ≤ dqgmres_tol)
      @test solver.stats.solved

      # Test with Jacobi (or diagonal) preconditioner
      A, b, M = square_preconditioned()
      (x, stats) = dqgmres(A, b, M=M)
      r = b - A * x
      resid = norm(M * r) / norm(M * b)
      @test(resid ≤ dqgmres_tol)
      @test(stats.solved)

      # Right preconditioning
      A, b, N = square_preconditioned()
      (x, stats) = dqgmres(A, b, N=N)
      r = b - A * x
      resid = norm(r) / norm(b)
      @test(resid ≤ dqgmres_tol)
      @test(stats.solved)

      # Split preconditioning
      A, b, M, N = two_preconditioners()
      (x, stats) = dqgmres(A, b, M=M, N=N)
      r = b - A * x
      resid = norm(M * r) / norm(M * b)
      @test(resid ≤ dqgmres_tol)
      @test(stats.solved)
    end
  end
end
