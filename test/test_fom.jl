@testset "fom" begin
  fom_tol = 1.0e-6

  for FC in (Float64,)
    @testset "Data Type: $FC" begin

      # Symmetric and positive definite system.
      A, b = symmetric_definite()
      (x, stats) = fom(A, b)
      r = b - A * x
      resid = norm(r) / norm(b)
      @test(resid ≤ fom_tol)
      @test(stats.solved)

      # Symmetric indefinite variant.
      A, b = symmetric_indefinite()
      (x, stats) = fom(A, b)
      r = b - A * x
      resid = norm(r) / norm(b)
      @test(resid ≤ fom_tol)
      @test(stats.solved)

      # Nonsymmetric and positive definite systems.
      A, b = nonsymmetric_definite()
      (x, stats) = fom(A, b)
      r = b - A * x
      resid = norm(r) / norm(b)
      @test(resid ≤ fom_tol)
      @test(stats.solved)

      # Nonsymmetric indefinite variant.
      A, b = nonsymmetric_indefinite()
      (x, stats) = fom(A, b)
      r = b - A * x
      resid = norm(r) / norm(b)
      @test(resid ≤ fom_tol)
      @test(stats.solved)

      # Code coverage.
      (x, stats) = fom(Matrix(A), b)

      # Sparse Laplacian.
      A, b = sparse_laplacian()
      (x, stats) = fom(A, b)
      r = b - A * x
      resid = norm(r) / norm(b)
      @test(resid ≤ fom_tol)
      @test(stats.solved)

      # Symmetric indefinite variant, almost singular.
      A, b = almost_singular()
      (x, stats) = fom(A, b)
      r = b - A * x
      resid = norm(r) / norm(b)
      @test(resid ≤ 100 * fom_tol)
      @test(stats.solved)

      # Singular system.
      A, b = square_inconsistent()
      (x, stats) = fom(A, b)
      @test(stats.inconsistent)

      # Test b == 0
      A, b = zero_rhs()
      (x, stats) = fom(A, b)
      @test x == zeros(size(A,1))
      @test stats.status == "x = 0 is a zero-residual solution"

      # Poisson equation in polar coordinates.
      A, b = polar_poisson()
      (x, stats) = fom(A, b, reorthogonalization=true)
      r = b - A * x
      resid = norm(r) / norm(b)
      @test(resid ≤ fom_tol)
      @test(stats.solved)

      # Test restart
      A, b = restart()
      solver = FomSolver(A, b, 20)
      fom!(solver, A, b, itmax=50)
      @test !solver.stats.solved
      fom!(solver, A, b, restart=true)
      r = b - A * solver.x
      resid = norm(r) / norm(b)
      @test(resid ≤ fom_tol)
      @test solver.stats.solved

      # Test with Jacobi (or diagonal) preconditioner
      A, b, M = square_preconditioned()
      (x, stats) = fom(A, b, M=M)
      r = b - A * x
      resid = norm(M * r) / norm(M * b)
      @test(resid ≤ fom_tol)
      @test(stats.solved)

      # Right preconditioning
      A, b, N = square_preconditioned()
      (x, stats) = fom(A, b, N=N)
      r = b - A * x
      resid = norm(r) / norm(b)
      @test(resid ≤ fom_tol)
      @test(stats.solved)

      # Split preconditioning
      A, b, M, N = two_preconditioners()
      (x, stats) = fom(A, b, M=M, N=N)
      r = b - A * x
      resid = norm(M * r) / norm(M * b)
      @test(resid ≤ fom_tol)
      @test(stats.solved)
    end
  end
end
