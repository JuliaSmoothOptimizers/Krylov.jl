@testset "diom" begin
  diom_tol = 1.0e-6

  for FC in (Float64, ComplexF64)
    @testset "Data Type: $FC" begin

      # Symmetric and positive definite system.
      A, b = symmetric_definite(FC=FC)
      (x, stats) = diom(A, b)
      r = b - A * x
      resid = norm(r) / norm(b)
      @test(resid ≤ diom_tol)
      @test(stats.solved)

      # Symmetric indefinite variant.
      A, b = symmetric_indefinite(FC=FC)
      (x, stats) = diom(A, b)
      r = b - A * x
      resid = norm(r) / norm(b)
      @test(resid ≤ diom_tol)
      @test(stats.solved)

      # Nonsymmetric and positive definite systems.
      A, b = nonsymmetric_definite(FC=FC)
      (x, stats) = diom(A, b)
      r = b - A * x
      resid = norm(r) / norm(b)
      @test(resid ≤ diom_tol)
      @test(stats.solved)

      # Nonsymmetric indefinite variant.
      A, b = nonsymmetric_indefinite(FC=FC)
      (x, stats) = diom(A, b)
      r = b - A * x
      resid = norm(r) / norm(b)
      @test(resid ≤ diom_tol)
      @test(stats.solved)

      # Sparse Laplacian.
      A, b = sparse_laplacian(FC=FC)
      (x, stats) = diom(A, b)
      r = b - A * x
      resid = norm(r) / norm(b)
      @test(resid ≤ diom_tol)
      @test(stats.solved)

      # Symmetric indefinite variant, almost singular.
      A, b = almost_singular(FC=FC)
      (x, stats) = diom(A, b, reorthogonalization=true)
      r = b - A * x
      resid = norm(r) / norm(b)
      @test(resid ≤ diom_tol)
      @test(stats.solved)

      # Test b == 0
      A, b = zero_rhs(FC=FC)
      (x, stats) = diom(A, b)
      @test norm(x) == 0
      @test stats.status == "x = 0 is a zero-residual solution"

      # Poisson equation in polar coordinates.
      A, b = polar_poisson(FC=FC)
      (x, stats) = diom(A, b, memory=200)
      r = b - A * x
      resid = norm(r) / norm(b)
      @test(resid ≤ diom_tol)
      @test(stats.solved)

      # Test with Jacobi (or diagonal) preconditioner
      A, b, M = square_preconditioned(FC=FC)
      (x, stats) = diom(A, b, M=M)
      r = b - A * x
      resid = norm(M * r) / norm(M * b)
      @test(resid ≤ diom_tol)
      @test(stats.solved)

      # Right preconditioning
      A, b, N = square_preconditioned(FC=FC)
      (x, stats) = diom(A, b, N=N)
      r = b - A * x
      resid = norm(r) / norm(b)
      @test(resid ≤ diom_tol)
      @test(stats.solved)

      # Split preconditioning
      A, b, M, N = two_preconditioners(FC=FC)
      (x, stats) = diom(A, b, M=M, N=N)
      r = b - A * x
      resid = norm(M * r) / norm(M * b)
      @test(resid ≤ diom_tol)
      @test(stats.solved)

      # test callback function
      solver = DiomSolver(A, b)
      storage_vec = similar(b, size(A, 1))
      tol = 1.0e-1
      diom!(solver, A, b,
              callback = (args...) -> test_callback_n2(args..., storage_vec = storage_vec, tol = tol))
      @test solver.stats.status == "user-requested exit"
      @test test_callback_n2(solver, A, b, storage_vec = storage_vec, tol = tol)

      @test_throws TypeError diom(A, b, callback = (args...) -> "string", history = true)
    end
  end
end
