@testset "gmres" begin
  gmres_tol = 1.0e-6

  for FC in (Float64, ComplexF64)
    @testset "Data Type: $FC" begin

      # Symmetric and positive definite system.
      A, b = symmetric_definite(FC=FC)
      (x, stats) = gmres(A, b)
      r = b - A * x
      resid = norm(r) / norm(b)
      @test(resid ≤ gmres_tol)
      @test(stats.solved)

      # Symmetric indefinite variant.
      A, b = symmetric_indefinite(FC=FC)
      (x, stats) = gmres(A, b)
      r = b - A * x
      resid = norm(r) / norm(b)
      @test(resid ≤ gmres_tol)
      @test(stats.solved)

      # Nonsymmetric and positive definite systems.
      A, b = nonsymmetric_definite(FC=FC)
      (x, stats) = gmres(A, b)
      r = b - A * x
      resid = norm(r) / norm(b)
      @test(resid ≤ gmres_tol)
      @test(stats.solved)

      # Nonsymmetric indefinite variant.
      A, b = nonsymmetric_indefinite(FC=FC)
      (x, stats) = gmres(A, b)
      r = b - A * x
      resid = norm(r) / norm(b)
      @test(resid ≤ gmres_tol)
      @test(stats.solved)

      # Symmetric indefinite variant, almost singular.
      A, b = almost_singular(FC=FC)
      (x, stats) = gmres(A, b)
      r = b - A * x
      resid = norm(r) / norm(b)
      @test(resid ≤ 100 * gmres_tol)
      @test(stats.solved)

      # Singular system.
      A, b = square_inconsistent(FC=FC)
      (x, stats) = gmres(A, b)
      r = b - A * x
      Aresid = norm(A' * r) / norm(A' * b)
      @test(Aresid ≤ gmres_tol)
      @test(stats.inconsistent)

      # Test b == 0
      A, b = zero_rhs(FC=FC)
      (x, stats) = gmres(A, b)
      @test norm(x) == 0
      @test stats.status == "x is a zero-residual solution"

      # Poisson equation in polar coordinates.
      A, b = polar_poisson(FC=FC)
      (x, stats) = gmres(A, b, reorthogonalization=true)
      r = b - A * x
      resid = norm(r) / norm(b)
      @test(resid ≤ gmres_tol)
      @test(stats.solved)

      # Left preconditioning
      A, b, M = square_preconditioned(FC=FC)
      (x, stats) = gmres(A, b, M=M)
      r = b - A * x
      resid = norm(M * r) / norm(M * b)
      @test(resid ≤ gmres_tol)
      @test(stats.solved)

      # Right preconditioning
      A, b, N = square_preconditioned(FC=FC)
      (x, stats) = gmres(A, b, N=N)
      r = b - A * x
      resid = norm(r) / norm(b)
      @test(resid ≤ gmres_tol)
      @test(stats.solved)

      # Split preconditioning
      A, b, M, N = two_preconditioners(FC=FC)
      (x, stats) = gmres(A, b, M=M, N=N)
      r = b - A * x
      resid = norm(M * r) / norm(M * b)
      @test(resid ≤ gmres_tol)
      @test(stats.solved)

      # Restart
      for restart ∈ (false, true)
        memory = 10

        A, b = sparse_laplacian(FC=FC)
        (x, stats) = gmres(A, b, restart=restart, memory=memory)
        r = b - A * x
        resid = norm(r) / norm(b)
        @test(resid ≤ gmres_tol)
        @test(stats.niter > memory)
        @test(stats.solved)

        M = Diagonal(1 ./ diag(A))
        (x, stats) = gmres(A, b, M=M, restart=restart, memory=memory)
        r = b - A * x
        resid = norm(M * r) / norm(M * b)
        @test(resid ≤ gmres_tol)
        @test(stats.niter > memory)
        @test(stats.solved)

        N = Diagonal(1 ./ diag(A))
        (x, stats) = gmres(A, b, N=N, restart=restart, memory=memory)
        r = b - A * x
        resid = norm(r) / norm(b)
        @test(resid ≤ gmres_tol)
        @test(stats.niter > memory)
        @test(stats.solved)

        N = Diagonal(1 ./ sqrt.(diag(A)))
        N = Diagonal(1 ./ sqrt.(diag(A)))
        (x, stats) = gmres(A, b, M=M, N=N, restart=restart, memory=memory)
        r = b - A * x
        resid = norm(M * r) / norm(M * b)
        @test(resid ≤ gmres_tol)
        @test(stats.niter > memory)
        @test(stats.solved)
      end

      # test callback function
      A, b = sparse_laplacian(FC=FC)
      solver = GmresSolver(A, b)
      tol = 1.0e-1
      N = Diagonal(1 ./ diag(A))
      stor = StorageGetxRestartedGmres(solver, N = N)
      storage_vec = similar(b)
      gmres!(solver, A, b, N = N, atol = 0.0, rtol = 0.0, restart = true, 
             callback = solver -> restarted_gmres_callback_n2(solver, A, b, stor, N, storage_vec, tol))
      @test solver.stats.status == "user-requested exit"
      @test norm(A * x - b) ≤ tol

      @test_throws TypeError gmres(A, b, callback = solver -> "string", history = true)
    end
  end
end
