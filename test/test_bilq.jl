@testset "bilq" begin
  bilq_tol = 1.0e-6

  for FC in (Float64, ComplexF64)
    @testset "Data Type: $FC" begin

      # Symmetric and positive definite system.
      A, b = symmetric_definite(FC=FC)
      (x, stats) = bilq(A, b)
      r = b - A * x
      resid = norm(r) / norm(b)
      @test(resid ≤ bilq_tol)
      @test(stats.solved)

      # Symmetric indefinite variant.
      A, b = symmetric_indefinite(FC=FC)
      (x, stats) = bilq(A, b)
      r = b - A * x
      resid = norm(r) / norm(b)
      @test(resid ≤ bilq_tol)
      @test(stats.solved)

      # Nonsymmetric and positive definite systems.
      A, b = nonsymmetric_definite(FC=FC)
      (x, stats) = bilq(A, b)
      r = b - A * x
      resid = norm(r) / norm(b)
      @test(resid ≤ bilq_tol)
      @test(stats.solved)

      # Nonsymmetric indefinite variant.
      A, b = nonsymmetric_indefinite(FC=FC)
      (x, stats) = bilq(A, b)
      r = b - A * x
      resid = norm(r) / norm(b)
      @test(resid ≤ bilq_tol)
      @test(stats.solved)

      # Sparse Laplacian.
      A, b = sparse_laplacian(FC=FC)
      (x, stats) = bilq(A, b)
      r = b - A * x
      resid = norm(r) / norm(b)
      @test(resid ≤ bilq_tol)
      @test(stats.solved)

      # Test b == 0
      A, b = zero_rhs(FC=FC)
      (x, stats) = bilq(A, b)
      @test norm(x) == 0
      @test stats.status == "x = 0 is a zero-residual solution"

      # System that cause a breakdown with the Lanczos biorthogonalization process.
      A, b, c = unsymmetric_breakdown(FC=FC)
      (x, stats) = bilq(A, b, c=c)
      r = b - A * x
      resid = norm(r) / norm(b)
      @test(resid ≤ bilq_tol)
      @test(stats.solved)

      # Poisson equation in polar coordinates.
      A, b = polar_poisson(FC=FC)
      (x, stats) = bilq(A, b)
      r = b - A * x
      resid = norm(r) / norm(b)
      @test(resid ≤ bilq_tol)
      @test(stats.solved)

      # Left preconditioning
      A, b, M = square_preconditioned(FC=FC)
      (x, stats) = bilq(A, b, M=M)
      r = b - A * x
      resid = norm(M * r) / norm(M * b)
      @test(resid ≤ bilq_tol)
      @test(stats.solved)

      # Right preconditioning
      A, b, N = square_preconditioned(FC=FC)
      (x, stats) = bilq(A, b, N=N)
      r = b - A * x
      resid = norm(r) / norm(b)
      @test(resid ≤ bilq_tol)
      @test(stats.solved)

      # Split preconditioning
      A, b, M, N = two_preconditioners(FC=FC)
      (x, stats) = bilq(A, b, M=M, N=N)
      r = b - A * x
      resid = norm(M * r) / norm(M * b)
      @test(resid ≤ bilq_tol)
      @test(stats.solved)

      # Test bᴴc == 0
      A, b, c = bc_breakdown(FC=FC)
      (x, stats) = bilq(A, b, c=c)
      @test stats.status == "Breakdown bᴴc = 0"
      
      # test callback function
      A, b = polar_poisson(FC=FC)
      solver = BilqSolver(A, b)
      tol = 1.0e-1
      cb_n2 = TestCallbackN2(A, b, tol = tol)
      bilq!(solver, A, b, callback = cb_n2)
      @test solver.stats.status == "user-requested exit"
      @test cb_n2(solver)

      @test_throws TypeError bilq(A, b, callback = solver -> "string", history = true)
    end
  end
end
