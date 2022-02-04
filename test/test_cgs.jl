@testset "cgs" begin
  cgs_tol = 1.0e-6

  for FC in (Float64,)
    @testset "Data Type: $FC" begin

      # Symmetric and positive definite system.
      A, b = symmetric_definite(FC=FC)
      (x, stats) = cgs(A, b)
      r = b - A * x
      resid = norm(r) / norm(b)
      @test(resid ≤ cgs_tol)
      @test(stats.solved)

      # Symmetric indefinite variant.
      A, b = symmetric_indefinite(FC=FC)
      (x, stats) = cgs(A, b)
      r = b - A * x
      resid = norm(r) / norm(b)
      @test(resid ≤ cgs_tol)
      @test(stats.solved)

      # Nonsymmetric and positive definite systems.
      A, b = nonsymmetric_definite(FC=FC)
      (x, stats) = cgs(A, b)
      r = b - A * x
      resid = norm(r) / norm(b)
      @test(resid ≤ cgs_tol)
      @test(stats.solved)

      # Nonsymmetric indefinite variant.
      A, b = nonsymmetric_indefinite(FC=FC)
      (x, stats) = cgs(A, b)
      r = b - A * x
      resid = norm(r) / norm(b)
      @test(resid ≤ cgs_tol)
      @test(stats.solved)

      # Sparse Laplacian.
      A, b = sparse_laplacian(FC=FC)
      (x, stats) = cgs(A, b)
      r = b - A * x
      resid = norm(r) / norm(b)
      @test(resid ≤ cgs_tol)
      @test(stats.solved)

      # Test b == 0
      A, b = zero_rhs(FC=FC)
      (x, stats) = cgs(A, b)
      @test norm(x) == 0
      @test stats.status == "x = 0 is a zero-residual solution"

      # Left preconditioning
      A, b, M = square_preconditioned(FC=FC)
      (x, stats) = cgs(A, b, M=M)
      r = b - A * x
      resid = norm(r) / norm(b)
      @test(resid ≤ cgs_tol)
      @test(stats.solved)

      # Right preconditioning
      A, b, N = square_preconditioned(FC=FC)
      (x, stats) = cgs(A, b, N=N)
      r = b - A * x
      resid = norm(r) / norm(b)
      @test(resid ≤ cgs_tol)
      @test(stats.solved)

      # Split preconditioning
      A, b, M, N = two_preconditioners(FC=FC)
      (x, stats) = cgs(A, b, M=M, N=N)
      r = b - A * x
      resid = norm(r) / norm(b)
      @test(resid ≤ cgs_tol)
      @test(stats.solved)

      # Test bᵀc == 0
      A, b, c = bc_breakdown(FC=FC)
      (x, stats) = cgs(A, b, c=c)
      @test stats.status == "Breakdown bᵀc = 0"
    end
  end
end
