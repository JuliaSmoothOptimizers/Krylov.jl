@testset "cg_lanczos" begin
  cg_lanczos_tol = 1.0e-6
  n = 10

  for FC in (Float64, ComplexF64)
    @testset "Data Type: $FC" begin

      # Cubic splines matrix.
      A, b = symmetric_definite(FC=FC)
      (x, stats) = cg_lanczos(A, b, itmax=n)
      resid = norm(b - A * x) / norm(b)
      @test(resid ≤ cg_lanczos_tol)
      @test(stats.solved)

      # Test negative curvature detection.
      A[n-1,n-1] = -4.0
      (x, stats) = cg_lanczos(A, b, check_curvature=true)
      @test(stats.status == "negative curvature")

      # Test b == 0
      A, b = zero_rhs(FC=FC)
      (x, stats) = cg_lanczos(A, b)
      @test norm(x) == 0
      @test stats.status == "x = 0 is a zero-residual solution"

      # Test with Jacobi (or diagonal) preconditioner
      A, b, M = square_preconditioned(FC=FC)
      (x, stats) = cg_lanczos(A, b, M=M)
      r = b - A * x
      resid = norm(r) / norm(b)
      @test(resid ≤ cg_lanczos_tol)
      @test(stats.solved)
    end
  end
end
