function residuals(A, b, shifts, x)
  nshifts = length(shifts)
  r = [ (b - A * x[i] - shifts[i] * x[i]) for i = 1 : nshifts ]
  return r
end

@testset "cg_lanczos" begin
  cg_tol = 1.0e-6
  n = 10

  for FC in (Float64, ComplexF64)
    @testset "Data Type: $FC" begin

      # Cubic splines matrix.
      A, b = symmetric_definite(FC=FC)
      b_norm = norm(b)

      (x, stats) = cg_lanczos(A, b, itmax=n)
      resid = norm(b - A * x) / b_norm
      @test(resid ≤ cg_tol)
      @test(stats.solved)

      # Test negative curvature detection.
      A[n-1,n-1] = -4.0
      (x, stats) = cg_lanczos(A, b, check_curvature=true)
      @test(stats.status == "negative curvature")
      A[n-1,n-1] = 4.0

      shifts = [1.0; 2.0; 3.0; 4.0; 5.0; 6.0]
      (x, stats) = cg_lanczos(A, b, shifts, itmax=n)
      r = residuals(A, b, shifts, x)
      resids = map(norm, r) / b_norm
      @test(all(resids .≤ cg_tol))
      @test(stats.solved)

      # Test negative curvature detection.
      shifts = [-4.0; -3.0; 2.0]
      (x, stats) = cg_lanczos(A, b, shifts, check_curvature=true, itmax=n)
      @test(stats.indefinite == [true, true, false])

      # Test b == 0
      A, b = zero_rhs(FC=FC)
      (x, stats) = cg_lanczos(A, b)
      @test norm(x) == 0
      @test stats.status == "x = 0 is a zero-residual solution"
      (x, stats) = cg_lanczos(A, b, shifts)
      for xi ∈ x
        @test norm(xi) == 0
      end
      @test stats.status == "x = 0 is a zero-residual solution"

      # Test with Jacobi (or diagonal) preconditioner
      A, b, M = square_preconditioned(FC=FC)
      (x, stats) = cg_lanczos(A, b, M=M)
      r = b - A * x
      resid = norm(r) / norm(b)
      @test(resid ≤ cg_tol)
      @test(stats.solved)

      shifts = [1.0; 2.0; 3.0; 4.0; 5.0; 6.0]
      (x, stats) = cg_lanczos(A, b, shifts, M=M)
      r = residuals(A, b, shifts, x)
      resids = map(norm, r) / norm(b)
      @test(all(resids .≤ cg_tol))
      @test(stats.solved)
    end
  end
end
