function residuals(A, b, shifts, x)
  nshifts = length(shifts)
  r = [ (b - A * x[i] - shifts[i] * x[i]) for i = 1 : nshifts ]
  return r
end

@testset "cg_lanczos_shift" begin
  cg_lanczos_shift_tol = 1.0e-6
  n = 10

  for FC in (Float64, ComplexF64)
    @testset "Data Type: $FC" begin

      # Cubic splines matrix.
      A, b = symmetric_definite(FC=FC)
      shifts = [1.0; 2.0; 3.0; 4.0; 5.0; 6.0]
      (x, stats) = cg_lanczos_shift(A, b, shifts, itmax=n)
      r = residuals(A, b, shifts, x)
      resids = map(norm, r) / norm(b)
      @test(all(resids .≤ cg_lanczos_shift_tol))
      @test(stats.solved)

      # Test negative curvature detection.
      shifts = [-4.0; -3.0; 2.0]
      (x, stats) = cg_lanczos_shift(A, b, shifts, check_curvature=true, itmax=n)
      @test(stats.indefinite == [true, true, false])

      # Test b == 0
      A, b = zero_rhs(FC=FC)
      (x, stats) = cg_lanczos_shift(A, b, shifts)
      for xi ∈ x
        @test norm(xi) == 0
      end
      @test stats.status == "x = 0 is a zero-residual solution"

      # Test with Jacobi (or diagonal) preconditioner
      A, b, M = square_preconditioned(FC=FC)
      shifts = [1.0; 2.0; 3.0; 4.0; 5.0; 6.0]
      (x, stats) = cg_lanczos_shift(A, b, shifts, M=M)
      r = residuals(A, b, shifts, x)
      resids = map(norm, r) / norm(b)
      @test(all(resids .≤ cg_lanczos_shift_tol))
      @test(stats.solved)
    end
  end
end
