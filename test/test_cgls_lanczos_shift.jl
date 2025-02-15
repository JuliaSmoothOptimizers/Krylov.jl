function residuals_ls(A, b, shifts, x)
  nshifts = length(shifts)
  r = [ (A' * (b - A * x[i]) - shifts[i] * x[i]) for i = 1 : nshifts ]
  return r
end

@testset "cgls_lanczos_shift" begin
  cgls_lanczos_shift_tol = 1.0e-6

  for FC in (Float64, ComplexF64)
    @testset "Data Type: $FC" begin

      for npower = 1 : 4
        (b, A, D, HY, HZ, Acond, rnorm) = test(40, 40, 4, npower, 0)  # No regularization.
        shifts = [1.0; 2.0; 3.0; 4.0; 5.0; 6.0]
        (x, stats) = cgls_lanczos_shift(A, b, shifts)
        r = residuals_ls(A, b, shifts, x)
        resids = map(norm, r) / norm(A' * b)
        @test all(resids .≤ cgls_lanczos_shift_tol)
        @test(stats.solved)
      end

      # Test b == 0
      A, b = zero_rhs(FC=FC)
      (x, stats) = cgls_lanczos_shift(A, b, shifts)
      for xi ∈ x
        @test norm(xi) == 0
      end
      @test stats.status == "x is a zero-residual solution"

      # Not implemented
      # Test with Jacobi (or diagonal) preconditioner
      # A, b, M = square_preconditioned(FC=FC)
      # shifts = [1.0; 2.0; 3.0; 4.0; 5.0; 6.0]
      # (x, stats) = cgls_lanczos_shift(A, b, shifts, M=M)
      # r = residuals(A, b, shifts, x)
      # resids = map(norm, r) / norm(b)
      # @test(all(resids .≤ cgls_lanczos_shift_tol))
      # @test(stats.solved)

      # test callback function
      A, b = symmetric_definite(FC=FC)
      shifts = [1.0; 2.0; 3.0; 4.0; 5.0; 6.0]
      solver = CglsLanczosShiftSolver(A, b, length(shifts))
      tol = 1.0e-1
      cb_n2 = TestCallbackN2LSShifts(A, b, shifts, tol = tol)
      cgls_lanczos_shift!(solver, A, b, shifts, atol = 0.0, rtol = 0.0, callback = cb_n2)
      @test solver.stats.status == "user-requested exit"
      @test cb_n2(solver)

      @test_throws TypeError cg_lanczos_shift(A, b, shifts, callback = solver -> "string", history = true)
    end
  end
end
