function residuals_ls(A, b, shifts, x)
  nshifts = length(shifts)
  r = [ (A' * (b - A * x[i]) - shifts[i] * x[i]) for i = 1 : nshifts ]
  return r
end

@testset "lsqr_shift" begin
  lsqr_shift_tol = 1.0e-6

  for FC in (Float64, ComplexF64)
    @testset "Data Type: $FC" begin

      for npower = 1 : 4
        (b, A, D, HY, HZ, Acond, rnorm) = test(40, 40, 4, npower, 0)  # No regularization.
        shifts = [1.0; 2.0; 3.0; 4.0; 5.0; 6.0]
        (x, stats) = lsqr_shift(A, b, shifts)
        r = residuals_ls(A, b, shifts, x)
        resids = map(norm, r) / norm(A' * b)
        @test all(resids .≤ lsqr_shift_tol)
        @test(stats.solved)
      end

      # Test b == 0
      A, b = zero_rhs(FC=FC)
      (x, stats) = lsqr_shift(A, b, shifts)
      for xi ∈ x
        @test norm(xi) == 0
      end
      for status in stats.status
        @test status == "x = 0 is a zero-residual solution"
      end

      #=
      # Not implemented
      # Test with Jacobi (or diagonal) preconditioner
      A, b, M = square_preconditioned(FC=FC)
      shifts = [1.0; 2.0; 3.0; 4.0; 5.0; 6.0]
      (x, stats) = lsqr_shift(A, b, shifts, M=M)
      r = residuals(A, b, shifts, x)
      resids = map(norm, r) / norm(A' * b)
      @test(all(resids .≤ lsqr_shift_tol))
      @test(stats.solved)
      =#
      

      # test callback function
      A, b = symmetric_definite(FC=FC)
      shifts = [1.0; 2.0; 3.0; 4.0; 5.0; 6.0]
      solver = LsqrShiftSolver(A, b, length(shifts))
      tol = 1.0e-1
      cb_n2 = TestCallbackN2LSShifts(A, b, shifts, tol = tol)
      lsqr_shift!(solver, A, b, shifts, atol = 0.0, rtol = 0.0, callback = cb_n2)
      for status in solver.stats.status
        @test status == "user-requested exit"
      end
      @test cb_n2(solver)

      # Why BoundsError instead?
      # @test_throws TypeError lsqr_shift(A, b, shifts, callback = solver -> "string", history = true)
    end
  end
end
