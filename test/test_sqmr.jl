@testset "sqmr" begin
  sqmr_tol = 1.0e-6

  for FC in (Float64, ComplexF64)
    @testset "Data Type: $FC" begin

      # Symmetric and positive definite system.
      A, b = symmetric_definite(FC=FC)
      (x, stats) = sqmr(A, b)
      r = b - A * x
      resid = norm(r) / norm(b)
      @test(resid ≤ sqmr_tol)
      @test(stats.solved)

      # Symmetric indefinite variant.
      A, b = symmetric_indefinite(FC=FC)
      (x, stats) = sqmr(A, b)
      r = b - A * x
      resid = norm(r) / norm(b)
      @test(resid ≤ sqmr_tol)
      @test(stats.solved)

      # Sparse Laplacian.
      A, b = sparse_laplacian(FC=FC)
      (x, stats) = sqmr(A, b)
      r = b - A * x
      resid = norm(r) / norm(b)
      @test(resid ≤ sqmr_tol)
      @test(stats.solved)

      # Test b == 0
      A, b = zero_rhs(FC=FC)
      (x, stats) = sqmr(A, b)
      @test norm(x) == 0
      @test stats.status == "x is a zero-residual solution"

      # Symmetric indefinite preconditioning.
      A, b, M = square_preconditioned(FC=FC)
      (x, stats) = sqmr(A, b, M=M)
      r = b - A * x
      resid = norm(M * r) / norm(M * b)
      @test(resid ≤ sqmr_tol)
      @test(stats.solved)

      # SQMR via generic interface.
      (x, stats) = krylov_solve(Val(:sqmr), A, b, M=M)
      r = b - A * x
      resid = norm(M * r) / norm(M * b)
      @test(resid ≤ sqmr_tol)
      @test(stats.solved)

      # SQMR warm start.
      x0 = fill!(similar(b), FC(0.1))
      (x, stats) = sqmr(A, b, x0, M=M)
      r = b - A * x
      resid = norm(M * r) / norm(M * b)
      @test(resid ≤ sqmr_tol)
      @test(stats.solved)

      # SQMR workspace API.
      workspace = SqmrWorkspace(A, b)
      sqmr!(workspace, A, b, M=M)
      @test(workspace.stats.solved)

      # SQMR with left-division preconditioning.
      nA = size(A, 1)
      M_true = FC(nA) * I  # true preconditioner M = n*I, not its inverse
      (x, stats) = sqmr(A, b, M=M_true, ldiv=true)
      r = b - A * x
      resid = norm(M_true \ r) / norm(M_true \ b)
      @test(resid ≤ sqmr_tol)
      @test(stats.solved)

      # SQMR without preconditioner (M=I) matches MINRES.
      A2, b2 = symmetric_definite(FC=FC)
      (x_sqmr, stats_sqmr) = sqmr(A2, b2)
      (x_minres, stats_minres) = minres(A2, b2)
      @test(norm(x_sqmr - x_minres) ≤ sqmr_tol * norm(x_minres))
      @test(stats_sqmr.solved)
      @test(stats_minres.solved)

      # SQMR with general SPD preconditioner (Jacobi: diag(A)).
      M_spd = spdiagm(0 => 4 * ones(FC, size(A2, 1)))
      (x, stats) = sqmr(A2, b2, M=M_spd)
      r = b2 - A2 * x
      resid = norm(r) / norm(b2)
      @test(resid ≤ sqmr_tol)
      @test(stats.solved)

      # SQMR on symmetric indefinite system.
      A3, b3 = symmetric_indefinite(FC=FC)
      (x, stats) = sqmr(A3, b3)
      r = b3 - A3 * x
      resid = norm(r) / norm(b3)
      @test(resid ≤ sqmr_tol)
      @test(stats.solved)

      # SQMR with symmetric indefinite preconditioner (no breakdown).
      A4, b4 = symmetric_definite(FC=FC)
      nA4 = size(A4, 1)
      M_indef = spdiagm(0 => [ones(FC, 6); -ones(FC, nA4-6)])
      (x, stats) = sqmr(A4, b4, M=M_indef)
      @test !occursin("Breakdown", stats.status)
      @test(stats.solved)

      # SQMR on a 2x2 symmetric system.
      A = FC[2.0 1.0; 1.0 2.0]
      b = FC[1.0; 2.0]
      (x, stats) = sqmr(A, b)
      r = b - A * x
      resid = norm(r) / norm(b)
      @test(resid ≤ sqmr_tol)

      # Test that stats tracking works (history mode).
      A, b = sparse_laplacian(FC=FC)
      (x, stats) = sqmr(A, b, history=true)
      @test(stats.solved)
      @test(length(stats.residuals) > 0)

      # Test callback function.
      A, b = sparse_laplacian(FC=FC)
      workspace = SqmrWorkspace(A, b)
      tol = 1.0e-1
      cb_n2 = TestCallbackN2(A, b, tol = tol)
      sqmr!(workspace, A, b, callback = cb_n2)
      @test workspace.stats.status == "user-requested exit"
      @test cb_n2(workspace)

      @test_throws TypeError sqmr(A, b, callback = workspace -> "string", history = true)
    end
  end
end
