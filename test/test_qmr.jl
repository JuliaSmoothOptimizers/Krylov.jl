@testset "qmr" begin
  qmr_tol = 1.0e-6
  sqmr_tol = 1.0e-6

  for FC in (Float64, ComplexF64)
    @testset "Data Type: $FC" begin

      # Symmetric and positive definite system.
      A, b = symmetric_definite(FC=FC)
      (x, stats) = qmr(A, b)
      r = b - A * x
      resid = norm(r) / norm(b)
      @test(resid ≤ qmr_tol)
      @test(stats.solved)

      # Symmetric indefinite variant.
      A, b = symmetric_indefinite(FC=FC)
      (x, stats) = qmr(A, b)
      r = b - A * x
      resid = norm(r) / norm(b)
      @test(resid ≤ qmr_tol)
      @test(stats.solved)

      # Nonsymmetric and positive definite systems.
      A, b = nonsymmetric_definite(FC=FC)
      (x, stats) = qmr(A, b)
      r = b - A * x
      resid = norm(r) / norm(b)
      @test(resid ≤ qmr_tol)
      @test(stats.solved)

      # Nonsymmetric indefinite variant.
      A, b = nonsymmetric_indefinite(FC=FC)
      (x, stats) = qmr(A, b)
      r = b - A * x
      resid = norm(r) / norm(b)
      @test(resid ≤ qmr_tol)
      @test(stats.solved)

      # Sparse Laplacian.
      A, b = sparse_laplacian(FC=FC)
      (x, stats) = qmr(A, b)
      r = b - A * x
      resid = norm(r) / norm(b)
      @test(resid ≤ qmr_tol)
      @test(stats.solved)

      # Test b == 0
      A, b = zero_rhs(FC=FC)
      (x, stats) = qmr(A, b)
      @test norm(x) == 0
      @test stats.status == "x is a zero-residual solution"

      # Poisson equation in polar coordinates.
      A, b = polar_poisson(FC=FC)
      (x, stats) = qmr(A, b)
      r = b - A * x
      resid = norm(r) / norm(b)
      @test(resid ≤ qmr_tol)
      @test(stats.solved)

      # Left preconditioning
      A, b, M = square_preconditioned(FC=FC)
      (x, stats) = qmr(A, b, M=M)
      r = b - A * x
      resid = norm(M * r) / norm(M * b)
      @test(resid ≤ qmr_tol)
      @test(stats.solved)

      # Right preconditioning
      A, b, N = square_preconditioned(FC=FC)
      (x, stats) = qmr(A, b, N=N)
      r = b - A * x
      resid = norm(r) / norm(b)
      @test(resid ≤ qmr_tol)
      @test(stats.solved)

      # Split preconditioning
      A, b, M, N = two_preconditioners(FC=FC)
      (x, stats) = qmr(A, b, M=M, N=N)
      r = b - A * x
      resid = norm(M * r) / norm(M * b)
      @test(resid ≤ qmr_tol)
      @test(stats.solved)

      # SQMR with symmetric indefinite preconditioning.
      A, b, M = square_preconditioned(FC=FC)
      (x, stats) = sqmr(A, b, M=M)
      r = b - A * x
      resid = norm(M * r) / norm(M * b)
      @test(resid ≤ sqmr_tol)
      @test(stats.solved)

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
      M_indef = spdiagm(0 => [ones(FC, 5); -ones(FC, nA4-5)])
      (x, stats) = sqmr(A4, b4, M=M_indef)
      @test(stats.status != "Lanczos breakdown ⟨v̂ₖ₊₁, M⁻¹v̂ₖ₊₁⟩ = 0")

      # Test bᴴc == 0
      A, b, c = bc_breakdown(FC=FC)
      (x, stats) = qmr(A, b, c=c)
      @test stats.status == "Breakdown bᴴc = 0"

      # test callback function
      workspace = QmrWorkspace(A, b)
      tol = 1.0e-1
      cb_n2 = TestCallbackN2(A, b, tol = tol)
      qmr!(workspace, A, b, callback = cb_n2)
      @test workspace.stats.status == "user-requested exit"
      @test cb_n2(workspace)

      @test_throws TypeError qmr(A, b, callback = workspace -> "string", history = true)
    end
  end
end
