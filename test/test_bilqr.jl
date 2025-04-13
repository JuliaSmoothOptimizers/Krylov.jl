@testset "bilqr" begin
  bilqr_tol = 1.0e-6

  for FC in (Float64, ComplexF64)
    @testset "Data Type: $FC" begin

      # Test square adjoint systems.
      A, b, c = square_adjoint(FC=FC)
      (x, t, stats) = bilqr(A, b, c)

      r = b - A * x
      resid_primal = norm(r) / norm(b)
      @test(resid_primal ≤ bilqr_tol)
      @test(stats.solved_primal)

      s = c - A' * t
      resid_dual = norm(s) / norm(c)
      @test(resid_dual ≤ bilqr_tol)
      @test(stats.solved_dual)

      # Test adjoint ODEs.
      A, b, c = adjoint_ode(FC=FC)
      (x, t, stats) = bilqr(A, b, c)

      r = b - A * x
      resid_primal = norm(r) / norm(b)
      @test(resid_primal ≤ bilqr_tol)
      @test(stats.solved_primal)

      s = c - A' * t
      resid_dual = norm(s) / norm(c)
      @test(resid_dual ≤ bilqr_tol)
      @test(stats.solved_dual)

      # Test adjoint PDEs.
      A, b, c = adjoint_pde(FC=FC)
      (x, t, stats) = bilqr(A, b, c)

      r = b - A * x
      resid_primal = norm(r) / norm(b)
      @test(resid_primal ≤ bilqr_tol)
      @test(stats.solved_primal)

      s = c - A' * t
      resid_dual = norm(s) / norm(c)
      @test(resid_dual ≤ bilqr_tol)
      @test(stats.solved_dual)

      # Test bᴴc == 0
      A, b, c = bc_breakdown(FC=FC)
      (x, t, stats) = bilqr(A, b, c)
      @test stats.status == "Breakdown bᴴc = 0"

      # test callback function
      A, b, c = adjoint_pde(FC=FC)
      workspace = BilqrWorkspace(A, b)
      tol = 1.0e-1
      cb_n2 = TestCallbackN2Adjoint(A, b, c, tol = tol)
      bilqr!(workspace, A, b, c, atol = 0.0, rtol = 0.0, callback = cb_n2)
      @test workspace.stats.status == "user-requested exit"
      @test cb_n2(workspace)

      @test_throws TypeError bilqr(A, b, c, callback = solver -> "string", history = true)
    end
  end
end
