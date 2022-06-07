@testset "trilqr" begin
  trilqr_tol = 1.0e-6

  for FC in (Float64, ComplexF64)
    @testset "Data Type: $FC" begin

      # Test underdetermined adjoint systems.
      A, b, c = underdetermined_adjoint(FC=FC)
      (x, t, stats) = trilqr(A, b, c)

      r = b - A * x
      resid_primal = norm(r) / norm(b)
      @test(resid_primal ≤ trilqr_tol)
      @test(stats.solved_primal)

      s = c - A' * t
      resid_dual = norm(s) / norm(c)
      @test(resid_dual ≤ trilqr_tol)
      @test(stats.solved_dual)

      # Test square adjoint systems.
      A, b, c = square_adjoint(FC=FC)
      (x, t, stats) = trilqr(A, b, c)

      r = b - A * x
      resid_primal = norm(r) / norm(b)
      @test(resid_primal ≤ trilqr_tol)
      @test(stats.solved_primal)

      s = c - A' * t
      resid_dual = norm(s) / norm(c)
      @test(resid_dual ≤ trilqr_tol)
      @test(stats.solved_dual)

      # Test overdetermined adjoint systems
      A, b, c = overdetermined_adjoint(FC=FC)
      (x, t, stats) = trilqr(A, b, c)

      r = b - A * x
      resid_primal = norm(r) / norm(b)
      @test(resid_primal ≤ trilqr_tol)
      @test(stats.solved_primal)

      s = c - A' * t
      resid_dual = norm(s) / norm(c)
      @test(resid_dual ≤ trilqr_tol)
      @test(stats.solved_dual)

      # Test adjoint ODEs.
      A, b, c = adjoint_ode(FC=FC)
      (x, t, stats) = trilqr(A, b, c)

      r = b - A * x
      resid_primal = norm(r) / norm(b)
      @test(resid_primal ≤ trilqr_tol)
      @test(stats.solved_primal)

      s = c - A' * t
      resid_dual = norm(s) / norm(c)
      @test(resid_dual ≤ trilqr_tol)
      @test(stats.solved_dual)

      # Test adjoint PDEs.
      A, b, c = adjoint_pde(FC=FC)
      (x, t, stats) = trilqr(A, b, c)

      r = b - A * x
      resid_primal = norm(r) / norm(b)
      @test(resid_primal ≤ trilqr_tol)
      @test(stats.solved_primal)

      s = c - A' * t
      resid_dual = norm(s) / norm(c)
      @test(resid_dual ≤ trilqr_tol)
      @test(stats.solved_dual)

      # Test consistent Ax = b and inconsistent Aᵀt = c.
      A, b, c = rectangular_adjoint(FC=FC)
      (x, t, stats) = trilqr(A, b, c)

      r = b - A * x
      resid_primal = norm(r) / norm(b)
      @test(resid_primal ≤ trilqr_tol)
      @test(stats.solved_primal)

      s = c - A' * t
      resid_dual = norm(s) / norm(c)
      Aresid_dual = norm(A * s) / norm(A * c)
      @test(Aresid_dual ≤ trilqr_tol)
      @test(stats.solved_dual)

      # test callback function
      A, b, c = adjoint_pde(FC=FC)
      solver = TrilqrSolver(A, b)
      tol = 1.0e-1
      cb_n2 = TestCallbackN2Adjoint(A, b, c, tol = tol)
      trilqr!(solver, A, b, c, atol = 0.0, rtol = 0.0, callback = solver -> cb_n2(solver))
      @test solver.stats.status == "user-requested exit"
      @test cb_n2(solver)

      @test_throws TypeError trilqr(A, b, c, callback = solver -> "string", history = true)
    end
  end
end
