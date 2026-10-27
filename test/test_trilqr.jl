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

      # Test consistent Ax = b and inconsistent Aᴴt = c.
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

      # Test different types for input and output
      A, b, c = adjoint_ode(FC=FC)
      c = TestVector(c)
      workspace = TrilqrWorkspace(KrylovConstructor(b, c))
      trilqr!(workspace, A, b, c)
      @test typeof(workspace.x) === typeof(c)
      @test typeof(workspace.y) === typeof(b)
      @test workspace.stats.solved_primal
      @test workspace.stats.solved_dual

      # test callback function
      A, b, c = adjoint_pde(FC=FC)
      workspace = TrilqrWorkspace(A, b)
      tol = 1.0e-1
      cb_n2 = TestCallbackN2Adjoint(A, b, c, tol = tol)
      trilqr!(workspace, A, b, c, atol = 0.0, rtol = 0.0, callback = cb_n2)
      @test workspace.stats.status == "user-requested exit"
      @test cb_n2(workspace)

      @test_throws TypeError trilqr(A, b, c, callback = workspace -> "string", history = true)
    end
  end

  # The Krylov subspace is invariant after one iteration.
  @testset "exact breakdown" begin
    for FC in (Float64, ComplexF64)
      n = 6
      A = Matrix{FC}(2I, n, n)
      b = FC[1:n;]
      c = FC[n:-1:1;]
      for transfer_to_usymcg in (false, true)
        (x, t, stats) = trilqr(A, b, c, transfer_to_usymcg=transfer_to_usymcg)
        @test norm(x - b / 2) ≤ 1e-10 * norm(b / 2)
        @test norm(t - c / 2) ≤ 1e-10 * norm(c / 2)
        @test stats.solved_primal && stats.solved_dual
      end
    end
  end
end
