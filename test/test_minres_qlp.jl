@testset "minres_qlp" begin
  minres_qlp_tol = 1.0e-6

  for FC in (Float64, ComplexF64)
    @testset "Data Type: $FC" begin

      # Cubic spline matrix.
      A, b = symmetric_definite(FC=FC)
      (x, stats) = minres_qlp(A, b)
      r = b - A * x
      resid = norm(r) / norm(b)
      @test(resid ≤ minres_qlp_tol * norm(A) * norm(x))
      @test(stats.solved)

      # Symmetric indefinite variant.
      A, b = symmetric_indefinite(FC=FC)
      (x, stats) = minres_qlp(A, b)
      r = b - A * x
      resid = norm(r) / norm(b)
      @test(resid ≤ minres_qlp_tol * norm(A) * norm(x))
      @test(stats.solved)

      # Sparse Laplacian.
      A, b = sparse_laplacian(FC=FC)
      (x, stats) = minres_qlp(A, b)
      r = b - A * x
      resid = norm(r) / norm(b)
      @test(resid ≤ minres_qlp_tol * norm(A) * norm(x))
      @test(stats.solved)

      # Symmetric indefinite variant, almost singular.
      A, b = almost_singular(FC=FC)
      (x, stats) = minres_qlp(A, b)
      r = b - A * x
      resid = norm(r) / norm(b)
      @test(resid ≤ minres_qlp_tol * norm(A) * norm(x))
      @test(stats.solved)

      # Test b == 0
      A, b = zero_rhs(FC=FC)
      (x, stats) = minres_qlp(A, b)
      @test norm(x) == 0
      @test stats.status == "x is a zero-residual solution"

      # Singular inconsistent system
      A, b = square_inconsistent(FC=FC)
      (x, stats) = minres_qlp(A, b)
      r = b - A * x
      Aresid = norm(A*r) / norm(A*b)
      @test(Aresid ≤ minres_qlp_tol)
      @test stats.inconsistent

      # Symmetric inconsistent system
      A, b = symmetric_inconsistent()
      (x, stats) = minres_qlp(A, b)
      r = b - A * x
      Aresid = norm(A*r) / norm(A*b)
      @test(Aresid ≤ minres_qlp_tol)
      @test stats.inconsistent

      # Shifted system
      A, b = symmetric_indefinite(FC=FC)
      λ = 2.0
      (x, stats) = minres_qlp(A, b, λ=λ)
      r = b - (A + λ*I) * x
      resid = norm(r) / norm(b)
      @test(resid ≤ minres_qlp_tol * norm(A) * norm(x))
      @test(stats.solved)

      # Test with Jacobi (or diagonal) preconditioner
      A, b, M = square_preconditioned(FC=FC)
      (x, stats) = minres_qlp(A, b, M=M)
      r = b - A * x
      resid = sqrt(real(dot(r, M * r))) / norm(b)
      @test(resid ≤ minres_qlp_tol * norm(A) * norm(x))
      @test(stats.solved)

      # test callback function
      A, b = sparse_laplacian(FC=FC)
      workspace = MinresQlpWorkspace(A, b)
      tol = 1.0
      cb_n2 = TestCallbackN2(A, b, tol = tol)
      minres_qlp!(workspace, A, b, atol = 0.0, rtol = 0.0, Artol = 0.0, callback = cb_n2)
      @test workspace.stats.status == "user-requested exit"
      @test cb_n2(workspace)

      # Test linesearch
      A, b = symmetric_indefinite(FC=FC)
      workspace = MinresQlpWorkspace(A, b)
      minres_qlp!(workspace, A, b, linesearch=true)
      x, stats, npc_dir = workspace.x, workspace.stats, workspace.npc_dir
      @test stats.status == "nonpositive curvature"
      @test stats.indefinite == true
      # Verify that the returned direction indeed exhibits nonpositive curvature.
      # For both real and complex cases, ensure to take the real part.
      @test real(dot(npc_dir, A * npc_dir)) <= 0
    
      # Test Linesearch which would stop on the first call since A is negative definite
      A, b = symmetric_indefinite(FC=FC; shift = 5)
      workspace = MinresQlpWorkspace(A, b)
      minres_qlp!(workspace, A, b, linesearch=true)
      x, stats, npc_dir = workspace.x, workspace.stats, workspace.npc_dir
      @test stats.status == "nonpositive curvature"
      @test stats.niter == 1 
      @test all(x .== b)
      @test stats.solved == true
      @test stats.indefinite == true
      @test stats.npcCount == 1
      @test real(dot(npc_dir, A * npc_dir)) <= 0      

      # Test when b^TAb=0 and linesearch is true
      A, b = system_zero_quad(FC=FC)
      workspace = MinresQlpWorkspace(A, b)
      minres_qlp!(workspace, A, b, linesearch=true)
      x, stats, npc_dir = workspace.x, workspace.stats, workspace.npc_dir
      @test stats.status == "nonpositive curvature"
      @test all(x .== b)
      @test stats.solved == true
      @test stats.indefinite == true
      @test real(dot(npc_dir, A * npc_dir)) ≈ 0.0

      # Test if warm_start and linesearch are both true, it should throw an error
      A, b = symmetric_indefinite(FC=FC)
      @test_throws MethodError minres_qlp(A, b, warm_start = true, linesearch = true)

      @test_throws TypeError minres_qlp(A, b, callback = workspace -> "string", history = true)

      # Test: Ensure stats are reset when reusing workspace
      # (pᵀAp < 0)
      A = FC[
        10.0 0.0 0.0 0.0;
        0.0 8.0 0.0 0.0;
        0.0 0.0 5.0 0.0;
        0.0 0.0 0.0 -1.0
      ]
      b = FC[1.0, 1.0, 1.0, 0.1]
      
      # Initialize workspace and solve
      solver = MinresQlpWorkspace(A, b)
      minres_qlp!(solver, A, b; linesearch=true)
      
      # Verify the "npc" state was recorded
      @test solver.stats.npcCount == 1
      @test solver.stats.indefinite == true
      @test solver.stats.status == "nonpositive curvature"

      # Reuse the SAME solver on a Positive Definite System

      A = FC[
        10.0 0.0 0.0 0.0;
        0.0 8.0 0.0 0.0;
        0.0 0.0 5.0 0.0;
        0.0 0.0 0.0 1.0
      ]
      b = FC[1.0, 1.0, 1.0, 1.0]

      # Run the solver again on the same workspace
      minres_qlp!(solver, A, b; linesearch=true)

      # Verify the RESET works
      @test solver.stats.npcCount == 0
      @test solver.stats.indefinite == false
      @test solver.stats.solved == true
    end
  end

  @testset "complex_symmetric" begin
    for FC in (ComplexF64, ComplexF32)
      T = real(FC)
      tol = T === Float32 ? T(1.0e-3) : T(1.0e-6)

      # Nonsingular complex symmetric system.
      A, b = complex_symmetric_definite(FC=FC)
      (x, stats) = minres_qlp(A, b; complex_symmetric=true)
      @test norm(x - A \ b) / norm(A \ b) ≤ tol
      @test stats.solved

      # Warm start.
      x0 = (A \ b) .+ FC(0.01) .* ones(FC, size(A, 1))
      (x, stats) = minres_qlp(A, b, x0; complex_symmetric=true)
      @test norm(x - A \ b) / norm(A \ b) ≤ tol

      # Real shift.
      λ = T(0.3)
      (x, stats) = minres_qlp(A, b; complex_symmetric=true, λ=λ)
      r = b - (A + λ * I) * x
      @test norm(r) / norm(b) ≤ tol * norm(A) * norm(x)
      @test stats.solved

      # Singular, inconsistent complex symmetric system.
      As, bs = complex_symmetric_inconsistent(FC=FC)
      (x, stats) = minres_qlp(As, bs; complex_symmetric=true)
      r = bs - As * x
      Aresid = norm(As' * r) / norm(As' * bs)
      @test Aresid ≤ tol
      @test stats.inconsistent

      # krylov_solve generic interface.
      workspace = krylov_workspace(Val(:minres_qlp), A, b)
      krylov_solve!(workspace, A, b; complex_symmetric=true)
      @test norm(workspace.x - A \ b) / norm(A \ b) ≤ tol

      # Verbose output.
      io = IOBuffer()
      minres_qlp(A, b; complex_symmetric=true, verbose=1, iostream=io)
      @test occursin("MINRES-QLP: system of size", String(take!(io)))

      # Allocation-free in-place path. Isolated in its own `let` block: `x`
      # and `stats` above are reassigned from several different calls, which
      # makes them type-unstable in this scope, and @allocated on an
      # unrelated call right after can pick up GC bookkeeping for those
      # boxed values otherwise.
      nbytes = let ws = MinresQlpWorkspace(A, b)
        minres_qlp!(ws, A, b; complex_symmetric=true)
        @allocated minres_qlp!(ws, A, b; complex_symmetric=true)
      end
      @test nbytes == 0

      # complex_symmetric=true requires a complex element type.
      Ar, br = symmetric_indefinite(FC=T)
      @test_throws ErrorException minres_qlp(Ar, br; complex_symmetric=true)

      # complex_symmetric=true does not support preconditioning or linesearch.
      @test_throws ArgumentError minres_qlp(A, b; complex_symmetric=true, M=Diagonal(ones(FC, size(A, 1))))
      @test_throws ArgumentError minres_qlp(A, b; complex_symmetric=true, linesearch=true)
    end

    # Singular, consistent complex symmetric system: the minimum-norm solution.
    A, b = complex_symmetric_singular()
    (x, stats) = minres_qlp(A, b; complex_symmetric=true)
    @test norm(x - pinv(A) * b) / norm(pinv(A) * b) ≤ 1.0e-6
    @test stats.solved

    # Iterates and both estimates against a dense oracle at every k; the
    # ‖Arₖ₋₁‖ estimate lags the iterate by one step.
    A, b = complex_symmetric_dense()
    xprev = zeros(ComplexF64, length(b))
    for k = 1:8
      workspace = MinresQlpWorkspace(A, b)
      minres_qlp!(workspace, A, b; complex_symmetric=true, itmax=k, atol=0.0, rtol=0.0, Artol=0.0, history=true)
      x, stats = workspace.x, workspace.stats
      @test norm(x - cs_minres_qlp_oracle(A, b, k)) ≤ 1.0e-8 * norm(x)
      @test abs(stats.residuals[end] - norm(b - A * x)) ≤ 1.0e-8 * norm(b)
      k ≥ 2 && @test abs(stats.Aresiduals[end] - norm(A' * (b - A * xprev))) ≤ 1.0e-8 * norm(A' * b)
      xprev = copy(x)
    end

    # Complex{BigFloat}.
    A, b = complex_symmetric_definite(FC=Complex{BigFloat})
    (x, stats) = minres_qlp(A, b; complex_symmetric=true)
    @test norm(x - Matrix(A) \ b) / norm(Matrix(A) \ b) ≤ sqrt(eps(BigFloat))
    @test stats.solved
  end
end
