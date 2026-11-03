@testset "minares" begin
  minares_tol = 1.0e-6

  for FC in (Float64, ComplexF64)
    @testset "Data Type: $FC" begin

      # Cubic spline matrix.
      A, b = symmetric_definite(FC=FC)
      (x, stats) = minares(A, b)
      r = b - A * x
      resid = norm(r) / norm(b)
      @test(resid ≤ minares_tol * norm(A) * norm(x))
      @test(stats.solved)

      # Symmetric indefinite variant.
      A, b = symmetric_indefinite(FC=FC)
      (x, stats) = minares(A, b)
      r = b - A * x
      resid = norm(r) / norm(b)
      @test(resid ≤ minares_tol * norm(A) * norm(x))
      @test(stats.solved)

      # Sparse Laplacian.
      A, b = sparse_laplacian(FC=FC)
      (x, stats) = minares(A, b)
      r = b - A * x
      resid = norm(r) / norm(b)
      @test(resid ≤ minares_tol * norm(A) * norm(x))
      @test(stats.solved)

      # Symmetric indefinite variant, almost singular.
      A, b = almost_singular(FC=FC)
      (x, stats) = minares(A, b)
      r = b - A * x
      resid = norm(r) / norm(b)
      @test(resid ≤ minares_tol * norm(A) * norm(x))
      @test(stats.solved)

      # Test b == 0
      A, b = zero_rhs(FC=FC)
      (x, stats) = minares(A, b)
      @test norm(x) == 0
      @test stats.status == "x is a zero-residual solution"

      # Singular inconsistent system
      A, b = square_inconsistent(FC=FC)
      (x, stats) = minares(A, b)
      r = b - A * x
      Aresid = norm(A*r) / norm(A*b)
      @test(Aresid ≤ minares_tol)
      # @test stats.inconsistent

      # Symmetric inconsistent system
      A, b = symmetric_inconsistent()
      (x, stats) = minares(A, b)
      r = b - A * x
      Aresid = norm(A*r) / norm(A*b)
      @test(Aresid ≤ minares_tol)
      # @test stats.inconsistent

      # Shifted system
      A, b = symmetric_indefinite(FC=FC)
      λ = 2.0
      (x, stats) = minares(A, b, λ=λ)
      r = b - (A + λ*I) * x
      resid = norm(r) / norm(b)
      @test(resid ≤ minares_tol * norm(A) * norm(x))
      @test(stats.solved)

      # Test with Jacobi (or diagonal) preconditioner
      A, b, M = square_preconditioned(FC=FC)
      (x, stats) = minares(A, b, M=M)
      r = b - A * x
      resid = sqrt(real(dot(r, M * r))) / sqrt(real(dot(b, M * b)))
      @test(resid ≤ minares_tol * norm(A) * norm(x))
      @test(stats.solved)

      # Test with a dense Hermitian positive-definite preconditioner
      A, b = symmetric_indefinite(FC=FC)
      n = length(b)
      X = rand(FC, n, n)
      M = Matrix(Hermitian(X * X' + n * I))
      for (P, ldiv) in ((M, false), (cholesky(Hermitian(inv(M))), true))
        (x, stats) = minares(A, b, M=P, ldiv=ldiv, history=true)
        r = b - A * x
        @test(norm(r) ≤ minares_tol * norm(A) * norm(x))
        @test(stats.solved)
        # The residual norms are estimated in the M-norm.
        @test stats.residuals[end] ≈ sqrt(real(dot(r, M * r))) atol=minares_tol * norm(b)
      end

      # Warm start and shift with a preconditioner
      x0 = rand(FC, n)
      for λ in (0.0, 0.5)
        (x, stats) = minares(A, b, x0, M=M, λ=λ)
        r = b - (A + λ * I) * x
        @test(norm(r) ≤ minares_tol * norm(A) * norm(x))
        @test(stats.solved)
      end

      # The estimates are ‖rₖ‖_M and ‖AMrₖ‖_M with a preconditioner
      for k in 1:5
        (x, stats) = minares(A, b, M=M, itmax=k, atol=0.0, rtol=0.0, Artol=0.0, history=true)
        r = b - A * x
        @test stats.residuals[end] ≈ sqrt(real(dot(r, M * r)))
        @test stats.Aresiduals[end] ≈ sqrt(real(dot(A * M * r, M * (A * M * r))))
      end

      # Test the in-place method with and without a preconditioner
      workspace = MinaresWorkspace(A, b)
      for P in (M, I, M)
        minares!(workspace, A, b, M=P)
        r = b - A * workspace.x
        @test(norm(r) ≤ minares_tol * norm(A) * norm(workspace.x))
        @test(workspace.stats.solved)
      end

      # test callback function
      A, b = sparse_laplacian(FC=FC)
      workspace = MinaresWorkspace(A, b)
      tol = 1.0
      cb_n2 = TestCallbackN2(A, b, tol = tol)
      minares!(workspace, A, b, atol = 0.0, rtol = 0.0, Artol = 0.0, callback = cb_n2)
      @test workspace.stats.status == "user-requested exit"
      @test cb_n2(workspace)

      @test_throws TypeError minares(A, b, callback = workspace -> "string", history = true)
    end
  end

  @testset "complex_symmetric" begin
    for FC in (ComplexF64, ComplexF32)
      T = real(FC)
      tol = T === Float32 ? T(1.0e-3) : T(1.0e-6)

      # Nonsingular complex symmetric system.
      A, b = complex_symmetric_definite(FC=FC)
      (x, stats) = minares(A, b; complex_symmetric=true)
      @test norm(x - A \ b) / norm(A \ b) ≤ tol
      @test stats.solved

      # Tiny problem sizes n = 1, ..., 4.
      for n in 1:4
        An, bn = complex_symmetric_definite(n, FC=FC)
        xtrue = Matrix(An) \ bn
        (xn, statsn) = minares(An, bn; complex_symmetric=true)
        @test norm(xn - xtrue) / norm(xtrue) ≤ tol
        @test statsn.solved
      end

      # Warm start.
      x0 = (A \ b) .+ FC(0.01) .* ones(FC, size(A, 1))
      (x, stats) = minares(A, b, x0; complex_symmetric=true)
      @test norm(x - A \ b) / norm(A \ b) ≤ tol

      # Real shift.
      λ = T(0.3)
      (x, stats) = minares(A, b; complex_symmetric=true, λ=λ)
      r = b - (A + λ * I) * x
      @test norm(r) / norm(b) ≤ tol * norm(A) * norm(x)
      @test stats.solved

      # Singular, inconsistent: lifted to the minimum-norm solution.
      As, bs = complex_symmetric_inconsistent(FC=FC)
      (x, stats) = minares(As, bs; complex_symmetric=true)
      r = bs - As * x
      @test norm(As' * r) / norm(As' * bs) ≤ tol
      xmn = pinv(Matrix(As)) * bs
      @test norm(x - xmn) / norm(xmn) ≤ tol
      @test stats.inconsistent

      # krylov_solve generic interface.
      workspace = krylov_workspace(Val(:minares), A, b)
      krylov_solve!(workspace, A, b; complex_symmetric=true)
      @test norm(workspace.x - A \ b) / norm(A \ b) ≤ tol

      # complex_symmetric=true is a no-op for real data.
      Ar, br = symmetric_indefinite(FC=T)
      (xr, statsr) = minares(Ar, br)
      (xs, statss) = minares(Ar, br; complex_symmetric=true)
      @test xs == xr
      @test statss.niter == statsr.niter

      # complex_symmetric=true does not support preconditioning.
      @test_throws ArgumentError minares(A, b; complex_symmetric=true, M=Diagonal(ones(FC, size(A, 1))))
    end

    # Singular, consistent complex symmetric system: the minimum-norm solution.
    A, b = complex_symmetric_singular()
    (x, stats) = minares(A, b; complex_symmetric=true)
    @test norm(x - pinv(A) * b) / norm(pinv(A) * b) ≤ 1.0e-6
    @test stats.solved

    # Singular, inconsistent: lifted to the minimum-norm solution, same residual and stationarity.
    b = b + nullspace(Matrix(A'))[:, 1]
    xmn = pinv(A) * b
    (x, stats) = minares(A, b; complex_symmetric=true)
    r = b - A * x
    @test norm(x - xmn) / norm(xmn) ≤ 1.0e-6
    @test abs(norm(r) - norm(b - A * xmn)) ≤ 1.0e-10 * norm(b)
    @test norm(A' * r) ≤ 1.0e-8 * norm(A' * b)
    @test stats.inconsistent
    @test stats.status == "found approximate minimum least-squares solution"

    # Shifted: (A - λI) + λI = A.
    (x, stats) = minares(A - 0.5I, b; complex_symmetric=true, λ=0.5)
    @test norm(x - xmn) / norm(xmn) ≤ 1.0e-6
    @test stats.inconsistent

    # Warm start: no lift, which needs x₀ = 0.
    (x, stats) = minares(A, b, nullspace(Matrix(A))[:, 1]; complex_symmetric=true)
    @test norm(A' * (b - A * x)) ≤ 1.0e-8 * norm(A' * b)
    @test !stats.inconsistent

    # Ill-conditioned nonsingular: the ‖A conj(r)‖ test fires first; the lift would raise ‖r‖, so it is rejected.
    rng = Random.MersenneTwister(10)
    Q = Matrix(qr(randn(rng, ComplexF64, 10, 10)).Q)
    A = Q * Diagonal(exp10.(range(0, -4, length=10))) * transpose(Q)
    b = randn(rng, ComplexF64, 10)
    (x, stats) = minares(A, b; complex_symmetric=true)
    @test norm(b - A * x) ≤ 1.0e-6 * norm(b)
    @test !stats.inconsistent

    # Iterates and both estimates against a dense oracle at every k. Only the
    # ‖rₖ‖ estimate check sees the LQ factorization of Uₖ.
    A, b = complex_symmetric_dense()
    # Check the BGS trial space and projected normal-residual identity independently.
    k = 6
    V, β, H = bunse_gerstner_stover(A, b, k + 1)
    Q = bgs_trial_basis(A, b, k)
    @test norm(Q * Q' - conj.(V[:, 1:k]) * transpose(V[:, 1:k])) ≤ 1.0e-10
    y = ComplexF64.(1:k) / k
    t = -H[1:k+1, 1:k] * y
    t[1] += β
    r = b - A * (conj.(V[:, 1:k]) * y)
    @test norm(A' * r - conj.(V) * (conj.(H) * t)) ≤ 1.0e-10 * norm(A' * r)
    Qₖ = Matrix(qr(Matrix(H[1:k+1, 1:k])).Q)[:, 1:k]
    N = conj.(H) * Qₖ
    @test norm(triu(N, 1)) + norm(tril(N, -3)) ≤ 1.0e-10 * norm(N)
    for k = 1:8
      workspace = MinaresWorkspace(A, b)
      minares!(workspace, A, b; complex_symmetric=true, itmax=k, atol=0.0, rtol=0.0, Artol=0.0, history=true)
      x, stats = workspace.x, workspace.stats
      r = b - A * x
      @test norm(x - cs_minares_oracle(A, b, k)) ≤ 1.0e-8 * norm(x)
      @test abs(stats.residuals[end] - norm(r)) ≤ 1.0e-8 * norm(b)
      @test abs(stats.Aresiduals[end] - norm(A' * r)) ≤ 1.0e-8 * norm(A' * b)
    end

    # Complex{BigFloat}.
    A, b = complex_symmetric_definite(FC=Complex{BigFloat})
    (x, stats) = minares(A, b; complex_symmetric=true)
    @test norm(x - Matrix(A) \ b) / norm(Matrix(A) \ b) ≤ sqrt(eps(BigFloat))
    @test stats.solved
  end

  @testset "stats[1] closed form" begin
    # residuals[1] and Aresiduals[1] are both pushed before the main loop, so
    # they never depend on itmax. Both track r₀=b-(A+λI)x₀, not b:
    # residuals[1]=‖r₀‖; Aresiduals[1]: Hermitian ‖(A+λI)r₀‖, complex
    # symmetric ‖(A+λI)conj(r₀)‖. Checked at x₀=0, at a warm start, and with
    # a nonzero shift λ combined with a warm start.
    for FC in (Float64, ComplexF32, ComplexF64)
      T = real(FC)
      A, b = symmetric_indefinite(FC=FC)
      (_, stats) = minares(A, b; history=true)
      @test stats.residuals[1] ≈ norm(b)
      @test stats.Aresiduals[1] ≈ norm(A' * b)

      x0 = rand(FC, length(b))
      r0 = b - A * x0
      (_, stats0) = minares(A, b, x0; history=true)
      @test stats0.residuals[1] ≈ norm(r0)
      @test stats0.Aresiduals[1] ≈ norm(A' * r0)

      λ = T(0.7)
      if FC != ComplexF32  # symmetric_indefinite's shift*eye(n) silently promotes ComplexF32 to ComplexF64
        r0λ = b - (A + λ * I) * x0
        (_, statsλ) = minares(A, b, x0; λ=λ, history=true)
        @test statsλ.residuals[1] ≈ norm(r0λ)
        @test statsλ.Aresiduals[1] ≈ norm((A + λ * I)' * r0λ)
      end

      FC <: Complex || continue
      Acs, bcs = complex_symmetric_definite(FC=FC)
      (_, stats) = minares(Acs, bcs; complex_symmetric=true, history=true)
      @test stats.residuals[1] ≈ norm(bcs)
      @test stats.Aresiduals[1] ≈ norm(Acs * conj.(bcs))

      x0cs = rand(FC, size(Acs, 1))
      r0cs = bcs - Acs * x0cs
      (_, stats0cs) = minares(Acs, bcs, x0cs; complex_symmetric=true, history=true)
      @test stats0cs.residuals[1] ≈ norm(r0cs)
      @test stats0cs.Aresiduals[1] ≈ norm(Acs * conj.(r0cs))

      r0csλ = bcs - (Acs + λ * I) * x0cs
      (_, statscsλ) = minares(Acs, bcs, x0cs; complex_symmetric=true, λ=λ, history=true)
      @test statscsλ.residuals[1] ≈ norm(r0csλ)
      @test statscsλ.Aresiduals[1] ≈ norm((Acs + λ * I) * conj.(r0csλ))
    end
  end

  @testset "stats[2] closed form" begin
    # The in-loop push is current (not lagged): residuals[2]/Aresiduals[2]
    # are produced once the main loop has run its first iteration, from that
    # same run's x₁. Hermitian: ‖r₁‖, ‖Ar₁‖; complex symmetric: ‖r₁‖,
    # ‖A conj(r₁)‖.
    for FC in (Float64, ComplexF64)
      T = real(FC)
      A, b = symmetric_indefinite(FC=FC)
      (x1, stats) = minares(A, b; itmax=1, history=true, atol=zero(T), rtol=zero(T), Artol=zero(T))
      r1 = b - A * x1
      @test stats.residuals[2] ≈ norm(r1)
      @test stats.Aresiduals[2] ≈ norm(A' * r1)
    end
    for FC in (ComplexF64, ComplexF32)
      T = real(FC)
      Acs, bcs = complex_symmetric_definite(FC=FC)
      (x1cs, stats) = minares(Acs, bcs; complex_symmetric=true, itmax=1, history=true, atol=zero(T), rtol=zero(T), Artol=zero(T))
      r1cs = bcs - Acs * x1cs
      @test stats.residuals[2] ≈ norm(r1cs)
      @test stats.Aresiduals[2] ≈ norm(Acs * conj.(r1cs))
    end
  end
end
