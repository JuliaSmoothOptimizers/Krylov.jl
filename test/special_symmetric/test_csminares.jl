function check_reference(A, b)
    x, s = projected_solve(A, b; rtol=1e-11)
    target = pinv(A; rtol=1e-12) * b
    @test s.solved
    @test s.closed
    @test all(isfinite, x)
    @test x ≈ target rtol=2e-9 atol=2e-10
    @test norm(A' * (b-A*x)) <= 2e-10 * max(norm(A)*norm(b), 1)
    @test s.orthogonality <= 1e-12
    @test all(diff(s.aresiduals) .<= 1e-10 * max(s.aresiduals[1], 1))
    @test s.projected_aresiduals ≈ s.aresiduals atol=1e-10 rtol=1e-9
    @test s.basis_products <= s.niter + 1
    @test s.diagnostic_products == 1 + 2s.niter
    # Independent oracle: minimize over the returned iterate's trial space,
    # constructed from raw conjugate-power products instead of the
    # implementation's own projections.
    raw = reshape(copy(b), :, 1)
    for k in 1:s.niter
        Z = Matrix(qr(conj.(raw)).Q)[:, 1:k]
        oracle = Z * (pinv(A' * A * Z; rtol=1e-11) * (A' * b))
        @test norm(A'*(b-A*s.iterates[k])) ≈ norm(A'*(b-A*oracle)) atol=5e-9
        v = A * conj.(raw[:, end])
        raw = hcat(raw, iszero(norm(v)) ? v : v / norm(v))
    end
end

@testset "CS-MinAres full-basis reference" begin
    for n in (3, 6, 9), rank in (n, n-1)
        A = cs_matrix(n, rank)
        check_reference(A, randn(RNG, ComplexF64, n))
        check_reference(A, A * randn(RNG, ComplexF64, n))
    end
end

@testset "Adjoint objective and singular completion" begin
    A = ComplexF64[1 im; im -1]
    b = ComplexF64[1, im]
    @test norm(A*b) == 0
    @test norm(A'*b) > 0
    x, s = csminares_oracle(A, b)
    @test x ≈ pinv(A)*b
    @test norm(b-A*x) < 1e-13
    @test s.niter == 1
    A = Diagonal([1., 0.])
    b = [1., 1.]
    early, s = csminares_oracle(A, b; completion=:stationary)
    @test norm(A'*(b-A*early)) < 1e-12
    @test norm(early - pinv(A)*b) > 0.5
    lifted, applied = minimum_norm_refinement(A, b, early)
    @test applied
    @test lifted ≈ pinv(A)*b atol=1e-12
    full, s = csminares_oracle(A, b)
    @test full ≈ pinv(A)*b atol=1e-12
    @test s.niter == 2
    @test_throws ArgumentError minimum_norm_refinement(A, b, zeros(2))

    # A small ORDINARY residual does not by itself mean x is stationary
    # (small NORMAL residual); the stationarity check must run regardless.
    A = Diagonal([1e9, 1.0])
    b = [1e-9, 1.0]
    x = [0.0, 1.0]
    @test norm(b - A*x) <= 1e-8 * norm(b)
    @test_throws ArgumentError minimum_norm_refinement(A, b, x)
end

@testset "Edges, scales, sparse matrices and validation" begin
    for A in (zeros(3, 3), zeros(ComplexF64, 3, 3))
        for b in (zeros(3), ones(3))
            x, s = csminares_oracle(A, b)
            @test x == zeros(3)
            @test s.solved && s.niter == 0
        end
    end
    for scale in (1e-50, 1.0, 1e50)
        A = scale .* ComplexF64[2+im 1; 1 3-im]
        b = A * ComplexF64[1, -2im]
        x, s = csminares_oracle(A, b; rtol=1e-11)
        @test s.solved
        @test x ≈ [1, -2im] rtol=1e-10
    end
    A = ComplexF32[2+im 1; 1 3-im]
    x, s = csminares_oracle(A, A * ComplexF32[1, -2im])
    @test eltype(x) == ComplexF32
    @test x ≈ [1, -2im] rtol=1e-4
    A = im * spdiagm(-1=>fill(-1., 7), 0=>fill(3., 8), 1=>fill(-1., 7))
    x, s = csminares_oracle(A, A * ones(8))
    @test x ≈ ones(8) atol=1e-10
    x, s = csminares_oracle(reshape([2im], 1, 1), ComplexF64[3+im])
    @test x ≈ [(3+im)/(2im)]
    @test_throws DimensionMismatch csminares_oracle(ones(2, 3), ones(2))
    @test_throws DimensionMismatch csminares_oracle(ones(2, 2), ones(3))
    @test_throws ArgumentError csminares_oracle([1. 2; 0 1], ones(2))
    @test_throws ArgumentError csminares_oracle(zeros(2, 2), [NaN, 1])
    @test_throws ArgumentError csminares_oracle([Inf 0.; 0 1], ones(2))
    @test_throws ArgumentError csminares_oracle(ones(2, 2), ones(2); rtol=-1)
    @test_throws ArgumentError csminares_oracle(ones(2, 2), ones(2); maxiter=0)
    @test_throws ArgumentError csminares_oracle(zeros(0, 0), zeros(0))
    A = cs_matrix(8)
    _, s = csminares_oracle(A, ones(8); maxiter=1, rtol=0, atol=0)
    @test s.status == :iteration_limit
    @test s.niter == 1 && !s.closed
end

@testset "Saunders enhancement and subspace comparisons" begin
    for rank in (8, 5)
        A = cs_matrix(8, rank)
        b = randn(RNG, ComplexF64, 8)
        x, s = csminares_range(A, b; rtol=1e-11)
        @test s.solved
        @test x ≈ pinv(A)*b atol=1e-9
        N = nullspace(A)
        for xk in s.iterates
            @test norm(N'*xk) < 1e-10
        end
        @test s.aresiduals ≈ s.projected_aresiduals atol=1e-10
        # At even k, the original Saunders trial space contains the LSMR
        # normal-equation Krylov space with half as many basis vectors.
        for j in 1:3
            xcs, _ = csminares_oracle(A, b; maxiter=2j)
            G = reshape(A'*b, :, 1)
            for k in 2:j
                G = hcat(G, A'*(A*G[:, end]))
            end
            Q = Matrix(qr(G).Q)[:, 1:j]
            xl = Q * ((A'*A*Q) \ (A'*b))
            @test norm(A'*(b-A*xcs)) <= norm(A'*(b-A*xl)) + 1e-10
        end
    end
    # A general complex shift changes the Saunders space; do not reuse T-σI.
    A = Diagonal([1., 2., 4.])
    b = ComplexF64[1, im, 1+im]
    Q = Matrix(qr(hcat(b, A*conj.(b))).Q)[:, 1:2]
    shifted = (A-I)*conj.(b)
    @test norm(shifted - Q*(Q'*shifted)) > 0.1
    x, s = csminares_range(zeros(2, 2), ones(2))
    @test x == zeros(2) && s.solved
end

@testset "Product-only operators" begin
    A = ComplexF64[2+im 1; 1 3-im]
    b = ComplexF64[1, 2im]
    op = ProductOnly(A)
    for method in (csminares_oracle, csminares_range)
        x, stats = method(op, b; check=false)
        @test stats.solved
        @test x ≈ A \ b rtol=1e-9
    end
end

@testset "Staged bandwidth-two factorization" begin
    k = 6
    alphas = ComplexF64[2 + 0.2im, -1 + 0.4im, 3 - 0.7im, 0.5 + 0.3im,
                        -2 - 0.1im, 1.5 + 0.8im, 0.7 - 0.6im]
    betas = Float64[0, 0.8, 1.1, 0.6, 1.3, 0.9, 0.7, 1.2]
    function projected_tridiagonal(ncols)
        C = zeros(ComplexF64, ncols + 1, ncols)
        for j in 1:ncols
            C[j, j] = alphas[j]
            C[j + 1, j] = betas[j + 1]
            j < ncols && (C[j, j + 1] = betas[j + 1])
        end
        return C
    end

    Ck = projected_tridiagonal(k)
    Ckp1 = projected_tridiagonal(k + 1)
    F = qr(Ck)
    Qk, Rk = Matrix(F.Q), Matrix(F.R)
    Nk = conj.(Ckp1) * Qk
    Bk = conj.(Ckp1) * Ck
    Uk = Matrix(qr(Nk).R)

    @test Bk ≈ Nk * Rk atol=2e-14
    @test Nk[1:k, :] ≈ Rk' atol=2e-14
    @test norm(triu(Rk, 3)) ≤ 2e-14 * norm(Rk)
    @test norm(triu(Uk, 3)) ≤ 2e-14 * norm(Uk)
    Noutside = copy(Nk)
    for j in 1:k
        Noutside[j:min(j + 2, k + 2), j] .= 0
    end
    @test norm(Noutside) ≤ 2e-14 * norm(Nk)
end

@testset "Short recurrence versus full projected iterates" begin
    rng = MersenneTwister(20261008)
    for n in (6, 9), rank in (n, n-1), compatible in (false, true)
        A = cs_matrix(n, rank)
        b = randn(rng, ComplexF64, n)
        compatible && (b = A * b)
        target = pinv(A; rtol=1e-12) * b
        x, full = projected_solve(A, b; rtol=1e-10, factorization=:svd)
        for reorthogonalize in (true, false)
            y, s = recurrence_solve(A, b; rtol=1e-10, reorthogonalize, history=true)
            @test s.solved
            @test s.factorization == :short && s.reorthogonalized == reorthogonalize
            if reorthogonalize
                # A nullspace direction found one step before detected closure
                # ends the recurrence early with the same minimum-norm answer.
                @test s.status in (:invariant_subspace, :rank_truncated)
                @test s.niter <= full.niter
                s.status == :invariant_subspace && @test s.niter == full.niter && s.closed == full.closed
            else
                @test s.status in (:invariant_subspace, :roundoff, :rank_truncated)
            end
            @test y ≈ x atol=1e-10 rtol=1e-10
            @test y ≈ target atol=1e-10 rtol=1e-10
            m = min(s.niter, full.niter)
            for k in 1:m
                @test s.iterates[k] ≈ full.iterates[k] atol=1e-10 rtol=1e-10
            end
            @test s.projected_aresiduals[1:m+1] ≈ full.projected_aresiduals[1:m+1] atol=1e-10 rtol=1e-10
            @test s.aresiduals[end] ≈ norm(A' * (b - A * y)) atol=1e-13
            @test all(diff(s.aresiduals) .<= 1e-10 * max(s.aresiduals[1], 1))
            @test s.diagnostic_products == 1 + 2s.niter
            @test s.niter <= s.basis_products <= s.niter + 1
        end
    end
end

@testset "Short-recurrence range start and minimum norm" begin
    rng = MersenneTwister(20261010)
    for rank in (8, 6)
        A = cs_matrix(8, rank)
        b = randn(rng, ComplexF64, 8)
        x, full = csminares_range(A, b; completion=:invariant)
        y, s = csminares_short(A, b; start=:normal, reorthogonalize=true, history=true)
        @test s.niter == full.niter
        @test y ≈ x atol=1e-10
        @test y ≈ pinv(A) * b atol=1e-10
        N = nullspace(A)
        @test all(v -> norm(N' * v) < 1e-10, s.iterates)
    end
    # Exactly stationary early iterate versus the minimum-norm closure.
    A, b = Diagonal([1., 0.]), [1., 1.]
    early, se = csminares_short(A, b; completion=:stationary, history=true)
    full, sf = csminares_short(A, b)
    @test se.niter == 1 && early ≈ [1., 1.]
    @test sf.closed && sf.status == :invariant_subspace
    @test full ≈ [1., 0.] atol=1e-14
    @test sf.pivots[end] <= 1e-14
    # A*b = 0 but A'*b != 0: the adjoint must drive the objective.
    A = ComplexF64[1 im; im -1]
    for b in (ComplexF64[1, im], ComplexF64[1, 0])
        x, s = csminares_short(A, b)
        @test s.solved
        @test x ≈ pinv(A) * b atol=1e-12
    end
end

@testset "Short recurrence storage, precision and operators" begin
    rng = MersenneTwister(20261011)
    n = 60
    U = Matrix(qr(randn(rng, ComplexF64, n, n)).Q)
    A = U * Diagonal(range(0.5, 2; length=n)) * transpose(U)
    b = randn(rng, ComplexF64, n)
    x, s = csminares_short(A, b; rtol=1e-12)
    @test s.solved && s.status in (:invariant_subspace, :roundoff)
    @test x ≈ A \ b rtol=1e-9
    @test isempty(s.iterates) && length(s.residuals) == 2
    @test isnan(s.orthogonality)
    @test s.diagnostic_products == 3
    y, st = csminares_short(A, b; rtol=1e-8, completion=:stationary)
    @test st.status == :stationary && st.niter < s.niter
    @test norm(A' * (b - A * y)) <= 1e-7 * norm(A' * b)
    for op in (A, ProductOnly(A), sparse(A))
        z, so = csminares_short(op, b; check=false, rtol=1e-12)
        @test so.solved
        @test z ≈ x rtol=1e-10
    end
    for T in (Float32, ComplexF32)
        A = T[3 1; 1 2]
        T <: Complex && (A .*= T(1 + im))
        x, s = csminares_short(A, A * T[1, -2])
        @test eltype(x) == T && s.solved
        @test x ≈ T[1, -2] rtol=1e-4
    end
    for A in (zeros(3, 3), zeros(ComplexF64, 3, 3)), b in (zeros(3), ones(3))
        x, s = csminares_short(A, b)
        @test iszero(x) && s.status == :stationary_zero
    end
    x, s = csminares_short(Diagonal([1., 0.]), [0., 1.])
    @test iszero(x) && s.status == :stationary_zero
    x, s = csminares_short(reshape([2im], 1, 1), ComplexF64[3 + im])
    @test x ≈ [(3 + im) / (2im)] && s.closed && s.niter == 1
    _, s = csminares_short(cs_matrix(8), ones(ComplexF64, 8); maxiter=1, rtol=0, ranktol=0)
    @test s.niter == 1 && s.status == :iteration_limit
end

@testset "Short recurrence validation" begin
    @test_throws DimensionMismatch csminares_short(ones(2, 3), ones(2))
    @test_throws DimensionMismatch csminares_short(ones(2, 2), ones(3))
    @test_throws ArgumentError csminares_short(zeros(0, 0), zeros(0))
    @test_throws ArgumentError csminares_short([1. 2; 0 1], ones(2))
    @test_throws ArgumentError csminares_short(ones(2, 2), [NaN, 1.])
    for kw in ((; rtol=-1), (; atol=Inf), (; ranktol=NaN), (; breakdown_tol=-1),
               (; maxiter=0), (; maxiter=1.5), (; completion=:unknown), (; start=:unknown),
               (; history=1), (; reorthogonalize=1))
        @test_throws ArgumentError csminares_short(ones(2, 2), ones(2); kw...)
    end
end
