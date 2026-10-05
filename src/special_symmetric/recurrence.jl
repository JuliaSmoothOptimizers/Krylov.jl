# Short-recurrence CS-MinAres: an incremental counterpart of `projected_solve`.

# Apply the Hermitian reflection [c s; conj(s) -c] to the pair (x, y).
reflect(c, s, x, y) = (c*x + s*y, conj(s)*x - c*y)

"""
    recurrence_solve(A, b; start=:rhs, completion=:invariant, maxiter=4length(b),
                     atol=0, rtol=√eps, ranktol=n*eps, breakdown_tol=100eps,
                     reorthogonalize=false, history=false, check=true)

Short-recurrence CS-MinAres on the trial spaces of `projected_solve`: minimizes
norm(A'*(b-A*x)) by an incremental Givens QR of the nested banded projected
matrices B_k. R_k has upper bandwidth 4, so each solution direction uses the four
previous ones, and storage is a fixed number of length-n vectors. In exact
arithmetic B_k has full column rank before closure. When a pivot is below
`sqrt(eps)*norm(B)`, one extra product (`null_products`) tests whether the
would-be direction u satisfies `norm(A*u) <= sqrt(ranktol)*norm(A)*norm(u)`; such a
nullspace direction is projected out of x to select minimum norm, and the
iteration ends (`:rank_truncated` before detected closure). Without
reorthogonalization it also stops (`:roundoff`) after two consecutive steps
with the estimated normal residual below
`ranktol*(norm(B)*norm(x) + norm(A'*b))`, since closure may then go undetected.
`reorthogonalize=true` stores the basis for validation against
`projected_solve`. `history=true` adds two products per iteration for
explicit residuals and iterates; otherwise stopping uses the recurrence
estimate `projected_aresiduals`.
"""
function recurrence_solve(A, b::AbstractVector; start=:rhs, completion=:invariant,
                          maxiter=4length(b), atol=0, rtol=nothing, ranktol=nothing,
                          breakdown_tol=nothing, reorthogonalize=false, history=false,
                          check=true)
    T = validate(A, b; check)
    R = realtype(T)
    n = length(b)
    start in (:rhs, :normal) || throw(ArgumentError("unknown start"))
    completion in (:stationary, :invariant) || throw(ArgumentError("unknown completion"))
    maxiter isa Integer && maxiter > 0 || throw(ArgumentError("maxiter must be positive"))
    reorthogonalize isa Bool || throw(ArgumentError("reorthogonalize must be Boolean"))
    history isa Bool || throw(ArgumentError("history must be Boolean"))
    rtol = isnothing(rtol) ? sqrt(eps(R)) : R(rtol)
    ranktol = isnothing(ranktol) ? n * eps(R) : R(ranktol)
    breakdown_tol = isnothing(breakdown_tol) ? 100eps(R) : R(breakdown_tol)
    all(t -> isfinite(t) && t >= 0, (atol, rtol, ranktol, breakdown_tol)) ||
        throw(ArgumentError("tolerances must be finite and nonnegative"))
    b = Vector{T}(b)
    beta1 = norm(b)
    normal_b = normal_product(A, b)
    gamma = norm(normal_b)
    x = zeros(T, n)
    residuals, aresiduals, projected = R[beta1], R[gamma], R[gamma]
    iterates, pivots = Vector{T}[], R[]
    diagnostic_products = 1
    if beta1 == 0 || gamma == 0
        return x, (; niter=0, solved=true, status=:stationary_zero, closed=true,
            residuals, aresiduals, projected_aresiduals=projected, iterates, pivots,
            basis_products=0, diagnostic_products, null_products=0, orthogonality=zero(R),
            reorthogonalized=reorthogonalize, factorization=:short)
    end

    vbuf = [start == :rhs ? b / beta1 : conj.(normal_b) / gamma,
            zeros(T, n), zeros(T, n)]
    getv(j) = vbuf[mod1(j, 3)]
    basis = Vector{T}[]
    alphas, betas = T[], R[zero(R)]   # betas[j] is beta_j; T has no beta_1 entry
    nsteps, basis_products, null_products = 0, 0, 0
    opscale = zero(R)
    closed = false

    function step!()
        j = nsteps + 1
        vj = getv(j)
        q = A * conj.(vj)
        all(isfinite, q) || throw(ArgumentError("nonfinite operator product"))
        basis_products += 1
        scale = norm(q)
        opscale = max(opscale, scale)
        j > 1 && (q .-= betas[j] .* getv(j - 1))
        alpha = dot(vj, q)
        q .-= alpha .* vj
        if reorthogonalize
            push!(basis, copy(vj))
            for pass in 1:2, v in basis
                q .-= dot(v, q) .* v
            end
        end
        tail = norm(q)
        closed = (reorthogonalize && j == n) || tail <= breakdown_tol * scale
        push!(alphas, alpha)
        push!(betas, closed ? zero(R) : tail)
        closed || (getv(j + 1) .= q ./ tail)
        nsteps = j
        return nothing
    end

    α(j) = 1 <= j <= length(alphas) ? alphas[j] : zero(T)
    β(j) = 1 <= j <= length(betas) ? betas[j] : zero(R)
    # Superdiagonal T[j-1,j] equals the subdiagonal beta_j for CS.
    Tsup(j) = β(j)
    Tsub(j) = β(j + 1)

    rot = [(one(R), zero(T), one(R), zero(T)) for _ in 1:4]
    W = [zeros(T, n) for _ in 1:4]
    col = zeros(T, 7)                 # rows k-4:k+2 of column k
    rho1, rho2 = zero(T), zero(T)     # rotated right-hand side, rows k and k+1
    Bscale = zero(R)
    status, niter, floor_hits = :iteration_limit, 0, 0
    limit = reorthogonalize ? min(n, maxiter) : maxiter
    for k in 1:limit
        while !closed && nsteps < k + 1
            step!()
        end
        if k == 1
            rho1 = start == :rhs ? T(beta1 * conj(α(1))) : T(gamma)
            rho2 = start == :rhs ? T(beta1 * conj(β(2))) : zero(T)
        end
        final = closed && k == nsteps

        fill!(col, zero(T))
        col[3] = Tsup(k) * conj(Tsup(k - 1))
        col[4] = Tsup(k) * conj(α(k - 1)) + α(k) * conj(Tsup(k))
        col[5] = Tsup(k) * conj(Tsub(k - 1)) + α(k) * conj(α(k)) + Tsub(k) * conj(Tsup(k + 1))
        col[6] = α(k) * conj(Tsub(k)) + Tsub(k) * conj(α(k + 1))
        col[7] = Tsub(k) * conj(Tsub(k + 1))
        Bscale = max(Bscale, norm(col))
        for j in max(1, k - 4):k-1
            c2, s2, c1, s1 = rot[mod1(j, 4)]
            i = j - k + 5
            col[i+1], col[i+2] = reflect(c2, s2, col[i+1], col[i+2])
            col[i], col[i+1] = reflect(c1, s1, col[i], col[i+1])
        end
        c2, s2, r2 = cs_symortho(col[6], col[7])
        col[6], col[7] = r2, zero(T)
        c1, s1, r1 = cs_symortho(col[5], col[6])
        col[5], col[6] = r1, zero(T)
        rot[mod1(k, 4)] = (c2, s2, c1, s1)

        rho2, rho3 = reflect(c2, s2, rho2, zero(T))
        zk, rho2 = reflect(c1, s1, rho1, rho2)
        estimate = hypot(abs(rho2), abs(rho3))
        push!(projected, estimate)
        rho1, rho2 = rho2, rho3

        pivot = col[5]
        push!(pivots, abs(pivot))
        u = conj.(getv(k))
        for i in 1:min(4, k - 1)
            u -= col[5 - i] .* W[mod1(k - i, 4)]
        end
        # A small pivot of B = A'A-type products is ambiguous after squaring, so
        # test the would-be direction against A itself. A null u means every
        # least-squares solution is x + t*u; remove the u component.
        nu = norm(u)
        truncated = iszero(pivot) || iszero(nu)
        if !truncated && abs(pivot) <= sqrt(eps(R)) * Bscale
            null_products += 1
            # ranktol applies to B ~ A'A, i.e. sqrt(ranktol) to A, as in projected_solve.
            truncated = norm(A * u) <= sqrt(ranktol) * opscale * nu
        end
        if truncated
            nu > 0 && (x .-= u .* (dot(u, x) / nu^2))
        else
            W[mod1(k, 4)] .= u ./ pivot
            x .+= zk .* W[mod1(k, 4)]
        end

        niter = k
        if history
            r = b - A * x
            ar = normal_product(A, r)
            diagnostic_products += 2
            push!(residuals, norm(r))
            push!(aresiduals, norm(ar))
            push!(iterates, copy(x))
            converged = residual_converged(norm(r), norm(ar), beta1, gamma, atol, rtol)
        else
            converged = estimate <= atol + rtol * gamma
        end
        # Without reorthogonalization, closure may go undetected; past this
        # floor further steps only add ghost directions. One floor hit is not
        # enough: an inconsistent problem is stationary one step before the
        # closure step that removes its nullspace component.
        floor_hits = estimate <= ranktol * (Bscale * norm(x) + gamma) ? floor_hits + 1 : 0
        roundoff = !reorthogonalize && floor_hits >= 2
        if final
            status = :invariant_subspace
            break
        elseif truncated
            status = :rank_truncated
            break
        elseif roundoff
            status = :roundoff
            break
        elseif converged && completion == :stationary
            status = :stationary
            break
        end
    end

    if !history
        r = b - A * x
        diagnostic_products += 2
        push!(residuals, norm(r))
        push!(aresiduals, norm(normal_product(A, r)))
    end
    solved = residual_converged(residuals[end], aresiduals[end], beta1, gamma, atol, rtol)
    status == :invariant_subspace && !solved && (status = :invariant_subspace_unconverged)
    orthogonality = isempty(basis) ? R(NaN) :
        norm(adjoint(reduce(hcat, basis)) * reduce(hcat, basis) - I)
    return x, (; niter, solved, status, closed=closed && niter == nsteps,
        residuals, aresiduals, projected_aresiduals=projected, iterates, pivots,
        basis_products, diagnostic_products, null_products, orthogonality,
        reorthogonalized=reorthogonalize, factorization=:short)
end

"Short-recurrence CS-MinAres; see `recurrence_solve`."
csminares_short(A, b; kwargs...) = recurrence_solve(A, b; kwargs...)
