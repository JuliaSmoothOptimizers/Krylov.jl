module SpecialSymmetric

using LinearAlgebra
import ..Krylov

export csminares, csminares_range, csminares_short, recurrence_solve,
       projected_solve, minimum_norm_refinement

realtype(::Type{T}) where {T} = typeof(real(zero(T)))

function validate(A, b; check=true)
    n, m = size(A)
    n == m || throw(DimensionMismatch("A must be square"))
    length(b) == n || throw(DimensionMismatch("length(b) must equal size(A, 1)"))
    n > 0 || throw(ArgumentError("empty systems are not supported"))
    all(isfinite, b) || throw(ArgumentError("b must be finite"))
    T = promote_type(float(eltype(A)), float(eltype(b)))
    realtype(T) in (Float32, Float64) ||
        throw(ArgumentError("the reference implementation supports Float32 and Float64"))
    if check
        all(isfinite, A) || throw(ArgumentError("A must be finite"))
        norm(A - transpose(A)) <= 100eps(realtype(T)) * norm(A) ||
            throw(ArgumentError("A must be complex symmetric: transpose(A) == A"))
    end
    return T
end

# Normal residual using A' = conj(A) elementwise and the conjugated Saunders
# trial basis; no adjoint operator is required.
normal_product(A, r) = conj.(A * conj.(r))

residual_converged(nr, nar, nb, nab, atol, rtol, artol=rtol) =
    nr <= atol + rtol * nb || nar <= atol + artol * nab

function minnorm_ls(B, f, ranktol; factorization=:givens)
    factorization == :givens && return givens_minnorm_ls(B, f, ranktol)
    factorization == :svd || throw(ArgumentError("factorization must be :givens or :svd"))
    F = svd(B; full=false)
    isempty(F.S) && return zeros(eltype(B), size(B, 2))
    cutoff = ranktol * maximum(F.S)
    weights = adjoint(F.U) * f
    for i in eachindex(F.S)
        weights[i] = F.S[i] > cutoff ? weights[i] / F.S[i] : zero(eltype(weights))
    end
    return adjoint(F.Vt) * weights
end

"""
    projected_solve(A, b; start=:rhs, completion=:invariant, maxiter=length(b),
                    atol=0, rtol=√eps, ranktol=n*eps, breakdown_tol=100eps,
                    factorization=:givens, check=true)

Full-basis CS-MinAres reference, with two-pass reorthogonalization and a
rank-revealing Givens QR/QLP projected solve (`factorization=:svd` retains an
independent SVD oracle). Minimizes norm(A'*(b-A*x)) over the conjugate
Saunders trial space generated from `b` (`start=:rhs`) or from `A'*b`
(`start=:normal`, every trial vector in range(A')).

`:invariant` continues to subspace closure, selecting the smallest coefficient
norm at each step; in exact arithmetic the final answer is A†b. `:stationary`
allows early residual/normal-residual stopping and need not return A†b.
This is deliberately a research reference, not a short-recurrence solver; see
`recurrence_solve` / `csminares_short` for that. No A'*A is formed, but the
small product used here can square singular-value conditioning.
`check=false` permits matrix-free operators supporting size, eltype, and `*`;
the caller must then ensure complex-symmetric structure and finite outputs.
"""
function projected_solve(A, b::AbstractVector; start=:rhs, completion=:invariant,
                         maxiter=length(b), atol=0, rtol=nothing, ranktol=nothing,
                         breakdown_tol=nothing, factorization=:givens, check=true)
    T = validate(A, b; check)
    R = realtype(T)
    n = length(b)
    factorization in (:givens, :svd) || throw(ArgumentError("factorization must be :givens or :svd"))
    start in (:rhs, :normal) || throw(ArgumentError("unknown start"))
    completion in (:stationary, :invariant) || throw(ArgumentError("unknown completion"))
    maxiter isa Integer && maxiter > 0 || throw(ArgumentError("maxiter must be positive"))
    rtol = isnothing(rtol) ? sqrt(eps(R)) : R(rtol)
    ranktol = isnothing(ranktol) ? n * eps(R) : R(ranktol)
    breakdown_tol = isnothing(breakdown_tol) ? 100eps(R) : R(breakdown_tol)
    all(t -> isfinite(t) && t >= 0, (atol, rtol, ranktol, breakdown_tol)) ||
        throw(ArgumentError("tolerances must be finite and nonnegative"))
    b = Vector{T}(b)
    beta = norm(b)
    x = zeros(T, n)
    residuals = R[beta]
    normal_b = normal_product(A, b)
    aresiduals = R[norm(normal_b)]
    projected_aresiduals = R[aresiduals[1]]
    diagnostic_products = 1
    iterates = Vector{T}[]
    if beta == 0 || aresiduals[1] == 0
        return x, (; niter=0, solved=true, status=:stationary_zero,
            closed=true, residuals, aresiduals, projected_aresiduals, iterates,
            basis_products=0, diagnostic_products, orthogonality=zero(R), factorization)
    end
    limit = min(n, maxiter)
    V = zeros(T, n, min(n, limit + 2))
    H = zeros(T, min(n, limit + 2), min(n, limit + 1))
    V[:, 1] = start == :rhs ? b / beta : conj.(normal_b) / norm(normal_b)
    nbasis = 1
    ncolumns = 0
    closed = false

    function extend!()
        j = ncolumns + 1
        v = view(V, :, j)
        q = A * conj.(v)
        all(isfinite, q) || throw(ArgumentError("nonfinite operator product"))
        scale = norm(q)
        # Keep every computed projection: forcing tridiagonality after
        # reorthogonalization would invalidate the represented operator.
        for pass in 1:2
            for i in 1:j
                h = dot(view(V, :, i), q)
                H[i, j] += h
                q .-= h .* view(V, :, i)
            end
        end
        tail = norm(q)
        closed = j == n || tail <= breakdown_tol * scale
        if !closed
            H[j + 1, j] = tail
            V[:, j + 1] = q / tail
            nbasis = j + 1
        end
        ncolumns = j
        return nothing
    end

    status = :iteration_limit
    solved = false
    kdone = 0
    for k in 1:limit
        ncolumns < k && extend!()
        m = min(k + 1, nbasis)
        ncolumns < m && extend!() # one look-ahead product
        C = H[1:m, 1:k]
        f = zeros(T, m)
        f[1] = beta
        Hnext = H[1:nbasis, 1:m]
        D = conj.(Hnext)
        B = D * C
        g = D * f
        if start == :normal
            fill!(g, zero(T))
            g[1] = norm(normal_b)
        end
        y = minnorm_ls(B, g, ranktol; factorization)
        projected_norm = norm(g - B * y)
        Z = view(V, :, 1:k)
        x = conj.(Z) * y
        r = b - A * x
        ar = normal_product(A, r)
        diagnostic_products += 2
        push!(residuals, norm(r))
        push!(aresiduals, norm(ar))
        push!(projected_aresiduals, projected_norm)
        push!(iterates, copy(x))
        kdone = k
        solved = residual_converged(norm(r), norm(ar), beta, aresiduals[1], atol, rtol)
        terminal = closed && k == ncolumns
        if terminal
            status = solved ? :invariant_subspace : :invariant_subspace_unconverged
            break
        elseif solved && completion == :stationary
            status = :stationary
            break
        end
    end
    orthogonality = norm(adjoint(V[:, 1:nbasis]) * V[:, 1:nbasis] - I)
    return x, (; niter=kdone, solved, status, closed=closed && kdone == ncolumns,
        residuals, aresiduals, projected_aresiduals, iterates,
        basis_products=ncolumns, diagnostic_products, orthogonality, factorization)
end

"CS-MinAres: minimizes norm(A'*(b-A*x)) for complex symmetric A (transpose(A) == A)."
csminares(A, b; kwargs...) = projected_solve(A, b; kwargs...)
"""
    csminares_range(A, b; completion=:stationary, kwargs...)

CS-MinAres with v₁ = conjugate(A'*b)/norm(A'*b). Every trial vector is
in range(A'), so an exactly stationary iterate is already the pseudoinverse
solution. This range-start variant uses a different Saunders subspace from
`csminares`; its reduced right-hand side is norm(A'*b)*e1.
"""
csminares_range(A, b; completion=:stationary, kwargs...) =
    projected_solve(A, b; start=:normal, completion, kwargs...)

"""
    minimum_norm_refinement(A, b, x; rtol=1e-8, atol=0)

Project away conjugate(r) from a stationary CS-MinAres iterate `x`. The exact
minimum-norm theorem assumes `x` is stationary within the zero-start Saunders
subspace generated from `b`; this function checks stationarity but cannot
check the subspace assumption. Returns `(refined_x, applied)`.
"""
function minimum_norm_refinement(A, b, x; rtol=1e-8, atol=0)
    validate(A, b)
    length(x) == length(b) || throw(DimensionMismatch("x has the wrong length"))
    all(isfinite, x) || throw(ArgumentError("x must be finite"))
    all(t -> isfinite(t) && t >= 0, (rtol, atol)) || throw(ArgumentError("invalid tolerance"))
    r = b - A * x
    nr = norm(r)
    nr <= atol + rtol * norm(b) && return copy(x), false
    norm(normal_product(A, r)) <= atol + rtol * norm(normal_product(A, b)) ||
        throw(ArgumentError("minimum-norm refinement requires a stationary iterate"))
    p = conj.(r) / nr
    return x - p * dot(p, x), true
end

include("givens.jl")
include("recurrence.jl")

end
