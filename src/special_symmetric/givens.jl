# Rank-revealing complete orthogonal decomposition by stable Hermitian Givens
# reflections (SymOrtho, due to Sou-Cheng T. Choi and Michael A. Saunders).
# This is a full projected factorization, not an incremental short recurrence;
# see csminares.jl for the incremental CS-MinAres solver.

# Hermitian reflection [c s; conj(s) -c] * [a; b] = [r; 0], c real.
function cs_symortho(a::T, b::T) where {T<:Number}
    R = realtype(T)
    aa, ab = abs(a), abs(b)
    if isreal(a) && isreal(b)
        if iszero(b)
            return iszero(a) ? one(R) : sign(real(a)), zero(T), T(aa)
        elseif iszero(a)
            return zero(R), T(sign(real(b))), T(ab)
        end
        radius = hypot(aa, ab)
        return real(a) / radius, b / radius, T(radius)
    elseif iszero(b)
        return one(R), zero(T), a
    elseif iszero(a)
        return zero(R), one(T), b
    end
    if ab > aa
        t = aa / ab
        scale = inv(sqrt(one(R) + t*t))
        s = scale * conj(sign(b) / sign(a))
        return scale*t, s, b / conj(s)
    end
    t = ab / aa
    c = inv(sqrt(one(R) + t*t))
    s = c*t * conj(sign(b) / sign(a))
    return c, s, a / c
end

function reflect_rows!(R, k, i, firstcol, c, s)
    for j in firstcol:size(R, 2)
        a, b = R[k, j], R[i, j]
        R[k, j] = c*a + s*b
        R[i, j] = conj(s)*a - c*b
    end
    return R
end

# Minimum-norm projected least-squares solve by column-pivoted Givens QR and
# a right orthogonal reduction (QLP/complete orthogonal decomposition).
# No SVD, normal matrix, or inverse is formed. Rank is selected by trailing
# column norms relative to the largest initial column norm; it need not agree
# with an SVD singular-value cutoff close to the numerical rank threshold.
function givens_minnorm_ls(B::AbstractMatrix, f::AbstractVector, ranktol)
    m, n = size(B)
    length(f) == m || throw(DimensionMismatch("projected right-hand side"))
    isfinite(ranktol) && ranktol >= 0 || throw(ArgumentError("invalid rank tolerance"))
    T = promote_type(eltype(B), eltype(f))
    R, c = Matrix{T}(B), Vector{T}(f)
    x = zeros(T, n)
    min(m, n) == 0 && return x
    perm = collect(1:n)
    scale = maximum(norm(view(R, :, j)) for j in 1:n)
    iszero(scale) && return x
    cutoff = ranktol * scale
    rank = 0
    # Q* B P = [R11 R12; 0 R22], Q* f = c. Discard the trailing
    # block only once every remaining column norm is below the threshold.
    for k in 1:min(m, n)
        pivot, largest = k, norm(view(R, k:m, k))
        for j in k+1:n
            magnitude = norm(view(R, k:m, j))
            if magnitude > largest
                pivot, largest = j, magnitude
            end
        end
        largest <= cutoff && break
        if pivot != k
            for i in 1:m
                R[i, k], R[i, pivot] = R[i, pivot], R[i, k]
            end
            perm[k], perm[pivot] = perm[pivot], perm[k]
        end
        for i in m:-1:k+1
            iszero(R[i, k]) && continue
            cosine, sine, radius = cs_symortho(R[k, k], R[i, k])
            reflect_rows!(R, k, i, k, cosine, sine)
            a, b = c[k], c[i]
            c[k], c[i] = cosine*a + sine*b, conj(sine)*a - cosine*b
            R[k, k], R[i, k] = radius, zero(T)
        end
        rank = k
    end
    rank == 0 && return x
    # A second Givens QR of Rtop* gives Rtop Z = [L 0]. Solving
    # L u = c[1:rank] and applying Z to [u;0] selects minimum norm,
    # unlike simply zeroing free variables after the first pivoted QR.
    W = Matrix(adjoint(view(R, 1:rank, :)))
    rotations = Tuple{Int,Int,realtype(T),T}[]
    for k in 1:rank, i in n:-1:k+1
        iszero(W[i, k]) && continue
        cosine, sine, radius = cs_symortho(W[k, k], W[i, k])
        reflect_rows!(W, k, i, k, cosine, sine)
        W[k, k], W[i, k] = radius, zero(T)
        push!(rotations, (k, i, cosine, sine))
    end
    z = zeros(T, n)
    z[1:rank] = LowerTriangular(adjoint(W[1:rank, 1:rank])) \ c[1:rank]
    # Each reflection is Hermitian, so reverse their order to apply Z.
    for (k, i, cosine, sine) in Iterators.reverse(rotations)
        a, b = z[k], z[i]
        z[k], z[i] = cosine*a + sine*b, conj(sine)*a - cosine*b
    end
    x[perm] = z
    return x
end
