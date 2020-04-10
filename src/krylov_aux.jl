"""
Numerically stable symmetric Givens reflection.
Given `a` and `b`, return `(c, s, ρ)` such that

    [ c  s ] [ a ] = [ ρ ]
    [ s -c ] [ b ] = [ 0 ].
"""
function sym_givens(a :: T, b :: T) where T <: AbstractFloat
  #
  # Modeled after the corresponding Matlab function by M. A. Saunders and S.-C. Choi.
  # http://www.stanford.edu/group/SOL/dissertations/sou-cheng-choi-thesis.pdf
  # D. Orban, Montreal, May 2015.

  if b == 0
    if a == 0
      c = one(T)
    else
      c = sign(a)  # In Julia, sign(0) = 0.
    end
    s = zero(T)
    ρ = abs(a)

  elseif a == 0
    c = zero(T)
    s = sign(b)
    ρ = abs(b)

  elseif abs(b) > abs(a)
    t = a / b
    s = sign(b) / sqrt(one(T) + t * t)
    c = s * t
    ρ = b / s  # Computationally better than d = a / c since |c| ≤ |s|.

  else
    t = b / a
    c = sign(a) / sqrt(one(T) + t * t)
    s = c * t
    ρ = a / c  # Computationally better than d = b / s since |s| ≤ |c|
  end

  return (c, s, ρ)
end


"""
Find the real roots of the quadratic

    q(x) = q₂ x² + q₁ x + q₀,

where q₂, q₁ and q₀ are real. Care is taken to avoid numerical
cancellation. Optionally, `nitref` steps of iterative refinement
may be performed to improve accuracy. By default, `nitref=1`.
"""
function roots_quadratic(q₂ :: T, q₁ :: T, q₀ :: T;
                         nitref :: Int=1) where T <: AbstractFloat
  # Case where q(x) is linear.
  if q₂ == zero(T)
    if q₁ == zero(T)
      root = [zero(T)]
      q₀ == zero(T) || (root = T[])
    else
      root = [-q₀ / q₁]
    end
    return root
  end

  # Case where q(x) is indeed quadratic.
  rhs = √eps(T) * q₁ * q₁
  if abs(q₀ * q₂) > rhs
    ρ = q₁ * q₁ - 4 * q₂ * q₀
    ρ < 0 && return T[]
    d = -(q₁ + copysign(sqrt(ρ), q₁)) / 2
    roots = [d / q₂, q₀ / d]
  else
    # Ill-conditioned quadratic.
    roots = [-q₁ / q₂, zero(T)]
  end

  # Perform a few Newton iterations to improve accuracy.
  for k = 1 : 2
    root = roots[k]
    for it = 1 : nitref
      q = (q₂ * root + q₁) * root + q₀
      dq = 2 * q₂ * root + q₁
      dq == zero(T) && continue
      root = root - q / dq
    end
    roots[k] = root
  end
  return roots
end


"""
Given a trust-region radius `radius`, a vector `x` lying inside the
trust-region and a direction `d`, return `σ1` and `σ2` such that

    ‖x + σi d‖ = radius, i = 1, 2

in the Euclidean norm. If known, ‖x‖² may be supplied in `xNorm2`.

If `flip` is set to `true`, `σ1` and `σ2` are computed such that

    ‖x - σi d‖ = radius, i = 1, 2.
"""
function to_boundary(x :: Vector{T}, d :: Vector{T},
                     radius :: T; flip :: Bool=false, xNorm2 :: T=zero(T), dNorm2 :: T=zero(T)) where T <: Number
  radius > 0 || error("radius must be positive")

  # ‖d‖² σ² + 2 xᵀd σ + (‖x‖² - radius²).
  xd = dot(x, d)
  flip && (xd = -xd)
  dNorm2 == zero(T) && (dNorm2 = dot(d, d))
  dNorm2 == zero(T) && error("zero direction")
  xNorm2 == zero(T) && (xNorm2 = dot(x, x))
  (xNorm2 ≤ radius * radius) || error(@sprintf("outside of the trust region: ‖x‖²=%7.1e, Δ²=%7.1e", xNorm2, radius * radius))
  roots = roots_quadratic(dNorm2, 2 * xd, xNorm2 - radius * radius)
  return roots # `σ1` and `σ2`
end

"""
    kzeros(S, n)

Create an AbstractVector of storage type `S` of length `n` only composed of zero.
"""
@inline kzeros(S, n) = fill!(S(undef, n), zero(eltype(S)))

"""
    kones(S, n)

Create an AbstractVector of storage type `S` of length `n` only composed of one.
"""
@inline kones(S, n) = fill!(S(undef, n), one(eltype(S)))

@inline krylov_dot(n :: Int, x :: Vector{T}, dx :: Int, y :: Vector{T}, dy :: Int) where T <: BLAS.BlasReal = BLAS.dot(n, x, dx, y, dy)
@inline krylov_dot(n :: Int, x :: AbstractVector{T}, dx :: Int, y :: AbstractVector{T}, dy :: Int) where T <: Number = dot(x, y)

@inline krylov_norm2(n :: Int, x :: Vector{T}, dx :: Int) where T <: BLAS.BlasReal = BLAS.nrm2(n, x, dx)
@inline krylov_norm2(n :: Int, x :: AbstractVector{T}, dx :: Int) where T <: Number = norm(x)

@inline krylov_scal!(n :: Int, s :: T, x :: Vector{T}, dx :: Int) where T <: BLAS.BlasReal = BLAS.scal!(n, s, x, dx)
@inline krylov_scal!(n :: Int, s :: T, x :: AbstractVector{T}, dx :: Int) where T <: Number = (x .*= s)

@inline krylov_axpy!(n :: Int, s :: T, x :: Vector{T}, dx :: Int, y :: Vector{T}, dy :: Int) where T <: BLAS.BlasReal = BLAS.axpy!(n, s, x, dx, y, dy)
@inline krylov_axpy!(n :: Int, s :: T, x :: AbstractVector{T}, dx :: Int, y :: AbstractVector{T}, dy :: Int) where T <: Number = (y .+= s .* x)

@inline krylov_axpby!(n :: Int, s :: T, x :: Vector{T}, dx :: Int, t :: T, y :: Vector{T}, dy :: Int) where T <: BLAS.BlasReal = BLAS.axpby!(n, s, x, dx, t, y, dy)
@inline krylov_axpby!(n :: Int, s :: T, x :: AbstractVector{T}, dx :: Int, t :: T, y :: AbstractVector{T}, dy :: Int) where T <: Number = (y .= s .* x .+ t.* y)

function krylov_ref!(n :: Int, x :: AbstractVector{T}, dx :: Int, y :: AbstractVector{T}, dy :: Int, c :: T , s :: T) where T <: Number
  # assume dx = dy
  @inbounds @simd for i = 1:dx:n
    xi = x[i]
    yi = y[i]
    x[i] = c * xi + s * yi
    y[i] = s * xi - c * yi
  end
  return x, y
end

# the macros are just for readability, so we don't have to write the increments (always equal to 1)

macro kdot(n, x, y)
  return esc(:(krylov_dot($n, $x, 1, $y, 1)))
end

macro knrm2(n, x)
  return esc(:(krylov_norm2($n, $x, 1)))
end

macro kscal!(n, s, x)
  return esc(:(krylov_scal!($n, $s, $x, 1)))
end

macro kaxpy!(n, s, x, y)
  return esc(:(krylov_axpy!($n, $s, $x, 1, $y, 1)))
end

macro kaxpby!(n, s, x, t, y)
  return esc(:(krylov_axpby!($n, $s, $x, 1, $t, $y, 1)))
end

macro kswap(x, y)
  quote
    local tmp = $(esc(x))
    $(esc(x)) = $(esc(y))
    $(esc(y)) = tmp
  end
end

macro kref!(n, x, y, c, s)
  return esc(:(krylov_ref!($n, $x, 1, $y, 1, $c, $s)))
end
