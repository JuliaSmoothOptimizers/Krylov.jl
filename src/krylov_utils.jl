"""
    FloatOrComplex{T}
Union type of `T` and `Complex{T}` where T is an `AbstractFloat`.
"""
const FloatOrComplex{T} = Union{T, Complex{T}} where T <: AbstractFloat

"""
    (c, s, ρ) = sym_givens(a, b)

Numerically stable symmetric Givens reflection.
Given `a` and `b` reals, return `(c, s, ρ)` such that

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
    ρ = b / s  # Computationally better than ρ = a / c since |c| ≤ |s|.

  else
    t = b / a
    c = sign(a) / sqrt(one(T) + t * t)
    s = c * t
    ρ = a / c  # Computationally better than ρ = b / s since |s| ≤ |c|
  end

  return (c, s, ρ)
end

"""
Numerically stable symmetric Givens reflection.
Given `a` and `b` complexes, return `(c, s, ρ)` with
c real and (s, ρ) complexes such that

    [ c   s ] [ a ] = [ ρ ]
    [ s̅  -c ] [ b ] = [ 0 ].
"""
function sym_givens(a :: Complex{T}, b :: Complex{T}) where T <: AbstractFloat
  #
  # Modeled after the corresponding Fortran function by M. A. Saunders and S.-C. Choi.
  # A. Montoison, Montreal, March 2020.

  abs_a = abs(a)
  abs_b = abs(b)

  if abs_b == 0
    c = one(T)
    s = zero(Complex{T})
    ρ = a

  elseif abs_a == 0
    c = zero(T)
    s = one(Complex{T})
    ρ = b

  elseif abs_b > abs_a
    t = abs_a / abs_b
    c = one(T) / sqrt(one(T) + t * t)
    s = c * conj((b / abs_b) / (a / abs_a))
    c = c * t
    ρ = b / conj(s)

  else
    t = abs_b / abs_a
    c = one(T) / sqrt(one(T) + t * t)
    s = c * t * conj((b / abs_b) / (a / abs_a))
    ρ = a / c
  end

  return (c, s, ρ)
end

@inline sym_givens(a :: Complex{T}, b :: T) where T <: AbstractFloat = sym_givens(a, Complex{T}(b))
@inline sym_givens(a :: T, b :: Complex{T}) where T <: AbstractFloat = sym_givens(Complex{T}(a), b)

"""
    roots = roots_quadratic(q₂, q₁, q₀; nitref)

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
    roots = to_boundary(x, d, radius; flip, xNorm2, dNorm2)

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

  # ‖d‖² σ² + (xᴴd + dᴴx) σ + (‖x‖² - radius²).
  rxd = real(dot(x, d))
  flip && (rxd = -rxd)
  dNorm2 == zero(T) && (dNorm2 = dot(d, d))
  dNorm2 == zero(T) && error("zero direction")
  xNorm2 == zero(T) && (xNorm2 = dot(x, x))
  (xNorm2 ≤ radius * radius) || error(@sprintf("outside of the trust region: ‖x‖²=%7.1e, Δ²=%7.1e", xNorm2, radius * radius))
  roots = roots_quadratic(dNorm2, 2 * rxd, xNorm2 - radius * radius)
  return roots # `σ1` and `σ2`
end

"""
    s = vec2str(x; ndisp)

Display an array in the form

    [ -3.0e-01 -5.1e-01  1.9e-01 ... -2.3e-01 -4.4e-01  2.4e-01 ]

with (ndisp - 1)/2 elements on each side.
"""
function vec2str(x :: AbstractVector{T}; ndisp :: Int=7) where T <: Union{AbstractFloat, Missing}
  n = length(x)
  if n ≤ ndisp
    ndisp = n
    nside = n
  else
    nside = max(1, div(ndisp - 1, 2))
  end
  s = "["
  i = 1
  while i ≤ nside
    if x[i] !== missing
      s *= @sprintf("%8.1e ", x[i])
    else
      s *= " ✗✗✗✗ "
    end
      i += 1
  end
  if i ≤ div(n, 2)
    s *= "... "
  end
  i = max(i, n - nside + 1)
  while i ≤ n
    if x[i] !== missing
      s *= @sprintf("%8.1e ", x[i])
    else
      s *= " ✗✗✗✗ "
    end
    i += 1
  end
  s *= "]"
  return s
end

"""
    S = ktypeof(v)

Return a dense storage type `S` based on the type of `v`.
"""
function ktypeof end

function ktypeof(v::S) where S <: DenseVector
  return S
end

function ktypeof(v::S) where S <: SparseVector
  T = eltype(S)
  return Vector{T}
end

function ktypeof(v::S) where S <: AbstractSparseVector
  return S.types[2]  # return `CuVector` for a `CuSparseVector`
end

function ktypeof(v::S) where S <: AbstractVector
  T = eltype(S)
  return Vector{T}  # BlockArrays, FillArrays, etc...
end

function ktypeof(v::S) where S <: SubArray
  return ktypeof(v.parent)
end

"""
    v = kzeros(S, n)

Create an AbstractVector of storage type `S` of length `n` only composed of zero.
"""
kzeros(S, n) = fill!(S(undef, n), zero(eltype(S)))

"""
    v = kones(S, n)

Create an AbstractVector of storage type `S` of length `n` only composed of one.
"""
kones(S, n) = fill!(S(undef, n), one(eltype(S)))

allocate_if(bool, solver, v, S, n) = bool && isempty(solver.:($v)) && (solver.:($v) = S(undef, n))

kdisplay(iter, verbose) = (verbose > 0) && (mod(iter, verbose) == 0)

mulorldiv!(y, P, x, ldiv::Bool) = ldiv ? ldiv!(y, P, x) : mul!(y, P, x)

kdot(n :: Integer, x :: Vector{T}, dx :: Integer, y :: Vector{T}, dy :: Integer) where T <: BLAS.BlasReal = BLAS.dot(n, x, dx, y, dy)
kdot(n :: Integer, x :: Vector{T}, dx :: Integer, y :: Vector{T}, dy :: Integer) where T <: BLAS.BlasComplex = BLAS.dotc(n, x, dx, y, dy)
kdot(n :: Integer, x :: AbstractVector{T}, dx :: Integer, y :: AbstractVector{T}, dy :: Integer) where T <: FloatOrComplex = dot(x, y)

kdotr(n :: Integer, x :: AbstractVector{T}, dx :: Integer, y :: AbstractVector{T}, dy :: Integer) where T <: AbstractFloat = kdot(n, x, dx, y, dy)
kdotr(n :: Integer, x :: AbstractVector{Complex{T}}, dx :: Integer, y :: AbstractVector{Complex{T}}, dy :: Integer) where T <: AbstractFloat = real(kdot(n, x, dx, y, dy))

knrm2(n :: Integer, x :: Vector{T}, dx :: Integer) where T <: BLAS.BlasFloat = BLAS.nrm2(n, x, dx)
knrm2(n :: Integer, x :: AbstractVector{T}, dx :: Integer) where T <: FloatOrComplex = norm(x)

kscal!(n :: Integer, s :: T, x :: Vector{T}, dx :: Integer) where T <: BLAS.BlasFloat = BLAS.scal!(n, s, x, dx)
kscal!(n :: Integer, s :: T, x :: AbstractVector{T}, dx :: Integer) where T <: FloatOrComplex = (x .*= s)
kscal!(n :: Integer, s :: T, x :: AbstractVector{Complex{T}}, dx :: Integer) where T <: AbstractFloat = kscal!(n, Complex{T}(s), x, dx)

kaxpy!(n :: Integer, s :: T, x :: Vector{T}, dx :: Integer, y :: Vector{T}, dy :: Integer) where T <: BLAS.BlasFloat = BLAS.axpy!(n, s, x, dx, y, dy)
kaxpy!(n :: Integer, s :: T, x :: AbstractVector{T}, dx :: Integer, y :: AbstractVector{T}, dy :: Integer) where T <: FloatOrComplex = axpy!(s, x, y)
kaxpy!(n :: Integer, s :: T, x :: AbstractVector{Complex{T}}, dx :: Integer, y :: AbstractVector{Complex{T}}, dy :: Integer) where T <: AbstractFloat = kaxpy!(n, Complex{T}(s), x, dx, y, dy)

kaxpby!(n :: Integer, s :: T, x :: Vector{T}, dx :: Integer, t :: T, y :: Vector{T}, dy :: Integer) where T <: BLAS.BlasFloat = BLAS.axpby!(n, s, x, dx, t, y, dy)
kaxpby!(n :: Integer, s :: T, x :: AbstractVector{T}, dx :: Integer, t :: T, y :: AbstractVector{T}, dy :: Integer) where T <: FloatOrComplex = axpby!(s, x, t, y)
kaxpby!(n :: Integer, s :: T, x :: AbstractVector{Complex{T}}, dx :: Integer, t :: Complex{T}, y :: AbstractVector{Complex{T}}, dy :: Integer) where T <: AbstractFloat = kaxpby!(n, Complex{T}(s), x, dx, t, y, dy)
kaxpby!(n :: Integer, s :: Complex{T}, x :: AbstractVector{Complex{T}}, dx :: Integer, t :: T, y :: AbstractVector{Complex{T}}, dy :: Integer) where T <: AbstractFloat = kaxpby!(n, s, x, dx, Complex{T}(t), y, dy)
kaxpby!(n :: Integer, s :: T, x :: AbstractVector{Complex{T}}, dx :: Integer, t :: T, y :: AbstractVector{Complex{T}}, dy :: Integer) where T <: AbstractFloat = kaxpby!(n, Complex{T}(s), x, dx, Complex{T}(t), y, dy)

kcopy!(n :: Integer, x :: Vector{T}, dx :: Integer, y :: Vector{T}, dy :: Integer) where T <: BLAS.BlasFloat = BLAS.blascopy!(n, x, dx, y, dy)
kcopy!(n :: Integer, x :: AbstractVector{T}, dx :: Integer, y :: AbstractVector{T}, dy :: Integer) where T <: FloatOrComplex = copyto!(y, x)

# the macros are just for readability, so we don't have to write the increments (always equal to 1)
macro kdot(n, x, y)
  return esc(:(Krylov.kdot($n, $x, 1, $y, 1)))
end

macro kdotr(n, x, y)
  return esc(:(Krylov.kdotr($n, $x, 1, $y, 1)))
end

macro knrm2(n, x)
  return esc(:(Krylov.knrm2($n, $x, 1)))
end

macro kscal!(n, s, x)
  return esc(:(Krylov.kscal!($n, $s, $x, 1)))
end

macro kaxpy!(n, s, x, y)
  return esc(:(Krylov.kaxpy!($n, $s, $x, 1, $y, 1)))
end

macro kaxpby!(n, s, x, t, y)
  return esc(:(Krylov.kaxpby!($n, $s, $x, 1, $t, $y, 1)))
end

macro kcopy!(n, x, y)
  return esc(:(Krylov.kcopy!($n, $x, 1, $y, 1)))
end

macro kswap(x, y)
  quote
    local tmp = $(esc(x))
    $(esc(x)) = $(esc(y))
    $(esc(y)) = tmp
  end
end

macro kref!(n, x, y, c, s)
  return esc(:(reflect!($x, $y, $c, $s)))
end
