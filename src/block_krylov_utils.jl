# """
#     Q, R = gs(A)
#
# Gram-Schmidt orthogonalization for a reduced QR decomposition.
#
# #### Input argument
#
# * `A`: an n-by-k matrix, n ≥ k
#
# #### Output arguments
#
# * `Q` an n-by-k orthonormal matrix: QᴴQ = Iₖ
# * `R` an k-by-k upper triangular matrix: QR = A
# """
function gs(A::AbstractMatrix{FC}) where FC <: FloatOrComplex
  n, k = size(A)
  Q = copy(A)
  R = zeros(FC, k, k)
  v = zeros(FC, n)
  gs!(Q, R, v)
end

function gs!(Q::AbstractMatrix{FC}, R::AbstractMatrix{FC}, v::AbstractVector{FC}) where FC <: FloatOrComplex
  n, k = size(Q)
  aⱼ = v
  kfill!(R, zero(FC))
  for j = 1:k
    qⱼ = view(Q,:,j)
    kcopy!(n, aⱼ, qⱼ)
    for i = 1:j-1
      qᵢ = view(Q,:,i)
      R[i,j] = kdot(n, qᵢ, aⱼ)    # rᵢⱼ = ⟨qᵢ , aⱼ⟩
      kaxpy!(n, -R[i,j], qᵢ, qⱼ)  # qⱼ = qⱼ - rᵢⱼqᵢ
    end
    R[j,j] = knorm(n, qⱼ)  # rⱼⱼ = ‖qⱼ‖
    qⱼ ./= R[j,j]           # qⱼ = qⱼ / rⱼⱼ
  end
  return Q, R
end

# """
# Modified Gram-Schmidt orthogonalization for a reduced QR decomposition:
# Q, R = mgs(A)
#
# Input :
# A an n-by-k matrix, n ≥ k
#
# # Q an n-by-k orthonormal matrix: QᴴQ = Iₖ
# # R an k-by-k upper triangular matrix: QR = A
# """
function mgs(A::AbstractMatrix{FC}) where FC <: FloatOrComplex
  n, k = size(A)
  Q = copy(A)
  R = zeros(FC, k, k)
  mgs!(Q, R)
end

function mgs!(Q::AbstractMatrix{FC}, R::AbstractMatrix{FC}) where FC <: FloatOrComplex
  n, k = size(Q)
  kfill!(R, zero(FC))
  for i = 1:k
    qᵢ = view(Q,:,i)
    R[i,i] = knorm(n, qᵢ)  # rᵢᵢ = ‖qᵢ‖
    qᵢ ./= R[i,i]          # qᵢ = qᵢ / rᵢᵢ
    for j = i+1:k
      qⱼ = view(Q,:,j)
      R[i,j] = kdot(n, qᵢ, qⱼ)    # rᵢⱼ = ⟨qᵢ , qⱼ⟩
      kaxpy!(n, -R[i,j], qᵢ, qⱼ)  # qⱼ = qⱼ - rᵢⱼqᵢ
    end
  end
  return Q, R
end

# Reduced QR factorization with Givens reflections:
# Q, R = givens(A)
#
# Input :
# A an n-by-k matrix, n ≥ k
#
# # Q an n-by-k orthonormal matrix: QᴴQ = Iₖ
# # R an k-by-k upper triangular matrix: QR = A
# """
function givens(A::AbstractMatrix{FC}) where FC <: FloatOrComplex
  n, k = size(A)
  nr = n*k - div(k*(k+1), 2)
  T = real(FC)
  Q = copy(A)
  R = zeros(FC, k, k)
  C = zeros(T, nr)
  S = zeros(FC, nr)
  givens!(Q, R, C, S)
end

function givens!(Q::AbstractMatrix{FC}, R::AbstractMatrix{FC}, C::AbstractVector{T}, S::AbstractVector{FC}) where {T <: AbstractFloat, FC <: FloatOrComplex{T}}
  n, k = size(Q)
  kfill!(R, zero(FC))
  pos = 0
  for j = 1:k
    for i = n-1:-1:j
      pos += 1
      C[pos], S[pos], Q[i,j] = sym_givens(Q[i,j], Q[i+1,j])
      if j < k
        reflect!(view(Q, i, j+1:k), view(Q, i+1, j+1:k), C[pos], S[pos])
      end
    end
  end
  for j = 1:k
    for i = 1:j
      R[i,j] = Q[i,j]
    end
  end
  kfill!(Q, zero(FC))
  for i = 1:k
    Q[i,i] = one(FC)
  end
  for j = k:-1:1
    for i = j:n-1
      reflect!(view(Q, i, j:k), view(Q, i+1, j:k), C[pos], S[pos])
      pos -= 1
    end
  end
  return Q, R
end

function reduced_qr!(Q::AbstractMatrix{FC}, R::AbstractMatrix{FC}, algo::String) where FC <: FloatOrComplex
  n, k = size(Q)
  T = real(FC)
  if algo == "gs"
    v = zeros(FC, n)
    gs!(Q, R, v)
  elseif algo == "mgs"
    mgs!(Q, R)
  elseif algo == "givens"
    nr = n*k - div(k*(k+1), 2)
    C = zeros(T, nr)
    S = zeros(FC, nr)
    givens!(Q, R, C, S)
  elseif algo == "householder"
    τ = zeros(FC, k)
    householder!(Q, R, τ)
  else
    error("$algo is not a supported method to perform a reduced QR.")
  end
  return Q, R
end

function reduced_qr(A::AbstractMatrix{FC}, algo::String) where FC <: FloatOrComplex
  if algo == "gs"
    Q, R = gs(A)
  elseif algo == "mgs"
    Q, R = mgs(A)
  elseif algo == "givens"
    Q, R = givens(A)
  elseif algo == "householder"
    Q, R = householder(A)
  else
    error("$algo is not a supported method to perform a reduced QR.")
  end
  return Q, R
end

function copy_triangle(Q::AbstractMatrix{FC}, R::AbstractMatrix{FC}, k::Int) where FC <: FloatOrComplex
  if VERSION < v"1.11"
    for i = 1:k
      for j = i:k
        R[i,j] = Q[i,j]
      end
    end
  else
    copytrito!(R, Q, 'U')
  end
  return R
end

# Reduced QR factorization with Householder reflections:
# Q, R = householder(A)
#
# Input :
# A an n-by-k matrix, n ≥ k
#
# Output :
# Q an n-by-k orthonormal matrix: QᴴQ = Iₖ
# R an k-by-k upper triangular matrix: QR = A
function householder(A::Matrix{FC}; compact::Bool=false) where FC <: FloatOrComplex
  n, k = size(A)
  Q = copy(A)
  τ = zeros(FC, k)
  R = zeros(FC, k, k)
  householder!(Q, R, τ; compact)
end

function householder!(Q::AbstractMatrix{FC}, R::AbstractMatrix{FC}, τ::AbstractVector{FC}; compact::Bool=false) where FC <: FloatOrComplex
  n, k = size(Q)
  kfill!(R, zero(FC))
  kgeqrf!(Q, τ)
  copy_triangle(Q, R, k)
  !compact && kungqr!(Q, τ)
  return Q, R
end

function householder!(Q::AbstractMatrix{FC}, R::AbstractMatrix{FC}, τ::AbstractVector{FC}, buffer::AbstractVector{FC}; compact::Bool=false) where FC <: FloatOrComplex
  n, k = size(Q)
  kfill!(R, zero(FC))
  kgeqrf!(Q, τ, buffer)
  copy_triangle(Q, R, k)
  !compact && kungqr!(Q, τ, buffer)
  return Q, R
end

for (Xgeqrf, Xungqr, Xunmqr, T) in ((:sgeqrf_, :sorgqr_, :sormqr_, :Float32   ),
                                    (:dgeqrf_, :dorgqr_, :dormqr_, :Float64   ),
                                    (:cgeqrf_, :cungqr_, :cunmqr_, :ComplexF32),
                                    (:zgeqrf_, :zungqr_, :zunmqr_, :ComplexF64))
    @eval begin
        function $Xgeqrf(m, n, a, lda, tau, work, lwork, info)
          return ccall((@blasfunc($Xgeqrf), libblastrampoline), Cvoid,
                       (Ref{BlasInt}, Ref{BlasInt}, Ptr{$T}, Ref{BlasInt},
                        Ptr{$T}, Ptr{$T}, Ref{BlasInt}, Ref{BlasInt}),
                        m, n, a, lda, tau, work, lwork, info)
        end

        function kgeqrf_buffer!(A::Matrix{$T}, tau::Vector{$T})
            m, n = size(A)
            work = Ref{$T}(0)
            lda = max(1, stride(A, 2))
            $Xgeqrf(m, n, A, lda, tau, work, -1, 0)
            return work[] |> BlasInt
        end

        function kgeqrf!(A::Matrix{$T}, tau::Vector{$T}, work::Vector{$T})
            m, n = size(A)
            lwork = length(work)
            lda = max(1, stride(A, 2))
            $Xgeqrf(m, n, A, lda, tau, work, lwork, 0)
            return nothing
        end

        function $Xungqr(m, n, k, a, lda, tau, work, lwork, info)
            return ccall((@blasfunc($Xungqr), libblastrampoline), Cvoid,
                         (Ref{BlasInt}, Ref{BlasInt}, Ref{BlasInt}, Ptr{$T},
                          Ref{BlasInt}, Ptr{$T}, Ptr{$T}, Ref{BlasInt}, Ref{BlasInt}),
                          m, n, k, a, lda, tau, work, lwork, info)
        end

        function kungqr_buffer!(A::Matrix{$T}, tau::Vector{$T})
            m, n = size(A)
            k = length(tau)
            work = Ref{$T}(0)
            lda = max(1, stride(A, 2))
            $Xungqr(m, n, k, A, lda, tau, work, -1, 0)
            return work[] |> BlasInt
        end

        function kungqr!(A::Matrix{$T}, tau::Vector{$T}, work::Vector{$T})
            symb = @blasfunc($Xungqr)
            m, n = size(A)
            k = length(tau)
            lwork = length(work)
            lda = max(1, stride(A, 2))
            $Xungqr(m, n, k, A, lda, tau, work, lwork, 0)
            return nothing
        end

        function $Xunmqr(side, trans, m, n, k, a, lda, tau, c, ldc, work, lwork, info)
            return ccall((@blasfunc($Xunmqr), libblastrampoline), Cvoid,
                         (Ref{UInt8}, Ref{UInt8}, Ref{BlasInt}, Ref{BlasInt}, Ref{BlasInt}, Ptr{$T},
                          Ref{BlasInt}, Ptr{$T}, Ptr{$T}, Ref{BlasInt}, Ptr{$T}, Ref{BlasInt},
                          Ref{BlasInt}, Clong, Clong),
                          side, trans, m, n, k, a, lda, tau, c, ldc, work, lwork, info, 1, 1)
        end

        function kunmqr_buffer!(side::Char, trans::Char, A::Matrix{$T}, tau::Vector{$T}, C::Matrix{$T})
            m, n = size(A)
            k = length(tau)
            work = Ref{$T}(0)
            lda = max(1, stride(A, 2))
            ldc = max(1, stride(C, 2))
            $Xunmqr(side, trans, m, n, k, A, lda, tau, C, ldc, work, -1, 0)
            return work[] |> BlasInt
        end

        function kunmqr!(side::Char, trans::Char, A::Matrix{$T}, tau::Vector{$T}, C::Matrix{$T}, work::Vector{$T})
            m, n = size(A)
            k = length(tau)
            lwork = length(work)
            lda = max(1, stride(A, 2))
            ldc = max(1, stride(C, 2))
            $Xunmqr(side, trans, m, n, k, A, lda, tau, C, ldc, work, lwork, 0)
            return nothing
        end
    end
end

kgeqrf!(A :: AbstractMatrix{T}, tau :: AbstractVector{T}) where T <: BLAS.BlasFloat = LAPACK.geqrf!(A, tau)
kgeqrf!(A :: AbstractMatrix{T}, tau :: AbstractVector{T}, buffer:: AbstractVector{T}) where T <: BLAS.BlasFloat = LAPACK.geqrf!(A, tau)

kungqr!(A :: AbstractMatrix{T}, tau :: AbstractVector{T}) where T <: BLAS.BlasFloat = LAPACK.orgqr!(A, tau)
kungqr!(A :: AbstractMatrix{T}, tau :: AbstractVector{T}, buffer:: AbstractVector{T}) where T <: BLAS.BlasFloat = LAPACK.orgqr!(A, tau)

kunmqr!(side :: Char, trans :: Char, A :: AbstractMatrix{T}, tau :: AbstractVector{T}, C :: AbstractMatrix{T}) where T <: BLAS.BlasFloat = LAPACK.ormqr!(side, trans, A, tau, C)
kunmqr!(side :: Char, trans :: Char, A :: AbstractMatrix{T}, tau :: AbstractVector{T}, C :: AbstractMatrix{T}, buffer:: AbstractVector{T}) where T <: BLAS.BlasFloat = LAPACK.ormqr!(side, trans, A, tau, C)

# """
#     β, τ = larfg!(α, x)
#
# Generate an elementary Householder reflector `H = I - τ vvᴴ` such that
# `Hᴴ * [α; x] = [β; 0]` (LAPACK convention; `H` is not hermitian in the complex
# case), where `v = [1; y]` and `β` is the (signed) Euclidean norm of `[α; x]`.
# On output `x` is overwritten by the tail `y` of `v`.
# """
function larfg!(α::FC, x::AbstractVector{FC}) where FC <: FloatOrComplex
  T = real(FC)
  n = length(x)
  xnorm = knorm(n, x)
  if xnorm == zero(T) && imag(α) == zero(T)
    return real(α), zero(FC)
  end
  β = -copysign(hypot(abs(α), xnorm), real(α))
  τ = (β - α) / β
  kdiv!(n, x, α - β)
  return β, τ
end

# """
#     A = geqr2!(A, tau)
#
# Reduced QR factorization via Householder reflections.
# On output the upper triangle of `A` holds `R`, and the reflectors are stored
# below the diagonal together with the scalars `τ`.
# """
function geqr2!(A::AbstractMatrix{FC}, tau::AbstractVector{FC}) where FC <: FloatOrComplex
  m, n = size(A)
  k = min(m, n)
  for i = 1:k
    x = view(A, i+1:m, i)
    βi, τi = larfg!(A[i,i], x)
    tau[i] = τi
    if i < n && τi != zero(FC)
      A[i,i] = one(FC)
      v = view(A, i:m, i)
      p = m - i + 1
      for j = i+1:n
        c = view(A, i:m, j)
        s = kdot(p, v, c)
        kaxpy!(p, -conj(τi) * s, v, c)
      end
    end
    A[i,i] = βi
  end
  return A
end

# """
#     A = ung2r!(A, tau)
#
# Form the orthonormal factor `Q` from the reflectors produced by [`geqrf!`](@ref), overwriting `A`.
# """
function ung2r!(A::AbstractMatrix{FC}, tau::AbstractVector{FC}) where FC <: FloatOrComplex
  m, n = size(A)
  k = length(tau)
  k ≤ min(m,n) || error("The dimension of A (($m,$n)) and the length of tau ($k) are inconsistent.")
  for j = k+1:n
    for l = 1:m
      A[l,j] = zero(FC)
    end
    A[j,j] = one(FC)
  end
  for i = k:-1:1
    τi = tau[i]
    if i < n && τi != zero(FC)
      A[i,i] = one(FC)
      v = view(A, i:m, i)
      p = m - i + 1
      for j = i+1:n
        c = view(A, i:m, j)
        s = kdot(p, v, c)
        kaxpy!(p, -τi * s, v, c)
      end
    end
    for l = i+1:m
      A[l,i] = -τi * A[l,i]
    end
    A[i,i] = one(FC) - τi
    for l = 1:i-1
      A[l,i] = zero(FC)
    end
  end
  return A
end

# """
#     C = unm2r!(side, trans, A, tau, C)
#
# Apply Q or Qᴴ (stored as reflectors in `A` with scalars `τ`) to the matrix `C`
# from the left ('L') or the right ('R'), using the reflectors computed by [`geqrf!`](@ref).
# """
function unm2r!(side::Char, trans::Char, A::AbstractMatrix{FC}, tau::AbstractVector{FC}, C::AbstractMatrix{FC}) where FC <: FloatOrComplex
  m, n = size(C)
  k = length(tau)
  notran = (trans == 'N')
  if side == 'L'
    rng = notran ? (k:-1:1) : (1:k)
    for i in rng
      τi = notran ? tau[i] : conj(tau[i])
      τi == zero(FC) && continue
      Aii = A[i,i]
      A[i,i] = one(FC)
      v = view(A, i:m, i)
      p = m - i + 1
      for j = 1:n
        c = view(C, i:m, j)
        s = kdot(p, v, c)
        kaxpy!(p, -τi * s, v, c)
      end
      A[i,i] = Aii
    end
  else  # side == 'R'
    rng = notran ? (1:k) : (k:-1:1)
    for i in rng
      τi = notran ? tau[i] : conj(tau[i])
      τi == zero(FC) && continue
      Aii = A[i,i]
      A[i,i] = one(FC)
      v = view(A, i:n, i)
      q = n - i + 1
      for a = 1:m
        r = view(C, a, i:n)
        s = zero(FC)
        for t = 1:q
          s += r[t] * v[t]
        end
        d = τi * s
        for t = 1:q
          r[t] -= d * conj(v[t])
        end
      end
      A[i,i] = Aii
    end
  end
  return C
end

kgeqrf!(A :: AbstractMatrix{FC}, tau :: AbstractVector{FC}) where FC <: FloatOrComplex = geqr2!(A, tau)
kungqr!(A :: AbstractMatrix{FC}, tau :: AbstractVector{FC}) where FC <: FloatOrComplex = ung2r!(A, tau)
kunmqr!(side :: Char, trans :: Char, A :: AbstractMatrix{FC}, tau :: AbstractVector{FC}, C :: AbstractMatrix{FC}) where FC <: FloatOrComplex = unm2r!(side, trans, A, tau, C)

# Fallback methods for the buffered API.
# The extra `buffer` argument is currently ignored because only the unblocked algorithms (`geqr2!`, `ung2r!`, `unm2r!`) are implemented.
kgeqrf!(A :: AbstractMatrix{FC}, tau :: AbstractVector{FC}, buffer :: AbstractVector{FC}) where FC <: FloatOrComplex = geqr2!(A, tau)
kungqr!(A :: AbstractMatrix{FC}, tau :: AbstractVector{FC}, buffer :: AbstractVector{FC}) where FC <: FloatOrComplex = ung2r!(A, tau)
kunmqr!(side :: Char, trans :: Char, A :: AbstractMatrix{FC}, tau :: AbstractVector{FC}, C :: AbstractMatrix{FC}, buffer :: AbstractVector{FC}) where FC <: FloatOrComplex = unm2r!(side, trans, A, tau, C)

kgeqrf_buffer!(A :: AbstractMatrix{FC}, tau :: AbstractVector{FC}) where FC <: FloatOrComplex = 0
kungqr_buffer!(A :: AbstractMatrix{FC}, tau :: AbstractVector{FC}) where FC <: FloatOrComplex = 0
kunmqr_buffer!(side :: Char, trans :: Char, A :: AbstractMatrix{FC}, tau :: AbstractVector{FC}, C :: AbstractMatrix{FC}) where FC <: FloatOrComplex = 0
