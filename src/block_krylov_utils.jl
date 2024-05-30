# """
# Gram-Schmidt orthogonalization for a reduced QR decomposition:
# Q, R = gs(A)
#
# Input :
# A an n-by-k matrix, n ≥ k
#
# Output :
# Q an n-by-k orthonormal matrix: QᴴQ = Iₖ
# R an k-by-k upper triangular matrix: QR = A
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
  R .= zero(FC)
  for j = 1:k
    qⱼ = view(Q,:,j)
    aⱼ .= qⱼ
    for i = 1:j-1
      qᵢ = view(Q,:,i)
      R[i,j] = @kdot(n, qᵢ, aⱼ)    # rᵢⱼ = ⟨qᵢ , aⱼ⟩
      @kaxpy!(n, -R[i,j], qᵢ, qⱼ)  # qⱼ = qⱼ - rᵢⱼqᵢ
    end
    R[j,j] = @knrm2(n, qⱼ)  # rⱼⱼ = ‖qⱼ‖
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
  R .= zero(FC)
  for i = 1:k
    qᵢ = view(Q,:,i)
    R[i,i] = @knrm2(n, qᵢ)  # rᵢᵢ = ‖qᵢ‖
    qᵢ ./= R[i,i]           # qᵢ = qᵢ / rᵢᵢ
    for j = i+1:k
      qⱼ = view(Q,:,j)
      R[i,j] = @kdot(n, qᵢ, qⱼ)    # rᵢⱼ = ⟨qᵢ , qⱼ⟩
      @kaxpy!(n, -R[i,j], qᵢ, qⱼ)  # qⱼ = qⱼ - rᵢⱼqᵢ
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
  R .= zero(FC)
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
  Q .= zero(FC)
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
    mR, nR = size(R)
    mQ, nQ = size(Q)
    if (mR == mQ) && (nR == nQ)
      copytrito!(R, Q, 'U')
    else
      copytrito!(R, view(Q, 1:k, 1:k), 'U')
    end
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
function householder(A::AbstractMatrix{FC}; compact::Bool=false) where FC <: FloatOrComplex
  n, k = size(A)
  Q = copy(A)
  τ = zeros(FC, k)
  R = zeros(FC, k, k)
  householder!(Q, R, τ; compact)
end

function householder!(Q::AbstractMatrix{FC}, R::AbstractMatrix{FC}, τ::AbstractVector{FC}; compact::Bool=false) where FC <: FloatOrComplex
  n, k = size(Q)
  R .= zero(FC)
  @kgeqrf!(Q, τ)
  copy_triangle(Q, R, k)
  !compact && @korgqr!(Q, τ)
  return Q, R
end
