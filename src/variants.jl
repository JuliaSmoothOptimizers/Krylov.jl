using LinearAlgebra

# Define a generic linear operator with preallocation
function preallocated_LinearOperator(A :: AbstractMatrix, T)
  (n, m) = size(A)
  Ap = zeros(T, n)
  Atq = zeros(T, m)
  return LinearOperator(T, n, m, false, false, p -> mul!(Ap, A, p),
                        q -> mul!(Atq, transpose(A), q), q -> mul!(Atq, transpose(A), q))
end

# Variants where A is a matrix without specific properties
for fn in (:cgls, :cgne, :craig, :craigmr, :crls, :crmr, :lslq, :lsmr, :lsqr, :dqgmres, :diom, :cgs, :bilq, :qmr)
  @eval begin
    function $fn(A :: AbstractMatrix{TA}, b :: AbstractVector{Tb}, args...; kwargs...) where {TA, Tb <: Number}
      S = promote_type(TA, Tb)
      if S <: Integer || S <: Complex{<: Integer}
        S = promote_type(S, Float64)
      end
      $fn(preallocated_LinearOperator(A, S), convert(Vector{S}, b), args...; kwargs...)
    end

    $fn(A :: AbstractMatrix{T}, b :: SparseVector{T}, args...; kwargs...) where T <: AbstractFloat =
      $fn(preallocated_LinearOperator(A, T), convert(Vector{T}, b), args...; kwargs...)

    $fn(A :: AbstractMatrix{T}, b :: AbstractVector{T}, args...; kwargs...) where T <: AbstractFloat =
      $fn(preallocated_LinearOperator(A, T), b, args...; kwargs...)
  end
end

# Variants for USYMLQ, USYMQR, TriLQR and BiLQR
for fn in (:usymlq, :usymqr, :trilqr, :bilqr)
  @eval begin
    function $fn(A :: AbstractMatrix{TA}, b :: AbstractVector{Tb}, c :: AbstractVector{Tc}, args...; kwargs...) where {TA, Tb, Tc <: Number}
      S = promote_type(TA, Tb, Tc)
      if S <: Integer || S <: Complex{<: Integer}
        S = promote_type(S, Float64)
      end
      $fn(preallocated_LinearOperator(A, S), convert(Vector{S}, b), convert(Vector{S}, c), args...; kwargs...)
    end

    $fn(A :: AbstractMatrix{T}, b :: SparseVector{T}, c :: SparseVector{T}, args...; kwargs...) where T <: AbstractFloat =
      $fn(preallocated_LinearOperator(A, T), convert(Vector{T}, b), convert(Vector{T}, c), args...; kwargs...)

    $fn(A :: AbstractMatrix{T}, b :: AbstractVector{T}, c :: SparseVector{T}, args...; kwargs...) where T <: AbstractFloat =
      $fn(preallocated_LinearOperator(A, T), b, convert(Vector{T}, c), args...; kwargs...)

    $fn(A :: AbstractMatrix{T}, b :: SparseVector{T}, c :: AbstractVector{T}, args...; kwargs...) where T <: AbstractFloat =
      $fn(preallocated_LinearOperator(A, T), convert(Vector{T}, b), c, args...; kwargs...)

    $fn(A :: AbstractMatrix{T}, b :: AbstractVector{T}, c :: AbstractVector{T}, args...; kwargs...) where T <: AbstractFloat =
      $fn(preallocated_LinearOperator(A, T), b, c, args...; kwargs...)
  end
end

# Define a symmetric linear operator with preallocation
function preallocated_symmetric_LinearOperator(A :: AbstractMatrix, T)
  (n, m) = size(A)
  Ap = zeros(T, n)
  return LinearOperator(T, n, m, true, true, p -> mul!(Ap, A, p))
end

# Variants where A must be symmetric
for fn in (:cg_lanczos, :cg_lanczos_shift_seq, :cg, :cr, :minres, :minres_qlp, :symmlq)
  @eval begin
    function $fn(A :: AbstractMatrix{TA}, b :: AbstractVector{Tb}, args...; kwargs...) where {TA, Tb <: Number}
      S = promote_type(TA, Tb)
      if S <: Integer || S <: Complex{<: Integer}
        S = promote_type(S, Float64)
      end
      $fn(preallocated_symmetric_LinearOperator(A, S), convert(Vector{S}, b), args...; kwargs...)
    end

    $fn(A :: AbstractMatrix{T}, b :: SparseVector{T}, args...; kwargs...) where T <: AbstractFloat =
      $fn(preallocated_symmetric_LinearOperator(A, T), convert(Vector{T}, b), args...; kwargs...)

    $fn(A :: AbstractMatrix{T}, b :: AbstractVector{T}, args...; kwargs...) where T <: AbstractFloat =
      $fn(preallocated_symmetric_LinearOperator(A, T), b, args...; kwargs...)
  end
end
