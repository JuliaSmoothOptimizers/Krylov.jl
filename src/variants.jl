using LinearAlgebra

# Variants where A is a matrix without specific properties
for fn in (:cgls, :cgne, :craig, :craigmr, :crls, :crmr, :lslq, :lsmr, :lsqr, :dqgmres, :diom, :cgs, :bilq, :qmr)
  @eval begin
    $fn(A :: AbstractMatrix{T}, b :: SparseVector{T}, args...; kwargs...) where T <: AbstractFloat =
      $fn(PreallocatedLinearOperator(A), convert(Vector{T}, b), args...; kwargs...)

    $fn(A :: AbstractMatrix{T}, b :: AbstractVector{T}, args...; kwargs...) where T <: AbstractFloat =
      $fn(PreallocatedLinearOperator(A), b, args...; kwargs...)
  end
end

# Variants for USYMLQ, USYMQR, TriLQR and BiLQR
for fn in (:usymlq, :usymqr, :trilqr, :bilqr)
  @eval begin
    $fn(A :: AbstractMatrix{T}, b :: SparseVector{T}, c :: SparseVector{T}, args...; kwargs...) where T <: AbstractFloat =
      $fn(PreallocatedLinearOperator(A), convert(Vector{T}, b), convert(Vector{T}, c), args...; kwargs...)

    $fn(A :: AbstractMatrix{T}, b :: AbstractVector{T}, c :: SparseVector{T}, args...; kwargs...) where T <: AbstractFloat =
      $fn(PreallocatedLinearOperator(A), b, convert(Vector{T}, c), args...; kwargs...)

    $fn(A :: AbstractMatrix{T}, b :: SparseVector{T}, c :: AbstractVector{T}, args...; kwargs...) where T <: AbstractFloat =
      $fn(PreallocatedLinearOperator(A), convert(Vector{T}, b), c, args...; kwargs...)

    $fn(A :: AbstractMatrix{T}, b :: AbstractVector{T}, c :: AbstractVector{T}, args...; kwargs...) where T <: AbstractFloat =
      $fn(PreallocatedLinearOperator(A), b, c, args...; kwargs...)
  end
end

# Variants where A must be symmetric
for fn in (:cg_lanczos, :cg_lanczos_shift_seq, :cg, :cr, :minres, :minres_qlp, :symmlq)
  @eval begin
    $fn(A :: AbstractMatrix{T}, b :: SparseVector{T}, args...; kwargs...) where T <: AbstractFloat =
      $fn(PreallocatedLinearOperator(A, symmetric=true), convert(Vector{T}, b), args...; kwargs...)

    $fn(A :: AbstractMatrix{T}, b :: AbstractVector{T}, args...; kwargs...) where T <: AbstractFloat =
      $fn(PreallocatedLinearOperator(A, symmetric=true), b, args...; kwargs...)
  end
end
