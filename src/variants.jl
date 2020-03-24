using LinearAlgebra

# Wrap preconditioners in a linear operator with preallocation
function wrap_preconditioners(kwargs)
  if (haskey(kwargs, :M) && typeof(kwargs[:M]) <: AbstractMatrix) || (haskey(kwargs, :N) && typeof(kwargs[:N]) <: AbstractMatrix)
    k = keys(kwargs)
    # Matrix-vector products with Mᵀ and Nᵀ are not required, we can safely use one vector for products with M / Mᵀ and N / Nᵀ
    # One vector for products with M / Mᵀ and N / Nᵀ is used when the option symmetric is set to true with a PreallocatedLinearOperator
    v = Tuple(typeof(arg) <: AbstractMatrix ? PreallocatedLinearOperator(arg, symmetric=true) : arg for arg in values(kwargs))
    kwargs = Iterators.Pairs(NamedTuple{k, typeof(v)}(v), k)
  end
  return kwargs
end

# Variants where matrix-vector products with A and Aᵀ are required
for fn in (:cgls, :cgne, :craig, :craigmr, :crls, :crmr, :lslq, :lsmr, :lsqr, :bilq, :qmr)
  @eval begin
    $fn(A :: AbstractMatrix{T}, b :: SparseVector{T}, args...; kwargs...) where T <: AbstractFloat =
      $fn(PreallocatedLinearOperator(A), convert(Vector{T}, b), args...; wrap_preconditioners(kwargs)...)

    $fn(A :: AbstractMatrix{T}, b :: AbstractVector{T}, args...; kwargs...) where T <: AbstractFloat =
      $fn(PreallocatedLinearOperator(A), b, args...; wrap_preconditioners(kwargs)...)
  end
end

# Variants for USYMLQ, USYMQR, TriLQR and BiLQR
for fn in (:usymlq, :usymqr, :trilqr, :bilqr)
  @eval begin
    $fn(A :: AbstractMatrix{T}, b :: SparseVector{T}, c :: SparseVector{T}, args...; kwargs...) where T <: AbstractFloat =
      $fn(PreallocatedLinearOperator(A), convert(Vector{T}, b), convert(Vector{T}, c), args...; wrap_preconditioners(kwargs)...)

    $fn(A :: AbstractMatrix{T}, b :: AbstractVector{T}, c :: SparseVector{T}, args...; kwargs...) where T <: AbstractFloat =
      $fn(PreallocatedLinearOperator(A), b, convert(Vector{T}, c), args...; wrap_preconditioners(kwargs)...)

    $fn(A :: AbstractMatrix{T}, b :: SparseVector{T}, c :: AbstractVector{T}, args...; kwargs...) where T <: AbstractFloat =
      $fn(PreallocatedLinearOperator(A), convert(Vector{T}, b), c, args...; wrap_preconditioners(kwargs)...)

    $fn(A :: AbstractMatrix{T}, b :: AbstractVector{T}, c :: AbstractVector{T}, args...; kwargs...) where T <: AbstractFloat =
      $fn(PreallocatedLinearOperator(A), b, c, args...; wrap_preconditioners(kwargs)...)
  end
end

# Variants where matrix-vector products with A are only required
for fn in (:cg_lanczos, :cg_lanczos_shift_seq, :cg, :cr, :minres, :minres_qlp, :symmlq, :cgs, :diom, :dqgmres)
  @eval begin
    $fn(A :: AbstractMatrix{T}, b :: SparseVector{T}, args...; kwargs...) where T <: AbstractFloat =
      $fn(PreallocatedLinearOperator(A, symmetric=true), convert(Vector{T}, b), args...; wrap_preconditioners(kwargs)...)

    $fn(A :: AbstractMatrix{T}, b :: AbstractVector{T}, args...; kwargs...) where T <: AbstractFloat =
      $fn(PreallocatedLinearOperator(A, symmetric=true), b, args...; wrap_preconditioners(kwargs)...)
  end
end
