# Wrap preconditioners in a linear operator with preallocation
function wrap_preconditioners(kwargs)
  wrapM = haskey(kwargs, :M) && (typeof(kwargs[:M]) <: AbstractMatrix)
  wrapN = haskey(kwargs, :N) && (typeof(kwargs[:N]) <: AbstractMatrix)
  if (wrapM || wrapN)
    k = keys(kwargs)
    # Matrix-vector products with Mᵀ and Nᵀ are not required, we can safely use one vector for products with M / Mᵀ and N / Nᵀ
    # One vector for products with M / Mᵀ and N / Nᵀ is used when the option symmetric is set to true with a LinearOperator
    v = Tuple(typeof(arg) <: AbstractMatrix ? LinearOperator(arg, symmetric=true) : arg for arg in values(kwargs))
    kwargs = Iterators.Pairs(NamedTuple{k, typeof(v)}(v), k)
  end
  return kwargs
end

# Variants where matrix-vector products with A and Aᵀ are required
for fn in (:cgls, :cgne, :lnlq, :craig, :craigmr, :crls, :crmr, :lslq, :lsqr, :lsmr, :bilq, :qmr)
  @eval begin
    $fn(A :: AbstractMatrix{T}, b :: AbstractVector{T}; kwargs...) where T <: AbstractFloat =
      $fn(LinearOperator(A), b; wrap_preconditioners(kwargs)...)
  end
end

# Variants for USYMLQ, USYMQR, TriCG, TriMR, TriLQR and BiLQR
for fn in (:usymlq, :usymqr, :tricg, :trimr, :trilqr, :bilqr)
  @eval begin
    $fn(A :: AbstractMatrix{T}, b :: AbstractVector{T}, c :: AbstractVector{T}; kwargs...) where T <: AbstractFloat =
      $fn(LinearOperator(A), b, c; wrap_preconditioners(kwargs)...)
  end
end

# Variants where matrix-vector products with A are only required
for fn in (:cg_lanczos, :cg, :cr, :minres, :minres_qlp, :symmlq, :cgs, :bicgstab, :diom, :dqgmres)
  @eval begin
    $fn(A :: AbstractMatrix{T}, b :: AbstractVector{T}; kwargs...) where T <: AbstractFloat =
      $fn(LinearOperator(A, symmetric=true), b; wrap_preconditioners(kwargs)...)
  end
end

# Variants for CG-LANCZOS-SHIFT-SEQ
for fn in [:cg_lanczos_shift_seq]
  @eval begin
    $fn(A :: AbstractMatrix{T}, b :: AbstractVector{T}, shifts :: AbstractVector{T}; kwargs...) where T <: AbstractFloat =
      $fn(LinearOperator(A, symmetric=true), b, shifts; wrap_preconditioners(kwargs)...)
  end
end
