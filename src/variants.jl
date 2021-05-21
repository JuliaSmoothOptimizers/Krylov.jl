# Replace identity operator modelized with UniformScaling by opEye
function wrap_uniformscaling(U)
  if U.λ == 1
    return opEye()
  else
    @printf("-------------------------------------------------------------------------------------------\n")
    @printf("Diagonal preconditioners modeled with UniformScaling are not recommended.\n")
    @printf("They allocate a new vector at each operator-vector products, which is not memory efficient.\n")
    @printf("-------------------------------------------------------------------------------------------\n")
    return U
  end
end

# Wrap preconditioners in a linear operator with preallocation
function wrap_preconditioners(kwargs, S)
  wrapM = haskey(kwargs, :M) && (typeof(kwargs[:M]) <: Union{AbstractMatrix, UniformScaling})
  wrapN = haskey(kwargs, :N) && (typeof(kwargs[:N]) <: Union{AbstractMatrix, UniformScaling})
  if (wrapM || wrapN)
    k = keys(kwargs)
    # Matrix-vector products with Mᵀ and Nᵀ are not required, we can safely use one vector for products with M / Mᵀ and N / Nᵀ
    # One vector for products with M / Mᵀ and N / Nᵀ is used when the option symmetric is set to true with a PreallocatedLinearOperator
    v = Tuple(typeof(arg) <: AbstractMatrix ? PreallocatedLinearOperator(arg, storagetype=S, symmetric=true) : (typeof(arg) <: UniformScaling ? wrap_uniformscaling(arg) : arg) for arg in values(kwargs))
    kwargs = Iterators.Pairs(NamedTuple{k, typeof(v)}(v), k)
  end
  return kwargs
end

# Variants where matrix-vector products with A and Aᵀ are required
for fn in (:cgls, :cgne, :lnlq, :craig, :craigmr, :crls, :crmr, :lslq, :lsqr, :lsmr)
  @eval begin
    $fn(A :: AbstractMatrix{T}, b :: AbstractVector{T}; kwargs...) where T <: AbstractFloat =
      $fn(PreallocatedLinearOperator(A, storagetype=ktypeof(b)), b; wrap_preconditioners(kwargs, ktypeof(b))...)
  end
end

# Variants for TriCG, TriMR
for fn in (:tricg, :trimr)
  @eval begin
    $fn(A :: AbstractMatrix{T}, b :: AbstractVector{T}, c :: AbstractVector{T}; kwargs...) where T <: AbstractFloat =
      $fn(PreallocatedLinearOperator(A, storagetype=ktypeof(b)), b, c; wrap_preconditioners(kwargs, ktypeof(b))...)
  end
end

# Variants where matrix-vector products with A are only required
for fn in (:cg_lanczos, :cr, :minres, :minres_qlp, :symmlq, :cgs, :bicgstab, :diom, :dqgmres)
  @eval begin
    $fn(A :: AbstractMatrix{T}, b :: AbstractVector{T}; kwargs...) where T <: AbstractFloat =
      $fn(PreallocatedLinearOperator(A, storagetype=ktypeof(b), symmetric=true), b; wrap_preconditioners(kwargs, ktypeof(b))...)
  end
end

# Variants for CG-LANCZOS-SHIFT-SEQ
for fn in [:cg_lanczos_shift_seq]
  @eval begin
    $fn(A :: AbstractMatrix{T}, b :: AbstractVector{T}, shifts :: AbstractVector{T}; kwargs...) where T <: AbstractFloat =
      $fn(PreallocatedLinearOperator(A, storagetype=ktypeof(b), symmetric=true), b, shifts; wrap_preconditioners(kwargs, ktypeof(b))...)
  end
end
