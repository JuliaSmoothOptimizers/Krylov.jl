# Variants

for fn in (:cg_lanczos, :cg_lanczos_shift_seq, :cg, :cgls, :cgne,
           :cr, :craig, :craigmr, :crls, :crmr,
           :lslq, :lsmr, :lsqr,
           :minres)

  @eval begin

    # Variant for A given as a dense array
    $fn{TA <: Number, Tb <: Number}(A :: Array{TA,2}, b :: Vector{Tb}, args...; kwargs...) =
      $fn(LinearOperator(A), convert(Vector{Float64}, b), args...; kwargs...)

    # Variant for A given as a sparse matrix
    $fn{TA <: Number, Tb <: Number, IA <: Integer}(A :: SparseMatrixCSC{TA,IA}, b :: Array{Tb,1}, args...; kwargs...) =
      $fn(LinearOperator(A), convert(Vector{Float64}, b), args...; kwargs...)

  end
end
