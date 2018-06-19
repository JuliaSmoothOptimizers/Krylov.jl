# Variants

for fn in (:cg_lanczos, :cg_lanczos_shift_seq, :cg, :cgls, :cgne,
           :cr, :craig, :craigmr, :crls, :crmr,
           :lslq, :lsmr, :lsqr,
           :minres, :symmlq)

  @eval begin

    # Variant for A given as a dense array and b given as a dense vector
    $fn{TA <: Number, Tb <: Number}(A :: Array{TA,2}, b :: Vector{Tb}, args...; kwargs...) =
      $fn(LinearOperator(A), convert(Vector{Float64}, b), args...; kwargs...)

    # Variant for A given as a sparse matrix and b given as a dense vector
    $fn{TA <: Number, Tb <: Number, IA <: Integer}(A :: SparseMatrixCSC{TA,IA}, b :: Array{Tb,1}, args...; kwargs...) =
      $fn(LinearOperator(A), convert(Vector{Float64}, b), args...; kwargs...)

    # Variant for A given as a dense array and b given as a sparse vector
    $fn{TA <: Number, Tb <: Number, Ib <: Integer}(A :: Array{TA,2}, b :: SparseVector{Tb,Ib}, args...; kwargs...) =
      $fn(LinearOperator(A), convert(Vector{Float64}, b), args...; kwargs...)

    # Variant for A given as a sparse matrix and b given as a sparse vector
    $fn{TA <: Number, Tb <: Number, IA <: Integer, Ib <: Integer}(A :: SparseMatrixCSC{TA,IA}, b :: SparseVector{Tb,Ib}, args...; kwargs...) =
      $fn(LinearOperator(A), convert(Vector{Float64}, b), args...; kwargs...)

  end
end
