"When `A` is given as an `Array`, it is first converted to a `LinearOperator`"
lslq{TA <: Number, Tb <: Number}(A :: Array{TA,2}, b :: Vector{Tb}; kwargs...) =
  lslq(LinearOperator(A), b; kwargs...)

"When `A` is given as a `SparseMatrixCSC`, it is first converted to a `LinearOperator`"
lslq{TA <: Number, Tb <: Number, IA <: Integer}(A :: SparseMatrixCSC{TA,IA}, b :: Vector{Tb}; kwargs...) =
  lslq(LinearOperator(A), b; kwargs...)
