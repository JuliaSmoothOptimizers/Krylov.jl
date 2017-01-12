lslq{TA <: Number, Tb <: Number}(A :: Array{TA,2}, b :: Array{Tb,1}; kwargs...) =
  lslq(LinearOperator(A), b; kwargs...)

lslq{TA <: Number, Tb <: Number, IA <: Integer}(A :: SparseMatrixCSC{TA,IA}, b :: Array{Tb,1}; kwargs...) =
  lslq(LinearOperator(A), b; kwargs...)
