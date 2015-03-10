cgls{TA <: Number, Tb <: Number}(A :: Array{TA,2}, b :: Array{Tb,1};
                                 atol :: Float64=1.0e-8, rtol :: Float64=1.0e-6, itmax :: Int=0, verbose :: Bool=false) =
  cgls(LinearOperator(A), b, atol=atol, rtol=rtol, itmax=itmax, verbose=verbose);

cgls{TA <: Number, Tb <: Number, IA <: Integer}(A :: SparseMatrixCSC{TA,IA}, b :: Array{Tb,1};
                                                atol :: Float64=1.0e-8, rtol :: Float64=1.0e-6, itmax :: Int=0, verbose :: Bool=false) =
  cgls(LinearOperator(A), b, atol=atol, rtol=rtol, itmax=itmax, verbose=verbose);
