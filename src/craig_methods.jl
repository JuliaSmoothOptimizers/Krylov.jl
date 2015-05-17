craig{TA <: Number, Tb <: Number}(
  A :: Array{TA,2}, b :: Array{Tb,1};
  λ :: Float64=0.0, atol :: Float64=1.0e-8, rtol :: Float64=1.0e-6,
  conlim :: Float64=1.0e+8, itmax :: Int=0, verbose :: Bool=false) =
  craig(LinearOperator(A), b, λ=λ, atol=atol, rtol=rtol,
        conlim=conlim, itmax=itmax, verbose=verbose);

craig{TA <: Number, Tb <: Number, IA <: Integer}(
  A :: SparseMatrixCSC{TA,IA}, b :: Array{Tb,1};
  λ :: Float64=0.0, atol :: Float64=1.0e-8, rtol :: Float64=1.0e-6,
  conlim :: Float64=1.0e+8, itmax :: Int=0, verbose :: Bool=false) =
  craig(LinearOperator(A), b, λ=λ, atol=atol, rtol=rtol,
        conlim=conlim, itmax=itmax, verbose=verbose);
