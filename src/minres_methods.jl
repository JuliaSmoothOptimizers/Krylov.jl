minres{TA <: Number, Tb <: Number}(A :: Array{TA,2}, b :: Array{Tb,1};
                                   M :: AbstractLinearOperator=opEye(size(A,1)),
                                   λ :: Float64=0.0,
                                   atol :: Float64=1.0e-8, rtol :: Float64=1.0e-6,
                                   etol :: Float64=1.0e-8, window :: Int=5,
                                   itmax :: Int=0, conlim :: Float64=1.0e+8, verbose :: Bool=false) =
  minres(LinearOperator(A), b, M=M, λ=λ, rtol=rtol, etol=etol, window=window, itmax=itmax, conlim=conlim, verbose=verbose);

minres{TA <: Number, Tb <: Number, IA <: Integer}(A :: SparseMatrixCSC{TA,IA}, b :: Array{Tb,1};
                                                  M :: AbstractLinearOperator=opEye(size(A,1)),
                                                  λ :: Float64=0.0,
                                                  atol :: Float64=1.0e-8, rtol :: Float64=1.0e-6,
                                                  etol :: Float64=1.0e-8, window :: Int=5,
                                                  itmax :: Int=0, conlim :: Float64=1.0e+8, verbose :: Bool=false) =
  minres(LinearOperator(A), b, M=M, λ=λ, rtol=rtol, etol=etol, window=window, itmax=itmax, conlim=conlim, verbose=verbose);
