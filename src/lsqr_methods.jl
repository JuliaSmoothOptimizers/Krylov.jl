lsqr{TA <: Number, Tb <: Number}(A :: Array{TA,2}, b :: Array{Tb,1};
                                 M :: AbstractLinearOperator=opEye(size(A,1)), N :: AbstractLinearOperator=opEye(size(A,2)),
                                 sqd :: Bool=false,
                                 λ :: Float64=0.0, atol :: Float64=1.0e-8, btol :: Float64=1.0e-8,
                                 itmax :: Int=0, conlim :: Float64=1.0e+8, verbose :: Bool=false) =
  lsqr(LinearOperator(A), b, M=M, N=N, sqd=sqd, λ=λ, atol=atol, btol=btol, itmax=itmax, conlim=conlim, verbose=verbose);

lsqr{TA <: Number, Tb <: Number, IA <: Integer}(A :: SparseMatrixCSC{TA,IA}, b :: Array{Tb,1};
                                                M :: AbstractLinearOperator=opEye(size(A,1)), N :: AbstractLinearOperator=opEye(size(A,2)),
                                                sqd :: Bool=false,
                                                λ :: Float64=0.0, atol :: Float64=1.0e-8, btol :: Float64=1.0e-8,
                                                itmax :: Int=0, conlim :: Float64=1.0e+8, verbose :: Bool=false) =
  lsqr(LinearOperator(A), b, M=M, N=N, sqd=sqd, λ=λ, atol=atol, btol=btol, itmax=itmax, conlim=conlim, verbose=verbose);
