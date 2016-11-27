lslq{TA <: Number, Tb <: Number}(A :: Array{TA,2}, b :: Array{Tb,1}, xsol :: Vector{Tb};
                                 M :: AbstractLinearOperator=opEye(size(A,1)), N :: AbstractLinearOperator=opEye(size(A,2)),
                                 sqd :: Bool=false,
                                 λ :: Float64=0.0, σ :: Float64=0.0,
                                 atol :: Float64=1.0e-8, btol :: Float64=1.0e-8,
                                 etol :: Float64=1.0e-8, window :: Int=5,
                                 utol :: Float64=1.0e-8,
                                 itmax :: Int=0, conlim :: Float64=1.0e+8, verbose :: Bool=false) =
  lslq(LinearOperator(A), b, x, M=M, N=N, sqd=sqd, λ=λ, σ=σ, atol=atol, btol=btol, etol=etol, utol=utol, window=window, itmax=itmax, conlim=conlim, verbose=verbose);

lslq{TA <: Number, Tb <: Number, IA <: Integer}(A :: SparseMatrixCSC{TA,IA}, b :: Array{Tb,1}, xsol :: Vector{Tb};
                                                M :: AbstractLinearOperator=opEye(size(A,1)), N :: AbstractLinearOperator=opEye(size(A,2)),
                                                sqd :: Bool=false,
                                                λ :: Float64=0.0, σ :: Float64=0.0,
                                                atol :: Float64=1.0e-8, btol :: Float64=1.0e-8,
                                                etol :: Float64=1.0e-8, window :: Int=5,
                                                utol :: Float64=1.0e-8,
                                                itmax :: Int=0, conlim :: Float64=1.0e+8, verbose :: Bool=false) =
  lslq(LinearOperator(A), b, x, M=M, N=N, sqd=sqd, λ=λ, σ=σ, atol=atol, btol=btol, etol=etol, utol=utol, window=window, itmax=itmax, conlim=conlim, verbose=verbose);
