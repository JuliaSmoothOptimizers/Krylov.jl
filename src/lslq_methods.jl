lslq{TA <: Number, Tb <: Number, Tc <: Number}(A :: Array{TA,2}, b :: Array{Tb,1}, x_exact :: Array{Tc,1};
                                 M :: AbstractLinearOperator=opEye(size(A,1)), N :: AbstractLinearOperator=opEye(size(A,2)),
                                 sqd :: Bool=false,
                                 λ :: Float64=0.0, atol :: Float64=1.0e-8, btol :: Float64=1.0e-8,
                                 etol :: Float64=1.0e-8, window :: Int=5,
                                 itmax :: Int=0, conlim :: Float64=1.0e+8, σ_est :: Float64=0.0, verbose :: Bool=false) =
  lslq(LinearOperator(A), b, x_exact, M=M, N=N, sqd=sqd, λ=λ, atol=atol, btol=btol, etol=etol, window=window, itmax=itmax, conlim=conlim, σ_est=0.0, verbose=verbose);

lslq{TA <: Number, Tb <: Number, Tc <: Number, IA <: Integer}(A :: SparseMatrixCSC{TA,IA}, b :: Array{Tb,1}, x_exact :: Array{Tc,1};
                                                M :: AbstractLinearOperator=opEye(size(A,1)), N :: AbstractLinearOperator=opEye(size(A,2)),
                                                sqd :: Bool=false,
                                                λ :: Float64=0.0, atol :: Float64=1.0e-8, btol :: Float64=1.0e-8,
                                                etol :: Float64=1.0e-8, window :: Int=5,
                                                itmax :: Int=0, conlim :: Float64=1.0e+8, σ_est :: Float64=0.0, verbose :: Bool=false) =
  lslq(LinearOperator(A), b, x_exact, M=M, N=N, sqd=sqd, λ=λ, atol=atol, btol=btol, etol=etol, window=window, itmax=itmax, conlim=conlim, σ_est=0.0, verbose=verbose);
