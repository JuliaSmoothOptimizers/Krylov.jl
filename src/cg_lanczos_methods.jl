# Single system version.

cg_lanczos{TA <: Number, Tb <: Number}(A :: Array{TA,2}, b :: Array{Tb,1};
                                       atol :: Float64=1.0e-8, rtol :: Float64=1.0e-6, itmax :: Int=0,
                                       check_curvature :: Bool=false, verbose :: Bool=false) =
  cg_lanczos(LinearOperator(A), b, atol=atol, rtol=rtol, itmax=itmax, check_curvature=check_curvature, verbose=verbose);

cg_lanczos{TA <: Number, Tb <: Number, IA <: Integer}(A :: SparseMatrixCSC{TA,IA}, b :: Array{Tb,1};
                                                      atol :: Float64=1.0e-8, rtol :: Float64=1.0e-6, itmax :: Int=0,
                                                      check_curvature :: Bool=false, verbose :: Bool=false) =
  cg_lanczos(LinearOperator(A), b, atol=atol, rtol=rtol, itmax=itmax, check_curvature=check_curvature, verbose=verbose);


# Multiple shifts, sequential.

cg_lanczos_shift_seq{TA <: Number, Tb <: Number, Ts <: Number}(A :: Array{TA,2}, b :: Array{Tb,1}, shifts :: Array{Ts,1};
                                                               atol :: Float64=1.0e-8, rtol :: Float64=1.0e-6, itmax :: Int=0, check_curvature :: Bool=false, verbose :: Bool=false) =
  cg_lanczos_shift_seq(LinearOperator(A), b, shifts, atol=atol, rtol=rtol, itmax=itmax, check_curvature=check_curvature, verbose=verbose);

cg_lanczos_shift_seq{TA <: Number, Tb <: Number, Ts <: Number, IA <: Integer}(A :: SparseMatrixCSC{TA,IA}, b :: Array{Tb,1}, shifts :: Array{Ts,1};
                                                                             atol :: Float64=1.0e-8, rtol :: Float64=1.0e-6, itmax :: Int=0, check_curvature :: Bool=false, verbose :: Bool=false) =
  cg_lanczos_shift_seq(LinearOperator(A), b, shifts,  atol=atol, rtol=rtol, itmax=itmax, check_curvature=check_curvature, verbose=verbose);


# Multiple shifts, parallel.

cg_lanczos_shift_par{TA <: Number, Tb <: Number, Ts <: Number}(A :: Array{TA,2}, b :: Array{Tb,1}, shifts :: Array{Ts,1};
                                                               atol :: Float64=1.0e-8, rtol :: Float64=1.0e-6, itmax :: Int=0,
                                                               check_curvature :: Bool=false, verbose :: Bool=false) =
  cg_lanczos_shift_par(LinearOperator(A), b, shifts, atol=atol, rtol=rtol, itmax=itmax, check_curvature=check_curvature, verbose=verbose);

cg_lanczos_shift_par{TA <: Number, Tb <: Number, Ts <: Number, IA <: Integer}(A :: SparseMatrixCSC{TA,IA}, b :: Array{Tb,1}, shifts :: Array{Ts,1};
                                                                             atol :: Float64=1.0e-8, rtol :: Float64=1.0e-6, itmax :: Int=0,
                                                                             check_curvature :: Bool=false, verbose :: Bool=false) =
  cg_lanczos_shift_par(LinearOperator(A), b, shifts, atol=atol, rtol=rtol, itmax=itmax, check_curvature=check_curvature, verbose=verbose);
