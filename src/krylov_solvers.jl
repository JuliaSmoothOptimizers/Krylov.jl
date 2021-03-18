export KrylovSolver, MinresSolver

"Abstract type for using Krylov solvers in-place"
abstract type KrylovSolver{T, S} end

"""
Type for storing the vectors required by the in-place version of MINRES.

The outer constructor 

    opA, solver = MinresSolver(A, b :: AbstractVector{T}; window :: Int=5) where T <: AbstractFloat

may be used in order to create these vectors.
"""
mutable struct MinresSolver{T, S} <: KrylovSolver{T, S}
  x       :: S
  r1      :: S
  r2      :: S
  w1      :: S 
  w2      :: S
  err_vec :: Vector{T}
  stats   :: SimpleStats{T}
end

function MinresSolver(A, b :: AbstractVector{T}; window :: Int=5) where T <: AbstractFloat
  n, m = size(A)
  S = typeof(b)
  x  = S(undef, n)
  r1 = S(undef, n)
  r2 = S(undef, n)
  w1 = S(undef, n)
  w2 = S(undef, n)
  err_vec = zeros(T, window)
  opA = typeof(A) <: AbstractMatrix ? PreallocatedLinearOperator(A, storagetype=S, symmetric=true) : A
  stats = SimpleStats(false, true, T[], T[], "unknown")
  return opA, MinresSolver{T, S}(x, r1, r2, w1, w2, err_vec, stats)
end
