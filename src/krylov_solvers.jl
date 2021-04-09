export KrylovSolver, MinresSolver

"Abstract type for using Krylov solvers in-place"
abstract type KrylovSolver{T, S} end

"""
Type for storing the vectors required by the in-place version of MINRES.

The outer constructor 

    solver = MinresSolver(A, b :: AbstractVector{T}; window :: Int=5) where T <: AbstractFloat

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
  m, n = size(A)
  x = similar(b, n)
  r1 = similar(b, n)
  r2 = similar(b, n)
  w1 = similar(b, n)
  w2 = similar(b, n)
  err_vec = zeros(T, window)
  stats = SimpleStats(false, true, T[], T[], "unknown")
  return MinresSolver(x, r1, r2, w1, w2, err_vec, stats)
end
