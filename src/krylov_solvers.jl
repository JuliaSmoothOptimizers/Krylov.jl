export KrylovSolver, MinresSolver

"Abstract type for using Krylov solvers in-place"
abstract type KrylovSolver{T, S} end

"""
Type for storing the vectors required by the in-place version of MINRES.

The outer constructor 

    solver = MinresSolver(b :: AbstractVector{T}, window :: Int=5) where T <: AbstractFloat

may be used in order to create these vectors.
"""
mutable struct MinresSolver{T, S} <: KrylovSolver{T, S}
  x       :: S
  r1      :: S
  r2      :: S
  w1      :: S 
  w2      :: S
  err_vec :: Vector{T}
end

function MinresSolver(b :: AbstractVector{T}, window :: Int=5) where T <: AbstractFloat
  n = length(b)
  S = typeof(b)
  return MinresSolver{T, S}(kzeros(S, n), kzeros(S, n), kzeros(S, n), kzeros(S, n), kzeros(S, n), zeros(T, window))
end
