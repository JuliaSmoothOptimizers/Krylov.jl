export KrylovSolver, MinresSolver

"Abstract type for using Krylov solvers in-place"
abstract type KrylovSolver{S, T} end

"""
Type for storing the vectors required by the in-place version of MINRES.
The attributes are:
- x 
- r1 
- r2 
- w1 
- w2 
- err_vec 

The outer constructor 

    solver = MinresSolver(b :: AbstractVector{T}, window :: Int=5) where T <: AbstractFloat

may be used in order to create these vectors.
"""
mutable struct MinresSolver{S, T} <: KrylovSolver{S, T}
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
  return MinresSolver{S, T}(kzeros(S, n), kzeros(S, n), kzeros(S, n), kzeros(S, n), kzeros(S, n), zeros(T, window))
end
