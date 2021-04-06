export BicgstabSolver

abstract type KrylovSolver{T,S} end

struct BicgstabSolver{T,S} <: KrylovSolver{T,S}
  x     :: S
  r     :: S
  p     :: S
  v     :: S
  s     :: S
  stats :: SimpleStats{T}
end

function BicgstabSolver(A, b)
  n, m = size(A)
  S = typeof(b)
  T = eltype(b)
  x = S(undef, n)
  r = S(undef, n)
  p = S(undef, n)
  v = S(undef, n)
  s = S(undef, n)
  stats = SimpleStats(false, false, T[], T[], "unknown")
  return BicgstabSolver{T,S}(x, r, p, v, s, stats)
end
