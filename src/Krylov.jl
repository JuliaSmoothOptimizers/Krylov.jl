module Krylov

using LinearOperators

if VERSION >= v"0.4-"
  using DistributedArrays
end

# Documentation.
using Docile
@docstrings

@doc "Abstract type for statistics returned by a solver" ->
abstract KrylovStats;

@doc "Type for statistics returned by non-Lanczos solvers" ->
type SimpleStats <: KrylovStats
  solved :: Bool
  inconsistent :: Bool
  residuals :: Array{Float64,1}
  Aresiduals :: Array{Float64,1}
  status :: UTF8String
end

type LanczosStats <: KrylovStats
  solved :: Bool
  residuals :: Union(Array{Float64}, DArray{Float64,1,Array{Float64,1}})
  flagged :: Union(Bool, BitArray{1}, DArray{Bool,1,Array{Bool,1}})
  Anorm :: Float64
  Acond :: Float64
  status :: UTF8String
end

import Base.show

function show(io :: IO, stats :: SimpleStats)
  s  = "\Simple stats\n"
  s *= @sprintf("  solved: %s\n", stats.solved)
  s *= @sprintf("  inconsistent: %s\n", stats.inconsistent)
  s *= @sprintf("  residuals:  %s\n", vec2str(stats.residuals))
  s *= @sprintf("  Aresiduals: %s\n", vec2str(stats.Aresiduals))
  s *= @sprintf("  status: %s\n", stats.status)
  print(io, s)
end

function show(io :: IO, stats :: LanczosStats)
  s  = "\nCG Lanczos stats\n"
  s *= @sprintf("  solved: %s\n", stats.solved)
  s *= @sprintf("  residuals: %s\n", typeof(stats.residuals))
  s *= @sprintf("  flagged: %s\n", stats.flagged)
  s *= @sprintf("  ‖A‖F: %7.1e\n", stats.Anorm)
  s *= @sprintf("  κ₂(A): %7.1e\n", stats.Acond)
  s *= @sprintf("  status: %s\n", stats.status)
  print(io, s)
end

include("krylov_utils.jl")
include("cg.jl")
include("cg_lanczos.jl")
include("cgls.jl")
include("crls.jl")
include("cgne.jl")
include("crmr.jl")

end
