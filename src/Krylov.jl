module Krylov

using LinearOperators, LinearAlgebra, SparseArrays, Printf

"Abstract type for statistics returned by a solver"
abstract type KrylovStats{T} end

"Type for statistics returned by non-Lanczos solvers"
mutable struct SimpleStats{T} <: KrylovStats{T}
  solved :: Bool
  inconsistent :: Bool
  residuals :: Vector{T}
  Aresiduals :: Vector{T}
  status :: String
end

"Type for statistics returned by Lanczos solvers"
mutable struct LanczosStats{T} <: KrylovStats{T}
  solved :: Bool
  residuals :: Array{T}
  flagged :: Union{Bool, Array{Bool,1}, BitArray{1}}
  Anorm :: T
  Acond :: T
  status :: String
end

"Type for statistics returned by SYMMLQ"
mutable struct SymmlqStats{T} <: KrylovStats{T}
  solved :: Bool
  residuals :: Array{T}
  residualscg :: Array{Union{T, Missing}}
  errors :: Array{T}
  errorscg :: Array{Union{T, Missing}}
  Anorm :: T
  Acond :: T
  status :: String
end

"Type for statistics returned by adjoint systems solvers"
mutable struct AdjointStats{T} <: KrylovStats{T}
  solved_primal :: Bool
  solved_dual :: Bool
  residuals_primal :: Vector{T}
  residuals_dual :: Vector{T}
  status :: String
end

import Base.show

function show(io :: IO, stats :: SimpleStats)
  s  = "\nSimple stats\n"
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

function show(io :: IO, stats :: SymmlqStats)
  s  = "\nSYMMLQ stats\n"
  s *= @sprintf("  solved: %s\n", stats.solved)
  s *= @sprintf("  residuals: %s\n", vec2str(stats.residuals))
  s *= @sprintf("  residuals (cg): %s\n", vec2str(stats.residualscg))
  s *= @sprintf("  errors: %s\n", vec2str(stats.errors))
  s *= @sprintf("  errors(cg): %s\n", vec2str(stats.errorscg))
  s *= @sprintf("  ‖A‖F: %7.1e\n", stats.Anorm)
  s *= @sprintf("  κ₂(A): %7.1e\n", stats.Acond)
  s *= @sprintf("  status: %s\n", stats.status)
  print(io, s)
end

function show(io :: IO, stats :: AdjointStats)
  s  = "\nAdjoint stats\n"
  s *= @sprintf("  solved primal: %s\n", stats.solved_primal)
  s *= @sprintf("  solved dual:   %s\n", stats.solved_dual)
  s *= @sprintf("  residuals primal: %s\n", vec2str(stats.residuals_primal))
  s *= @sprintf("  residuals dual:   %s\n", vec2str(stats.residuals_dual))
  s *= @sprintf("  status: %s\n", stats.status)
  print(io, s)
end

include("krylov_aux.jl")
include("krylov_utils.jl")

include("cg.jl")
include("cr.jl")

include("symmlq.jl")
include("cg_lanczos.jl")
include("minres.jl")
include("minres_qlp.jl")

include("dqgmres.jl")
include("diom.jl")

include("usymlq.jl")
include("usymqr.jl")
include("tricg.jl")
include("trimr.jl")
include("trilqr.jl")

include("bilq.jl")
include("cgs.jl")
include("qmr.jl")
include("bilqr.jl")

include("cgls.jl")
include("crls.jl")

include("cgne.jl")
include("crmr.jl")

include("lslq.jl")
include("lsqr.jl")
include("lsmr.jl")

include("craig.jl")
include("craigmr.jl")

include("variants.jl")

end
