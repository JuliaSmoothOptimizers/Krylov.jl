"Abstract type for statistics returned by a solver"
abstract type KrylovStats{T} end

"""
Type for statistics returned by the majority of Krylov solvers, the attributes are:
- solved
- inconsistent
- residuals
- Aresiduals
- status
"""
mutable struct SimpleStats{T} <: KrylovStats{T}
  solved       :: Bool
  inconsistent :: Bool
  residuals    :: Vector{T}
  Aresiduals   :: Vector{T}
  status       :: String
end

function reset!(stats :: SimpleStats)
  empty!(stats.residuals)
  empty!(stats.Aresiduals)
end

"""
Type for statistics returned by CG-LANCZOS, the attributes are:
- solved
- residuals
- flagged
- Anorm
- Acond
- status
"""
mutable struct LanczosStats{T} <: KrylovStats{T}
  solved    :: Bool
  residuals :: VecOrMat{T}
  flagged   :: Union{Bool, Vector{Bool}, BitVector}
  Anorm     :: T
  Acond     :: T
  status    :: String
end

function reset!(stats :: LanczosStats)
  empty!(stats.residuals)
  isa(stats.flagged, AbstractVector) && empty!(stats.flagged)
end

"""
Type for statistics returned by SYMMLQ, the attributes are:
- solved
- residuals
- residualscg
- errors
- errorscg
- Anorm
- Acond
- status
"""
mutable struct SymmlqStats{T} <: KrylovStats{T}
  solved      :: Bool
  residuals   :: Vector{T}
  residualscg :: Vector{Union{T, Missing}}
  errors      :: Vector{T}
  errorscg    :: Vector{Union{T, Missing}}
  Anorm       :: T
  Acond       :: T
  status      :: String
end

function reset!(stats :: SymmlqStats)
  empty!(stats.residuals)
  empty!(stats.residualscg)
  empty!(stats.errors)
  empty!(stats.errorscg)
end

"""
Type for statistics returned by adjoint systems solvers BiLQR and TriLQR, the attributes are:
- solved_primal
- solved_dual
- residuals_primal
- residuals_dual
- status
"""
mutable struct AdjointStats{T} <: KrylovStats{T}
  solved_primal    :: Bool
  solved_dual      :: Bool
  residuals_primal :: Vector{T}
  residuals_dual   :: Vector{T}
  status           :: String
end

function reset!(stats :: AdjointStats)
  empty!(stats.residuals_primal)
  empty!(stats.residuals_dual)
end

"""
Type for statistics returned by the LNLQ method, the attributes are:
- solved
- residuals
- error_with_bnd
- error_bnd_x
- error_bnd_y
- status
"""
mutable struct LNLQStats{T} <: KrylovStats{T}
  solved         :: Bool
  residuals      :: Vector{T}
  error_with_bnd :: Bool
  error_bnd_x    :: Vector{T}
  error_bnd_y    :: Vector{T}
  status         :: String
end

function reset!(stats :: LNLQStats)
  empty!(stats.residuals)
  empty!(stats.error_bnd_x)
  empty!(stats.error_bnd_y)
end

"""
Type for statistics returned by the LSLQ method, the attributes are:
- solved
- inconsistent
- residuals
- Aresiduals
- err_lbnds
- error_with_bnd
- err_ubnds_lq
- err_ubnds_cg
- status
"""
mutable struct LSLQStats{T} <: KrylovStats{T}
  solved         :: Bool
  inconsistent   :: Bool
  residuals      :: Vector{T}
  Aresiduals     :: Vector{T}
  err_lbnds      :: Vector{T}
  error_with_bnd :: Bool
  err_ubnds_lq   :: Vector{T}
  err_ubnds_cg   :: Vector{T}
  status         :: String
end

function reset!(stats :: LSLQStats)
  empty!(stats.residuals)
  empty!(stats.Aresiduals)
  empty!(stats.err_lbnds)
  empty!(stats.err_ubnds_lq)
  empty!(stats.err_ubnds_cg)
end

import Base.show

special_fields = Dict(
  :residualscg => "residuals (cg)",
  :errorscg => "errors (cg)",
  :Anorm => "‖A‖F",
  :Acond => "κ₂(A)",
  :err_ubnds_lq => "error bound LQ",
  :err_ubnds_cg => "error bound CG",
)

for f in ["Simple", "Lanczos", "Symmlq", "Adjoint", "LNLQ", "LSLQ"]
  T = Meta.parse("Krylov." * f * "Stats{S}")
  @eval function show(io :: IO, stats :: $T) where S
    s  = $f * " stats\n"
    for field in fieldnames($T)
      field_name = if field ∈ keys(special_fields) 
        special_fields[field]
      else
        replace(string(field), "_" => " ")
      end
      s *=  " " * field_name * ":"
      statfield = getfield(stats, field)
      if isa(statfield, AbstractVector) && eltype(statfield) <: Union{Missing, AbstractFloat}
        s *= @sprintf " %s\n" vec2str(statfield)
      else
        s *= @sprintf " %s\n" statfield
      end
    end
    print(io, s)
  end
end
