export KrylovStats, SimpleStats, LsmrStats, LanczosStats, LanczosShiftStats,
SymmlqStats, AdjointStats, LNLQStats, LSLQStats

"Abstract type for statistics returned by a solver"
abstract type KrylovStats{T} end

"""
Type for statistics returned by the majority of Krylov solvers, the attributes are:
- niter
- solved
- inconsistent
- residuals
- Aresiduals
- Acond
- timer
- status
"""
mutable struct SimpleStats{T} <: KrylovStats{T}
  niter        :: Int
  solved       :: Bool
  inconsistent :: Bool
  residuals    :: Vector{T}
  Aresiduals   :: Vector{T}
  Acond        :: Vector{T}
  timer        :: Float64
  status       :: String
end

function reset!(stats :: SimpleStats)
  empty!(stats.residuals)
  empty!(stats.Aresiduals)
  empty!(stats.Acond)
end

"""
Type for statistics returned by LSMR. The attributes are:
- niter
- solved
- inconsistent
- residuals
- Aresiduals
- Acond
- Anorm
- xNorm
- timer
- status
"""
mutable struct LsmrStats{T} <: KrylovStats{T}
  niter        :: Int
  solved       :: Bool
  inconsistent :: Bool
  residuals    :: Vector{T}
  Aresiduals   :: Vector{T}
  residual     :: T
  Aresidual    :: T
  Acond        :: T
  Anorm        :: T
  xNorm        :: T
  timer        :: Float64
  status       :: String
end

function reset!(stats :: LsmrStats)
  empty!(stats.residuals)
  empty!(stats.Aresiduals)
end

"""
Type for statistics returned by CG-LANCZOS, the attributes are:
- niter
- solved
- residuals
- indefinite
- Anorm
- Acond
- timer
- status
"""
mutable struct LanczosStats{T} <: KrylovStats{T}
  niter      :: Int
  solved     :: Bool
  residuals  :: Vector{T}
  indefinite :: Bool
  Anorm      :: T
  Acond      :: T
  timer      :: Float64
  status     :: String
end

function reset!(stats :: LanczosStats)
  empty!(stats.residuals)
end

"""
Type for statistics returned by CG-LANCZOS with shifts, the attributes are:
- niter
- solved
- residuals
- indefinite
- Anorm
- Acond
- timer
- status
"""
mutable struct LanczosShiftStats{T} <: KrylovStats{T}
  niter      :: Int
  solved     :: Bool
  residuals  :: Vector{Vector{T}}
  indefinite :: BitVector
  Anorm      :: T
  Acond      :: T
  timer      :: Float64
  status     :: String
end

function reset!(stats :: LanczosShiftStats)
  for vec in stats.residuals
    empty!(vec)
  end
end

"""
Type for statistics returned by SYMMLQ, the attributes are:
- niter
- solved
- residuals
- residualscg
- errors
- errorscg
- Anorm
- Acond
- timer
- status
"""
mutable struct SymmlqStats{T} <: KrylovStats{T}
  niter       :: Int
  solved      :: Bool
  residuals   :: Vector{T}
  residualscg :: Vector{Union{T, Missing}}
  errors      :: Vector{T}
  errorscg    :: Vector{Union{T, Missing}}
  Anorm       :: T
  Acond       :: T
  timer       :: Float64
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
- niter
- solved_primal
- solved_dual
- residuals_primal
- residuals_dual
- timer
- status
"""
mutable struct AdjointStats{T} <: KrylovStats{T}
  niter            :: Int
  solved_primal    :: Bool
  solved_dual      :: Bool
  residuals_primal :: Vector{T}
  residuals_dual   :: Vector{T}
  timer            :: Float64
  status           :: String
end

function reset!(stats :: AdjointStats)
  empty!(stats.residuals_primal)
  empty!(stats.residuals_dual)
end

"""
Type for statistics returned by the LNLQ method, the attributes are:
- niter
- solved
- residuals
- error_with_bnd
- error_bnd_x
- error_bnd_y
- timer
- status
"""
mutable struct LNLQStats{T} <: KrylovStats{T}
  niter          :: Int
  solved         :: Bool
  residuals      :: Vector{T}
  error_with_bnd :: Bool
  error_bnd_x    :: Vector{T}
  error_bnd_y    :: Vector{T}
  timer          :: Float64
  status         :: String
end

function reset!(stats :: LNLQStats)
  empty!(stats.residuals)
  empty!(stats.error_bnd_x)
  empty!(stats.error_bnd_y)
end

"""
Type for statistics returned by the LSLQ method, the attributes are:
- niter
- solved
- inconsistent
- residuals
- Aresiduals
- err_lbnds
- error_with_bnd
- err_ubnds_lq
- err_ubnds_cg
- timer
- status
"""
mutable struct LSLQStats{T} <: KrylovStats{T}
  niter          :: Int
  solved         :: Bool
  inconsistent   :: Bool
  residuals      :: Vector{T}
  Aresiduals     :: Vector{T}
  err_lbnds      :: Vector{T}
  error_with_bnd :: Bool
  err_ubnds_lq   :: Vector{T}
  err_ubnds_cg   :: Vector{T}
  timer          :: Float64
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

function show(io :: IO, stats :: KrylovStats)
  kst = typeof(stats)
  s = string(kst.name.name) * "\n"
  nfield = fieldcount(kst)
  for i = 1 : nfield
    field = fieldname(kst, i)
    field_name = if field ∈ keys(special_fields)
      special_fields[field]
    else
      replace(string(field), "_" => " ")
    end
    s *=  " " * field_name * ":"
    statfield = getfield(stats, field)
    if isa(statfield, AbstractVector) && eltype(statfield) <: Union{Missing, AbstractFloat}
      s *= @sprintf " %s\n" vec2str(statfield)
    elseif field_name == "timer"
      (statfield < 1e-3) && (s *= @sprintf " %.2fμs\n" 1e6*statfield)
      (1e-3 ≤ statfield < 1.00) && (s *= @sprintf " %.2fms\n" 1e3*statfield)
      (statfield ≥ 1.00) && (s *= @sprintf " %.2fs\n" statfield)
    else
      s *= @sprintf " %s\n" statfield
    end
  end
  print(io, s)
end
