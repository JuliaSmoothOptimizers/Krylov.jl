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
- status
"""
mutable struct SimpleStats{T} <: KrylovStats{T}
  niter        :: Int
  solved       :: Bool
  inconsistent :: Bool
  residuals    :: Vector{T}
  Aresiduals   :: Vector{T}
  Acond        :: Vector{T}
  status       :: String
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
  status       :: String
end

"""
Type for statistics returned by CG-LANCZOS, the attributes are:
- niter
- solved
- residuals
- indefinite
- Anorm
- Acond
- status
"""
mutable struct LanczosStats{T} <: KrylovStats{T}
  niter      :: Int
  solved     :: Bool
  residuals  :: Vector{T}
  indefinite :: Bool
  Anorm      :: T
  Acond      :: T
  status     :: String
end

"""
Type for statistics returned by CG-LANCZOS with shifts, the attributes are:
- niter
- solved
- residuals
- indefinite
- Anorm
- Acond
- status
"""
mutable struct LanczosShiftStats{T} <: KrylovStats{T}
  niter      :: Int
  solved     :: Bool
  residuals  :: Vector{Vector{T}}
  indefinite :: BitVector
  Anorm      :: T
  Acond      :: T
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
  status      :: String
end

"""
Type for statistics returned by adjoint systems solvers BiLQR and TriLQR, the attributes are:
- niter
- solved_primal
- solved_dual
- residuals_primal
- residuals_dual
- status
"""
mutable struct AdjointStats{T} <: KrylovStats{T}
  niter            :: Int
  solved_primal    :: Bool
  solved_dual      :: Bool
  residuals_primal :: Vector{T}
  residuals_dual   :: Vector{T}
  status           :: String
end

"""
Type for statistics returned by the LNLQ method, the attributes are:
- niter
- solved
- residuals
- error_with_bnd
- error_bnd_x
- error_bnd_y
- status
"""
mutable struct LNLQStats{T} <: KrylovStats{T}
  niter          :: Int
  solved         :: Bool
  residuals      :: Vector{T}
  error_with_bnd :: Bool
  error_bnd_x    :: Vector{T}
  error_bnd_y    :: Vector{T}
  status         :: String
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
  status         :: String
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

for f in ["Simple", "Lsmr", "Adjoint", "LNLQ", "LSLQ", "Lanczos", "Symmlq"]
  T = Meta.parse("Krylov." * f * "Stats{S}")

  @eval function empty_field!(stats :: $T, i, ::Type{Vector{Si}}) where {S, Si}
    statfield = getfield(stats, i)
    empty!(statfield)
  end
  @eval empty_field!(stats :: $T, i, type) where S = stats

  @eval function reset!(stats :: $T) where S
    nfield = length($T.types)
    for i = 1 : nfield
      type  = fieldtype($T, i)
      empty_field!(stats, i, type)
    end
  end
end

for f in ["Simple", "Lsmr", "Lanczos", "LanczosShift", "Symmlq", "Adjoint", "LNLQ", "LSLQ"]
  T = Meta.parse("Krylov." * f * "Stats{S}")

  @eval function show(io :: IO, stats :: $T) where S
    s  = $f * " stats\n"
    nfield = length($T.types)
    for i = 1 : nfield
      field = fieldname($T, i)
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
