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
  solved :: Bool
  inconsistent :: Bool
  residuals :: Vector{T}
  Aresiduals :: Vector{T}
  status :: String
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
  solved :: Bool
  residuals :: Array{T}
  flagged :: Union{Bool, Array{Bool,1}, BitArray{1}}
  Anorm :: T
  Acond :: T
  status :: String
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
  solved :: Bool
  residuals :: Array{T}
  residualscg :: Array{Union{T, Missing}}
  errors :: Array{T}
  errorscg :: Array{Union{T, Missing}}
  Anorm :: T
  Acond :: T
  status :: String
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
  solved_primal :: Bool
  solved_dual :: Bool
  residuals_primal :: Vector{T}
  residuals_dual :: Vector{T}
  status :: String
end

"""
Type for statistics returned by adjoint systems solvers BiLQR and TriLQR, the attributes are:
- solved
- inconsistent
- residuals
- error_bnd_x
- error_bnd_y
- status
"""
mutable struct LNLQStats{T} <: KrylovStats{T}
  solved :: Bool
  inconsistent :: Bool
  residuals :: Vector{T}
  error_bnd_x :: Vector{T}
  error_bnd_y :: Vector{T}
  status :: String
end

import Base.show

special_fields = Dict(
  :residualscg => "residuals (cg)",
  :errorscg => "errors (cg)",
  :Anorm => "‖A‖F",
  :Acond => "κ₂(A)",
)

for f in ["Simple", "Lanczos", "Symmlq", "Adjoint"]
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
      if typeof(statfield) <: AbstractVector && eltype(statfield) <: Union{Missing, AbstractFloat}
        s *= @sprintf " %s\n" vec2str(statfield)
      else
        s *= @sprintf " %s\n" statfield
      end
    end
    print(io, s)
  end
end
