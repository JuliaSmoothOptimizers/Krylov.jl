export KrylovStats, SimpleStats, LsmrStats, LanczosStats, LanczosShiftStats,
SymmlqStats, AdjointStats, LNLQStats, LSLQStats

import Base.copyto!

"Abstract type for statistics returned by a solver"
abstract type KrylovStats{T} end

"""
Type for storing statistics returned by the majority of Krylov solvers.
The fields are as follows:
- `niter`: The total number of iterations completed by the solver;
- `solved`: Indicates whether the solver successfully reached convergence (`true` if solved, `false` otherwise);
- `inconsistent`: Flags whether the system was detected as inconsistent (i.e., when `b` is not in the range of `A`);
- `residuals`: A vector containing the residual norms at each iteration;
- `Aresiduals`: A vector of `A'`-residual norms at each iteration;
- `Acond`: An estimate of the condition number of matrix `A`;
- `storage`: Total number of bytes required for the workspace;
- `timer`: The elapsed time (in seconds) taken by the solver to complete all iterations;
- `status`: A string indicating the outcome of the solve, providing additional details beyond `solved`.
"""
mutable struct SimpleStats{T} <: KrylovStats{T}
  niter        :: Int
  solved       :: Bool
  inconsistent :: Bool
  residuals    :: Vector{T}
  Aresiduals   :: Vector{T}
  Acond        :: Vector{T}
  storage      :: Int
  timer        :: Float64
  status       :: String
end

function reset!(stats :: SimpleStats)
  empty!(stats.residuals)
  empty!(stats.Aresiduals)
  empty!(stats.Acond)
end

function copyto!(dest :: SimpleStats, src :: SimpleStats)
  dest.niter        = src.niter
  dest.solved       = src.solved
  dest.inconsistent = src.inconsistent
  dest.residuals    = copy(src.residuals)
  dest.Aresiduals   = copy(src.Aresiduals)
  dest.Acond        = copy(src.Acond)
  dest.storage      = src.storage
  dest.timer        = src.timer
  dest.status       = src.status
  return dest
end

"""
Type for storing statistics returned by LSMR.
The fields are as follows:
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
  storage      :: Int
  timer        :: Float64
  status       :: String
end

function reset!(stats :: LsmrStats)
  empty!(stats.residuals)
  empty!(stats.Aresiduals)
end

function copyto!(dest :: LsmrStats, src :: LsmrStats)
  dest.niter        = src.niter
  dest.solved       = src.solved
  dest.inconsistent = src.inconsistent
  dest.residuals    = copy(src.residuals)
  dest.Aresiduals   = copy(src.Aresiduals)
  dest.residual     = src.residual
  dest.Aresidual    = src.Aresidual
  dest.Acond        = src.Acond
  dest.Anorm        = src.Anorm
  dest.xNorm        = src.xNorm
  dest.storage      = src.storage
  dest.timer        = src.timer
  dest.status       = src.status
  return dest
end

"""
Type for storing statistics returned by CG-LANCZOS.
The fields are as follows:
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
  storage    :: Int
  timer      :: Float64
  status     :: String
end

function reset!(stats :: LanczosStats)
  empty!(stats.residuals)
end

function copyto!(dest :: LanczosStats, src :: LanczosStats)
  dest.niter      = src.niter
  dest.solved     = src.solved
  dest.residuals  = copy(src.residuals)
  dest.indefinite = src.indefinite
  dest.Anorm      = src.Anorm
  dest.Acond      = src.Acond
  dest.storage    = src.storage
  dest.timer      = src.timer
  dest.status     = src.status
  return dest
end

"""
Type for storing statistics returned by CG-LANCZOS-SHIFT and CGLS-LANCZOS-SHIFT.
The fields are as follows:
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
  storage    :: Int
  timer      :: Float64
  status     :: String
end

function reset!(stats :: LanczosShiftStats)
  for vec in stats.residuals
    empty!(vec)
  end
end

function copyto!(dest :: LanczosShiftStats, src :: LanczosShiftStats)
  dest.niter      = src.niter
  dest.solved     = src.solved
  dest.residuals  = deepcopy(src.residuals)
  dest.indefinite = src.indefinite
  dest.Anorm      = src.Anorm
  dest.Acond      = src.Acond
  dest.storage    = src.storage
  dest.timer      = src.timer
  dest.status     = src.status
  return dest
end

"""
Type for storing statistics returned by SYMMLQ.
The fields are as follows:
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
  storage     :: Int
  timer       :: Float64
  status      :: String
end

function reset!(stats :: SymmlqStats)
  empty!(stats.residuals)
  empty!(stats.residualscg)
  empty!(stats.errors)
  empty!(stats.errorscg)
end

function copyto!(dest :: SymmlqStats, src :: SymmlqStats)
  dest.niter       = src.niter
  dest.solved      = src.solved
  dest.residuals   = copy(src.residuals)
  dest.residualscg = copy(src.residualscg)
  dest.errors      = copy(src.errors)
  dest.errorscg    = copy(src.errorscg)
  dest.Anorm       = src.Anorm
  dest.Acond       = src.Acond
  dest.storage     = src.storage
  dest.timer       = src.timer
  dest.status      = src.status
  return dest
end

"""
Type for storing statistics returned by adjoint systems solvers BiLQR and TriLQR.
The fields are as follows:
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
  storage          :: Int
  timer            :: Float64
  status           :: String
end

function reset!(stats :: AdjointStats)
  empty!(stats.residuals_primal)
  empty!(stats.residuals_dual)
end

function copyto!(dest :: AdjointStats, src :: AdjointStats)
  dest.niter            = src.niter
  dest.solved_primal    = src.solved_primal
  dest.solved_dual      = src.solved_dual
  dest.residuals_primal = copy(src.residuals_primal)
  dest.residuals_dual   = copy(src.residuals_dual)
  dest.storage          = src.storage
  dest.timer            = src.timer
  dest.status           = src.status
  return dest
end


"""
Type for storing statistics returned by the LNLQ method.
The fields are as follows:
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
  storage        :: Int
  timer          :: Float64
  status         :: String
end

function reset!(stats :: LNLQStats)
  empty!(stats.residuals)
  empty!(stats.error_bnd_x)
  empty!(stats.error_bnd_y)
end

function copyto!(dest :: LNLQStats, src :: LNLQStats)
  dest.niter          = src.niter
  dest.solved         = src.solved
  dest.residuals      = copy(src.residuals)
  dest.error_with_bnd = src.error_with_bnd
  dest.error_bnd_x    = copy(src.error_bnd_x)
  dest.error_bnd_y    = copy(src.error_bnd_y)
  dest.storage        = src.storage
  dest.timer          = src.timer
  dest.status         = src.status
  return dest
end

"""
Type for storing statistics returned by the LSLQ method.
The fields are as follows:
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
  storage        :: Int
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

function copyto!(dest :: LSLQStats, src :: LSLQStats)
  dest.niter          = src.niter
  dest.solved         = src.solved
  dest.inconsistent   = src.inconsistent
  dest.residuals      = copy(src.residuals)
  dest.Aresiduals     = copy(src.Aresiduals)
  dest.err_lbnds      = copy(src.err_lbnds)
  dest.error_with_bnd = src.error_with_bnd
  dest.err_ubnds_lq   = copy(src.err_ubnds_lq)
  dest.err_ubnds_cg   = copy(src.err_ubnds_cg)
  dest.storage        = src.storage
  dest.timer          = src.timer
  dest.status         = src.status
  return dest
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
    elseif field_name == "storage"
      s *= @sprintf " %s\n" format_bytes(statfield)
    elseif field_name == "timer"
      s *= display_timer(statfield)
    else
      s *= @sprintf " %s\n" statfield
    end
  end
  print(io, s)
end

function display_timer(timer::Float64)
  (timer < 1e-3) && (s = @sprintf(" %.2fμs\n", 1e6 * timer))
  (1e-3 ≤ timer < 1.00) && (s = @sprintf(" %.2fms\n", 1e3 * timer))
  (timer ≥ 1.00) && (s = @sprintf(" %.2fs\n", timer))
  return s
end
