#!/usr/bin/env julia
# generate_stores.jl — generates src/c_stores.jl
# Run: julia --startup-file=no --project=<root> scripts/generate_stores.jl
#
# Re-run whenever new solvers are added to Krylov.jl.

# ---------------------------------------------------------------------------
# Solver table: (c_name, workspace_type_name)
# ---------------------------------------------------------------------------
include(joinpath(@__DIR__, "solver_table.jl"))

# ---------------------------------------------------------------------------
# Precision table: (dtype_int, suffix, T_real, FC, S)
# ---------------------------------------------------------------------------
const DTYPES = [
  (0, "f32",  "Float32", "Float32",    "Vector{Float32}"),
  (1, "f64",  "Float64", "Float64",    "Vector{Float64}"),
  (2, "cf32", "Float32", "ComplexF32", "Vector{ComplexF32}"),
  (3, "cf64", "Float64", "ComplexF64", "Vector{ComplexF64}"),
]

# Combo key: solver_idx * 4 + dtype_idx  (max = 33*4+3 = 135, fits UInt8)
combo_key(si, di) = UInt8(si * 4 + di)
store_name(sname, suffix) = "store_$(sname)_$(suffix)"
solver_idx(si) = si - 1   # 0-based, matches KrylovSolverType enum value

# Solvers that require two RHS vectors (b of size m and c of size n)
const TWO_RHS_SOLVERS = Set(["tricg", "trimr", "bilqr", "trilqr", "usymlq", "usymqr", "usymlqr"])

# gpmr uses (A, B, b, c): fptr_At slot is repurposed as the B operator (n×m)
const GPMR_SOLVER = "gpmr"

# Solver categories for extra KrylovOptions fields (solve-time)
const TAU_NU_SOLVERS = Set(["tricg", "trimr"])

# Least-squares / least-norm solvers and their preconditioners.  Krylov's
# convention is M = E⁻¹ (acts on the m-dimensional data space) and N = F⁻¹ (the
# n-dimensional solution space), so for these solvers the matvec_M callback sees
# length-m vectors and matvec_N length-n.  The normal-equations methods are the
# exception: CGLS/CRLS take a single preconditioner on the n-space (passed as
# matvec_M), and CGNE/CRMR a single one on the n-space (passed as matvec_N).
const LS_MN_RADIUS_SOLVERS = Set(["lsqr", "lsmr"])                     # M(m) + N(n) + λ + radius
const LS_MN_SOLVERS        = Set(["lslq", "craig", "craigmr", "lnlq"]) # M(m) + N(n) + λ
const LS_M_RADIUS_SOLVERS  = Set(["cgls", "crls"])                    # M(n) + λ + radius
const LS_N_SOLVERS         = Set(["cgne", "crmr"])                    # N(n) + λ

# Square (non-symmetric) solvers that accept BOTH a left (M) and a right (N)
# preconditioner — the only family where the C interface exposes matvec_N.
# Mirrors the Krylov.jl solvers whose keyword list contains both :M and :N for
# a square operator.  Symmetric solvers (cg, cr, minres, ...) take M only;
# least-squares / least-norm and gpmr families are handled by their own helpers.
const MN_SOLVERS = Set(["gmres", "fgmres", "fom", "diom", "dqgmres", "bicgstab", "cgs", "bilq", "qmr"])

# ---------------------------------------------------------------------------
# Optional scalar solve-time options, gated per solver (mirrors the Krylov.jl
# keyword lists).  These determine which generated helper a solver routes to,
# so an option is only ever forwarded to a solver that actually accepts it.
# ---------------------------------------------------------------------------
# M + N + restart + reorthogonalization
const GMRES_FAMILY    = Set(["gmres", "fgmres", "fom"])
# M + N + reorthogonalization (limited-memory Arnoldi, no restart)
const DIOM_FAMILY     = Set(["diom", "dqgmres"])
# M + radius + linesearch
const CG_FAMILY       = Set(["cg", "cr"])
# M + linesearch + λ (shift)
const MINRES_FAMILY   = Set(["minres", "minres_qlp"])
# M + λ (shift) — symmetric, no linesearch
const SYM_LAMBDA_SOLVERS = Set(["symmlq", "minares"])

# Solver categories for KrylovWorkspaceOptions fields (construction-time).
# These mirror exactly the Krylov.jl workspace constructors that accept the
# corresponding kwarg, e.g. GmresWorkspace(m, n, S; memory) — see
# src/krylov_workspaces.jl.  A workspace whose constructor does NOT accept the
# kwarg must never be handed it, so these sets gate the generated _do_create!.
const MEMORY_SOLVERS = Set(["gmres", "fgmres", "fom", "diom", "dqgmres", "gpmr"])
const WINDOW_SOLVERS = Set(["minres", "symmlq", "lsqr", "lsmr", "lslq"])

# Defaults used when the user leaves the sentinel 0 (must match Krylov.jl).
const MEMORY_DEFAULT = 20
const WINDOW_DEFAULT = 5

# ---------------------------------------------------------------------------
# Block Krylov solvers (BLOCK_SOLVERS defined in solver_table.jl) — a separate,
# matrix-based API (m×p block RHS).  Workspaces use SV=Vector{FC}, SM=Matrix{FC}.
# ---------------------------------------------------------------------------
const BLOCK_MEMORY_SOLVERS = Set(["block_gmres"])  # only block_gmres takes `memory`
const BLOCK_MN_SOLVERS = Set(["block_gmres"])      # only block_gmres takes a right preconditioner N
const BLOCK_MEMORY_DEFAULT = 5
block_combo_key(si, di) = UInt8(si * 4 + di)
block_store_name(sname, suffix) = "store_$(sname)_$(suffix)"

# ---------------------------------------------------------------------------
# Write c_stores.jl
# ---------------------------------------------------------------------------
out = joinpath(@__DIR__, "..", "src", "c_stores.jl")

open(out, "w") do io
  n_solvers = length(SOLVERS)
  n_combos  = n_solvers * length(DTYPES)

  println(io, """
# c_stores.jl — AUTO-GENERATED by scripts/generate_stores.jl — do not edit.
#
# $(n_solvers) solvers × $(length(DTYPES)) precisions = $(n_combos) typed workspace stores.
# Each store is a Dict{Ptr{Cvoid}, <ConcreteWorkspaceType>} that acts as a
# GC root while keeping types fully concrete for --trim=safe.
#
# Re-generate with:
#   julia --startup-file=no --project=<root> scripts/generate_stores.jl
""")

  # -----------------------------------------------------------------
  # 1. Typed store dicts
  # -----------------------------------------------------------------
  println(io, """
# ---------------------------------------------------------------------------
# Typed workspace stores
# ---------------------------------------------------------------------------
""")
  for (si, (sname, wstype, _)) in enumerate(SOLVERS)
    for (di, (_, suffix, T, FC, S)) in enumerate(DTYPES)
      println(io, "const $(store_name(sname, suffix)) = ",
                  "Dict{Ptr{Cvoid}, Krylov.$(wstype){$(T), $(FC), $(S)}}()")
    end
    println(io)
  end

  # -----------------------------------------------------------------
  # 2. Key index:  handle → UInt8 combo key
  # -----------------------------------------------------------------
  println(io, """
# Single index: handle → combo key (UInt8, max=$(n_combos-1))
const ws_key_store = Dict{Ptr{Cvoid}, UInt8}()
""")

  # -----------------------------------------------------------------
  # 3. Generic typed helpers — compiled once per concrete ws type
  # -----------------------------------------------------------------
  println(io, """
# ---------------------------------------------------------------------------
# Generic typed helpers — specialized per concrete workspace type by the compiler.
# Called from within if-elseif branches where ws has a known concrete type.
# FC is extracted via the where-clause so --trim=safe can infer it statically.
# ---------------------------------------------------------------------------

function _typed_niter(ws)
  Cint(Krylov.iteration_count(ws))
end

function _typed_is_solved(ws)
  Krylov.issolved(ws) ? Cint(1) : Cint(0)
end

function _typed_elapsed_time(ws)
  Cdouble(Krylov.elapsed_time(ws))
end

function _typed_get_x!(ws::Krylov.KrylovWorkspace{T, FC, S}, x_ptr, n) where {T, FC, S}
  x = Krylov.solution(ws, 1)
  copyto!(unsafe_wrap(Vector{FC}, Ptr{FC}(x_ptr), Int(n)), x)
  Cint(0)
end

function _typed_get_y!(ws::Krylov.KrylovWorkspace{T, FC, S}, y_ptr, m) where {T, FC, S}
  Krylov.solution_count(ws) == 2 || return Cint(-2)
  y = Krylov.solution(ws, 2)
  copyto!(unsafe_wrap(Vector{FC}, Ptr{FC}(y_ptr), Int(m)), y)
  Cint(0)
end

function _typed_warm_start!(ws::Krylov.KrylovWorkspace{T, FC, S}, x0_ptr, n) where {T, FC, S}
  # Bypass Krylov.warm_start! to avoid trim-incompatible string interpolation in its
  # error path.  We inline the same logic: allocate Δx if needed, copy x0, set flag.
  hasfield(typeof(ws), :Δx) || return Cint(-2)   # solver does not support warm start
  x0 = unsafe_wrap(Vector{FC}, Ptr{FC}(x0_ptr), Int(n))
  if isempty(ws.Δx)
    ws.Δx = similar(ws.x)
  end
  copyto!(ws.Δx, x0)
  ws.warm_start = true
  Cint(0)
end

# Two-solution warm start (BiLQR, TriLQR, USYMLQR, TriCG, TriMR, GPMR): seeds
# both the primal guess x0 (size of solution(ws, 1)) and the dual guess y0 (size
# of solution(ws, 2)).  Inlined for the same trim-safety reason as above.
# nx / ny are the buffer lengths the caller passed; they must match ws.x / ws.y.
function _typed_warm_start2!(ws::Krylov.KrylovWorkspace{T, FC, S}, x0_ptr, y0_ptr, nx, ny) where {T, FC, S}
  hasfield(typeof(ws), :Δy) || return Cint(-2)   # not a two-solution solver
  (Int(nx) == length(ws.x) && Int(ny) == length(ws.y)) || return Cint(-1)
  x0 = unsafe_wrap(Vector{FC}, Ptr{FC}(x0_ptr), Int(nx))
  y0 = unsafe_wrap(Vector{FC}, Ptr{FC}(y0_ptr), Int(ny))
  isempty(ws.Δx) && (ws.Δx = similar(ws.x))
  isempty(ws.Δy) && (ws.Δy = similar(ws.y))
  copyto!(ws.Δx, x0)
  copyto!(ws.Δy, y0)
  ws.warm_start = true
  Cint(0)
end

# ---------------------------------------------------------------------------
# _opts_kw — build base keyword arguments from a KrylovOptionsC struct.
# NaN sentinels fall back to the Julia solver default (√eps(T); timemax → Inf).
# itmax=0 is always passed explicitly; Krylov.jl interprets 0 as "use default".
# atol/rtol/itmax/verbose/timemax are accepted by EVERY solver, so they live here.
# The NamedTuple type is always the same concrete type — compatible with --trim=safe.
# ---------------------------------------------------------------------------
function _opts_kw(opts::KrylovOptionsC, ::Type{T}) where T
  atol_v = isnan(opts.atol) ? sqrt(eps(T)) : T(opts.atol)
  rtol_v = isnan(opts.rtol) ? sqrt(eps(T)) : T(opts.rtol)
  tmax_v = isnan(opts.timemax) ? Inf : Float64(opts.timemax)
  (atol=atol_v, rtol=rtol_v, itmax=Int(opts.itmax), verbose=Int(opts.verbose), timemax=tmax_v)
end

# ---------------------------------------------------------------------------
# _typed_solve! variants — one per option family
# ---------------------------------------------------------------------------

# Every one/two-RHS _typed_solve* helper shares the same argument list (fptr_M is
# the left preconditioner, fptr_N the right one) so _do_solve! calls any of them
# uniformly.  There is one helper per *option set* (which scalar kwargs a solver
# accepts), so an option is only ever forwarded to a solver that supports it.
# Each call site lists its kwargs as literals → a statically-known concrete
# NamedTuple, required by --trim=safe.

# M only — CAR, CGNE, CRMR (and the catch-all).  fptr_N ignored.
function _typed_solve!(ws::Krylov.KrylovWorkspace{T, FC, S}, fptr_A, fptr_At, fptr_M, fptr_N, b_ptr, c_ptr, userdata, opts) where {T, FC, S}
  A  = COperator{FC}(ws.m, ws.n, fptr_A, fptr_At, userdata)
  b  = unsafe_wrap(Vector{FC}, Ptr{FC}(b_ptr), ws.m)
  kw = _opts_kw(opts, T)
  if fptr_M != C_NULL
    M = CPreconditioner{FC}(ws.n, fptr_M, userdata)
    Krylov.krylov_solve!(ws, A, b; M=M, kw...)
  else
    Krylov.krylov_solve!(ws, A, b; kw...)
  end
  Cint(0)
end

# M + radius + linesearch — CG, CR.
function _typed_solve_cg!(ws::Krylov.KrylovWorkspace{T, FC, S}, fptr_A, fptr_At, fptr_M, fptr_N, b_ptr, c_ptr, userdata, opts) where {T, FC, S}
  A  = COperator{FC}(ws.m, ws.n, fptr_A, fptr_At, userdata)
  b  = unsafe_wrap(Vector{FC}, Ptr{FC}(b_ptr), ws.m)
  kw = _opts_kw(opts, T)
  rad = T(opts.radius); ls = opts.linesearch != 0
  if fptr_M != C_NULL
    M = CPreconditioner{FC}(ws.n, fptr_M, userdata)
    Krylov.krylov_solve!(ws, A, b; M=M, radius=rad, linesearch=ls, kw...)
  else
    Krylov.krylov_solve!(ws, A, b; radius=rad, linesearch=ls, kw...)
  end
  Cint(0)
end

# M + linesearch + λ-shift — MINRES, MINRES-QLP.
function _typed_solve_minres!(ws::Krylov.KrylovWorkspace{T, FC, S}, fptr_A, fptr_At, fptr_M, fptr_N, b_ptr, c_ptr, userdata, opts) where {T, FC, S}
  A  = COperator{FC}(ws.m, ws.n, fptr_A, fptr_At, userdata)
  b  = unsafe_wrap(Vector{FC}, Ptr{FC}(b_ptr), ws.m)
  kw = _opts_kw(opts, T)
  λ = T(opts.lambda); ls = opts.linesearch != 0
  if fptr_M != C_NULL
    M = CPreconditioner{FC}(ws.n, fptr_M, userdata)
    Krylov.krylov_solve!(ws, A, b; M=M, λ=λ, linesearch=ls, kw...)
  else
    Krylov.krylov_solve!(ws, A, b; λ=λ, linesearch=ls, kw...)
  end
  Cint(0)
end

# M + λ-shift — SYMMLQ, MINARES (symmetric, no linesearch).
function _typed_solve_sym_lambda!(ws::Krylov.KrylovWorkspace{T, FC, S}, fptr_A, fptr_At, fptr_M, fptr_N, b_ptr, c_ptr, userdata, opts) where {T, FC, S}
  A  = COperator{FC}(ws.m, ws.n, fptr_A, fptr_At, userdata)
  b  = unsafe_wrap(Vector{FC}, Ptr{FC}(b_ptr), ws.m)
  kw = _opts_kw(opts, T)
  λ = T(opts.lambda)
  if fptr_M != C_NULL
    M = CPreconditioner{FC}(ws.n, fptr_M, userdata)
    Krylov.krylov_solve!(ws, A, b; M=M, λ=λ, kw...)
  else
    Krylov.krylov_solve!(ws, A, b; λ=λ, kw...)
  end
  Cint(0)
end

# M + N — BiCGSTAB, CGS, BiLQ, QMR.  The four NULL/non-NULL combinations are
# spelled out so each krylov_solve! call site is concretely typed (--trim=safe).
function _typed_solve_mn!(ws::Krylov.KrylovWorkspace{T, FC, S}, fptr_A, fptr_At, fptr_M, fptr_N, b_ptr, c_ptr, userdata, opts) where {T, FC, S}
  A  = COperator{FC}(ws.m, ws.n, fptr_A, fptr_At, userdata)
  b  = unsafe_wrap(Vector{FC}, Ptr{FC}(b_ptr), ws.m)
  kw = _opts_kw(opts, T)
  if fptr_M != C_NULL && fptr_N != C_NULL
    M = CPreconditioner{FC}(ws.m, fptr_M, userdata)
    N = CPreconditioner{FC}(ws.n, fptr_N, userdata)
    Krylov.krylov_solve!(ws, A, b; M=M, N=N, kw...)
  elseif fptr_M != C_NULL
    M = CPreconditioner{FC}(ws.m, fptr_M, userdata)
    Krylov.krylov_solve!(ws, A, b; M=M, kw...)
  elseif fptr_N != C_NULL
    N = CPreconditioner{FC}(ws.n, fptr_N, userdata)
    Krylov.krylov_solve!(ws, A, b; N=N, kw...)
  else
    Krylov.krylov_solve!(ws, A, b; kw...)
  end
  Cint(0)
end

# M + N + reorthogonalization — DIOM, DQGMRES (limited-memory, no restart).
function _typed_solve_mn_reorth!(ws::Krylov.KrylovWorkspace{T, FC, S}, fptr_A, fptr_At, fptr_M, fptr_N, b_ptr, c_ptr, userdata, opts) where {T, FC, S}
  A  = COperator{FC}(ws.m, ws.n, fptr_A, fptr_At, userdata)
  b  = unsafe_wrap(Vector{FC}, Ptr{FC}(b_ptr), ws.m)
  kw = _opts_kw(opts, T)
  ro = opts.reorthogonalization != 0
  if fptr_M != C_NULL && fptr_N != C_NULL
    M = CPreconditioner{FC}(ws.m, fptr_M, userdata)
    N = CPreconditioner{FC}(ws.n, fptr_N, userdata)
    Krylov.krylov_solve!(ws, A, b; M=M, N=N, reorthogonalization=ro, kw...)
  elseif fptr_M != C_NULL
    M = CPreconditioner{FC}(ws.m, fptr_M, userdata)
    Krylov.krylov_solve!(ws, A, b; M=M, reorthogonalization=ro, kw...)
  elseif fptr_N != C_NULL
    N = CPreconditioner{FC}(ws.n, fptr_N, userdata)
    Krylov.krylov_solve!(ws, A, b; N=N, reorthogonalization=ro, kw...)
  else
    Krylov.krylov_solve!(ws, A, b; reorthogonalization=ro, kw...)
  end
  Cint(0)
end

# M + N + restart + reorthogonalization — GMRES, FGMRES, FOM.
function _typed_solve_gmres!(ws::Krylov.KrylovWorkspace{T, FC, S}, fptr_A, fptr_At, fptr_M, fptr_N, b_ptr, c_ptr, userdata, opts) where {T, FC, S}
  A  = COperator{FC}(ws.m, ws.n, fptr_A, fptr_At, userdata)
  b  = unsafe_wrap(Vector{FC}, Ptr{FC}(b_ptr), ws.m)
  kw = _opts_kw(opts, T)
  rs = opts.restart != 0; ro = opts.reorthogonalization != 0
  if fptr_M != C_NULL && fptr_N != C_NULL
    M = CPreconditioner{FC}(ws.m, fptr_M, userdata)
    N = CPreconditioner{FC}(ws.n, fptr_N, userdata)
    Krylov.krylov_solve!(ws, A, b; M=M, N=N, restart=rs, reorthogonalization=ro, kw...)
  elseif fptr_M != C_NULL
    M = CPreconditioner{FC}(ws.m, fptr_M, userdata)
    Krylov.krylov_solve!(ws, A, b; M=M, restart=rs, reorthogonalization=ro, kw...)
  elseif fptr_N != C_NULL
    N = CPreconditioner{FC}(ws.n, fptr_N, userdata)
    Krylov.krylov_solve!(ws, A, b; N=N, restart=rs, reorthogonalization=ro, kw...)
  else
    Krylov.krylov_solve!(ws, A, b; restart=rs, reorthogonalization=ro, kw...)
  end
  Cint(0)
end

# Least-squares / least-norm preconditioning follows Krylov's convention:
# M = E⁻¹ acts on the m-dimensional data space, N = F⁻¹ on the n-dimensional
# solution space (so M is sized with ws.m, N with ws.n).  λ is always passed
# (0.0 = none).  M/N combinations are spelled out for --trim=safe.

# λ + radius + M(m) + N(n) — LSQR, LSMR.
function _typed_solve_ls_mn_radius!(ws::Krylov.KrylovWorkspace{T, FC, S}, fptr_A, fptr_At, fptr_M, fptr_N, b_ptr, c_ptr, userdata, opts) where {T, FC, S}
  A  = COperator{FC}(ws.m, ws.n, fptr_A, fptr_At, userdata)
  b  = unsafe_wrap(Vector{FC}, Ptr{FC}(b_ptr), ws.m)
  kw = _opts_kw(opts, T); λ = T(opts.lambda); rad = T(opts.radius)
  if fptr_M != C_NULL && fptr_N != C_NULL
    M = CPreconditioner{FC}(ws.m, fptr_M, userdata)
    N = CPreconditioner{FC}(ws.n, fptr_N, userdata)
    Krylov.krylov_solve!(ws, A, b; M=M, N=N, λ=λ, radius=rad, kw...)
  elseif fptr_M != C_NULL
    M = CPreconditioner{FC}(ws.m, fptr_M, userdata)
    Krylov.krylov_solve!(ws, A, b; M=M, λ=λ, radius=rad, kw...)
  elseif fptr_N != C_NULL
    N = CPreconditioner{FC}(ws.n, fptr_N, userdata)
    Krylov.krylov_solve!(ws, A, b; N=N, λ=λ, radius=rad, kw...)
  else
    Krylov.krylov_solve!(ws, A, b; λ=λ, radius=rad, kw...)
  end
  Cint(0)
end

# λ + M(m) + N(n) — LSLQ, CRAIG, CRAIGMR, LNLQ (no radius).
function _typed_solve_ls_mn!(ws::Krylov.KrylovWorkspace{T, FC, S}, fptr_A, fptr_At, fptr_M, fptr_N, b_ptr, c_ptr, userdata, opts) where {T, FC, S}
  A  = COperator{FC}(ws.m, ws.n, fptr_A, fptr_At, userdata)
  b  = unsafe_wrap(Vector{FC}, Ptr{FC}(b_ptr), ws.m)
  kw = _opts_kw(opts, T); λ = T(opts.lambda)
  if fptr_M != C_NULL && fptr_N != C_NULL
    M = CPreconditioner{FC}(ws.m, fptr_M, userdata)
    N = CPreconditioner{FC}(ws.n, fptr_N, userdata)
    Krylov.krylov_solve!(ws, A, b; M=M, N=N, λ=λ, kw...)
  elseif fptr_M != C_NULL
    M = CPreconditioner{FC}(ws.m, fptr_M, userdata)
    Krylov.krylov_solve!(ws, A, b; M=M, λ=λ, kw...)
  elseif fptr_N != C_NULL
    N = CPreconditioner{FC}(ws.n, fptr_N, userdata)
    Krylov.krylov_solve!(ws, A, b; N=N, λ=λ, kw...)
  else
    Krylov.krylov_solve!(ws, A, b; λ=λ, kw...)
  end
  Cint(0)
end

# λ + radius + M(n) — CGLS, CRLS (CG/CR on the normal equations; single
# preconditioner on the n-space, passed via matvec_M).
function _typed_solve_ls_m_radius!(ws::Krylov.KrylovWorkspace{T, FC, S}, fptr_A, fptr_At, fptr_M, fptr_N, b_ptr, c_ptr, userdata, opts) where {T, FC, S}
  A  = COperator{FC}(ws.m, ws.n, fptr_A, fptr_At, userdata)
  b  = unsafe_wrap(Vector{FC}, Ptr{FC}(b_ptr), ws.m)
  kw = _opts_kw(opts, T); λ = T(opts.lambda); rad = T(opts.radius)
  if fptr_M != C_NULL
    M = CPreconditioner{FC}(ws.n, fptr_M, userdata)
    Krylov.krylov_solve!(ws, A, b; M=M, λ=λ, radius=rad, kw...)
  else
    Krylov.krylov_solve!(ws, A, b; λ=λ, radius=rad, kw...)
  end
  Cint(0)
end

# λ + N(n) — CGNE, CRMR (single preconditioner on the n-space, via matvec_N).
function _typed_solve_ls_n!(ws::Krylov.KrylovWorkspace{T, FC, S}, fptr_A, fptr_At, fptr_M, fptr_N, b_ptr, c_ptr, userdata, opts) where {T, FC, S}
  A  = COperator{FC}(ws.m, ws.n, fptr_A, fptr_At, userdata)
  b  = unsafe_wrap(Vector{FC}, Ptr{FC}(b_ptr), ws.m)
  kw = _opts_kw(opts, T); λ = T(opts.lambda)
  if fptr_N != C_NULL
    N = CPreconditioner{FC}(ws.n, fptr_N, userdata)
    Krylov.krylov_solve!(ws, A, b; N=N, λ=λ, kw...)
  else
    Krylov.krylov_solve!(ws, A, b; λ=λ, kw...)
  end
  Cint(0)
end

# TriCG / TriMR: two-RHS + τ and ν (quasi-definite diagonal parameters).
# NaN sentinel → use Krylov.jl defaults (τ=1.0, ν=-1.0).  M/N ignored.
function _typed_solve_tau_nu!(ws::Krylov.KrylovWorkspace{T, FC, S}, fptr_A, fptr_At, fptr_M, fptr_N, b_ptr, c_ptr, userdata, opts) where {T, FC, S}
  A  = COperator{FC}(ws.m, ws.n, fptr_A, fptr_At, userdata)
  b  = unsafe_wrap(Vector{FC}, Ptr{FC}(b_ptr), ws.m)
  c  = unsafe_wrap(Vector{FC}, Ptr{FC}(c_ptr), ws.n)
  kw = _opts_kw(opts, T)
  τ  = isnan(opts.tau) ? T(1)  : T(opts.tau)
  ν  = isnan(opts.nu)  ? T(-1) : T(opts.nu)
  Krylov.krylov_solve!(ws, A, b, c; τ=τ, ν=ν, kw...)
  Cint(0)
end

# Basic two-RHS solvers (BiLQR / TriLQR / USYMLQ / USYMQR / USYMLQR).  M/N ignored.
function _typed_solve_two_rhs!(ws::Krylov.KrylovWorkspace{T, FC, S}, fptr_A, fptr_At, fptr_M, fptr_N, b_ptr, c_ptr, userdata, opts) where {T, FC, S}
  A  = COperator{FC}(ws.m, ws.n, fptr_A, fptr_At, userdata)
  b  = unsafe_wrap(Vector{FC}, Ptr{FC}(b_ptr), ws.m)
  c  = unsafe_wrap(Vector{FC}, Ptr{FC}(c_ptr), ws.n)
  kw = _opts_kw(opts, T)
  Krylov.krylov_solve!(ws, A, b, c; kw...)
  Cint(0)
end

# GPMR — fptr_At slot is repurposed as B (n×m), distinct from A (m×n).
# GPMR has its own preconditioner family (C, D, E, F), not exposed: M/N ignored.
# It does accept reorthogonalization.
function _typed_solve_gpmr!(ws::Krylov.KrylovWorkspace{T, FC, S}, fptr_A, fptr_B, fptr_M, fptr_N, b_ptr, c_ptr, userdata, opts) where {T, FC, S}
  A  = COperator{FC}(ws.m, ws.n, fptr_A, C_NULL, userdata)
  B  = COperator{FC}(ws.n, ws.m, fptr_B, C_NULL, userdata)
  b  = unsafe_wrap(Vector{FC}, Ptr{FC}(b_ptr), ws.m)
  c  = unsafe_wrap(Vector{FC}, Ptr{FC}(c_ptr), ws.n)
  kw = _opts_kw(opts, T)
  Krylov.krylov_solve!(ws, A, B, b, c; reorthogonalization=(opts.reorthogonalization != 0), kw...)
  Cint(0)
end
""")

  # -----------------------------------------------------------------
  # 4. Dispatch functions — big if-elseif on the UInt8 combo key
  # -----------------------------------------------------------------
  println(io, """
# ---------------------------------------------------------------------------
# Dispatch functions — generated if-elseif chains over all $(n_combos) combos.
# Each branch has a statically known concrete type: compatible with --trim=safe.
# ---------------------------------------------------------------------------
""")

  # Helper: emit the if-elseif header for a dispatch function
  function emit_dispatch(io, fname, sig, guard, branches_expr, fallback)
    println(io, "function $(fname)($(sig))")
    println(io, "  $(guard) || return $(fallback)")
    println(io, "  k = ws_key_store[ws_ptr]")
    first = true
    for (si, (sname, _, _)) in enumerate(SOLVERS)
      for (di, (_, suffix, _, _, _)) in enumerate(DTYPES)
        key  = combo_key(si-1, di-1)
        sref = "$(store_name(sname, suffix))[ws_ptr]"
        kw   = first ? "if" : "elseif"
        println(io, "  $(kw) k == 0x$(string(key, base=16, pad=2)); $(branches_expr(sref))")
        first = false
      end
    end
    println(io, "  else; $(fallback); end")
    println(io, "end")
    println(io)
  end

  emit_dispatch(io, "_do_niter", "ws_ptr :: Ptr{Cvoid}",
    "haskey(ws_key_store, ws_ptr)",
    sref -> "_typed_niter($(sref))",
    "Cint(-1)")

  emit_dispatch(io, "_do_is_solved", "ws_ptr :: Ptr{Cvoid}",
    "haskey(ws_key_store, ws_ptr)",
    sref -> "_typed_is_solved($(sref))",
    "Cint(-1)")

  emit_dispatch(io, "_do_elapsed_time", "ws_ptr :: Ptr{Cvoid}",
    "haskey(ws_key_store, ws_ptr)",
    sref -> "_typed_elapsed_time($(sref))",
    "Cdouble(-1.0)")

  emit_dispatch(io, "_do_get_x!", "ws_ptr :: Ptr{Cvoid}, x_ptr :: Ptr{Cvoid}, n :: Cint",
    "haskey(ws_key_store, ws_ptr)",
    sref -> "_typed_get_x!($(sref), x_ptr, n)",
    "Cint(-1)")

  emit_dispatch(io, "_do_get_y!", "ws_ptr :: Ptr{Cvoid}, y_ptr :: Ptr{Cvoid}, m :: Cint",
    "haskey(ws_key_store, ws_ptr)",
    sref -> "_typed_get_y!($(sref), y_ptr, m)",
    "Cint(-1)")

  emit_dispatch(io, "_do_warm_start!", "ws_ptr :: Ptr{Cvoid}, x0_ptr :: Ptr{Cvoid}, n :: Cint",
    "haskey(ws_key_store, ws_ptr)",
    sref -> "_typed_warm_start!($(sref), x0_ptr, n)",
    "Cint(-1)")

  emit_dispatch(io, "_do_warm_start2!", "ws_ptr :: Ptr{Cvoid}, x0_ptr :: Ptr{Cvoid}, y0_ptr :: Ptr{Cvoid}, nx :: Cint, ny :: Cint",
    "haskey(ws_key_store, ws_ptr)",
    sref -> "_typed_warm_start2!($(sref), x0_ptr, y0_ptr, nx, ny)",
    "Cint(-1)")

  # _do_solve! — dispatches to the right helper per solver type
  let sig = join(["ws_ptr :: Ptr{Cvoid}", "fptr_A :: Ptr{Cvoid}", "fptr_At :: Ptr{Cvoid}",
                  "fptr_M :: Ptr{Cvoid}", "fptr_N :: Ptr{Cvoid}", "b_ptr :: Ptr{Cvoid}", "c_ptr :: Ptr{Cvoid}",
                  "userdata :: Ptr{Cvoid}", "opts_ptr :: Ptr{Cvoid}"], ", ")
    println(io, "function _do_solve!($(sig))")
    println(io, "  haskey(ws_key_store, ws_ptr) || return Cint(-1)")
    println(io, "  opts = opts_ptr == C_NULL ? KrylovOptionsC(NaN, NaN, Cint(0), Cint(0), 0.0, NaN, NaN, NaN, 0.0, Cint(0), Cint(0), Cint(0)) : unsafe_load(Ptr{KrylovOptionsC}(opts_ptr))")
    println(io, "  k = ws_key_store[ws_ptr]")
    first = true
    for (si, (sname, _, _)) in enumerate(SOLVERS)
      fn = if sname == GPMR_SOLVER;              "_typed_solve_gpmr!"
           elseif sname in TAU_NU_SOLVERS;       "_typed_solve_tau_nu!"
           elseif sname in TWO_RHS_SOLVERS;      "_typed_solve_two_rhs!"
           elseif sname in GMRES_FAMILY;         "_typed_solve_gmres!"
           elseif sname in DIOM_FAMILY;          "_typed_solve_mn_reorth!"
           elseif sname in MN_SOLVERS;           "_typed_solve_mn!"
           elseif sname in CG_FAMILY;            "_typed_solve_cg!"
           elseif sname in MINRES_FAMILY;        "_typed_solve_minres!"
           elseif sname in SYM_LAMBDA_SOLVERS;   "_typed_solve_sym_lambda!"
           elseif sname in LS_MN_RADIUS_SOLVERS; "_typed_solve_ls_mn_radius!"
           elseif sname in LS_MN_SOLVERS;        "_typed_solve_ls_mn!"
           elseif sname in LS_M_RADIUS_SOLVERS;  "_typed_solve_ls_m_radius!"
           elseif sname in LS_N_SOLVERS;         "_typed_solve_ls_n!"
           else                                  "_typed_solve!"
           end
      for (di, (_, suffix, _, _, _)) in enumerate(DTYPES)
        key  = combo_key(si-1, di-1)
        sref = "$(store_name(sname, suffix))[ws_ptr]"
        kw   = first ? "if" : "elseif"
        println(io, "  $(kw) k == 0x$(string(key, base=16, pad=2)); $(fn)($(sref), fptr_A, fptr_At, fptr_M, fptr_N, b_ptr, c_ptr, userdata, opts)")
        first = false
      end
    end
    println(io, "  else; Cint(-1); end")
    println(io, "end")
    println(io)
  end

  # _do_free! — deletes from the right store AND the key index
  println(io, "function _do_free!(ws_ptr :: Ptr{Cvoid})")
  println(io, "  haskey(ws_key_store, ws_ptr) || return Cint(1)")
  println(io, "  k = ws_key_store[ws_ptr]")
  println(io, "  delete!(ws_key_store, ws_ptr)")
  first = true
  for (si, (sname, _, _)) in enumerate(SOLVERS)
    for (di, (_, suffix, _, _, _)) in enumerate(DTYPES)
      key = combo_key(si-1, di-1)
      kw  = first ? "if" : "elseif"
      println(io, "  $(kw) k == 0x$(string(key, base=16, pad=2)); delete!($(store_name(sname, suffix)), ws_ptr)")
      first = false
    end
  end
  println(io, "  end")
  println(io, "  Cint(0)")
  println(io, "end")
  println(io)

  # _do_create! — dispatch on (solver_int, dtype_int), stores in right dict.
  # Construction-time options (memory / window) come from KrylovWorkspaceOptionsC;
  # each branch passes the kwarg ONLY if the corresponding constructor accepts it.
  println(io, "function _do_create!(solver_int :: Cint, m :: Cint, n :: Cint,")
  println(io, "                     dtype_int :: Cint, wopts :: KrylovWorkspaceOptionsC,")
  println(io, "                     ws_out :: Ptr{Ptr{Cvoid}})")
  println(io, "  mem = wopts.memory == 0 ? $(MEMORY_DEFAULT) : Int(wopts.memory)")
  println(io, "  win = wopts.window == 0 ? $(WINDOW_DEFAULT) : Int(wopts.window)")
  first = true
  for (si, (sname, wstype, _)) in enumerate(SOLVERS)
    ctor_kw = if sname in MEMORY_SOLVERS; "; memory = mem"
              elseif sname in WINDOW_SOLVERS; "; window = win"
              else ""
              end
    for (di, (dtype_int, suffix, T, FC, S)) in enumerate(DTYPES)
      key      = combo_key(si-1, di-1)
      kw       = first ? "if" : "elseif"
      sref     = store_name(sname, suffix)
      sidx     = solver_idx(si)
      println(io, "  $(kw) solver_int == Cint($(sidx)) && dtype_int == Cint($(dtype_int))")
      println(io, "    ws = Krylov.$(wstype)(Int(m), Int(n), $(S)$(ctor_kw))")
      println(io, "    r  = Base.pointer_from_objref(ws)")
      println(io, "    $(sref)[r] = ws; ws_key_store[r] = 0x$(string(key, base=16, pad=2))")
      println(io, "    unsafe_store!(ws_out, r)")
      first = false
    end
  end
  println(io, "  else")
  println(io, "    return Cint(-2)  # unknown (solver, dtype) combination")
  println(io, "  end")
  println(io, "  Cint(0)")
  println(io, "end")
  println(io)

  # =======================================================================
  # Block Krylov interface (block_gmres / block_minres)
  # =======================================================================
  n_block = length(BLOCK_SOLVERS) * length(DTYPES)
  println(io, """
# ---------------------------------------------------------------------------
# Block Krylov stores — $(length(BLOCK_SOLVERS)) solvers × $(length(DTYPES)) precisions = $(n_block) combos.
# SV = Vector{FC}, SM = Matrix{FC}.  Keyed separately from the scalar stores.
# ---------------------------------------------------------------------------
""")
  for (sname, wstype, _) in BLOCK_SOLVERS
    for (_, suffix, T, FC, S) in DTYPES
      println(io, "const $(block_store_name(sname, suffix)) = ",
                  "Dict{Ptr{Cvoid}, Krylov.$(wstype){$(T), $(FC), $(S), Matrix{$(FC)}}}()")
    end
    println(io)
  end
  println(io, "const block_ws_key_store = Dict{Ptr{Cvoid}, UInt8}()")
  println(io)

  # Generic typed helpers (specialized per concrete block workspace type)
  println(io, """
# ---------------------------------------------------------------------------
# Generic typed helpers for block workspaces.
# ---------------------------------------------------------------------------
function _typed_block_niter(ws)
  Cint(ws.stats.niter)
end

function _typed_block_is_solved(ws)
  ws.stats.solved ? Cint(1) : Cint(0)
end

function _typed_block_elapsed_time(ws)
  Cdouble(ws.stats.timer)
end

function _typed_block_get_X!(ws::Krylov.BlockKrylovWorkspace{T, FC, SV, SM}, X_ptr, n, p) where {T, FC, SV, SM}
  copyto!(unsafe_wrap(Matrix{FC}, Ptr{FC}(X_ptr), (Int(n), Int(p))), ws.X)
  Cint(0)
end

function _typed_block_warm_start!(ws::Krylov.BlockKrylovWorkspace{T, FC, SV, SM}, X0_ptr, n, p) where {T, FC, SV, SM}
  X0 = unsafe_wrap(Matrix{FC}, Ptr{FC}(X0_ptr), (Int(n), Int(p)))
  if size(ws.ΔX) != (ws.n, ws.p)
    ws.ΔX = SM(undef, ws.n, ws.p)
  end
  copyto!(ws.ΔX, X0)
  ws.warm_start = true
  Cint(0)
end

# Block solve — uses the GENERIC krylov_solve! (dispatch on the workspace type
# picks block_gmres! / block_minres!).  Going through the generic interface is
# what makes the call trim-safe: it forwards every keyword with its default
# (restart=false, ldiv=false, …) as a literal, so allocate_if(restart, …) is
# constant-folded away — a direct block_gmres! call is NOT trim-safe.
# Supports an optional left preconditioner M (applied as Y = M⁻¹X).
# fptr_N is part of the shared signature but ignored: block_minres takes M only.
function _typed_block_solve!(ws::Krylov.BlockKrylovWorkspace{T, FC, SV, SM}, fptr_A, fptr_M, fptr_N, B_ptr, userdata, opts) where {T, FC, SV, SM}
  A  = CBlockOperator{FC}(ws.m, ws.n, fptr_A, userdata)
  B  = unsafe_wrap(Matrix{FC}, Ptr{FC}(B_ptr), (ws.m, ws.p))
  kw = _opts_kw(opts, T)
  if fptr_M != C_NULL
    M = CBlockOperator{FC}(ws.n, ws.n, fptr_M, userdata)
    Krylov.krylov_solve!(ws, A, B; M=M, kw...)
  else
    Krylov.krylov_solve!(ws, A, B; kw...)
  end
  Cint(0)
end

# block_gmres accepts BOTH a left (M) and right (N) preconditioner.  As for the
# single-vector _typed_solve_mn!, the four combinations are spelled out to keep
# each krylov_solve! call site concretely typed for --trim=safe.
function _typed_block_solve_mn!(ws::Krylov.BlockKrylovWorkspace{T, FC, SV, SM}, fptr_A, fptr_M, fptr_N, B_ptr, userdata, opts) where {T, FC, SV, SM}
  A  = CBlockOperator{FC}(ws.m, ws.n, fptr_A, userdata)
  B  = unsafe_wrap(Matrix{FC}, Ptr{FC}(B_ptr), (ws.m, ws.p))
  kw = _opts_kw(opts, T)
  rs = opts.restart != 0; ro = opts.reorthogonalization != 0
  if fptr_M != C_NULL && fptr_N != C_NULL
    M = CBlockOperator{FC}(ws.m, ws.m, fptr_M, userdata)
    N = CBlockOperator{FC}(ws.n, ws.n, fptr_N, userdata)
    Krylov.krylov_solve!(ws, A, B; M=M, N=N, restart=rs, reorthogonalization=ro, kw...)
  elseif fptr_M != C_NULL
    M = CBlockOperator{FC}(ws.m, ws.m, fptr_M, userdata)
    Krylov.krylov_solve!(ws, A, B; M=M, restart=rs, reorthogonalization=ro, kw...)
  elseif fptr_N != C_NULL
    N = CBlockOperator{FC}(ws.n, ws.n, fptr_N, userdata)
    Krylov.krylov_solve!(ws, A, B; N=N, restart=rs, reorthogonalization=ro, kw...)
  else
    Krylov.krylov_solve!(ws, A, B; restart=rs, reorthogonalization=ro, kw...)
  end
  Cint(0)
end
""")

  # Block dispatch helper — emit an if-elseif over block_ws_key_store
  function emit_block_dispatch(io, fname, sig, branches_expr, fallback)
    println(io, "function $(fname)($(sig))")
    println(io, "  haskey(block_ws_key_store, ws_ptr) || return $(fallback)")
    println(io, "  k = block_ws_key_store[ws_ptr]")
    first = true
    for (si, (sname, _, _)) in enumerate(BLOCK_SOLVERS)
      for (di, (_, suffix, _, _, _)) in enumerate(DTYPES)
        key  = block_combo_key(si-1, di-1)
        sref = "$(block_store_name(sname, suffix))[ws_ptr]"
        kw   = first ? "if" : "elseif"
        println(io, "  $(kw) k == 0x$(string(key, base=16, pad=2)); $(branches_expr(sref))")
        first = false
      end
    end
    println(io, "  else; $(fallback); end")
    println(io, "end")
    println(io)
  end

  emit_block_dispatch(io, "_do_block_niter", "ws_ptr :: Ptr{Cvoid}",
    sref -> "_typed_block_niter($(sref))", "Cint(-1)")
  emit_block_dispatch(io, "_do_block_is_solved", "ws_ptr :: Ptr{Cvoid}",
    sref -> "_typed_block_is_solved($(sref))", "Cint(-1)")
  emit_block_dispatch(io, "_do_block_elapsed_time", "ws_ptr :: Ptr{Cvoid}",
    sref -> "_typed_block_elapsed_time($(sref))", "Cdouble(-1.0)")
  emit_block_dispatch(io, "_do_block_get_X!", "ws_ptr :: Ptr{Cvoid}, X_ptr :: Ptr{Cvoid}, n :: Cint, p :: Cint",
    sref -> "_typed_block_get_X!($(sref), X_ptr, n, p)", "Cint(-1)")
  emit_block_dispatch(io, "_do_block_warm_start!", "ws_ptr :: Ptr{Cvoid}, X0_ptr :: Ptr{Cvoid}, n :: Cint, p :: Cint",
    sref -> "_typed_block_warm_start!($(sref), X0_ptr, n, p)", "Cint(-1)")

  # _do_block_solve! — pick block_gmres! / block_minres! per solver
  let sig = join(["ws_ptr :: Ptr{Cvoid}", "fptr_A :: Ptr{Cvoid}", "fptr_M :: Ptr{Cvoid}", "fptr_N :: Ptr{Cvoid}",
                  "B_ptr :: Ptr{Cvoid}", "userdata :: Ptr{Cvoid}", "opts_ptr :: Ptr{Cvoid}"], ", ")
    println(io, "function _do_block_solve!($(sig))")
    println(io, "  haskey(block_ws_key_store, ws_ptr) || return Cint(-1)")
    println(io, "  opts = opts_ptr == C_NULL ? KrylovOptionsC(NaN, NaN, Cint(0), Cint(0), 0.0, NaN, NaN, NaN, 0.0, Cint(0), Cint(0), Cint(0)) : unsafe_load(Ptr{KrylovOptionsC}(opts_ptr))")
    println(io, "  k = block_ws_key_store[ws_ptr]")
    first = true
    for (si, (sname, _, _)) in enumerate(BLOCK_SOLVERS)
      fn = sname in BLOCK_MN_SOLVERS ? "_typed_block_solve_mn!" : "_typed_block_solve!"
      for (di, (_, suffix, _, _, _)) in enumerate(DTYPES)
        key  = block_combo_key(si-1, di-1)
        sref = "$(block_store_name(sname, suffix))[ws_ptr]"
        kw   = first ? "if" : "elseif"
        println(io, "  $(kw) k == 0x$(string(key, base=16, pad=2)); $(fn)($(sref), fptr_A, fptr_M, fptr_N, B_ptr, userdata, opts)")
        first = false
      end
    end
    println(io, "  else; Cint(-1); end")
    println(io, "end")
    println(io)
  end

  # _do_block_free!
  println(io, "function _do_block_free!(ws_ptr :: Ptr{Cvoid})")
  println(io, "  haskey(block_ws_key_store, ws_ptr) || return Cint(1)")
  println(io, "  k = block_ws_key_store[ws_ptr]")
  println(io, "  delete!(block_ws_key_store, ws_ptr)")
  first = true
  for (si, (sname, _, _)) in enumerate(BLOCK_SOLVERS)
    for (di, (_, suffix, _, _, _)) in enumerate(DTYPES)
      key = block_combo_key(si-1, di-1)
      kw  = first ? "if" : "elseif"
      println(io, "  $(kw) k == 0x$(string(key, base=16, pad=2)); delete!($(block_store_name(sname, suffix)), ws_ptr)")
      first = false
    end
  end
  println(io, "  end")
  println(io, "  Cint(0)")
  println(io, "end")
  println(io)

  # _do_block_create! — dispatch on (solver_int, dtype_int); takes block size p
  println(io, "function _do_block_create!(solver_int :: Cint, m :: Cint, n :: Cint, p :: Cint,")
  println(io, "                           dtype_int :: Cint, wopts :: KrylovWorkspaceOptionsC,")
  println(io, "                           ws_out :: Ptr{Ptr{Cvoid}})")
  println(io, "  mem = wopts.memory == 0 ? $(BLOCK_MEMORY_DEFAULT) : Int(wopts.memory)")
  first = true
  for (si, (sname, wstype, _)) in enumerate(BLOCK_SOLVERS)
    ctor_kw = sname in BLOCK_MEMORY_SOLVERS ? "; memory = mem" : ""
    for (di, (dtype_int, suffix, T, FC, S)) in enumerate(DTYPES)
      key  = block_combo_key(si-1, di-1)
      kw   = first ? "if" : "elseif"
      sref = block_store_name(sname, suffix)
      println(io, "  $(kw) solver_int == Cint($(si-1)) && dtype_int == Cint($(dtype_int))")
      println(io, "    ws = Krylov.$(wstype)(Int(m), Int(n), Int(p), $(S), Matrix{$(FC)}$(ctor_kw))")
      println(io, "    r  = Base.pointer_from_objref(ws)")
      println(io, "    $(sref)[r] = ws; block_ws_key_store[r] = 0x$(string(key, base=16, pad=2))")
      println(io, "    unsafe_store!(ws_out, r)")
      first = false
    end
  end
  println(io, "  else")
  println(io, "    return Cint(-2)  # unknown (block solver, dtype) combination")
  println(io, "  end")
  println(io, "  Cint(0)")
  println(io, "end")
end

println("Generated $out")
println("  $(length(SOLVERS)) solvers × $(length(DTYPES)) precisions = $(length(SOLVERS)*length(DTYPES)) combos")
println("  $(length(BLOCK_SOLVERS)) block solvers × $(length(DTYPES)) precisions = $(length(BLOCK_SOLVERS)*length(DTYPES)) combos")
