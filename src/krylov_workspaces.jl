export KrylovWorkspace, MinresWorkspace, CgWorkspace, CrWorkspace, SymmlqWorkspace, CgLanczosWorkspace,
CgLanczosShiftWorkspace, MinresQlpWorkspace, DqgmresWorkspace, DiomWorkspace, UsymlqWorkspace,
UsymqrWorkspace, TricgWorkspace, TrimrWorkspace, TrilqrWorkspace, CgsWorkspace, BicgstabWorkspace,
BilqWorkspace, QmrWorkspace, BilqrWorkspace, CglsWorkspace, CglsLanczosShiftWorkspace, CrlsWorkspace, CgneWorkspace,
CrmrWorkspace, LslqWorkspace, LsqrWorkspace, LsmrWorkspace, LnlqWorkspace, CraigWorkspace, CraigmrWorkspace,
GmresWorkspace, FomWorkspace, GpmrWorkspace, FgmresWorkspace, CarWorkspace, MinaresWorkspace

export KrylovConstructor

"""
    KrylovConstructor(vm; vm_empty=vm)
    KrylovConstructor(vm, vn; vm_empty=vm, vn_empty=vn)

Krylov methods require a workspace containing vectors of length `m` and `n` to solve linear problems of size `m × n`.
The `KrylovConstructor` facilitates the allocation of these vectors using `similar`.

For square problems (`m == n`), use the first constructor with a single vector `vm`.
For rectangular problems (`m ≠ n`), use the second constructor with `vm` and `vn`.

#### Input arguments

* `vm`: a vector of length `m`;
* `vn`: a vector of length `n`.

#### Keyword arguments

- `vm_empty`: an empty vector that may be replaced with a vector of length `m`;
- `vn_empty`: an empty vector that may be replaced with a vector of length `n`.

#### Note

Empty vectors `vm_empty` and `vn_empty` reduce storage requirements when features such as warm-start or preconditioners are unused.
These empty vectors will be replaced within a [`KrylovWorkspace`](@ref) only if required, such as when preconditioners are provided.
"""
struct KrylovConstructor{S}
  vm::S
  vn::S
  vm_empty::S
  vn_empty::S
end

function KrylovConstructor(vm; vm_empty=vm)
  return KrylovConstructor(vm, vm, vm_empty, vm_empty)
end

function KrylovConstructor(vm, vn; vm_empty=vm, vn_empty=vn)
  return KrylovConstructor(vm, vn, vm_empty, vn_empty)
end

"Abstract type for using Krylov solvers in-place."
abstract type KrylovWorkspace{T,FC,S} end

"""
Workspace for the in-place method [`minres!`](@ref).

The following outer constructors can be used to initialize this workspace:

    workspace = MinresWorkspace(m, n, S; window = 5)
    workspace = MinresWorkspace(A, b; window = 5)
    workspace = MinresWorkspace(kc::KrylovConstructor; window = 5)
"""
mutable struct MinresWorkspace{T,FC,S} <: KrylovWorkspace{T,FC,S}
  m          :: Int
  n          :: Int
  Δx         :: S
  x          :: S
  r1         :: S
  r2         :: S
  npc_dir    :: S
  w1         :: S
  w2         :: S
  y          :: S
  v          :: S
  err_vec    :: Vector{T}
  warm_start :: Bool
  stats      :: SimpleStats{T}
end

function MinresWorkspace(kc::KrylovConstructor; window::Integer = 5)
  S  = typeof(kc.vm)
  FC = eltype(S)
  T  = real(FC)
  m  = length(kc.vm)
  n  = length(kc.vn)
  Δx = similar(kc.vn_empty)
  x  = similar(kc.vn)
  r1 = similar(kc.vn)
  r2 = similar(kc.vn)
  npc_dir = similar(kc.vn_empty)
  w1 = similar(kc.vn)
  w2 = similar(kc.vn)
  y  = similar(kc.vn)
  v  = similar(kc.vn_empty)
  err_vec = zeros(T, window)
  stats = SimpleStats(0, false, false, false, 0, zero(T), zero(T), T[], T[], T[], 0.0, "unknown")
  workspace = MinresWorkspace{T,FC,S}(m, n, Δx, x, r1, r2, npc_dir, w1, w2, y, v, err_vec, false, stats)
  return workspace
end

function MinresWorkspace(m::Integer, n::Integer, S::Type; window::Integer = 5)
  FC = eltype(S)
  T  = real(FC)
  Δx = S(undef, 0)
  x  = S(undef, n)
  r1 = S(undef, n)
  r2 = S(undef, n)
  npc_dir = S(undef, 0)
  w1 = S(undef, n)
  w2 = S(undef, n)
  y  = S(undef, n)
  v  = S(undef, 0)
  err_vec = zeros(T, window)
  S = isconcretetype(S) ? S : typeof(x)
  stats = SimpleStats(0, false, false, false, 0, zero(T), zero(T), T[], T[], T[], 0.0, "unknown")
  workspace = MinresWorkspace{T,FC,S}(m, n, Δx, x, r1, r2, npc_dir, w1, w2, y, v, err_vec, false, stats)
  return workspace
end

function MinresWorkspace(A, b; window::Integer = 5)
  m, n = size(A)
  S = ktypeof(b)
  MinresWorkspace(m, n, S; window)
end

"""
Workspace for the in-place method [`minares!`](@ref).

The following outer constructors can be used to initialize this workspace:

    workspace = MinaresWorkspace(m, n, S)
    workspace = MinaresWorkspace(A, b)
    workspace = MinaresWorkspace(kc::KrylovConstructor)
"""
mutable struct MinaresWorkspace{T,FC,S} <: KrylovWorkspace{T,FC,S}
  m          :: Int
  n          :: Int
  Δx         :: S
  vₖ         :: S
  vₖ₊₁       :: S
  x          :: S
  wₖ₋₂       :: S
  wₖ₋₁       :: S
  dₖ₋₂       :: S
  dₖ₋₁       :: S
  q          :: S
  warm_start :: Bool
  stats      :: SimpleStats{T}
end

function MinaresWorkspace(kc::KrylovConstructor)
  S    = typeof(kc.vm)
  FC   = eltype(S)
  T    = real(FC)
  m    = length(kc.vm)
  n    = length(kc.vn)
  Δx   = similar(kc.vn_empty)
  vₖ   = similar(kc.vn)
  vₖ₊₁ = similar(kc.vn)
  x    = similar(kc.vn)
  wₖ₋₂ = similar(kc.vn)
  wₖ₋₁ = similar(kc.vn)
  dₖ₋₂ = similar(kc.vn)
  dₖ₋₁ = similar(kc.vn)
  q    = similar(kc.vn)
  stats = SimpleStats(0, false, false, false, 0, zero(T), zero(T), T[], T[], T[], 0.0, "unknown")
  workspace = MinaresWorkspace{T,FC,S}(m, n, Δx, vₖ, vₖ₊₁, x, wₖ₋₂, wₖ₋₁, dₖ₋₂, dₖ₋₁, q, false, stats)
  return workspace
end

function MinaresWorkspace(m::Integer, n::Integer, S::Type)
  FC   = eltype(S)
  T    = real(FC)
  Δx   = S(undef, 0)
  vₖ   = S(undef, n)
  vₖ₊₁ = S(undef, n)
  x    = S(undef, n)
  wₖ₋₂ = S(undef, n)
  wₖ₋₁ = S(undef, n)
  dₖ₋₂ = S(undef, n)
  dₖ₋₁ = S(undef, n)
  q    = S(undef, n)
  S = isconcretetype(S) ? S : typeof(x)
  stats = SimpleStats(0, false, false, false, 0, zero(T), zero(T), T[], T[], T[], 0.0, "unknown")
  workspace = MinaresWorkspace{T,FC,S}(m, n, Δx, vₖ, vₖ₊₁, x, wₖ₋₂, wₖ₋₁, dₖ₋₂, dₖ₋₁, q, false, stats)
  return workspace
end

function MinaresWorkspace(A, b)
  m, n = size(A)
  S = ktypeof(b)
  MinaresWorkspace(m, n, S)
end

"""
Workspace for the in-place method [`cg!`](@ref).

The following outer constructors can be used to initialize this workspace:

    workspace = CgWorkspace(m, n, S)
    workspace = CgWorkspace(A, b)
    workspace = CgWorkspace(kc::KrylovConstructor)
"""
mutable struct CgWorkspace{T,FC,S} <: KrylovWorkspace{T,FC,S}
  m          :: Int
  n          :: Int
  Δx         :: S
  x          :: S
  r          :: S
  npc_dir    :: S
  p          :: S
  Ap         :: S
  z          :: S
  warm_start :: Bool
  stats      :: SimpleStats{T}
end

function CgWorkspace(kc::KrylovConstructor)
  S  = typeof(kc.vm)
  FC = eltype(S)
  T  = real(FC)
  m  = length(kc.vm)
  n  = length(kc.vn)
  Δx = similar(kc.vn_empty)
  x  = similar(kc.vn)
  r  = similar(kc.vn)
  npc_dir = similar(kc.vn_empty)
  p  = similar(kc.vn)
  Ap = similar(kc.vn)
  z  = similar(kc.vn_empty)
  stats = SimpleStats(0, false, false, false, 0, zero(T), zero(T), T[], T[], T[], 0.0, "unknown")
  workspace = CgWorkspace{T,FC,S}(m, n, Δx, x, r, npc_dir, p, Ap, z, false, stats)
  return workspace
end

function CgWorkspace(m::Integer, n::Integer, S::Type)
  FC = eltype(S)
  T  = real(FC)
  Δx = S(undef, 0)
  x  = S(undef, n)
  r  = S(undef, n)
  npc_dir = S(undef, 0)
  p  = S(undef, n)
  Ap = S(undef, n)
  z  = S(undef, 0)
  S = isconcretetype(S) ? S : typeof(x)
  stats = SimpleStats(0, false, false, false, 0, zero(T), zero(T), T[], T[], T[], 0.0, "unknown")
  workspace = CgWorkspace{T,FC,S}(m, n, Δx, x, r, npc_dir, p, Ap, z, false, stats)
  return workspace
end

function CgWorkspace(A, b)
  m, n = size(A)
  S = ktypeof(b)
  CgWorkspace(m, n, S)
end

"""
Workspace for the in-place method [`cr!`](@ref).

The following outer constructors can be used to initialize this workspace:

    workspace = CrWorkspace(m, n, S)
    workspace = CrWorkspace(A, b)
    workspace = CrWorkspace(kc::KrylovConstructor)
"""
mutable struct CrWorkspace{T,FC,S} <: KrylovWorkspace{T,FC,S}
  m          :: Int
  n          :: Int
  Δx         :: S
  x          :: S
  r          :: S
  npc_dir    :: S
  p          :: S
  q          :: S
  Ar         :: S
  Mq         :: S
  warm_start :: Bool
  stats      :: SimpleStats{T}
end

function CrWorkspace(kc::KrylovConstructor)
  S  = typeof(kc.vm)
  FC = eltype(S)
  T  = real(FC)
  m  = length(kc.vm)
  n  = length(kc.vn)
  Δx = similar(kc.vn_empty)
  x  = similar(kc.vn)
  r  = similar(kc.vn)
  npc_dir = similar(kc.vn_empty)
  p  = similar(kc.vn)
  q  = similar(kc.vn)
  Ar = similar(kc.vn)
  Mq = similar(kc.vn_empty)
  stats = SimpleStats(0, false, false, false, 0, zero(T), zero(T), T[], T[], T[], 0.0, "unknown")
  workspace = CrWorkspace{T,FC,S}(m, n, Δx, x, r, npc_dir, p, q, Ar, Mq, false, stats)
  return workspace
end

function CrWorkspace(m::Integer, n::Integer, S::Type)
  FC = eltype(S)
  T  = real(FC)
  Δx = S(undef, 0)
  x  = S(undef, n)
  r  = S(undef, n)
  npc_dir = S(undef, 0)
  p  = S(undef, n)
  q  = S(undef, n)
  Ar = S(undef, n)
  Mq = S(undef, 0)
  S = isconcretetype(S) ? S : typeof(x)
  stats = SimpleStats(0, false, false, false, 0, zero(T), zero(T), T[], T[], T[], 0.0, "unknown")
  workspace = CrWorkspace{T,FC,S}(m, n, Δx, x, r, npc_dir, p, q, Ar, Mq, false, stats)
  return workspace
end

function CrWorkspace(A, b)
  m, n = size(A)
  S = ktypeof(b)
  CrWorkspace(m, n, S)
end

"""
Workspace for the in-place method [`car!`](@ref).

The following outer constructors can be used to initialize this workspace:

    workspace = CarWorkspace(m, n, S)
    workspace = CarWorkspace(A, b)
    workspace = CarWorkspace(kc::KrylovConstructor)
"""
mutable struct CarWorkspace{T,FC,S} <: KrylovWorkspace{T,FC,S}
  m          :: Int
  n          :: Int
  Δx         :: S
  x          :: S
  r          :: S
  p          :: S
  s          :: S
  q          :: S
  t          :: S
  u          :: S
  Mu         :: S
  warm_start :: Bool
  stats      :: SimpleStats{T}
end

function CarWorkspace(kc::KrylovConstructor)
  S  = typeof(kc.vm)
  FC = eltype(S)
  T  = real(FC)
  m  = length(kc.vm)
  n  = length(kc.vn)
  Δx = similar(kc.vn_empty)
  x  = similar(kc.vn)
  r  = similar(kc.vn)
  p  = similar(kc.vn)
  s  = similar(kc.vn)
  q  = similar(kc.vn)
  t  = similar(kc.vn)
  u  = similar(kc.vn)
  Mu = similar(kc.vn_empty)
  stats = SimpleStats(0, false, false, false, 0, zero(T), zero(T), T[], T[], T[], 0.0, "unknown")
  workspace = CarWorkspace{T,FC,S}(m, n, Δx, x, r, p, s, q, t, u, Mu, false, stats)
  return workspace
end

function CarWorkspace(m::Integer, n::Integer, S::Type)
  FC = eltype(S)
  T  = real(FC)
  Δx = S(undef, 0)
  x  = S(undef, n)
  r  = S(undef, n)
  p  = S(undef, n)
  s  = S(undef, n)
  q  = S(undef, n)
  t  = S(undef, n)
  u  = S(undef, n)
  Mu = S(undef, 0)
  S = isconcretetype(S) ? S : typeof(x)
  stats = SimpleStats(0, false, false, false, 0, zero(T), zero(T), T[], T[], T[], 0.0, "unknown")
  workspace = CarWorkspace{T,FC,S}(m, n, Δx, x, r, p, s, q, t, u, Mu, false, stats)
  return workspace
end

function CarWorkspace(A, b)
  m, n = size(A)
  S = ktypeof(b)
  CarWorkspace(m, n, S)
end

"""
Workspace for the in-place method [`symmlq!`](@ref).

The following outer constructors can be used to initialize this workspace:

    workspace = SymmlqWorkspace(m, n, S)
    workspace = SymmlqWorkspace(A, b)
    workspace = SymmlqWorkspace(kc::KrylovConstructor)
"""
mutable struct SymmlqWorkspace{T,FC,S} <: KrylovWorkspace{T,FC,S}
  m          :: Int
  n          :: Int
  Δx         :: S
  x          :: S
  Mvold      :: S
  Mv         :: S
  Mv_next    :: S
  w̅          :: S
  v          :: S
  clist      :: Vector{T}
  zlist      :: Vector{T}
  sprod      :: Vector{T}
  warm_start :: Bool
  stats      :: SymmlqStats{T}
end

function SymmlqWorkspace(kc::KrylovConstructor; window::Integer = 5)
  S       = typeof(kc.vm)
  FC      = eltype(S)
  T       = real(FC)
  m       = length(kc.vm)
  n       = length(kc.vn)
  Δx      = similar(kc.vn_empty)
  x       = similar(kc.vn)
  Mvold   = similar(kc.vn)
  Mv      = similar(kc.vn)
  Mv_next = similar(kc.vn)
  w̅       = similar(kc.vn)
  v       = similar(kc.vn_empty)
  clist   = zeros(T, window)
  zlist   = zeros(T, window)
  sprod   = ones(T, window)
  stats = SymmlqStats(0, false, T[], Union{T, Missing}[], T[], Union{T, Missing}[], T(NaN), T(NaN), 0.0, "unknown")
  workspace = SymmlqWorkspace{T,FC,S}(m, n, Δx, x, Mvold, Mv, Mv_next, w̅, v, clist, zlist, sprod, false, stats)
  return workspace
end

function SymmlqWorkspace(m::Integer, n::Integer, S::Type; window::Integer = 5)
  FC      = eltype(S)
  T       = real(FC)
  Δx      = S(undef, 0)
  x       = S(undef, n)
  Mvold   = S(undef, n)
  Mv      = S(undef, n)
  Mv_next = S(undef, n)
  w̅       = S(undef, n)
  v       = S(undef, 0)
  clist   = zeros(T, window)
  zlist   = zeros(T, window)
  sprod   = ones(T, window)
  S = isconcretetype(S) ? S : typeof(x)
  stats = SymmlqStats(0, false, T[], Union{T, Missing}[], T[], Union{T, Missing}[], T(NaN), T(NaN), 0.0, "unknown")
  workspace = SymmlqWorkspace{T,FC,S}(m, n, Δx, x, Mvold, Mv, Mv_next, w̅, v, clist, zlist, sprod, false, stats)
  return workspace
end

function SymmlqWorkspace(A, b; window::Integer = 5)
  m, n = size(A)
  S = ktypeof(b)
  SymmlqWorkspace(m, n, S; window)
end

"""
Workspace for the in-place method [`cg_lanczos!`](@ref).

The following outer constructors can be used to initialize this workspace:

    workspace = CgLanczosWorkspace(m, n, S)
    workspace = CgLanczosWorkspace(A, b)
    workspace = CgLanczosWorkspace(kc::KrylovConstructor)
"""
mutable struct CgLanczosWorkspace{T,FC,S} <: KrylovWorkspace{T,FC,S}
  m          :: Int
  n          :: Int
  Δx         :: S
  x          :: S
  Mv         :: S
  Mv_prev    :: S
  p          :: S
  Mv_next    :: S
  v          :: S
  warm_start :: Bool
  stats      :: LanczosStats{T}
end

function CgLanczosWorkspace(kc::KrylovConstructor)
  S       = typeof(kc.vm)
  FC      = eltype(S)
  T       = real(FC)
  m       = length(kc.vm)
  n       = length(kc.vn)
  Δx      = similar(kc.vn_empty)
  x       = similar(kc.vn)
  Mv      = similar(kc.vn)
  Mv_prev = similar(kc.vn)
  p       = similar(kc.vn)
  Mv_next = similar(kc.vn)
  v       = similar(kc.vn_empty)
  stats = LanczosStats(0, false, T[], false, T(NaN), T(NaN), 0.0, "unknown")
  workspace = CgLanczosWorkspace{T,FC,S}(m, n, Δx, x, Mv, Mv_prev, p, Mv_next, v, false, stats)
  return workspace
end

function CgLanczosWorkspace(m::Integer, n::Integer, S::Type)
  FC      = eltype(S)
  T       = real(FC)
  Δx      = S(undef, 0)
  x       = S(undef, n)
  Mv      = S(undef, n)
  Mv_prev = S(undef, n)
  p       = S(undef, n)
  Mv_next = S(undef, n)
  v       = S(undef, 0)
  S = isconcretetype(S) ? S : typeof(x)
  stats = LanczosStats(0, false, T[], false, T(NaN), T(NaN), 0.0, "unknown")
  workspace = CgLanczosWorkspace{T,FC,S}(m, n, Δx, x, Mv, Mv_prev, p, Mv_next, v, false, stats)
  return workspace
end

function CgLanczosWorkspace(A, b)
  m, n = size(A)
  S = ktypeof(b)
  CgLanczosWorkspace(m, n, S)
end

"""
Workspace for the in-place method [`cg_lanczos_shift!`](@ref).

The following outer constructors can be used to initialize this workspace:

    workspace = CgLanczosShiftWorkspace(m, n, nshifts, S)
    workspace = CgLanczosShiftWorkspace(A, b, nshifts)
    workspace = CgLanczosShiftWorkspace(kc::KrylovConstructor, nshifts)
"""
mutable struct CgLanczosShiftWorkspace{T,FC,S} <: KrylovWorkspace{T,FC,S}
  m          :: Int
  n          :: Int
  nshifts    :: Int
  Mv         :: S
  Mv_prev    :: S
  Mv_next    :: S
  v          :: S
  x          :: Vector{S}
  p          :: Vector{S}
  σ          :: Vector{T}
  δhat       :: Vector{T}
  ω          :: Vector{T}
  γ          :: Vector{T}
  rNorms     :: Vector{T}
  converged  :: BitVector
  not_cv     :: BitVector
  stats      :: LanczosShiftStats{T}
end

function CgLanczosShiftWorkspace(kc::KrylovConstructor, nshifts::Integer)
  S          = typeof(kc.vm)
  FC         = eltype(S)
  T          = real(FC)
  m          = length(kc.vm)
  n          = length(kc.vn)
  Mv         = similar(kc.vn)
  Mv_prev    = similar(kc.vn)
  Mv_next    = similar(kc.vn)
  v          = similar(kc.vn_empty)
  x          = S[similar(kc.vn) for i = 1 : nshifts]
  p          = S[similar(kc.vn) for i = 1 : nshifts]
  σ          = Vector{T}(undef, nshifts)
  δhat       = Vector{T}(undef, nshifts)
  ω          = Vector{T}(undef, nshifts)
  γ          = Vector{T}(undef, nshifts)
  rNorms     = Vector{T}(undef, nshifts)
  indefinite = BitVector(undef, nshifts)
  converged  = BitVector(undef, nshifts)
  not_cv     = BitVector(undef, nshifts)
  stats = LanczosShiftStats(0, false, Vector{T}[T[] for i = 1 : nshifts], indefinite, T(NaN), T(NaN), 0.0, "unknown")
  workspace = CgLanczosShiftWorkspace{T,FC,S}(m, n, nshifts, Mv, Mv_prev, Mv_next, v, x, p, σ, δhat, ω, γ, rNorms, converged, not_cv, stats)
  return workspace
end

function CgLanczosShiftWorkspace(m::Integer, n::Integer, nshifts::Integer, S::Type)
  FC         = eltype(S)
  T          = real(FC)
  Mv         = S(undef, n)
  Mv_prev    = S(undef, n)
  Mv_next    = S(undef, n)
  v          = S(undef, 0)
  x          = S[S(undef, n) for i = 1 : nshifts]
  p          = S[S(undef, n) for i = 1 : nshifts]
  σ          = Vector{T}(undef, nshifts)
  δhat       = Vector{T}(undef, nshifts)
  ω          = Vector{T}(undef, nshifts)
  γ          = Vector{T}(undef, nshifts)
  rNorms     = Vector{T}(undef, nshifts)
  indefinite = BitVector(undef, nshifts)
  converged  = BitVector(undef, nshifts)
  not_cv     = BitVector(undef, nshifts)
  S = isconcretetype(S) ? S : typeof(x)
  stats = LanczosShiftStats(0, false, Vector{T}[T[] for i = 1 : nshifts], indefinite, T(NaN), T(NaN), 0.0, "unknown")
  workspace = CgLanczosShiftWorkspace{T,FC,S}(m, n, nshifts, Mv, Mv_prev, Mv_next, v, x, p, σ, δhat, ω, γ, rNorms, converged, not_cv, stats)
  return workspace
end

function CgLanczosShiftWorkspace(A, b, nshifts::Integer)
  m, n = size(A)
  S = ktypeof(b)
  CgLanczosShiftWorkspace(m, n, nshifts, S)
end

"""
Workspace for the in-place method [`minres_qlp!`](@ref).

The following outer constructors can be used to initialize this workspace:

    workspace = MinresQlpWorkspace(m, n, S)
    workspace = MinresQlpWorkspace(A, b)
    workspace = MinresQlpWorkspace(kc::KrylovConstructor)
"""
mutable struct MinresQlpWorkspace{T,FC,S} <: KrylovWorkspace{T,FC,S}
  m          :: Int
  n          :: Int
  Δx         :: S
  wₖ₋₁       :: S
  wₖ         :: S
  M⁻¹vₖ₋₁    :: S
  M⁻¹vₖ      :: S
  x          :: S
  p          :: S
  vₖ         :: S
  warm_start :: Bool
  stats      :: SimpleStats{T}
end

function MinresQlpWorkspace(kc::KrylovConstructor)
  S       = typeof(kc.vm)
  FC      = eltype(S)
  T       = real(FC)
  m       = length(kc.vm)
  n       = length(kc.vn)
  Δx      = similar(kc.vn_empty)
  wₖ₋₁    = similar(kc.vn)
  wₖ      = similar(kc.vn)
  M⁻¹vₖ₋₁ = similar(kc.vn)
  M⁻¹vₖ   = similar(kc.vn)
  x       = similar(kc.vn)
  p       = similar(kc.vn)
  vₖ      = similar(kc.vn_empty)
  stats = SimpleStats(0, false, false, false, 0, zero(T), zero(T), T[], T[], T[], 0.0, "unknown")
  workspace = MinresQlpWorkspace{T,FC,S}(m, n, Δx, wₖ₋₁, wₖ, M⁻¹vₖ₋₁, M⁻¹vₖ, x, p, vₖ, false, stats)
  return workspace
end

function MinresQlpWorkspace(m::Integer, n::Integer, S::Type)
  FC      = eltype(S)
  T       = real(FC)
  Δx      = S(undef, 0)
  wₖ₋₁    = S(undef, n)
  wₖ      = S(undef, n)
  M⁻¹vₖ₋₁ = S(undef, n)
  M⁻¹vₖ   = S(undef, n)
  x       = S(undef, n)
  p       = S(undef, n)
  vₖ      = S(undef, 0)
  S = isconcretetype(S) ? S : typeof(x)
  stats = SimpleStats(0, false, false, false, 0, zero(T), zero(T), T[], T[], T[], 0.0, "unknown")
  workspace = MinresQlpWorkspace{T,FC,S}(m, n, Δx, wₖ₋₁, wₖ, M⁻¹vₖ₋₁, M⁻¹vₖ, x, p, vₖ, false, stats)
  return workspace
end

function MinresQlpWorkspace(A, b)
  m, n = size(A)
  S = ktypeof(b)
  MinresQlpWorkspace(m, n, S)
end

"""
Workspace for the in-place method [`dqgmres!`](@ref).

The following outer constructors can be used to initialize this workspace:

    workspace = DqgmresWorkspace(m, n, S; memory = 20)
    workspace = DqgmresWorkspace(A, b; memory = 20)
    workspace = DqgmresWorkspace(kc::KrylovConstructor; memory = 20)

`memory` is set to `n` if the value given is larger than `n`.
"""
mutable struct DqgmresWorkspace{T,FC,S} <: KrylovWorkspace{T,FC,S}
  m          :: Int
  n          :: Int
  Δx         :: S
  x          :: S
  t          :: S
  z          :: S
  w          :: S
  P          :: Vector{S}
  V          :: Vector{S}
  c          :: Vector{T}
  s          :: Vector{FC}
  H          :: Vector{FC}
  warm_start :: Bool
  stats      :: SimpleStats{T}
end

function DqgmresWorkspace(kc::KrylovConstructor; memory::Integer = 20)
  S      = typeof(kc.vm)
  FC     = eltype(S)
  T      = real(FC)
  m      = length(kc.vm)
  n      = length(kc.vn)
  memory = min(m, memory)
  Δx     = similar(kc.vn_empty)
  x      = similar(kc.vn)
  t      = similar(kc.vn)
  z      = similar(kc.vn_empty)
  w      = similar(kc.vn_empty)
  P      = S[similar(kc.vn) for i = 1 : memory]
  V      = S[similar(kc.vn) for i = 1 : memory]
  c      = Vector{T}(undef, memory)
  s      = Vector{FC}(undef, memory)
  H      = Vector{FC}(undef, memory+1)
  stats  = SimpleStats(0, false, false, false, 0, zero(T), zero(T), T[], T[], T[], 0.0, "unknown")
  workspace = DqgmresWorkspace{T,FC,S}(m, n, Δx, x, t, z, w, P, V, c, s, H, false, stats)
  return workspace
end

function DqgmresWorkspace(m::Integer, n::Integer, S::Type; memory::Integer = 20)
  memory = min(m, memory)
  FC = eltype(S)
  T  = real(FC)
  Δx = S(undef, 0)
  x  = S(undef, n)
  t  = S(undef, n)
  z  = S(undef, 0)
  w  = S(undef, 0)
  P  = S[S(undef, n) for i = 1 : memory]
  V  = S[S(undef, n) for i = 1 : memory]
  c  = Vector{T}(undef, memory)
  s  = Vector{FC}(undef, memory)
  H  = Vector{FC}(undef, memory+1)
  S = isconcretetype(S) ? S : typeof(x)
  stats = SimpleStats(0, false, false, false, 0, zero(T), zero(T), T[], T[], T[], 0.0, "unknown")
  workspace = DqgmresWorkspace{T,FC,S}(m, n, Δx, x, t, z, w, P, V, c, s, H, false, stats)
  return workspace
end

function DqgmresWorkspace(A, b; memory::Integer = 20)
  m, n = size(A)
  S = ktypeof(b)
  DqgmresWorkspace(m, n, S; memory)
end

"""
Workspace for the in-place method [`diom!`](@ref).

The following outer constructors can be used to initialize this workspace:

    workspace = DiomWorkspace(m, n, S; memory = 20)
    workspace = DiomWorkspace(A, b; memory = 20)
    workspace = DiomWorkspace(kc::KrylovConstructor; memory = 20)

`memory` is set to `n` if the value given is larger than `n`.
"""
mutable struct DiomWorkspace{T,FC,S} <: KrylovWorkspace{T,FC,S}
  m          :: Int
  n          :: Int
  Δx         :: S
  x          :: S
  t          :: S
  z          :: S
  w          :: S
  P          :: Vector{S}
  V          :: Vector{S}
  L          :: Vector{FC}
  H          :: Vector{FC}
  warm_start :: Bool
  stats      :: SimpleStats{T}
end

function DiomWorkspace(kc::KrylovConstructor; memory::Integer = 20)
  S      = typeof(kc.vm)
  FC     = eltype(S)
  T      = real(FC)
  m      = length(kc.vm)
  n      = length(kc.vn)
  memory = min(m, memory)
  Δx     = similar(kc.vn_empty)
  x      = similar(kc.vn)
  t      = similar(kc.vn)
  z      = similar(kc.vn_empty)
  w      = similar(kc.vn_empty)
  P      = S[similar(kc.vn) for i = 1 : memory-1]
  V      = S[similar(kc.vn) for i = 1 : memory]
  L      = Vector{FC}(undef, memory-1)
  H      = Vector{FC}(undef, memory)
  stats  = SimpleStats(0, false, false, false, 0, zero(T), zero(T), T[], T[], T[], 0.0, "unknown")
  workspace = DiomWorkspace{T,FC,S}(m, n, Δx, x, t, z, w, P, V, L, H, false, stats)
  return workspace
end

function DiomWorkspace(m::Integer, n::Integer, S::Type; memory::Integer = 20)
  memory = min(m, memory)
  FC  = eltype(S)
  T   = real(FC)
  Δx = S(undef, 0)
  x  = S(undef, n)
  t  = S(undef, n)
  z  = S(undef, 0)
  w  = S(undef, 0)
  P  = S[S(undef, n) for i = 1 : memory-1]
  V  = S[S(undef, n) for i = 1 : memory]
  L  = Vector{FC}(undef, memory-1)
  H  = Vector{FC}(undef, memory)
  S = isconcretetype(S) ? S : typeof(x)
  stats = SimpleStats(0, false, false, false, 0, zero(T), zero(T), T[], T[], T[], 0.0, "unknown")
  workspace = DiomWorkspace{T,FC,S}(m, n, Δx, x, t, z, w, P, V, L, H, false, stats)
  return workspace
end

function DiomWorkspace(A, b; memory::Integer = 20)
  m, n = size(A)
  S = ktypeof(b)
  DiomWorkspace(m, n, S; memory)
end

"""
Workspace for the in-place method [`usymlq!`](@ref).

The following outer constructors can be used to initialize this workspace:

    workspace = UsymlqWorkspace(m, n, S)
    workspace = UsymlqWorkspace(A, b)
    workspace = UsymlqWorkspace(kc::KrylovConstructor)
"""
mutable struct UsymlqWorkspace{T,FC,S} <: KrylovWorkspace{T,FC,S}
  m          :: Int
  n          :: Int
  uₖ₋₁       :: S
  uₖ         :: S
  p          :: S
  Δx         :: S
  x          :: S
  d̅          :: S
  vₖ₋₁       :: S
  vₖ         :: S
  q          :: S
  warm_start :: Bool
  stats      :: SimpleStats{T}
end

function UsymlqWorkspace(kc::KrylovConstructor)
  S    = typeof(kc.vm)
  FC   = eltype(S)
  T    = real(FC)
  m    = length(kc.vm)
  n    = length(kc.vn)
  uₖ₋₁ = similar(kc.vn)
  uₖ   = similar(kc.vn)
  p    = similar(kc.vn)
  Δx   = similar(kc.vn_empty)
  x    = similar(kc.vn)
  d̅    = similar(kc.vn)
  vₖ₋₁ = similar(kc.vm)
  vₖ   = similar(kc.vm)
  q    = similar(kc.vm)
  stats = SimpleStats(0, false, false, false, 0, zero(T), zero(T), T[], T[], T[], 0.0, "unknown")
  workspace = UsymlqWorkspace{T,FC,S}(m, n, uₖ₋₁, uₖ, p, Δx, x, d̅, vₖ₋₁, vₖ, q, false, stats)
  return workspace
end

function UsymlqWorkspace(m::Integer, n::Integer, S::Type)
  FC   = eltype(S)
  T    = real(FC)
  uₖ₋₁ = S(undef, n)
  uₖ   = S(undef, n)
  p    = S(undef, n)
  Δx   = S(undef, 0)
  x    = S(undef, n)
  d̅    = S(undef, n)
  vₖ₋₁ = S(undef, m)
  vₖ   = S(undef, m)
  q    = S(undef, m)
  S = isconcretetype(S) ? S : typeof(x)
  stats = SimpleStats(0, false, false, false, 0, zero(T), zero(T), T[], T[], T[], 0.0, "unknown")
  workspace = UsymlqWorkspace{T,FC,S}(m, n, uₖ₋₁, uₖ, p, Δx, x, d̅, vₖ₋₁, vₖ, q, false, stats)
  return workspace
end

function UsymlqWorkspace(A, b)
  m, n = size(A)
  S = ktypeof(b)
  UsymlqWorkspace(m, n, S)
end

"""
Workspace for the in-place method [`usymqr!`](@ref).

The following outer constructors can be used to initialize this workspace:

    workspace = UsymqrWorkspace(m, n, S)
    workspace = UsymqrWorkspace(A, b)
    workspace = UsymqrWorkspace(kc::KrylovConstructor)
"""
mutable struct UsymqrWorkspace{T,FC,S} <: KrylovWorkspace{T,FC,S}
  m          :: Int
  n          :: Int
  vₖ₋₁       :: S
  vₖ         :: S
  q          :: S
  Δx         :: S
  x          :: S
  wₖ₋₂       :: S
  wₖ₋₁       :: S
  uₖ₋₁       :: S
  uₖ         :: S
  p          :: S
  warm_start :: Bool
  stats      :: SimpleStats{T}
end

function UsymqrWorkspace(kc::KrylovConstructor)
  S    = typeof(kc.vm)
  FC   = eltype(S)
  T    = real(FC)
  m    = length(kc.vm)
  n    = length(kc.vn)
  vₖ₋₁ = similar(kc.vm)
  vₖ   = similar(kc.vm)
  q    = similar(kc.vm)
  Δx   = similar(kc.vn_empty)
  x    = similar(kc.vn)
  wₖ₋₂ = similar(kc.vn)
  wₖ₋₁ = similar(kc.vn)
  uₖ₋₁ = similar(kc.vn)
  uₖ   = similar(kc.vn)
  p    = similar(kc.vn)
  stats = SimpleStats(0, false, false, false, 0, zero(T), zero(T), T[], T[], T[], 0.0, "unknown")
  workspace = UsymqrWorkspace{T,FC,S}(m, n, vₖ₋₁, vₖ, q, Δx, x, wₖ₋₂, wₖ₋₁, uₖ₋₁, uₖ, p, false, stats)
  return workspace
end

function UsymqrWorkspace(m::Integer, n::Integer, S::Type)
  FC   = eltype(S)
  T    = real(FC)
  vₖ₋₁ = S(undef, m)
  vₖ   = S(undef, m)
  q    = S(undef, m)
  Δx   = S(undef, 0)
  x    = S(undef, n)
  wₖ₋₂ = S(undef, n)
  wₖ₋₁ = S(undef, n)
  uₖ₋₁ = S(undef, n)
  uₖ   = S(undef, n)
  p    = S(undef, n)
  S = isconcretetype(S) ? S : typeof(x)
  stats = SimpleStats(0, false, false, false, 0, zero(T), zero(T), T[], T[], T[], 0.0, "unknown")
  workspace = UsymqrWorkspace{T,FC,S}(m, n, vₖ₋₁, vₖ, q, Δx, x, wₖ₋₂, wₖ₋₁, uₖ₋₁, uₖ, p, false, stats)
  return workspace
end

function UsymqrWorkspace(A, b)
  m, n = size(A)
  S = ktypeof(b)
  UsymqrWorkspace(m, n, S)
end

"""
Workspace for the in-place method [`tricg!`](@ref).

The following outer constructors can be used to initialize this workspace:

    workspace = TricgWorkspace(m, n, S)
    workspace = TricgWorkspace(A, b)
    workspace = TricgWorkspace(kc::KrylovConstructor)
"""
mutable struct TricgWorkspace{T,FC,S} <: KrylovWorkspace{T,FC,S}
  m          :: Int
  n          :: Int
  y          :: S
  N⁻¹uₖ₋₁    :: S
  N⁻¹uₖ      :: S
  p          :: S
  gy₂ₖ₋₁     :: S
  gy₂ₖ       :: S
  x          :: S
  M⁻¹vₖ₋₁    :: S
  M⁻¹vₖ      :: S
  q          :: S
  gx₂ₖ₋₁     :: S
  gx₂ₖ       :: S
  Δx         :: S
  Δy         :: S
  uₖ         :: S
  vₖ         :: S
  warm_start :: Bool
  stats      :: SimpleStats{T}
end

function TricgWorkspace(kc::KrylovConstructor)
  S       = typeof(kc.vm)
  FC      = eltype(S)
  T       = real(FC)
  m       = length(kc.vm)
  n       = length(kc.vn)
  y       = similar(kc.vn)
  N⁻¹uₖ₋₁ = similar(kc.vn)
  N⁻¹uₖ   = similar(kc.vn)
  p       = similar(kc.vn)
  gy₂ₖ₋₁  = similar(kc.vn)
  gy₂ₖ    = similar(kc.vn)
  x       = similar(kc.vm)
  M⁻¹vₖ₋₁ = similar(kc.vm)
  M⁻¹vₖ   = similar(kc.vm)
  q       = similar(kc.vm)
  gx₂ₖ₋₁  = similar(kc.vm)
  gx₂ₖ    = similar(kc.vm)
  Δx      = similar(kc.vm_empty)
  Δy      = similar(kc.vn_empty)
  uₖ      = similar(kc.vn_empty)
  vₖ      = similar(kc.vm_empty)
  stats = SimpleStats(0, false, false, false, 0, zero(T), zero(T), T[], T[], T[], 0.0, "unknown")
  workspace = TricgWorkspace{T,FC,S}(m, n, y, N⁻¹uₖ₋₁, N⁻¹uₖ, p, gy₂ₖ₋₁, gy₂ₖ, x, M⁻¹vₖ₋₁, M⁻¹vₖ, q, gx₂ₖ₋₁, gx₂ₖ, Δx, Δy, uₖ, vₖ, false, stats)
  return workspace
end

function TricgWorkspace(m::Integer, n::Integer, S::Type)
  FC      = eltype(S)
  T       = real(FC)
  y       = S(undef, n)
  N⁻¹uₖ₋₁ = S(undef, n)
  N⁻¹uₖ   = S(undef, n)
  p       = S(undef, n)
  gy₂ₖ₋₁  = S(undef, n)
  gy₂ₖ    = S(undef, n)
  x       = S(undef, m)
  M⁻¹vₖ₋₁ = S(undef, m)
  M⁻¹vₖ   = S(undef, m)
  q       = S(undef, m)
  gx₂ₖ₋₁  = S(undef, m)
  gx₂ₖ    = S(undef, m)
  Δx      = S(undef, 0)
  Δy      = S(undef, 0)
  uₖ      = S(undef, 0)
  vₖ      = S(undef, 0)
  S = isconcretetype(S) ? S : typeof(x)
  stats = SimpleStats(0, false, false, false, 0, zero(T), zero(T), T[], T[], T[], 0.0, "unknown")
  workspace = TricgWorkspace{T,FC,S}(m, n, y, N⁻¹uₖ₋₁, N⁻¹uₖ, p, gy₂ₖ₋₁, gy₂ₖ, x, M⁻¹vₖ₋₁, M⁻¹vₖ, q, gx₂ₖ₋₁, gx₂ₖ, Δx, Δy, uₖ, vₖ, false, stats)
  return workspace
end

function TricgWorkspace(A, b)
  m, n = size(A)
  S = ktypeof(b)
  TricgWorkspace(m, n, S)
end

"""
Workspace for the in-place method [`trimr!`](@ref).

The following outer constructors can be used to initialize this workspace:

    workspace = TrimrWorkspace(m, n, S)
    workspace = TrimrWorkspace(A, b)
    workspace = TrimrWorkspace(kc::KrylovConstructor)
"""
mutable struct TrimrWorkspace{T,FC,S} <: KrylovWorkspace{T,FC,S}
  m          :: Int
  n          :: Int
  y          :: S
  N⁻¹uₖ₋₁    :: S
  N⁻¹uₖ      :: S
  p          :: S
  gy₂ₖ₋₃     :: S
  gy₂ₖ₋₂     :: S
  gy₂ₖ₋₁     :: S
  gy₂ₖ       :: S
  x          :: S
  M⁻¹vₖ₋₁    :: S
  M⁻¹vₖ      :: S
  q          :: S
  gx₂ₖ₋₃     :: S
  gx₂ₖ₋₂     :: S
  gx₂ₖ₋₁     :: S
  gx₂ₖ       :: S
  Δx         :: S
  Δy         :: S
  uₖ         :: S
  vₖ         :: S
  warm_start :: Bool
  stats      :: SimpleStats{T}
end

function TrimrWorkspace(kc::KrylovConstructor)
  S       = typeof(kc.vm)
  FC      = eltype(S)
  T       = real(FC)
  m       = length(kc.vm)
  n       = length(kc.vn)
  y       = similar(kc.vn)
  N⁻¹uₖ₋₁ = similar(kc.vn)
  N⁻¹uₖ   = similar(kc.vn)
  p       = similar(kc.vn)
  gy₂ₖ₋₃  = similar(kc.vn)
  gy₂ₖ₋₂  = similar(kc.vn)
  gy₂ₖ₋₁  = similar(kc.vn)
  gy₂ₖ    = similar(kc.vn)
  x       = similar(kc.vm)
  M⁻¹vₖ₋₁ = similar(kc.vm)
  M⁻¹vₖ   = similar(kc.vm)
  q       = similar(kc.vm)
  gx₂ₖ₋₃  = similar(kc.vm)
  gx₂ₖ₋₂  = similar(kc.vm)
  gx₂ₖ₋₁  = similar(kc.vm)
  gx₂ₖ    = similar(kc.vm)
  Δx      = similar(kc.vm_empty)
  Δy      = similar(kc.vn_empty)
  uₖ      = similar(kc.vn_empty)
  vₖ      = similar(kc.vm_empty)
  stats = SimpleStats(0, false, false, false, 0, zero(T), zero(T), T[], T[], T[], 0.0, "unknown")
  workspace = TrimrWorkspace{T,FC,S}(m, n, y, N⁻¹uₖ₋₁, N⁻¹uₖ, p, gy₂ₖ₋₃, gy₂ₖ₋₂, gy₂ₖ₋₁, gy₂ₖ, x, M⁻¹vₖ₋₁, M⁻¹vₖ, q, gx₂ₖ₋₃, gx₂ₖ₋₂, gx₂ₖ₋₁, gx₂ₖ, Δx, Δy, uₖ, vₖ, false, stats)
  return workspace
end

function TrimrWorkspace(m::Integer, n::Integer, S::Type)
  FC      = eltype(S)
  T       = real(FC)
  y       = S(undef, n)
  N⁻¹uₖ₋₁ = S(undef, n)
  N⁻¹uₖ   = S(undef, n)
  p       = S(undef, n)
  gy₂ₖ₋₃  = S(undef, n)
  gy₂ₖ₋₂  = S(undef, n)
  gy₂ₖ₋₁  = S(undef, n)
  gy₂ₖ    = S(undef, n)
  x       = S(undef, m)
  M⁻¹vₖ₋₁ = S(undef, m)
  M⁻¹vₖ   = S(undef, m)
  q       = S(undef, m)
  gx₂ₖ₋₃  = S(undef, m)
  gx₂ₖ₋₂  = S(undef, m)
  gx₂ₖ₋₁  = S(undef, m)
  gx₂ₖ    = S(undef, m)
  Δx      = S(undef, 0)
  Δy      = S(undef, 0)
  uₖ      = S(undef, 0)
  vₖ      = S(undef, 0)
  S = isconcretetype(S) ? S : typeof(x)
  stats = SimpleStats(0, false, false, false, 0, zero(T), zero(T), T[], T[], T[], 0.0, "unknown")
  workspace = TrimrWorkspace{T,FC,S}(m, n, y, N⁻¹uₖ₋₁, N⁻¹uₖ, p, gy₂ₖ₋₃, gy₂ₖ₋₂, gy₂ₖ₋₁, gy₂ₖ, x, M⁻¹vₖ₋₁, M⁻¹vₖ, q, gx₂ₖ₋₃, gx₂ₖ₋₂, gx₂ₖ₋₁, gx₂ₖ, Δx, Δy, uₖ, vₖ, false, stats)
  return workspace
end

function TrimrWorkspace(A, b)
  m, n = size(A)
  S = ktypeof(b)
  TrimrWorkspace(m, n, S)
end

"""
Workspace for the in-place method [`trilqr!`](@ref).

The following outer constructors can be used to initialize this workspace:

    workspace = TrilqrWorkspace(m, n, S)
    workspace = TrilqrWorkspace(A, b)
    workspace = TrilqrWorkspace(kc::KrylovConstructor)
"""
mutable struct TrilqrWorkspace{T,FC,S} <: KrylovWorkspace{T,FC,S}
  m          :: Int
  n          :: Int
  uₖ₋₁       :: S
  uₖ         :: S
  p          :: S
  d̅          :: S
  Δx         :: S
  x          :: S
  vₖ₋₁       :: S
  vₖ         :: S
  q          :: S
  Δy         :: S
  y          :: S
  wₖ₋₃       :: S
  wₖ₋₂       :: S
  warm_start :: Bool
  stats      :: AdjointStats{T}
end

function TrilqrWorkspace(kc::KrylovConstructor)
  S    = typeof(kc.vm)
  FC   = eltype(S)
  T    = real(FC)
  m    = length(kc.vm)
  n    = length(kc.vn)
  uₖ₋₁ = similar(kc.vn)
  uₖ   = similar(kc.vn)
  p    = similar(kc.vn)
  d̅    = similar(kc.vn)
  Δx   = similar(kc.vn_empty)
  x    = similar(kc.vn)
  vₖ₋₁ = similar(kc.vm)
  vₖ   = similar(kc.vm)
  q    = similar(kc.vm)
  Δy   = similar(kc.vm_empty)
  y    = similar(kc.vm)
  wₖ₋₃ = similar(kc.vm)
  wₖ₋₂ = similar(kc.vm)
  stats = AdjointStats(0, false, false, T[], T[], 0.0, "unknown")
  workspace = TrilqrWorkspace{T,FC,S}(m, n, uₖ₋₁, uₖ, p, d̅, Δx, x, vₖ₋₁, vₖ, q, Δy, y, wₖ₋₃, wₖ₋₂, false, stats)
  return workspace
end

function TrilqrWorkspace(m::Integer, n::Integer, S::Type)
  FC   = eltype(S)
  T    = real(FC)
  uₖ₋₁ = S(undef, n)
  uₖ   = S(undef, n)
  p    = S(undef, n)
  d̅    = S(undef, n)
  Δx   = S(undef, 0)
  x    = S(undef, n)
  vₖ₋₁ = S(undef, m)
  vₖ   = S(undef, m)
  q    = S(undef, m)
  Δy   = S(undef, 0)
  y    = S(undef, m)
  wₖ₋₃ = S(undef, m)
  wₖ₋₂ = S(undef, m)
  S = isconcretetype(S) ? S : typeof(x)
  stats = AdjointStats(0, false, false, T[], T[], 0.0, "unknown")
  workspace = TrilqrWorkspace{T,FC,S}(m, n, uₖ₋₁, uₖ, p, d̅, Δx, x, vₖ₋₁, vₖ, q, Δy, y, wₖ₋₃, wₖ₋₂, false, stats)
  return workspace
end

function TrilqrWorkspace(A, b)
  m, n = size(A)
  S = ktypeof(b)
  TrilqrWorkspace(m, n, S)
end

"""
Workspace for the in-place method [`cgs!`](@ref).

The following outer constructors can be used to initialize this workspace:s

    workspace = CgsWorkspace(m, n, S)
    workspace = CgsWorkspace(A, b)
    workspace = CgsWorkspace(kc::KrylovConstructor)
"""
mutable struct CgsWorkspace{T,FC,S} <: KrylovWorkspace{T,FC,S}
  m          :: Int
  n          :: Int
  Δx         :: S
  x          :: S
  r          :: S
  u          :: S
  p          :: S
  q          :: S
  ts         :: S
  yz         :: S
  vw         :: S
  warm_start :: Bool
  stats      :: SimpleStats{T}
end

function CgsWorkspace(kc::KrylovConstructor)
  S  = typeof(kc.vm)
  FC = eltype(S)
  T  = real(FC)
  m  = length(kc.vm)
  n  = length(kc.vn)
  Δx = similar(kc.vn_empty)
  x  = similar(kc.vn)
  r  = similar(kc.vn)
  u  = similar(kc.vn)
  p  = similar(kc.vn)
  q  = similar(kc.vn)
  ts = similar(kc.vn)
  yz = similar(kc.vn_empty)
  vw = similar(kc.vn_empty)
  stats = SimpleStats(0, false, false, false, 0, zero(T), zero(T), T[], T[], T[], 0.0, "unknown")
  workspace = CgsWorkspace{T,FC,S}(m, n, Δx, x, r, u, p, q, ts, yz, vw, false, stats)
  return workspace
end

function CgsWorkspace(m::Integer, n::Integer, S::Type)
  FC = eltype(S)
  T  = real(FC)
  Δx = S(undef, 0)
  x  = S(undef, n)
  r  = S(undef, n)
  u  = S(undef, n)
  p  = S(undef, n)
  q  = S(undef, n)
  ts = S(undef, n)
  yz = S(undef, 0)
  vw = S(undef, 0)
  S = isconcretetype(S) ? S : typeof(x)
  stats = SimpleStats(0, false, false, false, 0, zero(T), zero(T), T[], T[], T[], 0.0, "unknown")
  workspace = CgsWorkspace{T,FC,S}(m, n, Δx, x, r, u, p, q, ts, yz, vw, false, stats)
  return workspace
end

function CgsWorkspace(A, b)
  m, n = size(A)
  S = ktypeof(b)
  CgsWorkspace(m, n, S)
end

"""
Workspace for the in-place method [`bicgstab!`](@ref).

The following outer constructors can be used to initialize this workspace:

    workspace = BicgstabWorkspace(m, n, S)
    workspace = BicgstabWorkspace(A, b)
    workspace = BicgstabWorkspace(kc::KrylovConstructor)
"""
mutable struct BicgstabWorkspace{T,FC,S} <: KrylovWorkspace{T,FC,S}
  m          :: Int
  n          :: Int
  Δx         :: S
  x          :: S
  r          :: S
  p          :: S
  v          :: S
  s          :: S
  qd         :: S
  yz         :: S
  t          :: S
  warm_start :: Bool
  stats      :: SimpleStats{T}
end

function BicgstabWorkspace(kc::KrylovConstructor)
  S  = typeof(kc.vm)
  FC = eltype(S)
  T  = real(FC)
  m  = length(kc.vm)
  n  = length(kc.vn)
  Δx = similar(kc.vn_empty)
  x  = similar(kc.vn)
  r  = similar(kc.vn)
  p  = similar(kc.vn)
  v  = similar(kc.vn)
  s  = similar(kc.vn)
  qd = similar(kc.vn)
  yz = similar(kc.vn_empty)
  t  = similar(kc.vn_empty)
  stats = SimpleStats(0, false, false, false, 0, zero(T), zero(T), T[], T[], T[], 0.0, "unknown")
  workspace = BicgstabWorkspace{T,FC,S}(m, n, Δx, x, r, p, v, s, qd, yz, t, false, stats)
  return workspace
end

function BicgstabWorkspace(m::Integer, n::Integer, S::Type)
  FC = eltype(S)
  T  = real(FC)
  Δx = S(undef, 0)
  x  = S(undef, n)
  r  = S(undef, n)
  p  = S(undef, n)
  v  = S(undef, n)
  s  = S(undef, n)
  qd = S(undef, n)
  yz = S(undef, 0)
  t  = S(undef, 0)
  S = isconcretetype(S) ? S : typeof(x)
  stats = SimpleStats(0, false, false, false, 0, zero(T), zero(T), T[], T[], T[], 0.0, "unknown")
  workspace = BicgstabWorkspace{T,FC,S}(m, n, Δx, x, r, p, v, s, qd, yz, t, false, stats)
  return workspace
end

function BicgstabWorkspace(A, b)
  m, n = size(A)
  S = ktypeof(b)
  BicgstabWorkspace(m, n, S)
end

"""
Workspace for the in-place method [`bilq!`](@ref).

The following outer constructors can be used to initialize this workspace:

    workspace = BilqWorkspace(m, n, S)
    workspace = BilqWorkspace(A, b)
    workspace = BilqWorkspace(kc::KrylovConstructor)
"""
mutable struct BilqWorkspace{T,FC,S} <: KrylovWorkspace{T,FC,S}
  m          :: Int
  n          :: Int
  uₖ₋₁       :: S
  uₖ         :: S
  q          :: S
  vₖ₋₁       :: S
  vₖ         :: S
  p          :: S
  Δx         :: S
  x          :: S
  d̅          :: S
  t          :: S
  s          :: S
  warm_start :: Bool
  stats      :: SimpleStats{T}
end

function BilqWorkspace(kc::KrylovConstructor)
  S    = typeof(kc.vm)
  FC   = eltype(S)
  T    = real(FC)
  m    = length(kc.vm)
  n    = length(kc.vn)
  uₖ₋₁ = similar(kc.vn)
  uₖ   = similar(kc.vn)
  q    = similar(kc.vn)
  vₖ₋₁ = similar(kc.vn)
  vₖ   = similar(kc.vn)
  p    = similar(kc.vn)
  Δx   = similar(kc.vn_empty)
  x    = similar(kc.vn)
  d̅    = similar(kc.vn)
  t    = similar(kc.vn_empty)
  s    = similar(kc.vn_empty)
  stats = SimpleStats(0, false, false, false, 0, zero(T), zero(T), T[], T[], T[], 0.0, "unknown")
  workspace = BilqWorkspace{T,FC,S}(m, n, uₖ₋₁, uₖ, q, vₖ₋₁, vₖ, p, Δx, x, d̅, t, s, false, stats)
  return workspace
end

function BilqWorkspace(m::Integer, n::Integer, S::Type)
  FC   = eltype(S)
  T    = real(FC)
  uₖ₋₁ = S(undef, n)
  uₖ   = S(undef, n)
  q    = S(undef, n)
  vₖ₋₁ = S(undef, n)
  vₖ   = S(undef, n)
  p    = S(undef, n)
  Δx   = S(undef, 0)
  x    = S(undef, n)
  d̅    = S(undef, n)
  t    = S(undef, 0)
  s    = S(undef, 0)
  S = isconcretetype(S) ? S : typeof(x)
  stats = SimpleStats(0, false, false, false, 0, zero(T), zero(T), T[], T[], T[], 0.0, "unknown")
  workspace = BilqWorkspace{T,FC,S}(m, n, uₖ₋₁, uₖ, q, vₖ₋₁, vₖ, p, Δx, x, d̅, t, s, false, stats)
  return workspace
end

function BilqWorkspace(A, b)
  m, n = size(A)
  S = ktypeof(b)
  BilqWorkspace(m, n, S)
end

"""
Workspace for the in-place method [`qmr!`](@ref).

The following outer constructors can be used to initialize this workspace:

    workspace = QmrWorkspace(m, n, S)
    workspace = QmrWorkspace(A, b)
    workspace = QmrWorkspace(kc::KrylovConstructor)
"""
mutable struct QmrWorkspace{T,FC,S} <: KrylovWorkspace{T,FC,S}
  m          :: Int
  n          :: Int
  uₖ₋₁       :: S
  uₖ         :: S
  q          :: S
  vₖ₋₁       :: S
  vₖ         :: S
  p          :: S
  Δx         :: S
  x          :: S
  wₖ₋₂       :: S
  wₖ₋₁       :: S
  t          :: S
  s          :: S
  warm_start :: Bool
  stats      :: SimpleStats{T}
end

function QmrWorkspace(kc::KrylovConstructor)
  S    = typeof(kc.vm)
  FC   = eltype(S)
  T    = real(FC)
  m    = length(kc.vm)
  n    = length(kc.vn)
  uₖ₋₁ = similar(kc.vn)
  uₖ   = similar(kc.vn)
  q    = similar(kc.vn)
  vₖ₋₁ = similar(kc.vn)
  vₖ   = similar(kc.vn)
  p    = similar(kc.vn)
  Δx   = similar(kc.vn_empty)
  x    = similar(kc.vn)
  wₖ₋₂ = similar(kc.vn)
  wₖ₋₁ = similar(kc.vn)
  t    = similar(kc.vn_empty)
  s    = similar(kc.vn_empty)
  stats = SimpleStats(0, false, false, false, 0, zero(T), zero(T), T[], T[], T[], 0.0, "unknown")
  workspace = QmrWorkspace{T,FC,S}(m, n, uₖ₋₁, uₖ, q, vₖ₋₁, vₖ, p, Δx, x, wₖ₋₂, wₖ₋₁, t, s, false, stats)
  return workspace
end

function QmrWorkspace(m::Integer, n::Integer, S::Type)
  FC   = eltype(S)
  T    = real(FC)
  uₖ₋₁ = S(undef, n)
  uₖ   = S(undef, n)
  q    = S(undef, n)
  vₖ₋₁ = S(undef, n)
  vₖ   = S(undef, n)
  p    = S(undef, n)
  Δx   = S(undef, 0)
  x    = S(undef, n)
  wₖ₋₂ = S(undef, n)
  wₖ₋₁ = S(undef, n)
  t    = S(undef, 0)
  s    = S(undef, 0)
  S = isconcretetype(S) ? S : typeof(x)
  stats = SimpleStats(0, false, false, false, 0, zero(T), zero(T), T[], T[], T[], 0.0, "unknown")
  workspace = QmrWorkspace{T,FC,S}(m, n, uₖ₋₁, uₖ, q, vₖ₋₁, vₖ, p, Δx, x, wₖ₋₂, wₖ₋₁, t, s, false, stats)
  return workspace
end

function QmrWorkspace(A, b)
  m, n = size(A)
  S = ktypeof(b)
  QmrWorkspace(m, n, S)
end

"""
Workspace for the in-place method [`bilqr!`](@ref).

The following outer constructors can be used to initialize this workspace:

    workspace = BilqrWorkspace(m, n, S)
    workspace = BilqrWorkspace(A, b)
    workspace = BilqrWorkspace(kc::KrylovConstructor)
"""
mutable struct BilqrWorkspace{T,FC,S} <: KrylovWorkspace{T,FC,S}
  m          :: Int
  n          :: Int
  uₖ₋₁       :: S
  uₖ         :: S
  q          :: S
  vₖ₋₁       :: S
  vₖ         :: S
  p          :: S
  Δx         :: S
  x          :: S
  Δy         :: S
  y          :: S
  d̅          :: S
  wₖ₋₃       :: S
  wₖ₋₂       :: S
  warm_start :: Bool
  stats      :: AdjointStats{T}
end

function BilqrWorkspace(kc::KrylovConstructor)
  S    = typeof(kc.vm)
  FC   = eltype(S)
  T    = real(FC)
  m    = length(kc.vm)
  n    = length(kc.vn)
  uₖ₋₁ = similar(kc.vn)
  uₖ   = similar(kc.vn)
  q    = similar(kc.vn)
  vₖ₋₁ = similar(kc.vn)
  vₖ   = similar(kc.vn)
  p    = similar(kc.vn)
  Δx   = similar(kc.vn_empty)
  x    = similar(kc.vn)
  Δy   = similar(kc.vn_empty)
  y    = similar(kc.vn)
  d̅    = similar(kc.vn)
  wₖ₋₃ = similar(kc.vn)
  wₖ₋₂ = similar(kc.vn)
  stats = AdjointStats(0, false, false, T[], T[], 0.0, "unknown")
  workspace = BilqrWorkspace{T,FC,S}(m, n, uₖ₋₁, uₖ, q, vₖ₋₁, vₖ, p, Δx, x, Δy, y, d̅, wₖ₋₃, wₖ₋₂, false, stats)
  return workspace
end

function BilqrWorkspace(m::Integer, n::Integer, S::Type)
  FC   = eltype(S)
  T    = real(FC)
  uₖ₋₁ = S(undef, n)
  uₖ   = S(undef, n)
  q    = S(undef, n)
  vₖ₋₁ = S(undef, n)
  vₖ   = S(undef, n)
  p    = S(undef, n)
  Δx   = S(undef, 0)
  x    = S(undef, n)
  Δy   = S(undef, 0)
  y    = S(undef, n)
  d̅    = S(undef, n)
  wₖ₋₃ = S(undef, n)
  wₖ₋₂ = S(undef, n)
  S = isconcretetype(S) ? S : typeof(x)
  stats = AdjointStats(0, false, false, T[], T[], 0.0, "unknown")
  workspace = BilqrWorkspace{T,FC,S}(m, n, uₖ₋₁, uₖ, q, vₖ₋₁, vₖ, p, Δx, x, Δy, y, d̅, wₖ₋₃, wₖ₋₂, false, stats)
  return workspace
end

function BilqrWorkspace(A, b)
  m, n = size(A)
  S = ktypeof(b)
  BilqrWorkspace(m, n, S)
end

"""
Workspace for the in-place method [`cgls!`](@ref).

The following outer constructors can be used to initialize this workspace:

    workspace = CglsWorkspace(m, n, S)
    workspace = CglsWorkspace(A, b)
    workspace = CglsWorkspace(kc::KrylovConstructor)
"""
mutable struct CglsWorkspace{T,FC,S} <: KrylovWorkspace{T,FC,S}
  m     :: Int
  n     :: Int
  x     :: S
  p     :: S
  s     :: S
  r     :: S
  q     :: S
  Mr    :: S
  stats :: SimpleStats{T}
end

function CglsWorkspace(kc::KrylovConstructor)
  S  = typeof(kc.vm)
  FC = eltype(S)
  T  = real(FC)
  m  = length(kc.vm)
  n  = length(kc.vn)
  x  = similar(kc.vn)
  p  = similar(kc.vn)
  s  = similar(kc.vn)
  r  = similar(kc.vm)
  q  = similar(kc.vm)
  Mr = similar(kc.vm_empty)
  stats = SimpleStats(0, false, false, false, 0, zero(T), zero(T), T[], T[], T[], 0.0, "unknown")
  workspace = CglsWorkspace{T,FC,S}(m, n, x, p, s, r, q, Mr, stats)
  return workspace
end

function CglsWorkspace(m::Integer, n::Integer, S::Type)
  FC = eltype(S)
  T  = real(FC)
  x  = S(undef, n)
  p  = S(undef, n)
  s  = S(undef, n)
  r  = S(undef, m)
  q  = S(undef, m)
  Mr = S(undef, 0)
  S = isconcretetype(S) ? S : typeof(x)
  stats = SimpleStats(0, false, false, false, 0, zero(T), zero(T), T[], T[], T[], 0.0, "unknown")
  workspace = CglsWorkspace{T,FC,S}(m, n, x, p, s, r, q, Mr, stats)
  return workspace
end

function CglsWorkspace(A, b)
  m, n = size(A)
  S = ktypeof(b)
  CglsWorkspace(m, n, S)
end

"""
Workspace for the in-place method [`cgls_lanczos_shift!`](@ref).

The following outer constructors can be used to initialize this workspace::

    workspace = CglsLanczosShiftWorkspace(m, n, nshifts, S)
    workspace = CglsLanczosShiftWorkspace(A, b, nshifts)
    workspace = CglsLanczosShiftWorkspace(kc::KrylovConstructor, nshifts)
"""
mutable struct CglsLanczosShiftWorkspace{T,FC,S} <: KrylovWorkspace{T,FC,S}
  m         :: Int
  n         :: Int
  nshifts   :: Int
  Mv        :: S
  u_prev    :: S
  u_next    :: S
  u         :: S
  v         :: S
  x         :: Vector{S}
  p         :: Vector{S}
  σ         :: Vector{T}
  δhat      :: Vector{T}
  ω         :: Vector{T}
  γ         :: Vector{T}
  rNorms    :: Vector{T}
  converged :: BitVector
  not_cv    :: BitVector
  stats     :: LanczosShiftStats{T}
end

function CglsLanczosShiftWorkspace(kc::KrylovConstructor, nshifts::Integer)
  S          = typeof(kc.vm)
  FC         = eltype(S)
  T          = real(FC)
  m          = length(kc.vm)
  n          = length(kc.vn)
  Mv         = similar(kc.vn)
  u_prev     = similar(kc.vm)
  u_next     = similar(kc.vm)
  u          = similar(kc.vm)
  v          = similar(kc.vn_empty)
  x          = S[similar(kc.vn) for i = 1 : nshifts]
  p          = S[similar(kc.vn) for i = 1 : nshifts]
  σ          = Vector{T}(undef, nshifts)
  δhat       = Vector{T}(undef, nshifts)
  ω          = Vector{T}(undef, nshifts)
  γ          = Vector{T}(undef, nshifts)
  rNorms     = Vector{T}(undef, nshifts)
  indefinite = BitVector(undef, nshifts)
  converged  = BitVector(undef, nshifts)
  not_cv     = BitVector(undef, nshifts)
  stats = LanczosShiftStats(0, false, Vector{T}[T[] for i = 1 : nshifts], indefinite, T(NaN), T(NaN), 0.0, "unknown")
  workspace = CglsLanczosShiftWorkspace{T,FC,S}(m, n, nshifts, Mv, u_prev, u_next, u, v, x, p, σ, δhat, ω, γ, rNorms, converged, not_cv, stats)
  return workspace
end

function CglsLanczosShiftWorkspace(m::Integer, n::Integer, nshifts::Integer, S::Type)
  FC         = eltype(S)
  T          = real(FC)
  Mv         = S(undef, n)
  u_prev     = S(undef, m)
  u_next     = S(undef, m)
  u          = S(undef, m)
  v          = S(undef, 0)
  x          = S[S(undef, n) for i = 1 : nshifts]
  p          = S[S(undef, n) for i = 1 : nshifts]
  σ          = Vector{T}(undef, nshifts)
  δhat       = Vector{T}(undef, nshifts)
  ω          = Vector{T}(undef, nshifts)
  γ          = Vector{T}(undef, nshifts)
  rNorms     = Vector{T}(undef, nshifts)
  indefinite = BitVector(undef, nshifts)
  converged  = BitVector(undef, nshifts)
  not_cv     = BitVector(undef, nshifts)
  S = isconcretetype(S) ? S : typeof(x)
  stats = LanczosShiftStats(0, false, Vector{T}[T[] for i = 1 : nshifts], indefinite, T(NaN), T(NaN), 0.0, "unknown")
  workspace = CglsLanczosShiftWorkspace{T,FC,S}(m, n, nshifts, Mv, u_prev, u_next, u, v, x, p, σ, δhat, ω, γ, rNorms, converged, not_cv, stats)
  return workspace
end

function CglsLanczosShiftWorkspace(A, b, nshifts::Integer)
  m, n = size(A)
  S = ktypeof(b)
  CglsLanczosShiftWorkspace(m, n, nshifts, S)
end

"""
Workspace for the in-place method [`crls!`](@ref).

The following outer constructors can be used to initialize this workspace:

    workspace = CrlsWorkspace(m, n, S)
    workspace = CrlsWorkspace(A, b)
    workspace = CrlsWorkspace(kc::KrylovConstructor)
"""
mutable struct CrlsWorkspace{T,FC,S} <: KrylovWorkspace{T,FC,S}
  m     :: Int
  n     :: Int
  x     :: S
  p     :: S
  Ar    :: S
  q     :: S
  r     :: S
  Ap    :: S
  s     :: S
  Ms    :: S
  stats :: SimpleStats{T}
end

function CrlsWorkspace(kc::KrylovConstructor)
  S  = typeof(kc.vm)
  FC = eltype(S)
  T  = real(FC)
  m  = length(kc.vm)
  n  = length(kc.vn)
  x  = similar(kc.vn)
  p  = similar(kc.vn)
  Ar = similar(kc.vn)
  q  = similar(kc.vn)
  r  = similar(kc.vm)
  Ap = similar(kc.vm)
  s  = similar(kc.vm)
  Ms = similar(kc.vm_empty)
  stats = SimpleStats(0, false, false, false, 0, zero(T), zero(T), T[], T[], T[], 0.0, "unknown")
  workspace = CrlsWorkspace{T,FC,S}(m, n, x, p, Ar, q, r, Ap, s, Ms, stats)
  return workspace
end

function CrlsWorkspace(m::Integer, n::Integer, S::Type)
  FC = eltype(S)
  T  = real(FC)
  x  = S(undef, n)
  p  = S(undef, n)
  Ar = S(undef, n)
  q  = S(undef, n)
  r  = S(undef, m)
  Ap = S(undef, m)
  s  = S(undef, m)
  Ms = S(undef, 0)
  S = isconcretetype(S) ? S : typeof(x)
  stats = SimpleStats(0, false, false, false, 0, zero(T), zero(T), T[], T[], T[], 0.0, "unknown")
  workspace = CrlsWorkspace{T,FC,S}(m, n, x, p, Ar, q, r, Ap, s, Ms, stats)
  return workspace
end

function CrlsWorkspace(A, b)
  m, n = size(A)
  S = ktypeof(b)
  CrlsWorkspace(m, n, S)
end

"""
Workspace for the in-place method [`cgne!`](@ref).

The following outer constructors can be used to initialize this workspace:

    workspace = CgneWorkspace(m, n, S)
    workspace = CgneWorkspace(A, b)
    workspace = CgneWorkspace(kc::KrylovConstructor)
"""
mutable struct CgneWorkspace{T,FC,S} <: KrylovWorkspace{T,FC,S}
  m     :: Int
  n     :: Int
  x     :: S
  p     :: S
  Aᴴz   :: S
  r     :: S
  q     :: S
  s     :: S
  z     :: S
  stats :: SimpleStats{T}
end

function CgneWorkspace(kc::KrylovConstructor)
  S   = typeof(kc.vm)
  FC  = eltype(S)
  T   = real(FC)
  m   = length(kc.vm)
  n   = length(kc.vn)
  x   = similar(kc.vn)
  p   = similar(kc.vn)
  Aᴴz = similar(kc.vn)
  r   = similar(kc.vm)
  q   = similar(kc.vm)
  s   = similar(kc.vm_empty)
  z   = similar(kc.vm_empty)
  stats = SimpleStats(0, false, false, false, 0, zero(T), zero(T), T[], T[], T[], 0.0, "unknown")
  workspace = CgneWorkspace{T,FC,S}(m, n, x, p, Aᴴz, r, q, s, z, stats)
  return workspace
end

function CgneWorkspace(m::Integer, n::Integer, S::Type)
  FC  = eltype(S)
  T   = real(FC)
  x   = S(undef, n)
  p   = S(undef, n)
  Aᴴz = S(undef, n)
  r   = S(undef, m)
  q   = S(undef, m)
  s   = S(undef, 0)
  z   = S(undef, 0)
  S = isconcretetype(S) ? S : typeof(x)
  stats = SimpleStats(0, false, false, false, 0, zero(T), zero(T), T[], T[], T[], 0.0, "unknown")
  workspace = CgneWorkspace{T,FC,S}(m, n, x, p, Aᴴz, r, q, s, z, stats)
  return workspace
end

function CgneWorkspace(A, b)
  m, n = size(A)
  S = ktypeof(b)
  CgneWorkspace(m, n, S)
end

"""
Workspace for the in-place method [`crmr!`](@ref).

The following outer constructors can be used to initialize this workspace:

    workspace = CrmrWorkspace(m, n, S)
    workspace = CrmrWorkspace(A, b)
    workspace = CrmrWorkspace(kc::KrylovConstructor)
"""
mutable struct CrmrWorkspace{T,FC,S} <: KrylovWorkspace{T,FC,S}
  m     :: Int
  n     :: Int
  x     :: S
  p     :: S
  Aᴴr   :: S
  r     :: S
  q     :: S
  Nq    :: S
  s     :: S
  stats :: SimpleStats{T}
end

function CrmrWorkspace(kc::KrylovConstructor)
  S   = typeof(kc.vm)
  FC  = eltype(S)
  T   = real(FC)
  m   = length(kc.vm)
  n   = length(kc.vn)
  x   = similar(kc.vn)
  p   = similar(kc.vn)
  Aᴴr = similar(kc.vn)
  r   = similar(kc.vm)
  q   = similar(kc.vm)
  Nq  = similar(kc.vm_empty)
  s   = similar(kc.vm_empty)
  stats = SimpleStats(0, false, false, false, 0, zero(T), zero(T), T[], T[], T[], 0.0, "unknown")
  workspace = CrmrWorkspace{T,FC,S}(m, n, x, p, Aᴴr, r, q, Nq, s, stats)
  return workspace
end

function CrmrWorkspace(m::Integer, n::Integer, S::Type)
  FC  = eltype(S)
  T   = real(FC)
  x   = S(undef, n)
  p   = S(undef, n)
  Aᴴr = S(undef, n)
  r   = S(undef, m)
  q   = S(undef, m)
  Nq  = S(undef, 0)
  s   = S(undef, 0)
  S = isconcretetype(S) ? S : typeof(x)
  stats = SimpleStats(0, false, false, false, 0, zero(T), zero(T), T[], T[], T[], 0.0, "unknown")
  workspace = CrmrWorkspace{T,FC,S}(m, n, x, p, Aᴴr, r, q, Nq, s, stats)
  return workspace
end

function CrmrWorkspace(A, b)
  m, n = size(A)
  S = ktypeof(b)
  CrmrWorkspace(m, n, S)
end

"""
Workspace for the in-place method [`lslq!`](@ref).

The following outer constructors can be used to initialize this workspace:

    workspace = LslqWorkspace(m, n, S)
    workspace = LslqWorkspace(A, b)
    workspace = LslqWorkspace(kc::KrylovConstructor)
"""
mutable struct LslqWorkspace{T,FC,S} <: KrylovWorkspace{T,FC,S}
  m       :: Int
  n       :: Int
  x       :: S
  Nv      :: S
  Aᴴu     :: S
  w̄       :: S
  Mu      :: S
  Av      :: S
  u       :: S
  v       :: S
  err_vec :: Vector{T}
  stats   :: LSLQStats{T}
end

function LslqWorkspace(kc::KrylovConstructor; window::Integer = 5)
  S   = typeof(kc.vm)
  FC  = eltype(S)
  T   = real(FC)
  m   = length(kc.vm)
  n   = length(kc.vn)
  x   = similar(kc.vn)
  Nv  = similar(kc.vn)
  Aᴴu = similar(kc.vn)
  w̄   = similar(kc.vn)
  Mu  = similar(kc.vm)
  Av  = similar(kc.vm)
  u   = similar(kc.vm_empty)
  v   = similar(kc.vn_empty)
  err_vec = zeros(T, window)
  stats = LSLQStats(0, false, false, T[], T[], T[], false, T[], T[], 0.0, "unknown")
  workspace = LslqWorkspace{T,FC,S}(m, n, x, Nv, Aᴴu, w̄, Mu, Av, u, v, err_vec, stats)
  return workspace
end

function LslqWorkspace(m::Integer, n::Integer, S::Type; window::Integer = 5)
  FC  = eltype(S)
  T   = real(FC)
  x   = S(undef, n)
  Nv  = S(undef, n)
  Aᴴu = S(undef, n)
  w̄   = S(undef, n)
  Mu  = S(undef, m)
  Av  = S(undef, m)
  u   = S(undef, 0)
  v   = S(undef, 0)
  err_vec = zeros(T, window)
  S = isconcretetype(S) ? S : typeof(x)
  stats = LSLQStats(0, false, false, T[], T[], T[], false, T[], T[], 0.0, "unknown")
  workspace = LslqWorkspace{T,FC,S}(m, n, x, Nv, Aᴴu, w̄, Mu, Av, u, v, err_vec, stats)
  return workspace
end

function LslqWorkspace(A, b; window::Integer = 5)
  m, n = size(A)
  S = ktypeof(b)
  LslqWorkspace(m, n, S; window)
end

"""
Workspace for the in-place method [`lsqr!`](@ref).

The following outer constructors can be used to initialize this workspace:

    workspace = LsqrWorkspace(m, n, S)
    workspace = LsqrWorkspace(A, b)
    workspace = LsqrWorkspace(kc::KrylovConstructor)
"""
mutable struct LsqrWorkspace{T,FC,S} <: KrylovWorkspace{T,FC,S}
  m       :: Int
  n       :: Int
  x       :: S
  Nv      :: S
  Aᴴu     :: S
  w       :: S
  Mu      :: S
  Av      :: S
  u       :: S
  v       :: S
  err_vec :: Vector{T}
  stats   :: SimpleStats{T}
end

function LsqrWorkspace(kc::KrylovConstructor; window::Integer = 5)
  S   = typeof(kc.vm)
  FC  = eltype(S)
  T   = real(FC)
  m   = length(kc.vm)
  n   = length(kc.vn)
  x   = similar(kc.vn)
  Nv  = similar(kc.vn)
  Aᴴu = similar(kc.vn)
  w   = similar(kc.vn)
  Mu  = similar(kc.vm)
  Av  = similar(kc.vm)
  u   = similar(kc.vm_empty)
  v   = similar(kc.vn_empty)
  err_vec = zeros(T, window)
  stats = SimpleStats(0, false, false, false, 0, zero(T), zero(T), T[], T[], T[], 0.0, "unknown")
  workspace = LsqrWorkspace{T,FC,S}(m, n, x, Nv, Aᴴu, w, Mu, Av, u, v, err_vec, stats)
  return workspace
end

function LsqrWorkspace(m::Integer, n::Integer, S::Type; window::Integer = 5)
  FC  = eltype(S)
  T   = real(FC)
  x   = S(undef, n)
  Nv  = S(undef, n)
  Aᴴu = S(undef, n)
  w   = S(undef, n)
  Mu  = S(undef, m)
  Av  = S(undef, m)
  u   = S(undef, 0)
  v   = S(undef, 0)
  err_vec = zeros(T, window)
  S = isconcretetype(S) ? S : typeof(x)
  stats = SimpleStats(0, false, false, false, 0, zero(T), zero(T), T[], T[], T[], 0.0, "unknown")
  workspace = LsqrWorkspace{T,FC,S}(m, n, x, Nv, Aᴴu, w, Mu, Av, u, v, err_vec, stats)
  return workspace
end

function LsqrWorkspace(A, b; window::Integer = 5)
  m, n = size(A)
  S = ktypeof(b)
  LsqrWorkspace(m, n, S; window)
end

"""
Workspace for the in-place method [`lsmr!`](@ref).

The following outer constructors can be used to initialize this workspace:

    workspace = LsmrWorkspace(m, n, S)
    workspace = LsmrWorkspace(A, b)
    workspace = LsmrWorkspace(kc::KrylovConstructor)
"""
mutable struct LsmrWorkspace{T,FC,S} <: KrylovWorkspace{T,FC,S}
  m       :: Int
  n       :: Int
  x       :: S
  Nv      :: S
  Aᴴu     :: S
  h       :: S
  hbar    :: S
  Mu      :: S
  Av      :: S
  u       :: S
  v       :: S
  err_vec :: Vector{T}
  stats   :: LsmrStats{T}
end

function LsmrWorkspace(kc::KrylovConstructor; window::Integer = 5)
  S    = typeof(kc.vm)
  FC   = eltype(S)
  T    = real(FC)
  m    = length(kc.vm)
  n    = length(kc.vn)
  x    = similar(kc.vn)
  Nv   = similar(kc.vn)
  Aᴴu  = similar(kc.vn)
  h    = similar(kc.vn)
  hbar = similar(kc.vn)
  Mu   = similar(kc.vm)
  Av   = similar(kc.vm)
  u    = similar(kc.vm_empty)
  v    = similar(kc.vn_empty)
  err_vec = zeros(T, window)
  stats = LsmrStats(0, false, false, T[], T[], zero(T), zero(T), zero(T), zero(T), zero(T), 0.0, "unknown")
  workspace = LsmrWorkspace{T,FC,S}(m, n, x, Nv, Aᴴu, h, hbar, Mu, Av, u, v, err_vec, stats)
  return workspace
end

function LsmrWorkspace(m::Integer, n::Integer, S::Type; window::Integer = 5)
  FC   = eltype(S)
  T    = real(FC)
  x    = S(undef, n)
  Nv   = S(undef, n)
  Aᴴu  = S(undef, n)
  h    = S(undef, n)
  hbar = S(undef, n)
  Mu   = S(undef, m)
  Av   = S(undef, m)
  u    = S(undef, 0)
  v    = S(undef, 0)
  err_vec = zeros(T, window)
  S = isconcretetype(S) ? S : typeof(x)
  stats = LsmrStats(0, false, false, T[], T[], zero(T), zero(T), zero(T), zero(T), zero(T), 0.0, "unknown")
  workspace = LsmrWorkspace{T,FC,S}(m, n, x, Nv, Aᴴu, h, hbar, Mu, Av, u, v, err_vec, stats)
  return workspace
end

function LsmrWorkspace(A, b; window::Integer = 5)
  m, n = size(A)
  S = ktypeof(b)
  LsmrWorkspace(m, n, S; window)
end

"""
Workspace for the in-place method [`lnlq!`](@ref).

The following outer constructors can be used to initialize this workspace:

    workspace = LnlqWorkspace(m, n, S)
    workspace = LnlqWorkspace(A, b)
    workspace = LnlqWorkspace(kc::KrylovConstructor)
"""
mutable struct LnlqWorkspace{T,FC,S} <: KrylovWorkspace{T,FC,S}
  m     :: Int
  n     :: Int
  x     :: S
  Nv    :: S
  Aᴴu   :: S
  y     :: S
  w̄     :: S
  Mu    :: S
  Av    :: S
  u     :: S
  v     :: S
  q     :: S
  stats :: LNLQStats{T}
end

function LnlqWorkspace(kc::KrylovConstructor)
  S   = typeof(kc.vm)
  FC  = eltype(S)
  T   = real(FC)
  m   = length(kc.vm)
  n   = length(kc.vn)
  x   = similar(kc.vn)
  Nv  = similar(kc.vn)
  Aᴴu = similar(kc.vn)
  y   = similar(kc.vm)
  w̄   = similar(kc.vm)
  Mu  = similar(kc.vm)
  Av  = similar(kc.vm)
  u   = similar(kc.vm_empty)
  v   = similar(kc.vn_empty)
  q   = similar(kc.vn_empty)
  stats = LNLQStats(0, false, T[], false, T[], T[], 0.0, "unknown")
  workspace = LnlqWorkspace{T,FC,S}(m, n, x, Nv, Aᴴu, y, w̄, Mu, Av, u, v, q, stats)
  return workspace
end

function LnlqWorkspace(m::Integer, n::Integer, S::Type)
  FC  = eltype(S)
  T   = real(FC)
  x   = S(undef, n)
  Nv  = S(undef, n)
  Aᴴu = S(undef, n)
  y   = S(undef, m)
  w̄   = S(undef, m)
  Mu  = S(undef, m)
  Av  = S(undef, m)
  u   = S(undef, 0)
  v   = S(undef, 0)
  q   = S(undef, 0)
  S = isconcretetype(S) ? S : typeof(x)
  stats = LNLQStats(0, false, T[], false, T[], T[], 0.0, "unknown")
  workspace = LnlqWorkspace{T,FC,S}(m, n, x, Nv, Aᴴu, y, w̄, Mu, Av, u, v, q, stats)
  return workspace
end

function LnlqWorkspace(A, b)
  m, n = size(A)
  S = ktypeof(b)
  LnlqWorkspace(m, n, S)
end

"""
Workspace for the in-place method [`craig!`](@ref).

The following outer constructors can be used to initialize this workspace:

    workspace = CraigWorkspace(m, n, S)
    workspace = CraigWorkspace(A, b)
    workspace = CraigWorkspace(kc::KrylovConstructor)
"""
mutable struct CraigWorkspace{T,FC,S} <: KrylovWorkspace{T,FC,S}
  m     :: Int
  n     :: Int
  x     :: S
  Nv    :: S
  Aᴴu   :: S
  y     :: S
  w     :: S
  Mu    :: S
  Av    :: S
  u     :: S
  v     :: S
  w2    :: S
  stats :: SimpleStats{T}
end

function CraigWorkspace(kc::KrylovConstructor)
  S   = typeof(kc.vm)
  FC  = eltype(S)
  T   = real(FC)
  m   = length(kc.vm)
  n   = length(kc.vn)
  x   = similar(kc.vn)
  Nv  = similar(kc.vn)
  Aᴴu = similar(kc.vn)
  y   = similar(kc.vm)
  w   = similar(kc.vm)
  Mu  = similar(kc.vm)
  Av  = similar(kc.vm)
  u   = similar(kc.vm_empty)
  v   = similar(kc.vn_empty)
  w2  = similar(kc.vn_empty)
  stats = SimpleStats(0, false, false, false, 0, zero(T), zero(T), T[], T[], T[], 0.0, "unknown")
  workspace = CraigWorkspace{T,FC,S}(m, n, x, Nv, Aᴴu, y, w, Mu, Av, u, v, w2, stats)
  return workspace
end

function CraigWorkspace(m::Integer, n::Integer, S::Type)
  FC  = eltype(S)
  T   = real(FC)
  x   = S(undef, n)
  Nv  = S(undef, n)
  Aᴴu = S(undef, n)
  y   = S(undef, m)
  w   = S(undef, m)
  Mu  = S(undef, m)
  Av  = S(undef, m)
  u   = S(undef, 0)
  v   = S(undef, 0)
  w2  = S(undef, 0)
  S = isconcretetype(S) ? S : typeof(x)
  stats = SimpleStats(0, false, false, false, 0, zero(T), zero(T), T[], T[], T[], 0.0, "unknown")
  workspace = CraigWorkspace{T,FC,S}(m, n, x, Nv, Aᴴu, y, w, Mu, Av, u, v, w2, stats)
  return workspace
end

function CraigWorkspace(A, b)
  m, n = size(A)
  S = ktypeof(b)
  CraigWorkspace(m, n, S)
end

"""
Workspace for the in-place method [`craigmr!`](@ref).

The following outer constructors can be used to initialize this workspace:

    workspace = CraigmrWorkspace(m, n, S)
    workspace = CraigmrWorkspace(A, b)
    workspace = CraigmrWorkspace(kc::KrylovConstructor)
"""
mutable struct CraigmrWorkspace{T,FC,S} <: KrylovWorkspace{T,FC,S}
  m     :: Int
  n     :: Int
  x     :: S
  Nv    :: S
  Aᴴu   :: S
  d     :: S
  y     :: S
  Mu    :: S
  w     :: S
  wbar  :: S
  Av    :: S
  u     :: S
  v     :: S
  q     :: S
  stats :: SimpleStats{T}
end

function CraigmrWorkspace(kc::KrylovConstructor)
  S    = typeof(kc.vm)
  FC   = eltype(S)
  T    = real(FC)
  m    = length(kc.vm)
  n    = length(kc.vn)
  x    = similar(kc.vn)
  Nv   = similar(kc.vn)
  Aᴴu  = similar(kc.vn)
  d    = similar(kc.vn)
  y    = similar(kc.vm)
  Mu   = similar(kc.vm)
  w    = similar(kc.vm)
  wbar = similar(kc.vm)
  Av   = similar(kc.vm)
  u    = similar(kc.vm_empty)
  v    = similar(kc.vn_empty)
  q    = similar(kc.vn_empty)
  stats = SimpleStats(0, false, false, false, 0, zero(T), zero(T), T[], T[], T[], 0.0, "unknown")
  workspace = CraigmrWorkspace{T,FC,S}(m, n, x, Nv, Aᴴu, d, y, Mu, w, wbar, Av, u, v, q, stats)
  return workspace
end

function CraigmrWorkspace(m::Integer, n::Integer, S::Type)
  FC   = eltype(S)
  T    = real(FC)
  x    = S(undef, n)
  Nv   = S(undef, n)
  Aᴴu  = S(undef, n)
  d    = S(undef, n)
  y    = S(undef, m)
  Mu   = S(undef, m)
  w    = S(undef, m)
  wbar = S(undef, m)
  Av   = S(undef, m)
  u    = S(undef, 0)
  v    = S(undef, 0)
  q    = S(undef, 0)
  S = isconcretetype(S) ? S : typeof(x)
  stats = SimpleStats(0, false, false, false, 0, zero(T), zero(T), T[], T[], T[], 0.0, "unknown")
  workspace = CraigmrWorkspace{T,FC,S}(m, n, x, Nv, Aᴴu, d, y, Mu, w, wbar, Av, u, v, q, stats)
  return workspace
end

function CraigmrWorkspace(A, b)
  m, n = size(A)
  S = ktypeof(b)
  CraigmrWorkspace(m, n, S)
end

"""
Workspace for the in-place method [`gmres!`](@ref).

The following outer constructors can be used to initialize this workspace:

    workspace = GmresWorkspace(m, n, S; memory = 20)
    workspace = GmresWorkspace(A, b; memory = 20)
    workspace = GmresWorkspace(kc::KrylovConstructor; memory = 20)

`memory` is set to `n` if the value given is larger than `n`.
"""
mutable struct GmresWorkspace{T,FC,S} <: KrylovWorkspace{T,FC,S}
  m          :: Int
  n          :: Int
  Δx         :: S
  x          :: S
  w          :: S
  p          :: S
  q          :: S
  V          :: Vector{S}
  c          :: Vector{T}
  s          :: Vector{FC}
  z          :: Vector{FC}
  R          :: Vector{FC}
  warm_start :: Bool
  inner_iter :: Int
  stats      :: SimpleStats{T}
end

function GmresWorkspace(kc::KrylovConstructor; memory::Integer = 20)
  S  = typeof(kc.vm)
  FC = eltype(S)
  T  = real(FC)
  m  = length(kc.vm)
  n  = length(kc.vn)
  memory = min(m, memory)
  Δx = similar(kc.vn_empty)
  x  = similar(kc.vn)
  w  = similar(kc.vn)
  p  = similar(kc.vn_empty)
  q  = similar(kc.vn_empty)
  V  = S[similar(kc.vn) for i = 1 : memory]
  c  = Vector{T}(undef, memory)
  s  = Vector{FC}(undef, memory)
  z  = Vector{FC}(undef, memory)
  R  = Vector{FC}(undef, div(memory * (memory+1), 2))
  stats = SimpleStats(0, false, false, false, 0, zero(T), zero(T), T[], T[], T[], 0.0, "unknown")
  workspace = GmresWorkspace{T,FC,S}(m, n, Δx, x, w, p, q, V, c, s, z, R, false, 0, stats)
  return workspace
end

function GmresWorkspace(m::Integer, n::Integer, S::Type; memory::Integer = 20)
  memory = min(m, memory)
  FC = eltype(S)
  T  = real(FC)
  Δx = S(undef, 0)
  x  = S(undef, n)
  w  = S(undef, n)
  p  = S(undef, 0)
  q  = S(undef, 0)
  V  = S[S(undef, n) for i = 1 : memory]
  c  = Vector{T}(undef, memory)
  s  = Vector{FC}(undef, memory)
  z  = Vector{FC}(undef, memory)
  R  = Vector{FC}(undef, div(memory * (memory+1), 2))
  S = isconcretetype(S) ? S : typeof(x)
  stats = SimpleStats(0, false, false, false, 0, zero(T), zero(T), T[], T[], T[], 0.0, "unknown")
  workspace = GmresWorkspace{T,FC,S}(m, n, Δx, x, w, p, q, V, c, s, z, R, false, 0, stats)
  return workspace
end

function GmresWorkspace(A, b; memory::Integer = 20)
  m, n = size(A)
  S = ktypeof(b)
  GmresWorkspace(m, n, S; memory)
end

"""
Workspace for the in-place method [`fgmres!`](@ref).

The following outer constructors can be used to initialize this workspace:

    workspace = FgmresWorkspace(m, n, S; memory = 20)
    workspace = FgmresWorkspace(A, b; memory = 20)
    workspace = FgmresWorkspace(kc::KrylovConstructor; memory = 20)

`memory` is set to `n` if the value given is larger than `n`.
"""
mutable struct FgmresWorkspace{T,FC,S} <: KrylovWorkspace{T,FC,S}
  m          :: Int
  n          :: Int
  Δx         :: S
  x          :: S
  w          :: S
  q          :: S
  V          :: Vector{S}
  Z          :: Vector{S}
  c          :: Vector{T}
  s          :: Vector{FC}
  z          :: Vector{FC}
  R          :: Vector{FC}
  warm_start :: Bool
  inner_iter :: Int
  stats      :: SimpleStats{T}
end

function FgmresWorkspace(kc::KrylovConstructor; memory::Integer = 20)
  S  = typeof(kc.vm)
  FC = eltype(S)
  T  = real(FC)
  m  = length(kc.vm)
  n  = length(kc.vn)
  memory = min(m, memory)
  Δx = similar(kc.vn_empty)
  x  = similar(kc.vn)
  w  = similar(kc.vn)
  q  = similar(kc.vn_empty)
  V  = S[similar(kc.vn) for i = 1 : memory]
  Z  = S[similar(kc.vn) for i = 1 : memory]
  c  = Vector{T}(undef, memory)
  s  = Vector{FC}(undef, memory)
  z  = Vector{FC}(undef, memory)
  R  = Vector{FC}(undef, div(memory * (memory+1), 2))
  stats = SimpleStats(0, false, false, false, 0, zero(T), zero(T), T[], T[], T[], 0.0, "unknown")
  workspace = FgmresWorkspace{T,FC,S}(m, n, Δx, x, w, q, V, Z, c, s, z, R, false, 0, stats)
  return workspace
end

function FgmresWorkspace(m::Integer, n::Integer, S::Type; memory::Integer = 20)
  memory = min(m, memory)
  FC = eltype(S)
  T  = real(FC)
  Δx = S(undef, 0)
  x  = S(undef, n)
  w  = S(undef, n)
  q  = S(undef, 0)
  V  = S[S(undef, n) for i = 1 : memory]
  Z  = S[S(undef, n) for i = 1 : memory]
  c  = Vector{T}(undef, memory)
  s  = Vector{FC}(undef, memory)
  z  = Vector{FC}(undef, memory)
  R  = Vector{FC}(undef, div(memory * (memory+1), 2))
  S = isconcretetype(S) ? S : typeof(x)
  stats = SimpleStats(0, false, false, false, 0, zero(T), zero(T), T[], T[], T[], 0.0, "unknown")
  workspace = FgmresWorkspace{T,FC,S}(m, n, Δx, x, w, q, V, Z, c, s, z, R, false, 0, stats)
  return workspace
end

function FgmresWorkspace(A, b; memory::Integer = 20)
  m, n = size(A)
  S = ktypeof(b)
  FgmresWorkspace(m, n, S; memory)
end

"""
Workspace for the in-place method [`fom!`](@ref).

The following outer constructors can be used to initialize this workspace:

    workspace = FomWorkspace(m, n, S; memory = 20)
    workspace = FomWorkspace(A, b; memory = 20)
    workspace = FomWorkspace(kc::KrylovConstructor; memory = 20)

`memory` is set to `n` if the value given is larger than `n`.
"""
mutable struct FomWorkspace{T,FC,S} <: KrylovWorkspace{T,FC,S}
  m          :: Int
  n          :: Int
  Δx         :: S
  x          :: S
  w          :: S
  p          :: S
  q          :: S
  V          :: Vector{S}
  l          :: Vector{FC}
  z          :: Vector{FC}
  U          :: Vector{FC}
  warm_start :: Bool
  stats      :: SimpleStats{T}
end

function FomWorkspace(kc::KrylovConstructor; memory::Integer = 20)
  S  = typeof(kc.vm)
  FC = eltype(S)
  T  = real(FC)
  m  = length(kc.vm)
  n  = length(kc.vn)
  memory = min(m, memory)
  Δx = similar(kc.vn_empty)
  x  = similar(kc.vn)
  w  = similar(kc.vn)
  p  = similar(kc.vn_empty)
  q  = similar(kc.vn_empty)
  V  = S[similar(kc.vn) for i = 1 : memory]
  l  = Vector{FC}(undef, memory)
  z  = Vector{FC}(undef, memory)
  U  = Vector{FC}(undef, div(memory * (memory+1), 2))
  stats = SimpleStats(0, false, false, false, 0, zero(T), zero(T), T[], T[], T[], 0.0, "unknown")
  workspace = FomWorkspace{T,FC,S}(m, n, Δx, x, w, p, q, V, l, z, U, false, stats)
  return workspace
end

function FomWorkspace(m::Integer, n::Integer, S::Type; memory::Integer = 20)
  memory = min(m, memory)
  FC = eltype(S)
  T  = real(FC)
  Δx = S(undef, 0)
  x  = S(undef, n)
  w  = S(undef, n)
  p  = S(undef, 0)
  q  = S(undef, 0)
  V  = S[S(undef, n) for i = 1 : memory]
  l  = Vector{FC}(undef, memory)
  z  = Vector{FC}(undef, memory)
  U  = Vector{FC}(undef, div(memory * (memory+1), 2))
  S = isconcretetype(S) ? S : typeof(x)
  stats = SimpleStats(0, false, false, false, 0, zero(T), zero(T), T[], T[], T[], 0.0, "unknown")
  workspace = FomWorkspace{T,FC,S}(m, n, Δx, x, w, p, q, V, l, z, U, false, stats)
  return workspace
end

function FomWorkspace(A, b; memory::Integer = 20)
  m, n = size(A)
  S = ktypeof(b)
  FomWorkspace(m, n, S; memory)
end

"""
Workspace for the in-place method [`gpmr!`](@ref).

The following outer constructors can be used to initialize this workspace:

    workspace = GpmrWorkspace(m, n, S; memory = 20)
    workspace = GpmrWorkspace(A, b; memory = 20)
    workspace = GpmrWorkspace(kc::KrylovConstructor; memory = 20)

`memory` is set to `n + m` if the value given is larger than `n + m`.
"""
mutable struct GpmrWorkspace{T,FC,S} <: KrylovWorkspace{T,FC,S}
  m          :: Int
  n          :: Int
  wA         :: S
  wB         :: S
  dA         :: S
  dB         :: S
  Δx         :: S
  Δy         :: S
  x          :: S
  y          :: S
  q          :: S
  p          :: S
  V          :: Vector{S}
  U          :: Vector{S}
  gs         :: Vector{FC}
  gc         :: Vector{T}
  zt         :: Vector{FC}
  R          :: Vector{FC}
  warm_start :: Bool
  stats      :: SimpleStats{T}
end

function GpmrWorkspace(kc::KrylovConstructor; memory::Integer = 20)
  S  = typeof(kc.vm)
  FC = eltype(S)
  T  = real(FC)
  m  = length(kc.vm)
  n  = length(kc.vn)
  memory = min(n + m, memory)
  wA = similar(kc.vn_empty)
  wB = similar(kc.vm_empty)
  dA = similar(kc.vm)
  dB = similar(kc.vn)
  Δx = similar(kc.vm_empty)
  Δy = similar(kc.vn_empty)
  x  = similar(kc.vm)
  y  = similar(kc.vn)
  q  = similar(kc.vm_empty)
  p  = similar(kc.vn_empty)
  V  = S[similar(kc.vm) for i = 1 : memory]
  U  = S[similar(kc.vn) for i = 1 : memory]
  gs = Vector{FC}(undef, 4 * memory)
  gc = Vector{T}(undef, 4 * memory)
  zt = Vector{FC}(undef, 2 * memory)
  R  = Vector{FC}(undef, memory * (2 * memory + 1))
  stats = SimpleStats(0, false, false, false, 0, zero(T), zero(T), T[], T[], T[], 0.0, "unknown")
  workspace = GpmrWorkspace{T,FC,S}(m, n, wA, wB, dA, dB, Δx, Δy, x, y, q, p, V, U, gs, gc, zt, R, false, stats)
  return workspace
end

function GpmrWorkspace(m::Integer, n::Integer, S::Type; memory::Integer = 20)
  memory = min(n + m, memory)
  FC = eltype(S)
  T  = real(FC)
  wA = S(undef, 0)
  wB = S(undef, 0)
  dA = S(undef, m)
  dB = S(undef, n)
  Δx = S(undef, 0)
  Δy = S(undef, 0)
  x  = S(undef, m)
  y  = S(undef, n)
  q  = S(undef, 0)
  p  = S(undef, 0)
  V  = S[S(undef, m) for i = 1 : memory]
  U  = S[S(undef, n) for i = 1 : memory]
  gs = Vector{FC}(undef, 4 * memory)
  gc = Vector{T}(undef, 4 * memory)
  zt = Vector{FC}(undef, 2 * memory)
  R  = Vector{FC}(undef, memory * (2 * memory + 1))
  S = isconcretetype(S) ? S : typeof(x)
  stats = SimpleStats(0, false, false, false, 0, zero(T), zero(T), T[], T[], T[], 0.0, "unknown")
  workspace = GpmrWorkspace{T,FC,S}(m, n, wA, wB, dA, dB, Δx, Δy, x, y, q, p, V, U, gs, gc, zt, R, false, stats)
  return workspace
end

function GpmrWorkspace(A, b; memory::Integer = 20)
  m, n = size(A)
  S = ktypeof(b)
  GpmrWorkspace(m, n, S; memory)
end
