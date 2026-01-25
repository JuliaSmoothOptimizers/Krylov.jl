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
For rectangular problems (`m != n`), use the second constructor with `vm` and `vn`.

#### Input arguments

* `vm`: a vector of length `m`;
* `vn`: a vector of length `n`.

#### Keyword arguments

- `vm_empty`: an empty vector that may be replaced with a vector of length `m`;
- `vn_empty`: an empty vector that may be replaced with a vector of length `n`.

#### Notes

Each pair of vectors (`vm`, `vm_empty`) and (`vn`, `vn_empty`) must have the same type.
Empty vectors `vm_empty` and `vn_empty` reduce storage requirements when features such as warm-start or preconditioners are unused.
These empty vectors will be replaced within a [`KrylovWorkspace`](@ref) only if required, such as when preconditioners are provided.
"""
struct KrylovConstructor{Sm,Sn}
  vm::Sm
  vn::Sn
  vm_empty::Sm
  vn_empty::Sn

  function KrylovConstructor{Sm,Sn}(vm, vn, vm_empty, vn_empty) where {Sm,Sn}
    eltype(Sm) === eltype(Sn) || throw(ArgumentError("KrylovConstructor requires that eltype(Sm) == eltype(Sn), got $(eltype(Sm)) and $(eltype(Sn))"))
    return new{Sm,Sn}(vm, vn, vm_empty, vn_empty)
  end
end

function KrylovConstructor(vm::Sm, vn::Sn; vm_empty::Sm=vm, vn_empty::Sn=vn) where {Sm,Sn}
  return KrylovConstructor{Sm,Sn}(vm, vn, vm_empty, vn_empty)
end

function KrylovConstructor(vm::S; vm_empty::S=vm) where S
  return KrylovConstructor{S,S}(vm, vm, vm_empty, vm_empty)
end

# Krylov.jl v11.x -> Change to KrylovWorkspace{T,FC,Sm,Sn} and delete the alias below
abstract type _KrylovWorkspace{T,FC,Sm,Sn} end

"Abstract type for using Krylov solvers in-place."
const KrylovWorkspace{T,FC,S} = _KrylovWorkspace{T,FC,S,S}

"""
Workspace for the in-place methods [`minres!`](@ref) and [`krylov_solve!`](@ref).

The following outer constructors can be used to initialize this workspace:

    workspace = MinresWorkspace(m, n, S; window = 5)
    workspace = MinresWorkspace(A, b; window = 5)
    workspace = MinresWorkspace(kc::KrylovConstructor{S,S}; window = 5)

`m` and `n` denote the dimensions of the linear operator `A` passed to the in-place methods.
Since [`minres`](@ref) only supports square linear operators, `m` and `n` must be equal.
`S` is the storage type of the vectors in the workspace, such as `Vector{Float64}`.

[`KrylovConstructor`](@ref) facilitates the allocation of vectors in the workspace if `S(undef, n)` is not available.
"""
mutable struct MinresWorkspace{T,FC,S} <: _KrylovWorkspace{T,FC,S,S}
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

function MinresWorkspace(kc::KrylovConstructor{S,S}; window::Int = 5) where S
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
  stats = SimpleStats(0, false, false, false, 0, T[], T[], T[], 0.0, "unknown")
  workspace = MinresWorkspace{T,FC,S}(m, n, Δx, x, r1, r2, npc_dir, w1, w2, y, v, err_vec, false, stats)
  return workspace
end

function MinresWorkspace(m::Integer, n::Integer, S::Type; window::Int = 5)
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
  stats = SimpleStats(0, false, false, false, 0, T[], T[], T[], 0.0, "unknown")
  workspace = MinresWorkspace{T,FC,S}(m, n, Δx, x, r1, r2, npc_dir, w1, w2, y, v, err_vec, false, stats)
  return workspace
end

function MinresWorkspace(A, b; window::Int = 5)
  m, n = size(A)
  S = ktypeof(b)
  MinresWorkspace(m, n, S; window)
end

"""
Workspace for the in-place methods [`minares!`](@ref) and [`krylov_solve!`](@ref).

The following outer constructors can be used to initialize this workspace:

    workspace = MinaresWorkspace(m, n, S)
    workspace = MinaresWorkspace(A, b)
    workspace = MinaresWorkspace(kc::KrylovConstructor{S,S})

`m` and `n` denote the dimensions of the linear operator `A` passed to the in-place methods.
Since [`minares`](@ref) only supports square linear operators, `m` and `n` must be equal.
`S` is the storage type of the vectors in the workspace, such as `Vector{Float64}`.

[`KrylovConstructor`](@ref) facilitates the allocation of vectors in the workspace if `S(undef, n)` is not available.
"""
mutable struct MinaresWorkspace{T,FC,S} <: _KrylovWorkspace{T,FC,S,S}
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

function MinaresWorkspace(kc::KrylovConstructor{S,S}) where S
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
  stats = SimpleStats(0, false, false, false, 0, T[], T[], T[], 0.0, "unknown")
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
  stats = SimpleStats(0, false, false, false, 0, T[], T[], T[], 0.0, "unknown")
  workspace = MinaresWorkspace{T,FC,S}(m, n, Δx, vₖ, vₖ₊₁, x, wₖ₋₂, wₖ₋₁, dₖ₋₂, dₖ₋₁, q, false, stats)
  return workspace
end

function MinaresWorkspace(A, b)
  m, n = size(A)
  S = ktypeof(b)
  MinaresWorkspace(m, n, S)
end

"""
Workspace for the in-place methods [`cg!`](@ref) and [`krylov_solve!`](@ref).

The following outer constructors can be used to initialize this workspace:

    workspace = CgWorkspace(m, n, S)
    workspace = CgWorkspace(A, b)
    workspace = CgWorkspace(kc::KrylovConstructor{S,S})

`m` and `n` denote the dimensions of the linear operator `A` passed to the in-place methods.
Since [`cg`](@ref) only supports square linear operators, `m` and `n` must be equal.
`S` is the storage type of the vectors in the workspace, such as `Vector{Float64}`.

[`KrylovConstructor`](@ref) facilitates the allocation of vectors in the workspace if `S(undef, n)` is not available.
"""
mutable struct CgWorkspace{T,FC,S} <: _KrylovWorkspace{T,FC,S,S}
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

function CgWorkspace(kc::KrylovConstructor{S,S}) where S
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
  stats = SimpleStats(0, false, false, false, 0, T[], T[], T[], 0.0, "unknown")
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
  stats = SimpleStats(0, false, false, false, 0, T[], T[], T[], 0.0, "unknown")
  workspace = CgWorkspace{T,FC,S}(m, n, Δx, x, r, npc_dir, p, Ap, z, false, stats)
  return workspace
end

function CgWorkspace(A, b)
  m, n = size(A)
  S = ktypeof(b)
  CgWorkspace(m, n, S)
end

"""
Workspace for the in-place methods [`cr!`](@ref) and [`krylov_solve!`](@ref).

The following outer constructors can be used to initialize this workspace:

    workspace = CrWorkspace(m, n, S)
    workspace = CrWorkspace(A, b)
    workspace = CrWorkspace(kc::KrylovConstructor{S,S})

`m` and `n` denote the dimensions of the linear operator `A` passed to the in-place methods.
Since [`cr`](@ref) only supports square linear operators, `m` and `n` must be equal.
`S` is the storage type of the vectors in the workspace, such as `Vector{Float64}`.

[`KrylovConstructor`](@ref) facilitates the allocation of vectors in the workspace if `S(undef, n)` is not available.
"""
mutable struct CrWorkspace{T,FC,S} <: _KrylovWorkspace{T,FC,S,S}
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

function CrWorkspace(kc::KrylovConstructor{S,S}) where S
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
  stats = SimpleStats(0, false, false, false, 0, T[], T[], T[], 0.0, "unknown")
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
  stats = SimpleStats(0, false, false, false, 0, T[], T[], T[], 0.0, "unknown")
  workspace = CrWorkspace{T,FC,S}(m, n, Δx, x, r, npc_dir, p, q, Ar, Mq, false, stats)
  return workspace
end

function CrWorkspace(A, b)
  m, n = size(A)
  S = ktypeof(b)
  CrWorkspace(m, n, S)
end

"""
Workspace for the in-place methods [`car!`](@ref) and [`krylov_solve!`](@ref).

The following outer constructors can be used to initialize this workspace:

    workspace = CarWorkspace(m, n, S)
    workspace = CarWorkspace(A, b)
    workspace = CarWorkspace(kc::KrylovConstructor{S,S})

`m` and `n` denote the dimensions of the linear operator `A` passed to the in-place methods.
Since [`car`](@ref) only supports square linear operators, `m` and `n` must be equal.
`S` is the storage type of the vectors in the workspace, such as `Vector{Float64}`.

[`KrylovConstructor`](@ref) facilitates the allocation of vectors in the workspace if `S(undef, n)` is not available.
"""
mutable struct CarWorkspace{T,FC,S} <: _KrylovWorkspace{T,FC,S,S}
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

function CarWorkspace(kc::KrylovConstructor{S,S}) where S
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
  stats = SimpleStats(0, false, false, false, 0, T[], T[], T[], 0.0, "unknown")
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
  stats = SimpleStats(0, false, false, false, 0, T[], T[], T[], 0.0, "unknown")
  workspace = CarWorkspace{T,FC,S}(m, n, Δx, x, r, p, s, q, t, u, Mu, false, stats)
  return workspace
end

function CarWorkspace(A, b)
  m, n = size(A)
  S = ktypeof(b)
  CarWorkspace(m, n, S)
end

"""
Workspace for the in-place methods [`symmlq!`](@ref) and [`krylov_solve!`](@ref).

The following outer constructors can be used to initialize this workspace:

    workspace = SymmlqWorkspace(m, n, S)
    workspace = SymmlqWorkspace(A, b)
    workspace = SymmlqWorkspace(kc::KrylovConstructor{S,S})

`m` and `n` denote the dimensions of the linear operator `A` passed to the in-place methods.
Since [`symmlq`](@ref) only supports square linear operators, `m` and `n` must be equal.
`S` is the storage type of the vectors in the workspace, such as `Vector{Float64}`.

[`KrylovConstructor`](@ref) facilitates the allocation of vectors in the workspace if `S(undef, n)` is not available.
"""
mutable struct SymmlqWorkspace{T,FC,S} <: _KrylovWorkspace{T,FC,S,S}
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

function SymmlqWorkspace(kc::KrylovConstructor{S,S}; window::Int = 5) where S
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

function SymmlqWorkspace(m::Integer, n::Integer, S::Type; window::Int = 5)
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

function SymmlqWorkspace(A, b; window::Int = 5)
  m, n = size(A)
  S = ktypeof(b)
  SymmlqWorkspace(m, n, S; window)
end

"""
Workspace for the in-place methods [`cg_lanczos!`](@ref) and [`krylov_solve!`](@ref).

The following outer constructors can be used to initialize this workspace:

    workspace = CgLanczosWorkspace(m, n, S)
    workspace = CgLanczosWorkspace(A, b)
    workspace = CgLanczosWorkspace(kc::KrylovConstructor{S,S})

`m` and `n` denote the dimensions of the linear operator `A` passed to the in-place methods.
Since [`cg_lanczos`](@ref) only supports square linear operators, `m` and `n` must be equal.
`S` is the storage type of the vectors in the workspace, such as `Vector{Float64}`.

[`KrylovConstructor`](@ref) facilitates the allocation of vectors in the workspace if `S(undef, n)` is not available.
"""
mutable struct CgLanczosWorkspace{T,FC,S} <: _KrylovWorkspace{T,FC,S,S}
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

function CgLanczosWorkspace(kc::KrylovConstructor{S,S}) where S
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
Workspace for the in-place methods [`cg_lanczos_shift!`](@ref) and [`krylov_solve!`](@ref).

The following outer constructors can be used to initialize this workspace:

    workspace = CgLanczosShiftWorkspace(m, n, nshifts, S)
    workspace = CgLanczosShiftWorkspace(A, b, nshifts)
    workspace = CgLanczosShiftWorkspace(kc::KrylovConstructor{S,S}, nshifts)

`m` and `n` denote the dimensions of the linear operator `A` passed to the in-place methods.
`nshifts` denotes the length of the vector `shifts` passed to the in-place methods.
Since [`cg_lanczos_shift`](@ref) only supports square linear operators, `m` and `n` must be equal.
`S` is the storage type of the vectors in the workspace, such as `Vector{Float64}`.

[`KrylovConstructor`](@ref) facilitates the allocation of vectors in the workspace if `S(undef, n)` is not available.
"""
mutable struct CgLanczosShiftWorkspace{T,FC,S} <: _KrylovWorkspace{T,FC,S,S}
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

function CgLanczosShiftWorkspace(kc::KrylovConstructor{S,S}, nshifts::Integer) where S
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
Workspace for the in-place methods [`minres_qlp!`](@ref) and [`krylov_solve!`](@ref).

The following outer constructors can be used to initialize this workspace:

    workspace = MinresQlpWorkspace(m, n, S)
    workspace = MinresQlpWorkspace(A, b)
    workspace = MinresQlpWorkspace(kc::KrylovConstructor{S,S})

`m` and `n` denote the dimensions of the linear operator `A` passed to the in-place methods.
Since [`minres_qlp`](@ref) only supports square linear operators, `m` and `n` must be equal.
`S` is the storage type of the vectors in the workspace, such as `Vector{Float64}`.

[`KrylovConstructor`](@ref) facilitates the allocation of vectors in the workspace if `S(undef, n)` is not available.
"""
mutable struct MinresQlpWorkspace{T,FC,S} <: _KrylovWorkspace{T,FC,S,S}
  m          :: Int
  n          :: Int
  Δx         :: S
  wₖ₋₁       :: S
  wₖ         :: S
  M⁻¹vₖ₋₁    :: S
  M⁻¹vₖ      :: S
  npc_dir    :: S
  x          :: S
  p          :: S
  vₖ         :: S
  warm_start :: Bool
  stats      :: SimpleStats{T}
end

function MinresQlpWorkspace(kc::KrylovConstructor{S,S}) where S
  FC      = eltype(S)
  T       = real(FC)
  m       = length(kc.vm)
  n       = length(kc.vn)
  Δx      = similar(kc.vn_empty)
  wₖ₋₁    = similar(kc.vn)
  wₖ      = similar(kc.vn)
  M⁻¹vₖ₋₁ = similar(kc.vn)
  M⁻¹vₖ   = similar(kc.vn)
  npc_dir = similar(kc.vn_empty)
  x       = similar(kc.vn)
  p       = similar(kc.vn)
  vₖ      = similar(kc.vn_empty)
  stats = SimpleStats(0, false, false, false, 0, T[], T[], T[], 0.0, "unknown")
  workspace = MinresQlpWorkspace{T,FC,S}(m, n, Δx, wₖ₋₁, wₖ, M⁻¹vₖ₋₁, M⁻¹vₖ, npc_dir, x, p, vₖ, false, stats)
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
  npc_dir  = S(undef, 0)
  S = isconcretetype(S) ? S : typeof(x)
  stats = SimpleStats(0, false, false, false, 0, T[], T[], T[], 0.0, "unknown")
  workspace = MinresQlpWorkspace{T,FC,S}(m, n, Δx, wₖ₋₁, wₖ, M⁻¹vₖ₋₁, M⁻¹vₖ, npc_dir, x, p, vₖ, false, stats)
  return workspace
end

function MinresQlpWorkspace(A, b)
  m, n = size(A)
  S = ktypeof(b)
  MinresQlpWorkspace(m, n, S)
end

"""
Workspace for the in-place methods [`dqgmres!`](@ref) and [`krylov_solve!`](@ref).

The following outer constructors can be used to initialize this workspace:

    workspace = DqgmresWorkspace(m, n, S; memory = 20)
    workspace = DqgmresWorkspace(A, b; memory = 20)
    workspace = DqgmresWorkspace(kc::KrylovConstructor{S,S}; memory = 20)

`m` and `n` denote the dimensions of the linear operator `A` passed to the in-place methods.
Since [`dqgmres`](@ref) only supports square linear operators, `m` and `n` must be equal.
`S` is the storage type of the vectors in the workspace, such as `Vector{Float64}`.
`memory` is set to `n` if the value given is larger than `n`.

[`KrylovConstructor`](@ref) facilitates the allocation of vectors in the workspace if `S(undef, n)` is not available.
"""
mutable struct DqgmresWorkspace{T,FC,S} <: _KrylovWorkspace{T,FC,S,S}
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

function DqgmresWorkspace(kc::KrylovConstructor{S,S}; memory::Int = 20) where S
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
  stats  = SimpleStats(0, false, false, false, 0, T[], T[], T[], 0.0, "unknown")
  workspace = DqgmresWorkspace{T,FC,S}(m, n, Δx, x, t, z, w, P, V, c, s, H, false, stats)
  return workspace
end

function DqgmresWorkspace(m::Integer, n::Integer, S::Type; memory::Int = 20)
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
  stats = SimpleStats(0, false, false, false, 0, T[], T[], T[], 0.0, "unknown")
  workspace = DqgmresWorkspace{T,FC,S}(m, n, Δx, x, t, z, w, P, V, c, s, H, false, stats)
  return workspace
end

function DqgmresWorkspace(A, b; memory::Int = 20)
  m, n = size(A)
  S = ktypeof(b)
  DqgmresWorkspace(m, n, S; memory)
end

"""
Workspace for the in-place methods [`diom!`](@ref) and [`krylov_solve!`](@ref).

The following outer constructors can be used to initialize this workspace:

    workspace = DiomWorkspace(m, n, S; memory = 20)
    workspace = DiomWorkspace(A, b; memory = 20)
    workspace = DiomWorkspace(kc::KrylovConstructor{S,S}; memory = 20)

`m` and `n` denote the dimensions of the linear operator `A` passed to the in-place methods.
Since [`diom`](@ref) only supports square linear operators, `m` and `n` must be equal.
`S` is the storage type of the vectors in the workspace, such as `Vector{Float64}`.
`memory` is set to `n` if the value given is larger than `n`.

[`KrylovConstructor`](@ref) facilitates the allocation of vectors in the workspace if `S(undef, n)` is not available.
"""
mutable struct DiomWorkspace{T,FC,S} <: _KrylovWorkspace{T,FC,S,S}
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

function DiomWorkspace(kc::KrylovConstructor{S,S}; memory::Int = 20) where S
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
  stats  = SimpleStats(0, false, false, false, 0, T[], T[], T[], 0.0, "unknown")
  workspace = DiomWorkspace{T,FC,S}(m, n, Δx, x, t, z, w, P, V, L, H, false, stats)
  return workspace
end

function DiomWorkspace(m::Integer, n::Integer, S::Type; memory::Int = 20)
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
  stats = SimpleStats(0, false, false, false, 0, T[], T[], T[], 0.0, "unknown")
  workspace = DiomWorkspace{T,FC,S}(m, n, Δx, x, t, z, w, P, V, L, H, false, stats)
  return workspace
end

function DiomWorkspace(A, b; memory::Int = 20)
  m, n = size(A)
  S = ktypeof(b)
  DiomWorkspace(m, n, S; memory)
end

"""
Workspace for the in-place methods [`usymlq!`](@ref) and [`krylov_solve!`](@ref).

The following outer constructors can be used to initialize this workspace:

    workspace = UsymlqWorkspace(m, n, Sm, Sn)
    workspace = UsymlqWorkspace(m, n, S)
    workspace = UsymlqWorkspace(A, b)
    workspace = UsymlqWorkspace(A, b, c)
    workspace = UsymlqWorkspace(kc::KrylovConstructor{Sm,Sn})

`m` and `n` denote the dimensions of the linear operator `A` passed to the in-place methods.
Since [`usymlq`](@ref) supports rectangular linear operators, `m` and `n` can differ.
`Sm` and `Sn` are the storage types of the workspace vectors of length `m` and `n`, respectively.
If the same storage type can be used for both, a single type `S` may be provided, such as `Vector{Float64}`.

[`KrylovConstructor`](@ref) facilitates the allocation of vectors in the workspace if `Sm(undef, m)` and `Sn(undef, n)` are not available.
"""
mutable struct UsymlqWorkspace{T,FC,Sm,Sn} <: _KrylovWorkspace{T,FC,Sm,Sn}
  m          :: Int
  n          :: Int
  uₖ₋₁       :: Sn
  uₖ         :: Sn
  p          :: Sn
  Δx         :: Sn
  x          :: Sn
  d̅          :: Sn
  vₖ₋₁       :: Sm
  vₖ         :: Sm
  q          :: Sm
  warm_start :: Bool
  stats      :: SimpleStats{T}
end

function UsymlqWorkspace(kc::KrylovConstructor{Sm,Sn}) where {Sm,Sn}
  FC   = eltype(Sm)
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
  stats = SimpleStats(0, false, false, false, 0, T[], T[], T[], 0.0, "unknown")
  workspace = UsymlqWorkspace{T,FC,Sm,Sn}(m, n, uₖ₋₁, uₖ, p, Δx, x, d̅, vₖ₋₁, vₖ, q, false, stats)
  return workspace
end

function UsymlqWorkspace(m::Integer, n::Integer, Sm::Type, Sn::Type)
  FC   = eltype(Sm)
  T    = real(FC)
  uₖ₋₁ = Sn(undef, n)
  uₖ   = Sn(undef, n)
  p    = Sn(undef, n)
  Δx   = Sn(undef, 0)
  x    = Sn(undef, n)
  d̅    = Sn(undef, n)
  vₖ₋₁ = Sm(undef, m)
  vₖ   = Sm(undef, m)
  q    = Sm(undef, m)
  Sm = isconcretetype(Sm) ? Sm : typeof(q)
  Sn = isconcretetype(Sn) ? Sn : typeof(x)
  stats = SimpleStats(0, false, false, false, 0, T[], T[], T[], 0.0, "unknown")
  workspace = UsymlqWorkspace{T,FC,Sm,Sn}(m, n, uₖ₋₁, uₖ, p, Δx, x, d̅, vₖ₋₁, vₖ, q, false, stats)
  return workspace
end

function UsymlqWorkspace(m::Integer, n::Integer, S::Type)
  UsymlqWorkspace(m, n, S, S)
end

function UsymlqWorkspace(A, b)
  m, n = size(A)
  S = ktypeof(b)
  UsymlqWorkspace(m, n, S)
end

function UsymlqWorkspace(A, b, c)
  m, n = size(A)
  Sm = ktypeof(b)
  Sn = ktypeof(c)
  UsymlqWorkspace(m, n, Sm, Sn)
end

"""
Workspace for the in-place methods [`usymqr!`](@ref) and [`krylov_solve!`](@ref).

The following outer constructors can be used to initialize this workspace:

    workspace = UsymqrWorkspace(m, n, Sm, Sn)
    workspace = UsymqrWorkspace(m, n, S)
    workspace = UsymqrWorkspace(A, b)
    workspace = UsymqrWorkspace(A, b, c)
    workspace = UsymqrWorkspace(kc::KrylovConstructor{Sm,Sn})

`m` and `n` denote the dimensions of the linear operator `A` passed to the in-place methods.
Since [`usymqr`](@ref) supports rectangular linear operators, `m` and `n` can differ.
`Sm` and `Sn` are the storage types of the workspace vectors of length `m` and `n`, respectively.
If the same storage type can be used for both, a single type `S` may be provided, such as `Vector{Float64}`.

[`KrylovConstructor`](@ref) facilitates the allocation of vectors in the workspace if `Sm(undef, m)` and `Sn(undef, n)` are not available.
"""
mutable struct UsymqrWorkspace{T,FC,Sm,Sn} <: _KrylovWorkspace{T,FC,Sm,Sn}
  m          :: Int
  n          :: Int
  vₖ₋₁       :: Sm
  vₖ         :: Sm
  q          :: Sm
  Δx         :: Sn
  x          :: Sn
  wₖ₋₂       :: Sn
  wₖ₋₁       :: Sn
  uₖ₋₁       :: Sn
  uₖ         :: Sn
  p          :: Sn
  warm_start :: Bool
  stats      :: SimpleStats{T}
end

function UsymqrWorkspace(kc::KrylovConstructor{Sm,Sn}) where {Sm,Sn}
  FC   = eltype(Sm)
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
  stats = SimpleStats(0, false, false, false, 0, T[], T[], T[], 0.0, "unknown")
  workspace = UsymqrWorkspace{T,FC,Sm,Sn}(m, n, vₖ₋₁, vₖ, q, Δx, x, wₖ₋₂, wₖ₋₁, uₖ₋₁, uₖ, p, false, stats)
  return workspace
end

function UsymqrWorkspace(m::Integer, n::Integer, Sm::Type, Sn::Type)
  FC   = eltype(Sm)
  T    = real(FC)
  vₖ₋₁ = Sm(undef, m)
  vₖ   = Sm(undef, m)
  q    = Sm(undef, m)
  Δx   = Sn(undef, 0)
  x    = Sn(undef, n)
  wₖ₋₂ = Sn(undef, n)
  wₖ₋₁ = Sn(undef, n)
  uₖ₋₁ = Sn(undef, n)
  uₖ   = Sn(undef, n)
  p    = Sn(undef, n)
  Sm = isconcretetype(Sm) ? Sm : typeof(q)
  Sn = isconcretetype(Sn) ? Sn : typeof(x)
  stats = SimpleStats(0, false, false, false, 0, T[], T[], T[], 0.0, "unknown")
  workspace = UsymqrWorkspace{T,FC,Sm,Sn}(m, n, vₖ₋₁, vₖ, q, Δx, x, wₖ₋₂, wₖ₋₁, uₖ₋₁, uₖ, p, false, stats)
  return workspace
end

function UsymqrWorkspace(m::Integer, n::Integer, S::Type)
  UsymqrWorkspace(m, n, S, S)
end

function UsymqrWorkspace(A, b)
  m, n = size(A)
  S = ktypeof(b)
  UsymqrWorkspace(m, n, S)
end

function UsymqrWorkspace(A, b, c)
  m, n = size(A)
  Sm = ktypeof(b)
  Sn = ktypeof(c)
  UsymqrWorkspace(m, n, Sm, Sn)
end

"""
Workspace for the in-place methods [`tricg!`](@ref) and [`krylov_solve!`](@ref).

The following outer constructors can be used to initialize this workspace:

    workspace = TricgWorkspace(m, n, Sm, Sn)
    workspace = TricgWorkspace(m, n, S)
    workspace = TricgWorkspace(A, b)
    workspace = TricgWorkspace(A, b, c)
    workspace = TricgWorkspace(kc::KrylovConstructor{Sm,Sn})

`m` and `n` denote the dimensions of the linear operator `A` passed to the in-place methods.
Since [`tricg`](@ref) supports rectangular linear operators, `m` and `n` can differ.
`Sm` and `Sn` are the storage types of the workspace vectors of length `m` and `n`, respectively.
If the same storage type can be used for both, a single type `S` may be provided, such as `Vector{Float64}`.

[`KrylovConstructor`](@ref) facilitates the allocation of vectors in the workspace if `Sm(undef, m)` and `Sn(undef, n)` are not available.
"""
mutable struct TricgWorkspace{T,FC,Sm,Sn} <: _KrylovWorkspace{T,FC,Sm,Sn}
  m          :: Int
  n          :: Int
  y          :: Sn
  N⁻¹uₖ₋₁    :: Sn
  N⁻¹uₖ      :: Sn
  p          :: Sn
  gy₂ₖ₋₁     :: Sn
  gy₂ₖ       :: Sn
  x          :: Sm
  M⁻¹vₖ₋₁    :: Sm
  M⁻¹vₖ      :: Sm
  q          :: Sm
  gx₂ₖ₋₁     :: Sm
  gx₂ₖ       :: Sm
  Δx         :: Sm
  Δy         :: Sn
  uₖ         :: Sn
  vₖ         :: Sm
  warm_start :: Bool
  stats      :: SimpleStats{T}
end

function TricgWorkspace(kc::KrylovConstructor{Sm,Sn}) where {Sm,Sn}
  FC      = eltype(Sm)
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
  stats = SimpleStats(0, false, false, false, 0, T[], T[], T[], 0.0, "unknown")
  workspace = TricgWorkspace{T,FC,Sm,Sn}(m, n, y, N⁻¹uₖ₋₁, N⁻¹uₖ, p, gy₂ₖ₋₁, gy₂ₖ, x, M⁻¹vₖ₋₁, M⁻¹vₖ, q, gx₂ₖ₋₁, gx₂ₖ, Δx, Δy, uₖ, vₖ, false, stats)
  return workspace
end

function TricgWorkspace(m::Integer, n::Integer, Sm::Type, Sn::Type)
  FC      = eltype(Sm)
  T       = real(FC)
  y       = Sn(undef, n)
  N⁻¹uₖ₋₁ = Sn(undef, n)
  N⁻¹uₖ   = Sn(undef, n)
  p       = Sn(undef, n)
  gy₂ₖ₋₁  = Sn(undef, n)
  gy₂ₖ    = Sn(undef, n)
  x       = Sm(undef, m)
  M⁻¹vₖ₋₁ = Sm(undef, m)
  M⁻¹vₖ   = Sm(undef, m)
  q       = Sm(undef, m)
  gx₂ₖ₋₁  = Sm(undef, m)
  gx₂ₖ    = Sm(undef, m)
  Δx      = Sm(undef, 0)
  Δy      = Sn(undef, 0)
  uₖ      = Sn(undef, 0)
  vₖ      = Sm(undef, 0)
  Sm = isconcretetype(Sm) ? Sm : typeof(x)
  Sn = isconcretetype(Sn) ? Sn : typeof(y)
  stats = SimpleStats(0, false, false, false, 0, T[], T[], T[], 0.0, "unknown")
  workspace = TricgWorkspace{T,FC,Sm,Sn}(m, n, y, N⁻¹uₖ₋₁, N⁻¹uₖ, p, gy₂ₖ₋₁, gy₂ₖ, x, M⁻¹vₖ₋₁, M⁻¹vₖ, q, gx₂ₖ₋₁, gx₂ₖ, Δx, Δy, uₖ, vₖ, false, stats)
  return workspace
end

function TricgWorkspace(m::Integer, n::Integer, S::Type)
  TricgWorkspace(m, n, S, S)
end

function TricgWorkspace(A, b)
  m, n = size(A)
  S = ktypeof(b)
  TricgWorkspace(m, n, S)
end

function TricgWorkspace(A, b, c)
  m, n = size(A)
  Sm = ktypeof(b)
  Sn = ktypeof(c)
  TricgWorkspace(m, n, Sm, Sn)
end

"""
Workspace for the in-place methods [`trimr!`](@ref) and [`krylov_solve!`](@ref).

The following outer constructors can be used to initialize this workspace:

    workspace = TrimrWorkspace(m, n, Sm, Sn)
    workspace = TrimrWorkspace(m, n, S)
    workspace = TrimrWorkspace(A, b)
    workspace = TrimrWorkspace(A, b, c)
    workspace = TrimrWorkspace(kc::KrylovConstructor{Sm,Sn})

`m` and `n` denote the dimensions of the linear operator `A` passed to the in-place methods.
Since [`trimr`](@ref) supports rectangular linear operators, `m` and `n` can differ.
`Sm` and `Sn` are the storage types of the workspace vectors of length `m` and `n`, respectively.
If the same storage type can be used for both, a single type `S` may be provided, such as `Vector{Float64}`.

[`KrylovConstructor`](@ref) facilitates the allocation of vectors in the workspace if `Sm(undef, m)` and `Sn(undef, n)` are not available.
"""
mutable struct TrimrWorkspace{T,FC,Sm,Sn} <: _KrylovWorkspace{T,FC,Sm,Sn}
  m          :: Int
  n          :: Int
  y          :: Sn
  N⁻¹uₖ₋₁    :: Sn
  N⁻¹uₖ      :: Sn
  p          :: Sn
  gy₂ₖ₋₃     :: Sn
  gy₂ₖ₋₂     :: Sn
  gy₂ₖ₋₁     :: Sn
  gy₂ₖ       :: Sn
  x          :: Sm
  M⁻¹vₖ₋₁    :: Sm
  M⁻¹vₖ      :: Sm
  q          :: Sm
  gx₂ₖ₋₃     :: Sm
  gx₂ₖ₋₂     :: Sm
  gx₂ₖ₋₁     :: Sm
  gx₂ₖ       :: Sm
  Δx         :: Sm
  Δy         :: Sn
  uₖ         :: Sn
  vₖ         :: Sm
  warm_start :: Bool
  stats      :: SimpleStats{T}
end

function TrimrWorkspace(kc::KrylovConstructor{Sm,Sn}) where {Sm,Sn}
  FC      = eltype(Sm)
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
  stats = SimpleStats(0, false, false, false, 0, T[], T[], T[], 0.0, "unknown")
  workspace = TrimrWorkspace{T,FC,Sm,Sn}(m, n, y, N⁻¹uₖ₋₁, N⁻¹uₖ, p, gy₂ₖ₋₃, gy₂ₖ₋₂, gy₂ₖ₋₁, gy₂ₖ, x, M⁻¹vₖ₋₁, M⁻¹vₖ, q, gx₂ₖ₋₃, gx₂ₖ₋₂, gx₂ₖ₋₁, gx₂ₖ, Δx, Δy, uₖ, vₖ, false, stats)
  return workspace
end

function TrimrWorkspace(m::Integer, n::Integer, Sm::Type, Sn::Type)
  FC      = eltype(Sm)
  T       = real(FC)
  y       = Sn(undef, n)
  N⁻¹uₖ₋₁ = Sn(undef, n)
  N⁻¹uₖ   = Sn(undef, n)
  p       = Sn(undef, n)
  gy₂ₖ₋₃  = Sn(undef, n)
  gy₂ₖ₋₂  = Sn(undef, n)
  gy₂ₖ₋₁  = Sn(undef, n)
  gy₂ₖ    = Sn(undef, n)
  x       = Sm(undef, m)
  M⁻¹vₖ₋₁ = Sm(undef, m)
  M⁻¹vₖ   = Sm(undef, m)
  q       = Sm(undef, m)
  gx₂ₖ₋₃  = Sm(undef, m)
  gx₂ₖ₋₂  = Sm(undef, m)
  gx₂ₖ₋₁  = Sm(undef, m)
  gx₂ₖ    = Sm(undef, m)
  Δx      = Sm(undef, 0)
  Δy      = Sn(undef, 0)
  uₖ      = Sn(undef, 0)
  vₖ      = Sm(undef, 0)
  Sm = isconcretetype(Sm) ? Sm : typeof(x)
  Sn = isconcretetype(Sn) ? Sn : typeof(y)
  stats = SimpleStats(0, false, false, false, 0, T[], T[], T[], 0.0, "unknown")
  workspace = TrimrWorkspace{T,FC,Sm,Sn}(m, n, y, N⁻¹uₖ₋₁, N⁻¹uₖ, p, gy₂ₖ₋₃, gy₂ₖ₋₂, gy₂ₖ₋₁, gy₂ₖ, x, M⁻¹vₖ₋₁, M⁻¹vₖ, q, gx₂ₖ₋₃, gx₂ₖ₋₂, gx₂ₖ₋₁, gx₂ₖ, Δx, Δy, uₖ, vₖ, false, stats)
  return workspace
end

function TrimrWorkspace(m::Integer, n::Integer, S::Type)
  TrimrWorkspace(m, n, S, S)
end

function TrimrWorkspace(A, b)
  m, n = size(A)
  S = ktypeof(b)
  TrimrWorkspace(m, n, S)
end

function TrimrWorkspace(A, b, c)
  m, n = size(A)
  Sm = ktypeof(b)
  Sn = ktypeof(c)
  TrimrWorkspace(m, n, Sm, Sn)
end

"""
Workspace for the in-place methods [`trilqr!`](@ref) and [`krylov_solve!`](@ref).

The following outer constructors can be used to initialize this workspace:

    workspace = TrilqrWorkspace(m, n, Sm, Sn)
    workspace = TrilqrWorkspace(m, n, S)
    workspace = TrilqrWorkspace(A, b)
    workspace = TrilqrWorkspace(A, b, c)
    workspace = TrilqrWorkspace(kc::KrylovConstructor{Sm,Sn})

`m` and `n` denote the dimensions of the linear operator `A` passed to the in-place methods.
Since [`trilqr`](@ref) supports rectangular linear operators, `m` and `n` can differ.
`Sm` and `Sn` are the storage types of the workspace vectors of length `m` and `n`, respectively.
If the same storage type can be used for both, a single type `S` may be provided, such as `Vector{Float64}`.

[`KrylovConstructor`](@ref) facilitates the allocation of vectors in the workspace if `Sm(undef, m)` and `Sn(undef, n)` are not available.
"""
mutable struct TrilqrWorkspace{T,FC,Sm,Sn} <: _KrylovWorkspace{T,FC,Sm,Sn}
  m          :: Int
  n          :: Int
  uₖ₋₁       :: Sn
  uₖ         :: Sn
  p          :: Sn
  d̅          :: Sn
  Δx         :: Sn
  x          :: Sn
  vₖ₋₁       :: Sm
  vₖ         :: Sm
  q          :: Sm
  Δy         :: Sm
  y          :: Sm
  wₖ₋₃       :: Sm
  wₖ₋₂       :: Sm
  warm_start :: Bool
  stats      :: AdjointStats{T}
end

function TrilqrWorkspace(kc::KrylovConstructor{Sm,Sn}) where {Sm,Sn}
  FC   = eltype(Sm)
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
  workspace = TrilqrWorkspace{T,FC,Sm,Sn}(m, n, uₖ₋₁, uₖ, p, d̅, Δx, x, vₖ₋₁, vₖ, q, Δy, y, wₖ₋₃, wₖ₋₂, false, stats)
  return workspace
end

function TrilqrWorkspace(m::Integer, n::Integer, Sm::Type, Sn::Type)
  FC   = eltype(Sm)
  T    = real(FC)
  uₖ₋₁ = Sn(undef, n)
  uₖ   = Sn(undef, n)
  p    = Sn(undef, n)
  d̅    = Sn(undef, n)
  Δx   = Sn(undef, 0)
  x    = Sn(undef, n)
  vₖ₋₁ = Sm(undef, m)
  vₖ   = Sm(undef, m)
  q    = Sm(undef, m)
  Δy   = Sm(undef, 0)
  y    = Sm(undef, m)
  wₖ₋₃ = Sm(undef, m)
  wₖ₋₂ = Sm(undef, m)
  Sm = isconcretetype(Sm) ? Sm : typeof(y)
  Sn = isconcretetype(Sn) ? Sn : typeof(x)
  stats = AdjointStats(0, false, false, T[], T[], 0.0, "unknown")
  workspace = TrilqrWorkspace{T,FC,Sm,Sn}(m, n, uₖ₋₁, uₖ, p, d̅, Δx, x, vₖ₋₁, vₖ, q, Δy, y, wₖ₋₃, wₖ₋₂, false, stats)
  return workspace
end

function TrilqrWorkspace(m::Integer, n::Integer, S::Type)
  TrilqrWorkspace(m, n, S, S)
end

function TrilqrWorkspace(A, b)
  m, n = size(A)
  S = ktypeof(b)
  TrilqrWorkspace(m, n, S)
end

function TrilqrWorkspace(A, b, c)
  m, n = size(A)
  Sm = ktypeof(b)
  Sn = ktypeof(c)
  TrilqrWorkspace(m, n, Sm, Sn)
end

"""
Workspace for the in-place methods [`cgs!`](@ref) and [`krylov_solve!`](@ref).

The following outer constructors can be used to initialize this workspace:s

    workspace = CgsWorkspace(m, n, S)
    workspace = CgsWorkspace(A, b)
    workspace = CgsWorkspace(kc::KrylovConstructor{S,S})

`m` and `n` denote the dimensions of the linear operator `A` passed to the in-place methods.
Since [`cgs`](@ref) only supports square linear operators, `m` and `n` must be equal.
`S` is the storage type of the vectors in the workspace, such as `Vector{Float64}`.

[`KrylovConstructor`](@ref) facilitates the allocation of vectors in the workspace if `S(undef, n)` is not available.
"""
mutable struct CgsWorkspace{T,FC,S} <: _KrylovWorkspace{T,FC,S,S}
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

function CgsWorkspace(kc::KrylovConstructor{S,S}) where S
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
  stats = SimpleStats(0, false, false, false, 0, T[], T[], T[], 0.0, "unknown")
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
  stats = SimpleStats(0, false, false, false, 0, T[], T[], T[], 0.0, "unknown")
  workspace = CgsWorkspace{T,FC,S}(m, n, Δx, x, r, u, p, q, ts, yz, vw, false, stats)
  return workspace
end

function CgsWorkspace(A, b)
  m, n = size(A)
  S = ktypeof(b)
  CgsWorkspace(m, n, S)
end

"""
Workspace for the in-place methods [`bicgstab!`](@ref) and [`krylov_solve!`](@ref).

The following outer constructors can be used to initialize this workspace:

    workspace = BicgstabWorkspace(m, n, S)
    workspace = BicgstabWorkspace(A, b)
    workspace = BicgstabWorkspace(kc::KrylovConstructor{S,S})

`m` and `n` denote the dimensions of the linear operator `A` passed to the in-place methods.
Since [`bicgstab`](@ref) only supports square linear operators, `m` and `n` must be equal.
`S` is the storage type of the vectors in the workspace, such as `Vector{Float64}`.

[`KrylovConstructor`](@ref) facilitates the allocation of vectors in the workspace if `S(undef, n)` is not available.
"""
mutable struct BicgstabWorkspace{T,FC,S} <: _KrylovWorkspace{T,FC,S,S}
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

function BicgstabWorkspace(kc::KrylovConstructor{S,S}) where S
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
  stats = SimpleStats(0, false, false, false, 0, T[], T[], T[], 0.0, "unknown")
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
  stats = SimpleStats(0, false, false, false, 0, T[], T[], T[], 0.0, "unknown")
  workspace = BicgstabWorkspace{T,FC,S}(m, n, Δx, x, r, p, v, s, qd, yz, t, false, stats)
  return workspace
end

function BicgstabWorkspace(A, b)
  m, n = size(A)
  S = ktypeof(b)
  BicgstabWorkspace(m, n, S)
end

"""
Workspace for the in-place methods [`bilq!`](@ref) and [`krylov_solve!`](@ref).

The following outer constructors can be used to initialize this workspace:

    workspace = BilqWorkspace(m, n, S)
    workspace = BilqWorkspace(A, b)
    workspace = BilqWorkspace(kc::KrylovConstructor{S,S})

`m` and `n` denote the dimensions of the linear operator `A` passed to the in-place methods.
Since [`bilq`](@ref) only supports square linear operators, `m` and `n` must be equal.
`S` is the storage type of the vectors in the workspace, such as `Vector{Float64}`.

[`KrylovConstructor`](@ref) facilitates the allocation of vectors in the workspace if `S(undef, n)` is not available.
"""
mutable struct BilqWorkspace{T,FC,S} <: _KrylovWorkspace{T,FC,S,S}
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

function BilqWorkspace(kc::KrylovConstructor{S,S}) where S
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
  stats = SimpleStats(0, false, false, false, 0, T[], T[], T[], 0.0, "unknown")
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
  stats = SimpleStats(0, false, false, false, 0, T[], T[], T[], 0.0, "unknown")
  workspace = BilqWorkspace{T,FC,S}(m, n, uₖ₋₁, uₖ, q, vₖ₋₁, vₖ, p, Δx, x, d̅, t, s, false, stats)
  return workspace
end

function BilqWorkspace(A, b)
  m, n = size(A)
  S = ktypeof(b)
  BilqWorkspace(m, n, S)
end

"""
Workspace for the in-place methods [`qmr!`](@ref) and [`krylov_solve!`](@ref).

The following outer constructors can be used to initialize this workspace:

    workspace = QmrWorkspace(m, n, S)
    workspace = QmrWorkspace(A, b)
    workspace = QmrWorkspace(kc::KrylovConstructor{S,S})

`m` and `n` denote the dimensions of the linear operator `A` passed to the in-place methods.
Since [`qmr`](@ref) only supports square linear operators, `m` and `n` must be equal.
`S` is the storage type of the vectors in the workspace, such as `Vector{Float64}`.

[`KrylovConstructor`](@ref) facilitates the allocation of vectors in the workspace if `S(undef, n)` is not available.
"""
mutable struct QmrWorkspace{T,FC,S} <: _KrylovWorkspace{T,FC,S,S}
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

function QmrWorkspace(kc::KrylovConstructor{S,S}) where S
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
  stats = SimpleStats(0, false, false, false, 0, T[], T[], T[], 0.0, "unknown")
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
  stats = SimpleStats(0, false, false, false, 0, T[], T[], T[], 0.0, "unknown")
  workspace = QmrWorkspace{T,FC,S}(m, n, uₖ₋₁, uₖ, q, vₖ₋₁, vₖ, p, Δx, x, wₖ₋₂, wₖ₋₁, t, s, false, stats)
  return workspace
end

function QmrWorkspace(A, b)
  m, n = size(A)
  S = ktypeof(b)
  QmrWorkspace(m, n, S)
end

"""
Workspace for the in-place methods [`bilqr!`](@ref) and [`krylov_solve!`](@ref).

The following outer constructors can be used to initialize this workspace:

    workspace = BilqrWorkspace(m, n, S)
    workspace = BilqrWorkspace(A, b)
    workspace = BilqrWorkspace(A, b, c)
    workspace = BilqrWorkspace(kc::KrylovConstructor{S,S})

`m` and `n` denote the dimensions of the linear operator `A` passed to the in-place methods.
Since [`bilqr`](@ref) only supports square linear operators, `m` and `n` must be equal.
`S` is the storage type of the vectors in the workspace, such as `Vector{Float64}`.

[`KrylovConstructor`](@ref) facilitates the allocation of vectors in the workspace if `S(undef, n)` is not available.
"""
mutable struct BilqrWorkspace{T,FC,S} <: _KrylovWorkspace{T,FC,S,S}
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

function BilqrWorkspace(kc::KrylovConstructor{S,S}) where S
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

function BilqrWorkspace(A, b, c)
  m, n = size(A)
  S = ktypeof(b)
  BilqrWorkspace(m, n, S)
end

"""
Workspace for the in-place methods [`cgls!`](@ref) and [`krylov_solve!`](@ref).

The following outer constructors can be used to initialize this workspace:

    workspace = CglsWorkspace(m, n, Sm, Sn)
    workspace = CglsWorkspace(m, n, S)
    workspace = CglsWorkspace(A, b)
    workspace = CglsWorkspace(kc::KrylovConstructor{Sm,Sn})

`m` and `n` denote the dimensions of the linear operator `A` passed to the in-place methods.
Since [`cgls`](@ref) supports rectangular linear operators, `m` and `n` can differ.
`Sm` and `Sn` are the storage types of the workspace vectors of length `m` and `n`, respectively.
If the same storage type can be used for both, a single type `S` may be provided, such as `Vector{Float64}`.

[`KrylovConstructor`](@ref) facilitates the allocation of vectors in the workspace if `Sm(undef, m)` and `Sn(undef, n)` are not available.
"""
mutable struct CglsWorkspace{T,FC,Sm,Sn} <: _KrylovWorkspace{T,FC,Sm,Sn}
  m     :: Int
  n     :: Int
  x     :: Sn
  p     :: Sn
  s     :: Sn
  r     :: Sm
  q     :: Sm
  Mr    :: Sm
  stats :: SimpleStats{T}
end

function CglsWorkspace(kc::KrylovConstructor{Sm,Sn}) where {Sm,Sn}
  FC = eltype(Sm)
  T  = real(FC)
  m  = length(kc.vm)
  n  = length(kc.vn)
  x  = similar(kc.vn)
  p  = similar(kc.vn)
  s  = similar(kc.vn)
  r  = similar(kc.vm)
  q  = similar(kc.vm)
  Mr = similar(kc.vm_empty)
  stats = SimpleStats(0, false, false, false, 0, T[], T[], T[], 0.0, "unknown")
  workspace = CglsWorkspace{T,FC,Sm,Sn}(m, n, x, p, s, r, q, Mr, stats)
  return workspace
end

function CglsWorkspace(m::Integer, n::Integer, Sm::Type, Sn::Type)
  FC = eltype(Sm)
  T  = real(FC)
  x  = Sn(undef, n)
  p  = Sn(undef, n)
  s  = Sn(undef, n)
  r  = Sm(undef, m)
  q  = Sm(undef, m)
  Mr = Sm(undef, 0)
  Sm = isconcretetype(Sm) ? Sm : typeof(r)
  Sn = isconcretetype(Sn) ? Sn : typeof(x)
  stats = SimpleStats(0, false, false, false, 0, T[], T[], T[], 0.0, "unknown")
  workspace = CglsWorkspace{T,FC,Sm,Sn}(m, n, x, p, s, r, q, Mr, stats)
  return workspace
end

function CglsWorkspace(m::Integer, n::Integer, S::Type)
  CglsWorkspace(m, n, S, S)
end

function CglsWorkspace(A, b)
  m, n = size(A)
  S = ktypeof(b)
  CglsWorkspace(m, n, S)
end

"""
Workspace for the in-place methods [`cgls_lanczos_shift!`](@ref) and [`krylov_solve!`](@ref).

The following outer constructors can be used to initialize this workspace:

    workspace = CglsLanczosShiftWorkspace(m, n, nshifts, Sm, Sn)
    workspace = CglsLanczosShiftWorkspace(m, n, nshifts, S)
    workspace = CglsLanczosShiftWorkspace(A, b, nshifts)
    workspace = CglsLanczosShiftWorkspace(kc::KrylovConstructor{Sm,Sn}, nshifts)

`m` and `n` denote the dimensions of the linear operator `A` passed to the in-place methods.
`nshifts` denotes the length of the vector `shifts` passed to the in-place methods.
Since [`cgls_lanczos_shift`](@ref) supports rectangular linear operators, `m` and `n` can differ.
`Sm` and `Sn` are the storage types of the workspace vectors of length `m` and `n`, respectively.
If the same storage type can be used for both, a single type `S` may be provided, such as `Vector{Float64}`.

[`KrylovConstructor`](@ref) facilitates the allocation of vectors in the workspace if `Sm(undef, m)` and `Sn(undef, n)` are not available.
"""
mutable struct CglsLanczosShiftWorkspace{T,FC,Sm,Sn} <: _KrylovWorkspace{T,FC,Sm,Sn}
  m         :: Int
  n         :: Int
  nshifts   :: Int
  Mv        :: Sn
  u_prev    :: Sm
  u_next    :: Sm
  u         :: Sm
  v         :: Sn
  x         :: Vector{Sn}
  p         :: Vector{Sn}
  σ         :: Vector{T}
  δhat      :: Vector{T}
  ω         :: Vector{T}
  γ         :: Vector{T}
  rNorms    :: Vector{T}
  converged :: BitVector
  not_cv    :: BitVector
  stats     :: LanczosShiftStats{T}
end

function CglsLanczosShiftWorkspace(kc::KrylovConstructor{Sm,Sn}, nshifts::Integer) where {Sm,Sn}
  FC         = eltype(Sm)
  T          = real(FC)
  m          = length(kc.vm)
  n          = length(kc.vn)
  Mv         = similar(kc.vn)
  u_prev     = similar(kc.vm)
  u_next     = similar(kc.vm)
  u          = similar(kc.vm)
  v          = similar(kc.vn_empty)
  x          = Sn[similar(kc.vn) for i = 1 : nshifts]
  p          = Sn[similar(kc.vn) for i = 1 : nshifts]
  σ          = Vector{T}(undef, nshifts)
  δhat       = Vector{T}(undef, nshifts)
  ω          = Vector{T}(undef, nshifts)
  γ          = Vector{T}(undef, nshifts)
  rNorms     = Vector{T}(undef, nshifts)
  indefinite = BitVector(undef, nshifts)
  converged  = BitVector(undef, nshifts)
  not_cv     = BitVector(undef, nshifts)
  stats = LanczosShiftStats(0, false, Vector{T}[T[] for i = 1 : nshifts], indefinite, T(NaN), T(NaN), 0.0, "unknown")
  workspace = CglsLanczosShiftWorkspace{T,FC,Sm,Sn}(m, n, nshifts, Mv, u_prev, u_next, u, v, x, p, σ, δhat, ω, γ, rNorms, converged, not_cv, stats)
  return workspace
end

function CglsLanczosShiftWorkspace(m::Integer, n::Integer, nshifts::Integer, Sm::Type, Sn::Type)
  FC         = eltype(Sm)
  T          = real(FC)
  Mv         = Sn(undef, n)
  u_prev     = Sm(undef, m)
  u_next     = Sm(undef, m)
  u          = Sm(undef, m)
  v          = Sn(undef, 0)
  x          = Sn[Sn(undef, n) for i = 1 : nshifts]
  p          = Sn[Sn(undef, n) for i = 1 : nshifts]
  σ          = Vector{T}(undef, nshifts)
  δhat       = Vector{T}(undef, nshifts)
  ω          = Vector{T}(undef, nshifts)
  γ          = Vector{T}(undef, nshifts)
  rNorms     = Vector{T}(undef, nshifts)
  indefinite = BitVector(undef, nshifts)
  converged  = BitVector(undef, nshifts)
  not_cv     = BitVector(undef, nshifts)
  Sm = isconcretetype(Sm) ? Sm : typeof(u)
  Sn = isconcretetype(Sn) ? Sn : typeof(Mv)
  stats = LanczosShiftStats(0, false, Vector{T}[T[] for i = 1 : nshifts], indefinite, T(NaN), T(NaN), 0.0, "unknown")
  workspace = CglsLanczosShiftWorkspace{T,FC,Sm,Sn}(m, n, nshifts, Mv, u_prev, u_next, u, v, x, p, σ, δhat, ω, γ, rNorms, converged, not_cv, stats)
  return workspace
end

function CglsLanczosShiftWorkspace(m::Integer, n::Integer, nshifts::Integer, S::Type)
  CglsLanczosShiftWorkspace(m, n, nshifts, S, S)
end

function CglsLanczosShiftWorkspace(A, b, nshifts::Integer)
  m, n = size(A)
  S = ktypeof(b)
  CglsLanczosShiftWorkspace(m, n, nshifts, S)
end

"""
Workspace for the in-place methods [`crls!`](@ref) and [`krylov_solve!`](@ref).

The following outer constructors can be used to initialize this workspace:

    workspace = CrlsWorkspace(m, n, Sm, Sn)
    workspace = CrlsWorkspace(m, n, S)
    workspace = CrlsWorkspace(A, b)
    workspace = CrlsWorkspace(kc::KrylovConstructor{Sm,Sn})

`m` and `n` denote the dimensions of the linear operator `A` passed to the in-place methods.
Since [`crls`](@ref) supports rectangular linear operators, `m` and `n` can differ.
`Sm` and `Sn` are the storage types of the workspace vectors of length `m` and `n`, respectively.
If the same storage type can be used for both, a single type `S` may be provided, such as `Vector{Float64}`.

[`KrylovConstructor`](@ref) facilitates the allocation of vectors in the workspace if `Sm(undef, m)` and `Sn(undef, n)` are not available.
"""
mutable struct CrlsWorkspace{T,FC,Sm,Sn} <: _KrylovWorkspace{T,FC,Sm,Sn}
  m     :: Int
  n     :: Int
  x     :: Sn
  p     :: Sn
  Ar    :: Sn
  q     :: Sn
  r     :: Sm
  Ap    :: Sm
  s     :: Sm
  Ms    :: Sm
  stats :: SimpleStats{T}
end

function CrlsWorkspace(kc::KrylovConstructor{Sm,Sn}) where {Sm,Sn}
  FC = eltype(Sm)
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
  stats = SimpleStats(0, false, false, false, 0, T[], T[], T[], 0.0, "unknown")
  workspace = CrlsWorkspace{T,FC,Sm,Sn}(m, n, x, p, Ar, q, r, Ap, s, Ms, stats)
  return workspace
end

function CrlsWorkspace(m::Integer, n::Integer, Sm::Type, Sn::Type)
  FC = eltype(Sm)
  T  = real(FC)
  x  = Sn(undef, n)
  p  = Sn(undef, n)
  Ar = Sn(undef, n)
  q  = Sn(undef, n)
  r  = Sm(undef, m)
  Ap = Sm(undef, m)
  s  = Sm(undef, m)
  Ms = Sm(undef, 0)
  Sm = isconcretetype(Sm) ? Sm : typeof(r)
  Sn = isconcretetype(Sn) ? Sn : typeof(x)
  stats = SimpleStats(0, false, false, false, 0, T[], T[], T[], 0.0, "unknown")
  workspace = CrlsWorkspace{T,FC,Sm,Sn}(m, n, x, p, Ar, q, r, Ap, s, Ms, stats)
  return workspace
end

function CrlsWorkspace(m::Integer, n::Integer, S::Type)
  CrlsWorkspace(m, n, S, S)
end

function CrlsWorkspace(A, b)
  m, n = size(A)
  S = ktypeof(b)
  CrlsWorkspace(m, n, S)
end

"""
Workspace for the in-place methods [`cgne!`](@ref) and [`krylov_solve!`](@ref).

The following outer constructors can be used to initialize this workspace:

    workspace = CgneWorkspace(m, n, Sm, Sn)
    workspace = CgneWorkspace(m, n, S)
    workspace = CgneWorkspace(A, b)
    workspace = CgneWorkspace(kc::KrylovConstructor{Sm,Sn})

`m` and `n` denote the dimensions of the linear operator `A` passed to the in-place methods.
Since [`cgne`](@ref) supports rectangular linear operators, `m` and `n` can differ.
`Sm` and `Sn` are the storage types of the workspace vectors of length `m` and `n`, respectively.
If the same storage type can be used for both, a single type `S` may be provided, such as `Vector{Float64}`.

[`KrylovConstructor`](@ref) facilitates the allocation of vectors in the workspace if `Sm(undef, m)` and `Sn(undef, n)` are not available.
"""
mutable struct CgneWorkspace{T,FC,Sm,Sn} <: _KrylovWorkspace{T,FC,Sm,Sn}
  m     :: Int
  n     :: Int
  x     :: Sn
  p     :: Sn
  Aᴴz   :: Sn
  r     :: Sm
  q     :: Sm
  s     :: Sm
  z     :: Sm
  stats :: SimpleStats{T}
end

function CgneWorkspace(kc::KrylovConstructor{Sm,Sn}) where {Sm,Sn}
  FC  = eltype(Sm)
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
  stats = SimpleStats(0, false, false, false, 0, T[], T[], T[], 0.0, "unknown")
  workspace = CgneWorkspace{T,FC,Sm,Sn}(m, n, x, p, Aᴴz, r, q, s, z, stats)
  return workspace
end

function CgneWorkspace(m::Integer, n::Integer, Sm::Type, Sn::Type)
  FC  = eltype(Sm)
  T   = real(FC)
  x   = Sn(undef, n)
  p   = Sn(undef, n)
  Aᴴz = Sn(undef, n)
  r   = Sm(undef, m)
  q   = Sm(undef, m)
  s   = Sm(undef, 0)
  z   = Sm(undef, 0)
  Sm = isconcretetype(Sm) ? Sm : typeof(r)
  Sn = isconcretetype(Sn) ? Sn : typeof(x)
  stats = SimpleStats(0, false, false, false, 0, T[], T[], T[], 0.0, "unknown")
  workspace = CgneWorkspace{T,FC,Sm,Sn}(m, n, x, p, Aᴴz, r, q, s, z, stats)
  return workspace
end

function CgneWorkspace(m::Integer, n::Integer, S::Type)
  CgneWorkspace(m, n, S, S)
end

function CgneWorkspace(A, b)
  m, n = size(A)
  S = ktypeof(b)
  CgneWorkspace(m, n, S)
end

"""
Workspace for the in-place methods [`crmr!`](@ref) and [`krylov_solve!`](@ref).

The following outer constructors can be used to initialize this workspace:

    workspace = CrmrWorkspace(m, n, Sm, Sn)
    workspace = CrmrWorkspace(m, n, S)
    workspace = CrmrWorkspace(A, b)
    workspace = CrmrWorkspace(kc::KrylovConstructor{Sm,Sn})

`m` and `n` denote the dimensions of the linear operator `A` passed to the in-place methods.
Since [`crmr`](@ref) supports rectangular linear operators, `m` and `n` can differ.
`Sm` and `Sn` are the storage types of the workspace vectors of length `m` and `n`, respectively.
If the same storage type can be used for both, a single type `S` may be provided, such as `Vector{Float64}`.

[`KrylovConstructor`](@ref) facilitates the allocation of vectors in the workspace if `Sm(undef, m)` and `Sn(undef, n)` are not available.
"""
mutable struct CrmrWorkspace{T,FC,Sm,Sn} <: _KrylovWorkspace{T,FC,Sm,Sn}
  m     :: Int
  n     :: Int
  x     :: Sn
  p     :: Sn
  Aᴴr   :: Sn
  r     :: Sm
  q     :: Sm
  Nq    :: Sm
  s     :: Sm
  stats :: SimpleStats{T}
end

function CrmrWorkspace(kc::KrylovConstructor{Sm,Sn}) where {Sm,Sn}
  FC  = eltype(Sm)
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
  stats = SimpleStats(0, false, false, false, 0, T[], T[], T[], 0.0, "unknown")
  workspace = CrmrWorkspace{T,FC,Sm,Sn}(m, n, x, p, Aᴴr, r, q, Nq, s, stats)
  return workspace
end

function CrmrWorkspace(m::Integer, n::Integer, Sm::Type, Sn::Type)
  FC  = eltype(Sm)
  T   = real(FC)
  x   = Sn(undef, n)
  p   = Sn(undef, n)
  Aᴴr = Sn(undef, n)
  r   = Sm(undef, m)
  q   = Sm(undef, m)
  Nq  = Sm(undef, 0)
  s   = Sm(undef, 0)
  Sm = isconcretetype(Sm) ? Sm : typeof(r)
  Sn = isconcretetype(Sn) ? Sn : typeof(x)
  stats = SimpleStats(0, false, false, false, 0, T[], T[], T[], 0.0, "unknown")
  workspace = CrmrWorkspace{T,FC,Sm,Sn}(m, n, x, p, Aᴴr, r, q, Nq, s, stats)
  return workspace
end

function CrmrWorkspace(m::Integer, n::Integer, S::Type)
  CrmrWorkspace(m, n, S, S)
end
 
function CrmrWorkspace(A, b)
  m, n = size(A)
  S = ktypeof(b)
  CrmrWorkspace(m, n, S)
end

"""
Workspace for the in-place methods [`lslq!`](@ref) and [`krylov_solve!`](@ref).

The following outer constructors can be used to initialize this workspace:

    workspace = LslqWorkspace(m, n, Sm, Sn)
    workspace = LslqWorkspace(m, n, S)
    workspace = LslqWorkspace(A, b)
    workspace = LslqWorkspace(kc::KrylovConstructor{Sm,Sn})

`m` and `n` denote the dimensions of the linear operator `A` passed to the in-place methods.
Since [`lslq`](@ref) supports rectangular linear operators, `m` and `n` can differ.
`Sm` and `Sn` are the storage types of the workspace vectors of length `m` and `n`, respectively.
If the same storage type can be used for both, a single type `S` may be provided, such as `Vector{Float64}`.

[`KrylovConstructor`](@ref) facilitates the allocation of vectors in the workspace if `Sm(undef, m)` and `Sn(undef, n)` are not available.
"""
mutable struct LslqWorkspace{T,FC,Sm,Sn} <: _KrylovWorkspace{T,FC,Sm,Sn}
  m       :: Int
  n       :: Int
  x       :: Sn
  Nv      :: Sn
  Aᴴu     :: Sn
  w̄       :: Sn
  Mu      :: Sm
  Av      :: Sm
  u       :: Sm
  v       :: Sn
  err_vec :: Vector{T}
  stats   :: LSLQStats{T}
end

function LslqWorkspace(kc::KrylovConstructor{Sm,Sn}; window::Int = 5) where {Sm,Sn}
  FC  = eltype(Sm)
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
  workspace = LslqWorkspace{T,FC,Sm,Sn}(m, n, x, Nv, Aᴴu, w̄, Mu, Av, u, v, err_vec, stats)
  return workspace
end

function LslqWorkspace(m::Integer, n::Integer, Sm::Type, Sn::Type; window::Int = 5)
  FC  = eltype(Sm)
  T   = real(FC)
  x   = Sn(undef, n)
  Nv  = Sn(undef, n)
  Aᴴu = Sn(undef, n)
  w̄   = Sn(undef, n)
  Mu  = Sm(undef, m)
  Av  = Sm(undef, m)
  u   = Sm(undef, 0)
  v   = Sn(undef, 0)
  err_vec = zeros(T, window)
  Sm = isconcretetype(Sm) ? Sm : typeof(Av)
  Sn = isconcretetype(Sn) ? Sn : typeof(x)
  stats = LSLQStats(0, false, false, T[], T[], T[], false, T[], T[], 0.0, "unknown")
  workspace = LslqWorkspace{T,FC,Sm,Sn}(m, n, x, Nv, Aᴴu, w̄, Mu, Av, u, v, err_vec, stats)
  return workspace
end

function LslqWorkspace(m::Integer, n::Integer, S::Type; window::Int = 5)
  LslqWorkspace(m, n, S, S; window)
end

function LslqWorkspace(A, b; window::Int = 5)
  m, n = size(A)
  S = ktypeof(b)
  LslqWorkspace(m, n, S; window)
end

"""
Workspace for the in-place methods [`lsqr!`](@ref) and [`krylov_solve!`](@ref).

The following outer constructors can be used to initialize this workspace:

    workspace = LsqrWorkspace(m, n, Sm, Sn)
    workspace = LsqrWorkspace(m, n, S)
    workspace = LsqrWorkspace(A, b)
    workspace = LsqrWorkspace(kc::KrylovConstructor{Sm,Sn})

`m` and `n` denote the dimensions of the linear operator `A` passed to the in-place methods.
Since [`lsqr`](@ref) supports rectangular linear operators, `m` and `n` can differ.
`Sm` and `Sn` are the storage types of the workspace vectors of length `m` and `n`, respectively.
If the same storage type can be used for both, a single type `S` may be provided, such as `Vector{Float64}`.

[`KrylovConstructor`](@ref) facilitates the allocation of vectors in the workspace if `Sm(undef, m)` and `Sn(undef, n)` are not available.
"""
mutable struct LsqrWorkspace{T,FC,Sm,Sn} <: _KrylovWorkspace{T,FC,Sm,Sn}
  m       :: Int
  n       :: Int
  x       :: Sn
  Nv      :: Sn
  Aᴴu     :: Sn
  w       :: Sn
  Mu      :: Sm
  Av      :: Sm
  u       :: Sm
  v       :: Sn
  err_vec :: Vector{T}
  stats   :: SimpleStats{T}
end

function LsqrWorkspace(kc::KrylovConstructor{Sm,Sn}; window::Int = 5) where {Sm,Sn}
  FC  = eltype(Sm)
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
  stats = SimpleStats(0, false, false, false, 0, T[], T[], T[], 0.0, "unknown")
  workspace = LsqrWorkspace{T,FC,Sm,Sn}(m, n, x, Nv, Aᴴu, w, Mu, Av, u, v, err_vec, stats)
  return workspace
end

function LsqrWorkspace(m::Integer, n::Integer, Sm::Type, Sn::Type; window::Int = 5)
  FC  = eltype(Sm)
  T   = real(FC)
  x   = Sn(undef, n)
  Nv  = Sn(undef, n)
  Aᴴu = Sn(undef, n)
  w   = Sn(undef, n)
  Mu  = Sm(undef, m)
  Av  = Sm(undef, m)
  u   = Sm(undef, 0)
  v   = Sn(undef, 0)
  err_vec = zeros(T, window)
  Sm = isconcretetype(Sm) ? Sm : typeof(Av)
  Sn = isconcretetype(Sn) ? Sn : typeof(x)
  stats = SimpleStats(0, false, false, false, 0, T[], T[], T[], 0.0, "unknown")
  workspace = LsqrWorkspace{T,FC,Sm,Sn}(m, n, x, Nv, Aᴴu, w, Mu, Av, u, v, err_vec, stats)
  return workspace
end

function LsqrWorkspace(m::Integer, n::Integer, S::Type; window::Int = 5)
  LsqrWorkspace(m, n, S, S; window)
end

function LsqrWorkspace(A, b; window::Int = 5)
  m, n = size(A)
  S = ktypeof(b)
  LsqrWorkspace(m, n, S; window)
end

"""
Workspace for the in-place methods [`lsmr!`](@ref) and [`krylov_solve!`](@ref).

The following outer constructors can be used to initialize this workspace:

    workspace = LsmrWorkspace(m, n, Sm, Sn)
    workspace = LsmrWorkspace(m, n, S)
    workspace = LsmrWorkspace(A, b)
    workspace = LsmrWorkspace(kc::KrylovConstructor{Sm,Sn})

`m` and `n` denote the dimensions of the linear operator `A` passed to the in-place methods.
Since [`lsmr`](@ref) supports rectangular linear operators, `m` and `n` can differ.
`Sm` and `Sn` are the storage types of the workspace vectors of length `m` and `n`, respectively.
If the same storage type can be used for both, a single type `S` may be provided, such as `Vector{Float64}`.

[`KrylovConstructor`](@ref) facilitates the allocation of vectors in the workspace if `Sm(undef, m)` and `Sn(undef, n)` are not available.
"""
mutable struct LsmrWorkspace{T,FC,Sm,Sn} <: _KrylovWorkspace{T,FC,Sm,Sn}
  m       :: Int
  n       :: Int
  x       :: Sn
  Nv      :: Sn
  Aᴴu     :: Sn
  h       :: Sn
  hbar    :: Sn
  Mu      :: Sm
  Av      :: Sm
  u       :: Sm
  v       :: Sn
  err_vec :: Vector{T}
  stats   :: LsmrStats{T}
end

function LsmrWorkspace(kc::KrylovConstructor{Sm,Sn}; window::Int = 5) where {Sm,Sn}
  FC   = eltype(Sm)
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
  workspace = LsmrWorkspace{T,FC,Sm,Sn}(m, n, x, Nv, Aᴴu, h, hbar, Mu, Av, u, v, err_vec, stats)
  return workspace
end

function LsmrWorkspace(m::Integer, n::Integer, Sm::Type, Sn::Type; window::Int = 5)
  FC   = eltype(Sm)
  T    = real(FC)
  x    = Sn(undef, n)
  Nv   = Sn(undef, n)
  Aᴴu  = Sn(undef, n)
  h    = Sn(undef, n)
  hbar = Sn(undef, n)
  Mu   = Sm(undef, m)
  Av   = Sm(undef, m)
  u    = Sm(undef, 0)
  v    = Sn(undef, 0)
  err_vec = zeros(T, window)
  Sm = isconcretetype(Sm) ? Sm : typeof(Av)
  Sn = isconcretetype(Sn) ? Sn : typeof(x)
  stats = LsmrStats(0, false, false, T[], T[], zero(T), zero(T), zero(T), zero(T), zero(T), 0.0, "unknown")
  workspace = LsmrWorkspace{T,FC,Sm,Sn}(m, n, x, Nv, Aᴴu, h, hbar, Mu, Av, u, v, err_vec, stats)
  return workspace
end

function LsmrWorkspace(m::Integer, n::Integer, S::Type; window::Int = 5)
  LsmrWorkspace(m, n, S, S; window)
end

function LsmrWorkspace(A, b; window::Int = 5)
  m, n = size(A)
  S = ktypeof(b)
  LsmrWorkspace(m, n, S; window)
end

"""
Workspace for the in-place methods [`lnlq!`](@ref) and [`krylov_solve!`](@ref).

The following outer constructors can be used to initialize this workspace:

    workspace = LnlqWorkspace(m, n, Sm, Sn)
    workspace = LnlqWorkspace(m, n, S)
    workspace = LnlqWorkspace(A, b)
    workspace = LnlqWorkspace(kc::KrylovConstructor{Sm,Sn})

`m` and `n` denote the dimensions of the linear operator `A` passed to the in-place methods.
Since [`lnlq`](@ref) supports rectangular linear operators, `m` and `n` can differ.
`Sm` and `Sn` are the storage types of the workspace vectors of length `m` and `n`, respectively.
If the same storage type can be used for both, a single type `S` may be provided, such as `Vector{Float64}`.

[`KrylovConstructor`](@ref) facilitates the allocation of vectors in the workspace if `Sm(undef, m)` and `Sn(undef, n)` are not available.
"""
mutable struct LnlqWorkspace{T,FC,Sm,Sn} <: _KrylovWorkspace{T,FC,Sm,Sn}
  m     :: Int
  n     :: Int
  x     :: Sn
  Nv    :: Sn
  Aᴴu   :: Sn
  y     :: Sm
  w̄     :: Sm
  Mu    :: Sm
  Av    :: Sm
  u     :: Sm
  v     :: Sn
  q     :: Sn
  stats :: LNLQStats{T}
end

function LnlqWorkspace(kc::KrylovConstructor{Sm,Sn}) where {Sm,Sn}
  FC  = eltype(Sm)
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
  workspace = LnlqWorkspace{T,FC,Sm,Sn}(m, n, x, Nv, Aᴴu, y, w̄, Mu, Av, u, v, q, stats)
  return workspace
end

function LnlqWorkspace(m::Integer, n::Integer, Sm::Type, Sn::Type)
  FC  = eltype(Sm)
  T   = real(FC)
  x   = Sn(undef, n)
  Nv  = Sn(undef, n)
  Aᴴu = Sn(undef, n)
  y   = Sm(undef, m)
  w̄   = Sm(undef, m)
  Mu  = Sm(undef, m)
  Av  = Sm(undef, m)
  u   = Sm(undef, 0)
  v   = Sn(undef, 0)
  q   = Sn(undef, 0)
  Sm = isconcretetype(Sm) ? Sm : typeof(y)
  Sn = isconcretetype(Sn) ? Sn : typeof(x)
  stats = LNLQStats(0, false, T[], false, T[], T[], 0.0, "unknown")
  workspace = LnlqWorkspace{T,FC,Sm,Sn}(m, n, x, Nv, Aᴴu, y, w̄, Mu, Av, u, v, q, stats)
  return workspace
end

function LnlqWorkspace(m::Integer, n::Integer, S::Type)
  LnlqWorkspace(m, n, S, S)
end

function LnlqWorkspace(A, b)
  m, n = size(A)
  S = ktypeof(b)
  LnlqWorkspace(m, n, S)
end

"""
Workspace for the in-place methods [`craig!`](@ref) and [`krylov_solve!`](@ref).

The following outer constructors can be used to initialize this workspace:

    workspace = CraigWorkspace(m, n, Sm, Sn)
    workspace = CraigWorkspace(m, n, S)
    workspace = CraigWorkspace(A, b)
    workspace = CraigWorkspace(kc::KrylovConstructor{Sm,Sn})

`m` and `n` denote the dimensions of the linear operator `A` passed to the in-place methods.
Since [`craig`](@ref) supports rectangular linear operators, `m` and `n` can differ.
`Sm` and `Sn` are the storage types of the workspace vectors of length `m` and `n`, respectively.
If the same storage type can be used for both, a single type `S` may be provided, such as `Vector{Float64}`.

[`KrylovConstructor`](@ref) facilitates the allocation of vectors in the workspace if `Sm(undef, m)` and `Sn(undef, n)` are not available.
"""
mutable struct CraigWorkspace{T,FC,Sm,Sn} <: _KrylovWorkspace{T,FC,Sm,Sn}
  m     :: Int
  n     :: Int
  x     :: Sn
  Nv    :: Sn
  Aᴴu   :: Sn
  y     :: Sm
  w     :: Sm
  Mu    :: Sm
  Av    :: Sm
  u     :: Sm
  v     :: Sn
  w2    :: Sn
  stats :: SimpleStats{T}
end

function CraigWorkspace(kc::KrylovConstructor{Sm,Sn}) where {Sm,Sn}
  FC  = eltype(Sm)
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
  stats = SimpleStats(0, false, false, false, 0, T[], T[], T[], 0.0, "unknown")
  workspace = CraigWorkspace{T,FC,Sm,Sn}(m, n, x, Nv, Aᴴu, y, w, Mu, Av, u, v, w2, stats)
  return workspace
end

function CraigWorkspace(m::Integer, n::Integer, Sm::Type, Sn::Type)
  FC  = eltype(Sm)
  T   = real(FC)
  x   = Sn(undef, n)
  Nv  = Sn(undef, n)
  Aᴴu = Sn(undef, n)
  y   = Sm(undef, m)
  w   = Sm(undef, m)
  Mu  = Sm(undef, m)
  Av  = Sm(undef, m)
  u   = Sm(undef, 0)
  v   = Sn(undef, 0)
  w2  = Sn(undef, 0)
  Sm = isconcretetype(Sm) ? Sm : typeof(y)
  Sn = isconcretetype(Sn) ? Sn : typeof(x)
  stats = SimpleStats(0, false, false, false, 0, T[], T[], T[], 0.0, "unknown")
  workspace = CraigWorkspace{T,FC,Sm,Sn}(m, n, x, Nv, Aᴴu, y, w, Mu, Av, u, v, w2, stats)
  return workspace
end

function CraigWorkspace(m::Integer, n::Integer, S::Type)
  CraigWorkspace(m, n, S, S)
end

function CraigWorkspace(A, b)
  m, n = size(A)
  S = ktypeof(b)
  CraigWorkspace(m, n, S)
end

"""
Workspace for the in-place methods [`craigmr!`](@ref) and [`krylov_solve!`](@ref).

The following outer constructors can be used to initialize this workspace:

    workspace = CraigmrWorkspace(m, n, S)
    workspace = CraigmrWorkspace(A, b)
    workspace = CraigmrWorkspace(kc::KrylovConstructor{Sm,Sn})

`m` and `n` denote the dimensions of the linear operator `A` passed to the in-place methods.
Since [`craigmr`](@ref) supports rectangular linear operators, `m` and `n` can differ.
`Sm` and `Sn` are the storage types of the workspace vectors of length `m` and `n`, respectively.
If the same storage type can be used for both, a single type `S` may be provided, such as `Vector{Float64}`.

[`KrylovConstructor`](@ref) facilitates the allocation of vectors in the workspace if `Sm(undef, m)` and `Sn(undef, n)` are not available.
"""
mutable struct CraigmrWorkspace{T,FC,Sm,Sn} <: _KrylovWorkspace{T,FC,Sm,Sn}
  m     :: Int
  n     :: Int
  x     :: Sn
  Nv    :: Sn
  Aᴴu   :: Sn
  d     :: Sm
  y     :: Sm
  Mu    :: Sm
  w     :: Sm
  wbar  :: Sm
  Av    :: Sm
  u     :: Sm
  v     :: Sn
  q     :: Sn
  stats :: SimpleStats{T}
end

function CraigmrWorkspace(kc::KrylovConstructor{Sm,Sn}) where {Sm,Sn}
  FC   = eltype(Sm)
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
  stats = SimpleStats(0, false, false, false, 0, T[], T[], T[], 0.0, "unknown")
  workspace = CraigmrWorkspace{T,FC,Sm,Sn}(m, n, x, Nv, Aᴴu, d, y, Mu, w, wbar, Av, u, v, q, stats)
  return workspace
end

function CraigmrWorkspace(m::Integer, n::Integer, Sm::Type, Sn::Type)
  FC   = eltype(Sm)
  T    = real(FC)
  x    = Sn(undef, n)
  Nv   = Sn(undef, n)
  Aᴴu  = Sn(undef, n)
  d    = Sn(undef, n)
  y    = Sm(undef, m)
  Mu   = Sm(undef, m)
  w    = Sm(undef, m)
  wbar = Sm(undef, m)
  Av   = Sm(undef, m)
  u    = Sm(undef, 0)
  v    = Sn(undef, 0)
  q    = Sn(undef, 0)
  Sm = isconcretetype(Sm) ? Sm : typeof(y)
  Sn = isconcretetype(Sn) ? Sn : typeof(x)
  stats = SimpleStats(0, false, false, false, 0, T[], T[], T[], 0.0, "unknown")
  workspace = CraigmrWorkspace{T,FC,Sm,Sn}(m, n, x, Nv, Aᴴu, d, y, Mu, w, wbar, Av, u, v, q, stats)
  return workspace
end

function CraigmrWorkspace(m::Integer, n::Integer, S::Type)
  CraigmrWorkspace(m, n, S, S)
end

function CraigmrWorkspace(A, b)
  m, n = size(A)
  S = ktypeof(b)
  CraigmrWorkspace(m, n, S)
end

"""
Workspace for the in-place methods [`gmres!`](@ref) and [`krylov_solve!`](@ref).

The following outer constructors can be used to initialize this workspace:

    workspace = GmresWorkspace(m, n, S; memory = 20)
    workspace = GmresWorkspace(A, b; memory = 20)
    workspace = GmresWorkspace(kc::KrylovConstructor{S,S}; memory = 20)

`m` and `n` denote the dimensions of the linear operator `A` passed to the in-place methods.
Since [`gmres`](@ref) only supports square linear operators, `m` and `n` must be equal.
`S` is the storage type of the vectors in the workspace, such as `Vector{Float64}`.
`memory` is set to `n` if the value given is larger than `n`.

[`KrylovConstructor`](@ref) facilitates the allocation of vectors in the workspace if `S(undef, n)` is not available.
"""
mutable struct GmresWorkspace{T,FC,S} <: _KrylovWorkspace{T,FC,S,S}
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

function GmresWorkspace(kc::KrylovConstructor{S,S}; memory::Int = 20) where S
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
  stats = SimpleStats(0, false, false, false, 0, T[], T[], T[], 0.0, "unknown")
  workspace = GmresWorkspace{T,FC,S}(m, n, Δx, x, w, p, q, V, c, s, z, R, false, 0, stats)
  return workspace
end

function GmresWorkspace(m::Integer, n::Integer, S::Type; memory::Int = 20)
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
  stats = SimpleStats(0, false, false, false, 0, T[], T[], T[], 0.0, "unknown")
  workspace = GmresWorkspace{T,FC,S}(m, n, Δx, x, w, p, q, V, c, s, z, R, false, 0, stats)
  return workspace
end

function GmresWorkspace(A, b; memory::Int = 20)
  m, n = size(A)
  S = ktypeof(b)
  GmresWorkspace(m, n, S; memory)
end

"""
Workspace for the in-place methods [`fgmres!`](@ref) and [`krylov_solve!`](@ref).

The following outer constructors can be used to initialize this workspace:

    workspace = FgmresWorkspace(m, n, S; memory = 20)
    workspace = FgmresWorkspace(A, b; memory = 20)
    workspace = FgmresWorkspace(kc::KrylovConstructor{S,S}; memory = 20)

`m` and `n` denote the dimensions of the linear operator `A` passed to the in-place methods.
Since [`fgmres`](@ref) only supports square linear operators, `m` and `n` must be equal.
`S` is the storage type of the vectors in the workspace, such as `Vector{Float64}`.
`memory` is set to `n` if the value given is larger than `n`.

[`KrylovConstructor`](@ref) facilitates the allocation of vectors in the workspace if `S(undef, n)` is not available.
"""
mutable struct FgmresWorkspace{T,FC,S} <: _KrylovWorkspace{T,FC,S,S}
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

function FgmresWorkspace(kc::KrylovConstructor{S,S}; memory::Int = 20) where S
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
  stats = SimpleStats(0, false, false, false, 0, T[], T[], T[], 0.0, "unknown")
  workspace = FgmresWorkspace{T,FC,S}(m, n, Δx, x, w, q, V, Z, c, s, z, R, false, 0, stats)
  return workspace
end

function FgmresWorkspace(m::Integer, n::Integer, S::Type; memory::Int = 20)
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
  stats = SimpleStats(0, false, false, false, 0, T[], T[], T[], 0.0, "unknown")
  workspace = FgmresWorkspace{T,FC,S}(m, n, Δx, x, w, q, V, Z, c, s, z, R, false, 0, stats)
  return workspace
end

function FgmresWorkspace(A, b; memory::Int = 20)
  m, n = size(A)
  S = ktypeof(b)
  FgmresWorkspace(m, n, S; memory)
end

"""
Workspace for the in-place methods [`fom!`](@ref) and [`krylov_solve!`](@ref).

The following outer constructors can be used to initialize this workspace:

    workspace = FomWorkspace(m, n, S; memory = 20)
    workspace = FomWorkspace(A, b; memory = 20)
    workspace = FomWorkspace(kc::KrylovConstructor{S,S}; memory = 20)

`m` and `n` denote the dimensions of the linear operator `A` passed to the in-place methods.
Since [`fom`](@ref) only supports square linear operators, `m` and `n` must be equal.
`S` is the storage type of the vectors in the workspace, such as `Vector{Float64}`.
`memory` is set to `n` if the value given is larger than `n`.

[`KrylovConstructor`](@ref) facilitates the allocation of vectors in the workspace if `S(undef, n)` is not available.
"""
mutable struct FomWorkspace{T,FC,S} <: _KrylovWorkspace{T,FC,S,S}
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

function FomWorkspace(kc::KrylovConstructor{S,S}; memory::Int = 20) where S
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
  stats = SimpleStats(0, false, false, false, 0, T[], T[], T[], 0.0, "unknown")
  workspace = FomWorkspace{T,FC,S}(m, n, Δx, x, w, p, q, V, l, z, U, false, stats)
  return workspace
end

function FomWorkspace(m::Integer, n::Integer, S::Type; memory::Int = 20)
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
  stats = SimpleStats(0, false, false, false, 0, T[], T[], T[], 0.0, "unknown")
  workspace = FomWorkspace{T,FC,S}(m, n, Δx, x, w, p, q, V, l, z, U, false, stats)
  return workspace
end

function FomWorkspace(A, b; memory::Int = 20)
  m, n = size(A)
  S = ktypeof(b)
  FomWorkspace(m, n, S; memory)
end

"""
Workspace for the in-place methods [`gpmr!`](@ref) and [`krylov_solve!`](@ref).

The following outer constructors can be used to initialize this workspace:

    workspace = GpmrWorkspace(m, n, Sm, Sn; memory = 20)
    workspace = GpmrWorkspace(m, n, S; memory = 20)
    workspace = GpmrWorkspace(A, b; memory = 20)
    workspace = GpmrWorkspace(A, b, c; memory = 20)
    workspace = GpmrWorkspace(A, B, b, c; memory = 20)
    workspace = GpmrWorkspace(kc::KrylovConstructor{Sm,Sn}; memory = 20)

`m` and `n` denote the dimensions of the linear operator `A` passed to the in-place methods.
Since [`gpmr`](@ref) supports rectangular linear operators, `m` and `n` can differ.
`Sm` and `Sn` are the storage types of the workspace vectors of length `m` and `n`, respectively.
If the same storage type can be used for both, a single type `S` may be provided, such as `Vector{Float64}`.
`memory` is set to `n + m` if the value given is larger than `n + m`.

[`KrylovConstructor`](@ref) facilitates the allocation of vectors in the workspace if `Sm(undef, m)` and `Sn(undef, n)` are not available.
"""
mutable struct GpmrWorkspace{T,FC,Sm,Sn} <: _KrylovWorkspace{T,FC,Sm,Sn}
  m          :: Int
  n          :: Int
  wA         :: Sn
  wB         :: Sm
  dA         :: Sm
  dB         :: Sn
  Δx         :: Sm
  Δy         :: Sn
  x          :: Sm
  y          :: Sn
  q          :: Sm
  p          :: Sn
  V          :: Vector{Sm}
  U          :: Vector{Sn}
  gs         :: Vector{FC}
  gc         :: Vector{T}
  zt         :: Vector{FC}
  R          :: Vector{FC}
  warm_start :: Bool
  stats      :: SimpleStats{T}
end

function GpmrWorkspace(kc::KrylovConstructor{Sm,Sn}; memory::Int = 20) where {Sm,Sn}
  FC = eltype(Sm)
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
  V  = Sm[similar(kc.vm) for i = 1 : memory]
  U  = Sn[similar(kc.vn) for i = 1 : memory]
  gs = Vector{FC}(undef, 4 * memory)
  gc = Vector{T}(undef, 4 * memory)
  zt = Vector{FC}(undef, 2 * memory)
  R  = Vector{FC}(undef, memory * (2 * memory + 1))
  stats = SimpleStats(0, false, false, false, 0, T[], T[], T[], 0.0, "unknown")
  workspace = GpmrWorkspace{T,FC,Sm,Sn}(m, n, wA, wB, dA, dB, Δx, Δy, x, y, q, p, V, U, gs, gc, zt, R, false, stats)
  return workspace
end

function GpmrWorkspace(m::Integer, n::Integer, Sm::Type, Sn::Type; memory::Int = 20)
  memory = min(n + m, memory)
  FC = eltype(Sm)
  T  = real(FC)
  wA = Sn(undef, 0)
  wB = Sm(undef, 0)
  dA = Sm(undef, m)
  dB = Sn(undef, n)
  Δx = Sm(undef, 0)
  Δy = Sn(undef, 0)
  x  = Sm(undef, m)
  y  = Sn(undef, n)
  q  = Sm(undef, 0)
  p  = Sn(undef, 0)
  V  = Sm[Sm(undef, m) for i = 1 : memory]
  U  = Sn[Sn(undef, n) for i = 1 : memory]
  gs = Vector{FC}(undef, 4 * memory)
  gc = Vector{T}(undef, 4 * memory)
  zt = Vector{FC}(undef, 2 * memory)
  R  = Vector{FC}(undef, memory * (2 * memory + 1))
  Sm = isconcretetype(Sm) ? Sm : typeof(x)
  Sn = isconcretetype(Sn) ? Sn : typeof(y)
  stats = SimpleStats(0, false, false, false, 0, T[], T[], T[], 0.0, "unknown")
  workspace = GpmrWorkspace{T,FC,Sm,Sn}(m, n, wA, wB, dA, dB, Δx, Δy, x, y, q, p, V, U, gs, gc, zt, R, false, stats)
  return workspace
end

function GpmrWorkspace(m::Integer, n::Integer, S::Type; memory::Int = 20)
  GpmrWorkspace(m, n, S, S; memory)
end

function GpmrWorkspace(A, b; memory::Int = 20)
  m, n = size(A)
  S = ktypeof(b)
  GpmrWorkspace(m, n, S; memory)
end

function GpmrWorkspace(A, b, c; memory::Int = 20)
  m, n = size(A)
  Sm = ktypeof(b)
  Sn = ktypeof(c)
  GpmrWorkspace(m, n, Sm, Sn; memory)
end

function GpmrWorkspace(A, B, b, c; memory::Int = 20)
  m, n = size(A)
  Sm = ktypeof(b)
  Sn = ktypeof(c)
  GpmrWorkspace(m, n, Sm, Sn; memory)
end
