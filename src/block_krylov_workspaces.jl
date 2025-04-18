export BlockKrylovWorkspace, BlockMinresWorkspace, BlockGmresWorkspace

"Abstract type for using block Krylov solvers in-place."
abstract type BlockKrylovWorkspace{T,FC,SV,SM} end

"""
Workspace for the in-place method [`block_minres!`](@ref).

The following outer constructors can be used to initialize this workspace:

    workspace = BlockMinresWorkspace(m, n, p, SV, SM)
    workspace = BlockMinresWorkspace(A, B)
"""
mutable struct BlockMinresWorkspace{T,FC,SV,SM} <: BlockKrylovWorkspace{T,FC,SV,SM}
  m          :: Int
  n          :: Int
  p          :: Int
  ΔX         :: SM
  X          :: SM
  P          :: SM
  Q          :: SM
  C          :: SM
  D          :: SM
  Φ          :: SM
  Vₖ₋₁       :: SM
  Vₖ         :: SM
  wₖ₋₂       :: SM
  wₖ₋₁       :: SM
  Hₖ₋₂       :: SM
  Hₖ₋₁       :: SM
  τₖ₋₂       :: SV
  τₖ₋₁       :: SV
  warm_start :: Bool
  stats      :: SimpleStats{T}
end

function BlockMinresWorkspace(m::Integer, n::Integer, p::Integer, SV::Type, SM::Type)
  FC   = eltype(SV)
  T    = real(FC)
  ΔX   = SM(undef, 0, 0)
  X    = SM(undef, n, p)
  P    = SM(undef, 0, 0)
  Q    = SM(undef, n, p)
  C    = SM(undef, p, p)
  D    = SM(undef, 2p, p)
  Φ    = SM(undef, p, p)
  Vₖ₋₁ = SM(undef, n, p)
  Vₖ   = SM(undef, n, p)
  wₖ₋₂ = SM(undef, n, p)
  wₖ₋₁ = SM(undef, n, p)
  Hₖ₋₂ = SM(undef, 2p, p)
  Hₖ₋₁ = SM(undef, 2p, p)
  τₖ₋₂ = SV(undef, p)
  τₖ₋₁ = SV(undef, p)
  SV = isconcretetype(SV) ? SV : typeof(τₖ₋₁)
  SM = isconcretetype(SM) ? SM : typeof(X)
  stats = SimpleStats(0, false, false, false, T[], T[], T[], 0.0, "unknown")
  workspace = BlockMinresWorkspace{T,FC,SV,SM}(m, n, p, ΔX, X, P, Q, C, D, Φ, Vₖ₋₁, Vₖ, wₖ₋₂, wₖ₋₁, Hₖ₋₂, Hₖ₋₁, τₖ₋₂, τₖ₋₁, false, stats)
  return workspace
end

function BlockMinresWorkspace(A, B)
  m, n = size(A)
  s, p = size(B)
  SM = typeof(B)
  SV = matrix_to_vector(SM)
  BlockMinresWorkspace(m, n, p, SV, SM)
end

"""
Workspace for the in-place method [`block_gmres!`](@ref).

The following outer constructors can be used to initialize this workspace:

    workspace = BlockGmresWorkspace(m, n, p, SV, SM; memory = 5)
    workspace = BlockGmresWorkspace(A, B; memory = 5)

`memory` is set to `div(n,p)` if the value given is larger than `div(n,p)`.
"""
mutable struct BlockGmresWorkspace{T,FC,SV,SM} <: BlockKrylovWorkspace{T,FC,SV,SM}
  m          :: Int
  n          :: Int
  p          :: Int
  ΔX         :: SM
  X          :: SM
  W          :: SM
  P          :: SM
  Q          :: SM
  C          :: SM
  D          :: SM
  V          :: Vector{SM}
  Z          :: Vector{SM}
  R          :: Vector{SM}
  H          :: Vector{SM}
  τ          :: Vector{SV}
  warm_start :: Bool
  stats      :: SimpleStats{T}
end

function BlockGmresWorkspace(m::Integer, n::Integer, p::Integer, SV::Type, SM::Type; memory::Integer = 5)
  memory = min(div(n,p), memory)
  FC = eltype(SV)
  T  = real(FC)
  ΔX = SM(undef, 0, 0)
  X  = SM(undef, n, p)
  W  = SM(undef, n, p)
  P  = SM(undef, 0, 0)
  Q  = SM(undef, 0, 0)
  C  = SM(undef, p, p)
  D  = SM(undef, 2p, p)
  V  = SM[SM(undef, n, p) for i = 1 : memory]
  Z  = SM[SM(undef, p, p) for i = 1 : memory]
  R  = SM[SM(undef, p, p) for i = 1 : div(memory * (memory+1), 2)]
  H  = SM[SM(undef, 2p, p) for i = 1 : memory]
  τ  = SV[SV(undef, p) for i = 1 : memory]
  SV = isconcretetype(SV) ? SV : typeof(τ)
  SM = isconcretetype(SM) ? SM : typeof(X)
  stats = SimpleStats(0, false, false, false, T[], T[], T[], 0.0, "unknown")
  workspace = BlockGmresWorkspace{T,FC,SV,SM}(m, n, p, ΔX, X, W, P, Q, C, D, V, Z, R, H, τ, false, stats)
  return workspace
end

function BlockGmresWorkspace(A, B; memory::Integer = 5)
  m, n = size(A)
  s, p = size(B)
  SM = typeof(B)
  SV = matrix_to_vector(SM)
  BlockGmresWorkspace(m, n, p, SV, SM; memory)
end
