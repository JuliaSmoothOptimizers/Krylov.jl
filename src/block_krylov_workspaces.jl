export BlockKrylovWorkspace, BlockMinresWorkspace, BlockGmresWorkspace, BlockCgWorkspace

"Abstract type for using block Krylov solvers in-place."
abstract type BlockKrylovWorkspace{T,FC,SV,SM} end

"""
Workspace for the in-place methods [`block_minres!`](@ref) and [`krylov_solve!`](@ref).

The following outer constructors can be used to initialize this workspace:

    workspace = BlockMinresWorkspace(m, n, p, SV, SM)
    workspace = BlockMinresWorkspace(A, B)

`m` and `n` denote the dimensions of the linear operator `A` passed to the in-place methods.
`p` denotes the number of columns of the right-hand side `B` passed to the in-place methods.
`SV` is the storage type of the vectors in the workspace, such as `Vector{Float64}`.
`SM` is the storage type of the matrices in the workspace, such as `Matrix{Float64}`.
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
  Ψₖ         :: SM
  Ωₖ         :: SM
  Ψₖ₊₁       :: SM
  Πₖ₋₂       :: SM
  Γbarₖ₋₁    :: SM
  Γₖ₋₁       :: SM
  Λbarₖ      :: SM
  Λₖ         :: SM
  Vₖ₋₁       :: SM
  Vₖ         :: SM
  wₖ₋₂       :: SM
  wₖ₋₁       :: SM
  wₖ         :: SM
  Hₖ₋₂       :: SM
  Hₖ₋₁       :: SM
  τₖ₋₂       :: SV
  τₖ₋₁       :: SV
  buffer     :: Vector{FC}
  warm_start :: Bool
  stats      :: SimpleStats{T}
end

function BlockMinresWorkspace(m::Integer, n::Integer, p::Integer, SV::Type, SM::Type)
  start_allocation_time = time_ns()
  FC   = eltype(SV)
  T    = real(FC)
  ΔX   = SM(undef, 0, 0)
  X    = SM(undef, n, p)
  P    = SM(undef, 0, 0)
  Q    = SM(undef, n, p)
  C    = SM(undef, p, p)
  D    = SM(undef, 2p, p)
  Φ    = SM(undef, p, p)
  Ψₖ   = SM(undef, p, p)
  Ωₖ   = SM(undef, p, p)
  Ψₖ₊₁ = SM(undef, p, p)
  Πₖ₋₂ = SM(undef, p, p)
  Γbarₖ₋₁ = SM(undef, p, p)
  Γₖ₋₁ = SM(undef, p, p)
  Λbarₖ = SM(undef, p, p)
  Λₖ   = SM(undef, p, p)
  Vₖ₋₁ = SM(undef, n, p)
  Vₖ   = SM(undef, n, p)
  wₖ₋₂ = SM(undef, n, p)
  wₖ₋₁ = SM(undef, n, p)
  wₖ   = SM(undef, n, p)
  Hₖ₋₂ = SM(undef, 2p, p)
  Hₖ₋₁ = SM(undef, 2p, p)
  τₖ₋₂ = SV(undef, p)
  τₖ₋₁ = SV(undef, p)
  SV = isconcretetype(SV) ? SV : typeof(τₖ₋₁)
  SM = isconcretetype(SM) ? SM : typeof(X)
  stats = SimpleStats(0, false, false, false, 0, T[], T[], T[], T[], 0.0, 0.0, "unknown")
  size_buffer = max(kgeqrf_buffer!(Vₖ, τₖ₋₁), kgeqrf_buffer!(Hₖ₋₁, τₖ₋₁),
                    kungqr_buffer!(Vₖ, τₖ₋₁), kungqr_buffer!(Hₖ₋₁, τₖ₋₁),
                    kunmqr_buffer!('L', FC <: AbstractFloat ? 'T' : 'C', Hₖ₋₁, τₖ₋₁, D))
  buffer = SV(undef, size_buffer)
  workspace = BlockMinresWorkspace{T,FC,SV,SM}(m, n, p, ΔX, X, P, Q, C, D, Φ, Ψₖ, Ωₖ, Ψₖ₊₁, Πₖ₋₂, Γbarₖ₋₁, Γₖ₋₁, Λbarₖ, Λₖ,
                                               Vₖ₋₁, Vₖ, wₖ₋₂, wₖ₋₁, wₖ, Hₖ₋₂, Hₖ₋₁, τₖ₋₂, τₖ₋₁, buffer, false, stats)
  workspace.stats.allocation_timer = start_allocation_time |> ktimer
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
Workspace for the in-place methods [`block_gmres!`](@ref) and [`krylov_solve!`](@ref).

The following outer constructors can be used to initialize this workspace:

    workspace = BlockGmresWorkspace(m, n, p, SV, SM; memory = 5)
    workspace = BlockGmresWorkspace(A, B; memory = 5)

`m` and `n` denote the dimensions of the linear operator `A` passed to the in-place methods.
`p` denotes the number of columns of the right-hand side `B` passed to the in-place methods.
`SV` is the storage type of the vectors in the workspace, such as `Vector{Float64}`.
`SM` is the storage type of the matrices in the workspace, such as `Matrix{Float64}`.
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
  buffer     :: Vector{FC}
  warm_start :: Bool
  stats      :: SimpleStats{T}
end

function BlockGmresWorkspace(m::Integer, n::Integer, p::Integer, SV::Type, SM::Type; memory::Int = 5)
  start_allocation_time = time_ns()
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
  size_buffer = max(kgeqrf_buffer!(V[1], τ[1]), kgeqrf_buffer!(H[1], τ[1]),
                    kungqr_buffer!(V[1], τ[1]), kungqr_buffer!(H[1], τ[1]),
                    kunmqr_buffer!('L', FC <: AbstractFloat ? 'T' : 'C', H[1], τ[1], D))
  buffer = SV(undef, size_buffer)
  stats = SimpleStats(0, false, false, false, 0, T[], T[], T[], T[], 0.0, 0.0, "unknown")
  workspace = BlockGmresWorkspace{T,FC,SV,SM}(m, n, p, ΔX, X, W, P, Q, C, D, V, Z, R, H, τ, buffer, false, stats)
  workspace.stats.allocation_timer = start_allocation_time |> ktimer
  return workspace
end

function BlockGmresWorkspace(A, B; memory::Int = 5)
  m, n = size(A)
  s, p = size(B)
  SM = typeof(B)
  SV = matrix_to_vector(SM)
  BlockGmresWorkspace(m, n, p, SV, SM; memory)
end

"""
Workspace for the in-place methods [`block_cg!`](@ref) and [`krylov_solve!`](@ref).

The following outer constructors can be used to initialize this workspace:

    workspace = BlockCgWorkspace(m, n, p, SV, SM; nullspace = nothing, grounding = nothing)
    workspace = BlockCgWorkspace(A, B; nullspace = nothing, grounding = nothing)

`m` and `n` denote the dimensions of the linear operator `A` passed to the in-place methods.
`p` denotes the number of columns of the right-hand side `B` passed to the in-place methods.
`SV` is the storage type of the vectors in the workspace, such as `Vector{Float64}`.
`SM` is the storage type of the matrices in the workspace, such as `Matrix{Float64}`.

For a singular `A`, `nullspace` is a basis of its null space (an `n × k` matrix or a vector,
not necessarily orthonormal), and `grounding` (`n × k` or a vector) defines the solution among
all solutions: `Wᴴx = 0` for `W = grounding`. By default `W` is the null space itself, which
gives the solution orthogonal to it. `WᴴV` must be invertible. Both are given on the host and
copied into the storage type `SM`. See [`block_cg`](@ref).

After a solve, `workspace.bnorm` and `workspace.rnorm` hold the norms of the projected
right-hand sides and the final residual norms column by column, and
`workspace.defect` the relative part `‖(I - Π)B‖_F / ‖B‖_F` of the right-hand
side outside the range of `A` that was removed.
"""
mutable struct BlockCgWorkspace{T,FC,SV,SM} <: BlockKrylovWorkspace{T,FC,SV,SM}
  m          :: Int
  n          :: Int
  p          :: Int
  ΔX         :: SM
  X          :: SM
  R          :: SM
  Z          :: SM
  P          :: SM
  Q          :: SM
  V          :: SM
  W          :: SM
  F          :: SM
  K          :: SM
  G          :: SM
  C          :: SM
  mask       :: SM
  colsums    :: SM
  maskh      :: Matrix{FC}
  colsumsh   :: Vector{FC}
  Gh         :: Matrix{FC}
  Ch         :: Matrix{FC}
  Th         :: Matrix{FC}
  Eh         :: Matrix{FC}
  Lh         :: Matrix{FC}
  Wh         :: Matrix{FC}
  λh         :: Vector{T}
  perm       :: Vector{Int}
  chol_ok    :: Bool
  d          :: Vector{T}
  bnorm      :: Vector{T}
  rnorm      :: Vector{T}
  tol        :: Vector{T}
  active     :: Vector{Bool}
  defect     :: T
  warm_start :: Bool
  stats      :: SimpleStats{T}
end

function BlockCgWorkspace(m::Integer, n::Integer, p::Integer, SV::Type, SM::Type; nullspace = nothing, grounding = nothing)
  start_allocation_time = time_ns()
  FC = eltype(SV)
  T  = real(FC)
  if nullspace === nothing
    grounding === nothing || throw(ArgumentError("a grounding needs a null space basis"))
    Vh = Wh = Fh = Matrix{FC}(undef, n, 0)
  else
    Vh = Matrix{FC}(reshape(collect(nullspace), size(nullspace, 1), :))
    size(Vh, 1) == n || throw(DimensionMismatch("the null space basis must have $n rows"))
    Vh = Matrix(qr(Vh).Q)  # orthonormal basis of the null space
    k = size(Vh, 2)
    Wh = grounding === nothing ? copy(Vh) : Matrix{FC}(reshape(collect(grounding), size(grounding, 1), :))
    size(Wh) == (n, k) || throw(DimensionMismatch("the grounding must be $n × $k"))
    WV = Wh' * Vh
    abs(det(WV)) > eps(T) * opnorm(Wh) || throw(ArgumentError("Wᴴ V must be invertible"))
    Fh = Vh / WV
  end
  k = size(Vh, 2)
  ΔX = SM(undef, 0, 0)
  X  = SM(undef, n, p)
  R  = SM(undef, n, p)
  Z  = SM(undef, n, p)
  P  = SM(undef, n, p)
  Q  = SM(undef, n, p)
  V  = copyto!(SM(undef, n, k), Vh)
  W  = copyto!(SM(undef, n, k), Wh)
  F  = copyto!(SM(undef, n, k), Fh)
  K  = SM(undef, k, p)
  G  = SM(undef, p, p)
  C  = SM(undef, p, p)
  mask = SM(undef, 1, p)
  colsums = SM(undef, 1, p)
  SV = isconcretetype(SV) ? SV : matrix_to_vector(typeof(X))
  SM = isconcretetype(SM) ? SM : typeof(X)
  stats = SimpleStats(0, false, false, false, 0, T[], T[], T[], T[], 0.0, 0.0, "unknown")
  host() = zeros(FC, p, p)
  workspace = BlockCgWorkspace{T,FC,SV,SM}(m, n, p, ΔX, X, R, Z, P, Q, V, W, F, K, G, C, mask, colsums,
                                           ones(FC, 1, p), zeros(FC, p), host(), host(), host(), host(), host(), host(),
                                           zeros(T, p), collect(1:p), true, zeros(T, p), zeros(T, p), zeros(T, p),
                                           zeros(T, p), fill(true, p), zero(T), false, stats)
  workspace.stats.allocation_timer = start_allocation_time |> ktimer
  return workspace
end

function BlockCgWorkspace(A, B; nullspace = nothing, grounding = nothing)
  m, n = size(A)
  s, p = size(B)
  SM = typeof(B)
  SV = matrix_to_vector(SM)
  BlockCgWorkspace(m, n, p, SV, SM; nullspace, grounding)
end

