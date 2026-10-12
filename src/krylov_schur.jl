# An implementation of the Krylov-Schur method for the computation of a few eigenvalues
# and eigenvectors of a square (non-Hermitian) linear operator A.
#
# This method is described in
#
# G. W. Stewart, A Krylov-Schur algorithm for large eigenproblems.
# SIAM Journal on Matrix Analysis and Applications, 23(3), pp. 601--614, 2001.

export krylov_schur, krylov_schur!, KrylovSchurWorkspace

"""
    (λ, X, stats) = krylov_schur(A, b::AbstractVector{FC};
                                 nev::Int=1, memory::Int=20, which=:LM,
                                 atol::T=zero(T), rtol::T=√eps(T), itmax::Int=0,
                                 timemax::Float64=Inf, verbose::Int=0, history::Bool=false,
                                 callback=workspace->false, iostream::IO=kstdout)

`T` is an `AbstractFloat` such as `Float32` or `Float64`.
`FC` is `T` or `Complex{T}`.

Compute `nev` eigenvalues `λ` and eigenvectors `X` of the square linear operator `A` of size `n`
with the Krylov-Schur method, a restarted Arnoldi method.

The Arnoldi process builds an orthonormal basis `Vₘ₊₁` of a Krylov subspace of dimension
`m = memory` and the relation `AVₘ = Vₘ₊₁Bₘ₊₁.ₘ`.
The Ritz values are the eigenvalues of `Bₘ`.
The method is restarted with the `k < m` leading Schur vectors `VₘZₘ[:, 1:k]` of a Schur
decomposition `Bₘ = ZₘTₘZₘᴴ` reordered such that the wanted Ritz values come first.
These Schur vectors span a Krylov subspace that contains the wanted Ritz vectors.

If `A` and `b` are real, the Krylov basis is real and the method uses the real Schur form,
in which the complex conjugate eigenvalues of `A` are kept together.
The eigenvalues `λ` and the eigenvectors `X` are always complex.

A Ritz pair `(θ, x = Vₘy)` with `‖y‖ = 1` is converged if its residual
`‖Ax - θx‖ = |βₘ eₘᵀy|` satisfies `‖Ax - θx‖ ≤ atol + rtol * |θ|`.

The eigenvalues of `A` closest to a shift `σ` are computed with the shift-invert spectral
transformation: pass an operator that applies `(A - σI)⁻¹` (for instance with a factorization
or an inner Krylov solver), compute its eigenvalues `μ` of largest magnitude, and recover
`λ = σ + 1 / μ`. The eigenvectors are the same.
For the generalized eigenvalue problem `Ax = λMx`, the operator is `(A - σM)⁻¹M`.

#### Input arguments

* `A`: a linear operator that models a matrix of dimension `n`;
* `b`: a nonzero vector of length `n`, the starting vector of the Arnoldi process.

#### Keyword arguments

* `nev`: the number of wanted eigenvalues, `1 ≤ nev < memory`;
* `memory`: the dimension `m` of the Krylov subspace before a restart. A value of at least `2nev + 1` is recommended;
* `which`: the wanted eigenvalues: `:LM` (largest magnitude), `:SM` (smallest magnitude), `:LR` / `:SR` (largest / smallest real part), `:LI` / `:SI` (largest / smallest imaginary part), or a function `θ -> score` where the eigenvalues with the largest scores are wanted;
* `atol`: absolute stopping tolerance on the residual norms of the Ritz pairs;
* `rtol`: relative stopping tolerance on the residual norms of the Ritz pairs;
* `itmax`: the maximum number of restart cycles. If `itmax=0`, the default number of cycles is set to `100`;
* `timemax`: the time limit in seconds;
* `verbose`: additional details can be displayed if verbose mode is enabled (verbose > 0). Information will be displayed every `verbose` cycles;
* `history`: collect the largest residual norm of the `nev` wanted Ritz pairs at each cycle;
* `callback`: function or functor called as `callback(workspace)` that returns `true` if the method should terminate, and `false` otherwise;
* `iostream`: stream to which output is logged.

#### Output arguments

* `λ`: a vector of `p ≥ nev` complex eigenvalues ordered according to `which`. `p > nev` if more than `nev` Ritz pairs converged;
* `X`: a dense `n × p` complex matrix whose columns are the normalized eigenvectors (Ritz vectors);
* `stats`: statistics collected on the run in an [`EigenStats`](@ref) structure.

#### Reference

* G. W. Stewart, [*A Krylov-Schur algorithm for large eigenproblems*](https://doi.org/10.1137/S0895479800371529), SIAM Journal on Matrix Analysis and Applications, 23(3), pp. 601--614, 2001.
"""
function krylov_schur end

"""
    workspace = krylov_schur!(workspace::KrylovSchurWorkspace, A, b; kwargs...)

In this call, `kwargs` are keyword arguments of [`krylov_schur`](@ref).
The keyword argument `memory` is the only exception.
It is only supported by [`krylov_schur`](@ref) and is required to create a `KrylovSchurWorkspace`.
It cannot be changed later.

The eigenvalues, eigenvectors and statistics are available with
`eigenvalues(workspace)`, `eigenvectors(workspace)` and `statistics(workspace)`.
The eigenvalues and eigenvectors are views of the storage of the workspace.

See [`KrylovSchurWorkspace`](@ref) for instructions on how to create the `workspace`.
"""
function krylov_schur! end

"""
Type for storing statistics returned by eigenvalue solvers such as [`krylov_schur`](@ref).

The fields are as follows:
- `niter`: The number of restart cycles (Arnoldi expansions) completed by the solver;
- `nmatvec`: The number of operator-vector products;
- `nconv`: The number of converged Ritz pairs, counted in the order given by `which`;
- `solved`: Indicates whether the `nev` wanted eigenpairs converged;
- `residuals`: The residual norms ‖Ax - θx‖ of the returned Ritz pairs;
- `history`: If `history = true`, the largest residual norm of the `nev` wanted Ritz pairs at each cycle;
- `allocation_timer`: The elapsed time (in seconds) spent on allocations;
- `timer`: The elapsed time (in seconds) taken by the solver to complete all iterations;
- `status`: A string indicating the outcome of the solve.
"""
mutable struct EigenStats{T} <: KrylovStats{T}
  niter            :: Int
  nmatvec          :: Int
  nconv            :: Int
  solved           :: Bool
  residuals        :: Vector{T}
  history          :: Vector{T}
  allocation_timer :: Float64
  timer            :: Float64
  status           :: String
end

EigenStats(T::Type) = EigenStats{T}(0, 0, 0, false, T[], T[], 0.0, 0.0, "unknown")

function reset!(stats :: EigenStats)
  empty!(stats.residuals)
  empty!(stats.history)
  stats.niter = 0
  stats.nmatvec = 0
  stats.nconv = 0
  stats.solved = false
end

function copyto!(dest :: EigenStats, src :: EigenStats)
  dest.niter            = src.niter
  dest.nmatvec          = src.nmatvec
  dest.nconv            = src.nconv
  dest.solved           = src.solved
  dest.residuals        = copy(src.residuals)
  dest.history          = copy(src.history)
  dest.allocation_timer = src.allocation_timer
  dest.timer            = src.timer
  dest.status           = src.status
  return dest
end

"""
Workspace for the in-place method [`krylov_schur!`](@ref).

The following outer constructors can be used to initialize this workspace:

    workspace = KrylovSchurWorkspace(m, n, S; memory = 20)
    workspace = KrylovSchurWorkspace(A, b; memory = 20)

`m` and `n` denote the dimensions of the linear operator `A` passed to the in-place method; `A` must be square (`m = n`).
`S` is the storage type of the vectors in the workspace, such as `Vector{Float64}` or `Vector{ComplexF64}`.
A real `S` requires a real operator `A`.
`memory` is the dimension of the Krylov subspace; it is set to `n` if the value given is larger than `n`.
The Krylov basis is stored in a dense matrix of size `n × (memory+1)`.
The complex eigenvectors are stored in a dense matrix of size `n × p`, where `p` is the number of
returned eigenpairs, which is allocated by the first call to [`krylov_schur!`](@ref) and grown if needed.
"""
mutable struct KrylovSchurWorkspace{T,FC<:FloatOrComplex{T},S,M,Mc}
  n      :: Int
  memory :: Int
  V      :: M                   # Krylov basis Vₘ₊₁, n × (m+1)
  Vtmp   :: M                   # row block buffer for the basis rotations, nchunk × m
  Qd     :: M                   # rotation matrix in the storage of V, m × m
  hd     :: S                   # Gram-Schmidt coefficients in the storage of V, length m+1
  X      :: Mc                  # eigenvectors, n × p
  h      :: Vector{FC}          # Gram-Schmidt coefficients
  htmp   :: Vector{FC}          # Gram-Schmidt coefficients of one pass
  B      :: Matrix{FC}          # AVₘ = Vₘ₊₁Bₘ₊₁.ₘ, (m+1) × m
  Tm     :: Matrix{FC}          # work matrix for the eigenvalue and Schur decompositions of Bₘ, m × m
  Q      :: Matrix{FC}          # Schur vectors of Bₘ, m × m
  θ      :: Vector{Complex{T}}  # Ritz values ordered according to `which`
  Y      :: Matrix{Complex{T}}  # eigenvectors of Bₘ ordered as θ, m × m
  res    :: Vector{T}           # residual norms of the Ritz pairs
  score  :: Vector{T}           # scores of the Ritz values
  perm   :: Vector{Int}         # permutation that sorts the Ritz values
  select :: Vector{Bool}        # Schur vectors kept at a restart
  nev    :: Int                 # number of returned eigenpairs
  stats  :: EigenStats{T}
end

function KrylovSchurWorkspace(m::Integer, n::Integer, S::Type; memory::Int = 20)
  m == n || error("The operator must be square")
  start_allocation_time = time_ns()
  memory = min(n, memory)
  FC = eltype(S)
  T  = real(FC)
  # The matrices are allocated with `similar` so that their types are concrete,
  # even if `S` is not (e.g., `CuVector{T}` without its memory type parameter).
  hd = S(undef, memory+1)
  V  = similar(hd, FC, n, memory+1)
  Vtmp = similar(hd, FC, min(n, 4096), memory)
  Qd = similar(hd, FC, memory, memory)
  X  = similar(hd, Complex{T}, n, 0)
  S  = typeof(hd)
  M  = typeof(V)
  Mc = typeof(X)
  h  = Vector{FC}(undef, memory+1)
  htmp = Vector{FC}(undef, memory+1)
  B  = zeros(FC, memory+1, memory)
  Tm = zeros(FC, memory, memory)
  Q  = zeros(FC, memory, memory)
  θ  = zeros(Complex{T}, memory)
  Y  = zeros(Complex{T}, memory, memory)
  res = zeros(T, memory)
  score = zeros(T, memory)
  perm = collect(1:memory)
  select = fill(false, memory)
  stats = EigenStats(T)
  workspace = KrylovSchurWorkspace{T,FC,S,M,Mc}(n, memory, V, Vtmp, Qd, hd, X, h, htmp, B, Tm, Q, θ, Y,
                                                res, score, perm, select, 0, stats)
  workspace.stats.allocation_timer = start_allocation_time |> ktimer
  return workspace
end

function KrylovSchurWorkspace(A, b; memory::Int = 20)
  m, n = size(A)
  m == n || error("The operator must be square")
  S = ktypeof(b)
  KrylovSchurWorkspace(m, n, S; memory)
end

eigenvalues(workspace::KrylovSchurWorkspace) = view(workspace.θ, 1:workspace.nev)
eigenvectors(workspace::KrylovSchurWorkspace) = view(workspace.X, :, 1:workspace.nev)
statistics(workspace::KrylovSchurWorkspace) = workspace.stats
results(workspace::KrylovSchurWorkspace) = (eigenvalues(workspace), eigenvectors(workspace), workspace.stats)
elapsed_time(workspace::KrylovSchurWorkspace) = workspace.stats.timer
iteration_count(workspace::KrylovSchurWorkspace) = workspace.stats.niter
Aprod_count(workspace::KrylovSchurWorkspace) = workspace.stats.nmatvec

function sizeof(workspace :: KrylovSchurWorkspace)
  return sum(ksizeof(getfield(workspace, i)) for i = 1:fieldcount(typeof(workspace)))
end

function show(io :: IO, workspace :: KrylovSchurWorkspace{T,FC,S}) where {T,FC,S}
  @printf(io, "KrylovSchurWorkspace{%s,%s,%s} of size %d with memory %d (%s)\n", T, FC, S,
          workspace.n, workspace.memory, format_bytes(sizeof(workspace)))
end

# Score of a Ritz value: the eigenvalues with the largest scores are wanted.
function eigs_score(which)
  which isa Function && return which
  which === :LM && return abs
  which === :SM && return θ -> -abs(θ)
  which === :LR && return real
  which === :SR && return θ -> -real(θ)
  which === :LI && return imag
  which === :SI && return θ -> -imag(θ)
  error("which = $which is not supported; use :LM, :SM, :LR, :SR, :LI, :SI or a function")
end

# In real arithmetic, the eigenvalues of a complex conjugate pair are given the same score.
pair_score(score, θ, ::Type{FC}) where FC <: Real = max(score(θ), score(conj(θ)))
pair_score(score, θ, ::Type{FC}) where FC <: Complex = score(θ)

# Orthogonalize q against V[:, 1:j] with classical Gram-Schmidt and one reorthogonalization (CGS2).
# The coefficients are accumulated in workspace.h[1:j].
function cgs2!(workspace, q, j::Int)
  Vj   = view(workspace.V, :, 1:j)
  hd   = view(workspace.hd, 1:j)
  h    = workspace.h
  htmp = workspace.htmp
  FC   = eltype(h)
  fill!(view(h, 1:j), zero(FC))
  for pass = 1 : 2
    mul!(hd, Vj', q)
    mul!(q, Vj, hd, -one(FC), one(FC))
    copyto!(htmp, 1, hd, 1, j)
    for i = 1 : j
      h[i] += htmp[i]
    end
  end
  return h
end

# Expand the Krylov-Schur factorization AVₖ = Vₖ₊₁Bₖ₊₁.ₖ to AVₘ = Vₘ₊₁Bₘ₊₁.ₘ with the Arnoldi process.
function krylov_schur_expand!(workspace, A, k::Int, m::Int)
  n, V, B, h = workspace.n, workspace.V, workspace.B, workspace.h
  FC = eltype(B)
  T = real(FC)
  btol = eps(T)^(3//4)
  for j = k+1 : m
    vⱼ = view(V, :, j)
    q = view(V, :, j+1)
    kmul!(q, A, vⱼ)
    workspace.stats.nmatvec += 1
    qNorm₀ = knorm(n, q)
    cgs2!(workspace, q, j)
    for i = 1 : j
      B[i,j] = h[i]
    end
    βⱼ = knorm(n, q)
    if βⱼ ≤ btol * qNorm₀
      # Breakdown: Vⱼ spans an invariant subspace of A.
      # Continue with a new vector orthogonal to Vⱼ, which does not change AVⱼ = VⱼBⱼ.
      B[j+1,j] = zero(FC)
      q .= sin.(T(j + 1) .* T.(1:n)) .+ inv.(T.(1:n))
      cgs2!(workspace, q, j)
      cgs2!(workspace, q, j)
      kdiv!(n, q, knorm(n, q))
    else
      B[j+1,j] = βⱼ
      kdiv!(n, q, βⱼ)
    end
  end
  return workspace
end

# Ritz values θ and eigenvectors Y of Bₘ ordered by decreasing score, and residual norms of the Ritz pairs.
function krylov_schur_ritz!(workspace, score, m::Int)
  B, Tm, θ, Y, res, sc, perm = workspace.B, workspace.Tm, workspace.θ, workspace.Y, workspace.res, workspace.score, workspace.perm
  Tm .= view(B, 1:m, 1:m)
  F = eigen!(Tm)
  for i = 1 : m
    sc[i] = pair_score(score, F.values[i], eltype(B))
  end
  sortperm!(perm, sc, rev=true)
  βₘ = abs(B[m+1,m])
  for i = 1 : m
    pᵢ = perm[i]
    θ[i] = F.values[pᵢ]
    for l = 1 : m
      Y[l,i] = F.vectors[l,pᵢ]
    end
    yNorm = norm(view(Y, :, i))
    view(Y, :, i) ./= yNorm
    res[i] = βₘ * abs(Y[m,i])
  end
  return workspace
end

# Reorder a Schur decomposition of Bₘ such that its k leading Schur vectors span the invariant subspace
# associated with the k Ritz values of largest scores, and restart the factorization with these vectors.
# In real arithmetic, k is increased (or decreased) by one to keep complex conjugate pairs together.
function krylov_schur_restart!(workspace, score, m::Int, k::Int)
  V, B, Tm, Q, sc, perm, select = workspace.V, workspace.B, workspace.Tm, workspace.Q, workspace.score, workspace.perm, workspace.select
  FC = eltype(B)
  βₘ = B[m+1,m]
  Tm .= view(B, 1:m, 1:m)
  F = schur!(Tm)
  for i = 1 : m
    sc[i] = pair_score(score, F.values[i], FC)
  end
  sortperm!(perm, sc, rev=true)
  fill!(select, false)
  for i = 1 : k
    select[perm[i]] = true
  end
  if FC <: Real
    # A complex conjugate pair is a 2 × 2 diagonal block of the real Schur form.
    for i = 1 : m-1
      if F.T[i+1,i] != 0 && select[i] != select[i+1]
        select[i] = select[i+1] = true
      end
    end
    # Both eigenvalues of a pair have the same score, so at most one pair has to be dropped.
    if count(select) == m
      q = perm[m]
      q = (q > 1 && F.T[q,q-1] != 0) ? q-1 : q
      select[q] = select[q+1] = false
    end
    k = count(select)
  end
  ordschur!(F, select)
  Q .= F.Z

  # Vₖ₊₁ ← [VₘQ[:, 1:k]  vₘ₊₁] and Bₖ₊₁.ₖ ← [T[1:k, 1:k]; βₘQ[m, 1:k]].
  rotate_basis!(workspace, Q, m, k)
  view(V, :, k+1) .= view(V, :, m+1)
  fill!(B, zero(FC))
  for j = 1 : k
    for i = 1 : k
      B[i,j] = F.T[i,j]
    end
    B[k+1,j] = βₘ * Q[m,j]
  end
  return k
end

# W[:, 1:k] ← V[:, 1:m] * R[1:m, 1:k] by blocks of rows, where W is V or the matrix of eigenvectors X.
function rotate_basis!(workspace, R::Matrix, m::Int, k::Int; W = workspace.V)
  V, Vtmp, Qd = workspace.V, workspace.Vtmp, workspace.Qd
  n = size(V, 1)
  nchunk = size(Vtmp, 1)
  # If V is real and R is complex, the real and imaginary parts are computed one after the other.
  parts = (eltype(R) <: Complex && eltype(V) <: Real) ? (real, imag) : (identity,)
  for (ipart, part) in enumerate(parts)
    copyto!(view(Qd, 1:m, 1:k), part.(view(R, 1:m, 1:k)))
    for r1 = 1 : nchunk : n
      rows = r1 : min(r1 + nchunk - 1, n)
      tmp = view(Vtmp, 1:length(rows), 1:k)
      mul!(tmp, view(V, rows, 1:m), view(Qd, 1:m, 1:k))
      if ipart == 1
        view(W, rows, 1:k) .= tmp
      else
        view(W, rows, 1:k) .+= im .* tmp
      end
    end
  end
  return W
end

function krylov_schur(A, b::AbstractVector{FC}; memory::Int = 20, kwargs...) where FC <: FloatOrComplex
  workspace = KrylovSchurWorkspace(A, b; memory)
  krylov_schur!(workspace, A, b; kwargs...)
  λ = copy(eigenvalues(workspace))
  X = copy(eigenvectors(workspace))
  return λ, X, workspace.stats
end

function krylov_schur!(workspace :: KrylovSchurWorkspace{T,FC,S}, A, b :: AbstractVector{FC};
                       nev :: Int = 1, which = :LM, atol :: T = zero(T), rtol :: T = √eps(T),
                       itmax :: Int = 0, timemax :: Float64 = Inf, verbose :: Int = 0,
                       history :: Bool = false, callback = workspace -> false,
                       iostream :: IO = kstdout) where {T <: AbstractFloat, FC <: FloatOrComplex{T}, S}

  # Timer
  start_time = time_ns()
  timemax_ns = 1e9 * timemax

  n, mem = workspace.n, workspace.memory
  m₁, n₁ = size(A)
  (m₁ == n && n₁ == n) || error("workspace.n = $n is inconsistent with size(A) = ($m₁, $n₁)")
  length(b) == n || error("Inconsistent problem size")
  1 ≤ nev < mem || error("nev = $nev must satisfy 1 ≤ nev < memory = $mem")
  score = eigs_score(which)
  (verbose > 0) && @printf(iostream, "Krylov-Schur: operator of size %d, nev = %d, memory = %d\n", n, nev, mem)

  V, B, θ, Y, res = workspace.V, workspace.B, workspace.θ, workspace.Y, workspace.res
  stats = workspace.stats
  reset!(stats)
  workspace.nev = 0

  # v₁ = b / ‖b‖
  v₁ = view(V, :, 1)
  copyto!(v₁, b)
  β = knorm(n, v₁)
  β == 0 && error("The starting vector b must be nonzero")
  kdiv!(n, v₁, β)

  itmax == 0 && (itmax = 100)
  k = 0      # Number of kept Schur vectors
  nconv = 0  # Number of converged Ritz pairs
  iter = 0
  fill!(B, zero(FC))

  (verbose > 0) && @printf(iostream, "%5s  %7s  %5s  %7s  %5s\n", "cycle", "matvecs", "nconv", "‖rₙₑᵥ‖", "timer")

  solved = false
  tired = false
  user_requested_exit = false
  overtimed = false

  while !(solved || tired || user_requested_exit || overtimed)
    krylov_schur_expand!(workspace, A, k, mem)
    iter += 1

    # Ritz pairs ordered according to `which` and their residuals ‖Axᵢ - θᵢxᵢ‖ = |βₘ eₘᵀyᵢ|.
    krylov_schur_ritz!(workspace, score, mem)
    nconv = 0
    while nconv < mem && res[nconv+1] ≤ atol + rtol * abs(θ[nconv+1])
      nconv += 1
    end
    rNev = maximum(view(res, 1:nev))
    history && push!(stats.history, rNev)

    user_requested_exit = callback(workspace) :: Bool
    solved = nconv ≥ nev
    tired = iter ≥ itmax
    overtimed = time_ns() - start_time > timemax_ns
    kdisplay(iter, verbose) && @printf(iostream, "%5d  %7d  %5d  %7.1e  %.2fs\n", iter, stats.nmatvec, nconv, rNev, start_time |> ktimer)

    if !(solved || tired || user_requested_exit || overtimed)
      # Restart with the Schur vectors of the converged Ritz values and half of the other ones.
      k = clamp(nconv + div(mem - nconv, 2), nev, mem - 1)
      k = krylov_schur_restart!(workspace, score, mem, k)
    end
  end
  (verbose > 0) && @printf(iostream, "\n")

  # Ritz vectors xᵢ = Vₘyᵢ of the p returned eigenvalues.
  p = min(max(nev, nconv), mem)
  if size(workspace.X, 2) < p
    workspace.X = similar(workspace.X, n, p)
  end
  rotate_basis!(workspace, Y, mem, p; W = workspace.X)
  for i = 1 : p
    xᵢ = view(workspace.X, :, i)
    kdiv!(n, xᵢ, knorm(n, xᵢ))
  end
  workspace.nev = p

  # Termination status
  status = "unknown"
  tired               && (status = "maximum number of iterations exceeded")
  solved              && (status = "eigenvalues converged given atol and rtol")
  user_requested_exit && (status = "user-requested exit")
  overtimed           && (status = "time limit exceeded")

  # Update stats
  stats.niter = iter
  stats.nconv = nconv
  stats.solved = solved
  append!(stats.residuals, view(res, 1:p))
  stats.timer = start_time |> ktimer
  stats.status = status
  return workspace
end
