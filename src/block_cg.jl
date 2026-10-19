# An implementation of the block conjugate gradient method for the solution of the Hermitian
# positive (semi)definite linear system AX = B, with an optional projection onto the
# complement of a known null space of A.
#
# The block iteration is the one of O'Leary (1980), in Galerkin form with search blocks that
# are orthonormalized by a rank-revealing eigendecomposition of their Gram matrix (SVQB), so
# that (nearly) linearly dependent right-hand sides and search directions are dropped instead
# of breaking down. Converged columns are removed from the block (deflation).
#
# For singular A with known null space V (e.g. the stiffness matrix of a pure Neumann problem,
# V = constants), every residual and every preconditioned residual is projected onto V⊥ =
# range(A), so the iteration never leaves V⊥; the component of the solution in V is fixed at
# the end by Wᴴx = 0.
#
# Daniel Boigk -- 2026.

export block_cg, block_cg!

"""
    (X, stats) = block_cg(A, B::AbstractMatrix{FC};
                          nullspace=nothing, grounding=nothing,
                          M=I, ldiv::Bool=false,
                          atol::T=√eps(T), rtol::T=√eps(T), itmax::Int=0,
                          rank_tol::T=√eps(T), recompute_every::Int=50,
                          timemax::Float64=Inf, verbose::Int=0, history::Bool=false,
                          callback=workspace->false, iostream::IO=kstdout)

`T` is an `AbstractFloat` such as `Float32`, `Float64` or `BigFloat`.
`FC` is `T` or `Complex{T}`.

    (X, stats) = block_cg(A, B, X0::AbstractMatrix; kwargs...)

Block-CG can be warm-started from an initial guess `X0` where `kwargs` are the same keyword arguments as above.

Solve the Hermitian positive definite linear system AX = B of size n with p right-hand sides using the block conjugate gradient method.

If `A` is only positive semidefinite, with a known null space spanned by the columns of `nullspace`, Block-CG solves the projected system `AX = ΠB`, where `Π` is the orthogonal projector onto the range of `A`, the orthogonal complement of the null space.
Every residual and every preconditioned residual is projected, so round-off and preconditioners that do not preserve the range cannot pollute the solution, and inconsistent right-hand sides are handled by solving the consistent projected system (`stats.inconsistent` is then set).
The component of `X` in the null space is fixed at the end by `Wᴴ X = 0`, where `W = grounding` (default: the null space itself, which gives the solution orthogonal to the null space).

Every search block is orthonormalized by a rank-revealing eigendecomposition of its Gram matrix, so dependent right-hand sides and directions are dropped instead of causing a breakdown.
Columns that have converged are removed from the block.
The residual of column `j` is compared with `atol + rtol * ‖Π bⱼ‖`, and the method stops when all columns are below their tolerance.

#### Interface

To easily switch between block Krylov methods, use the generic interface [`krylov_solve`](@ref) with `method = :block_cg`.

For an in-place variant that reuses memory across solves, see [`block_cg!`](@ref).

#### Input arguments

* `A`: a linear operator that models a Hermitian positive (semi)definite matrix of dimension `n`;
* `B`: a matrix of size `n × p`.

#### Optional argument

* `X0`: a matrix of size `n × p` that represents an initial guess of the solution `X`.

#### Keyword arguments

* `nullspace`: a basis of the null space of `A`, an `n × k` matrix or a vector (not necessarily orthonormal), or `nothing` for a positive definite `A`;
* `grounding`: an `n × k` matrix or a vector `W` that selects the solution `Wᴴ X = 0` among all solutions of the projected system, or `nothing` for `W = nullspace`;
* `M`: linear operator that models a Hermitian positive-definite matrix of size `n` used for centered preconditioning;
* `ldiv`: define whether the preconditioner uses `ldiv!` or `mul!`;
* `atol`: absolute stopping tolerance based on the residual norm of every column;
* `rtol`: relative stopping tolerance based on the residual norm of every column;
* `itmax`: the maximum number of iterations. If `itmax=0`, the default number of iterations is set to `2n` (deflation can reduce the block to a single column);
* `rank_tol`: relative eigenvalue threshold below which directions of a search block are considered linearly dependent and dropped;
* `recompute_every`: number of iterations after which the updated residual is replaced by the true residual `Π(B - AX)`, to limit the drift of the recurrence;
* `timemax`: the time limit in seconds;
* `verbose`: additional details can be displayed if verbose mode is enabled (verbose > 0). Information will be displayed every `verbose` iterations;
* `history`: collect additional statistics on the run such as residual norms;
* `callback`: function or functor called as `callback(workspace)` that returns `true` if the block-Krylov method should terminate, and `false` otherwise;
* `iostream`: stream to which output is logged.

#### Output arguments

* `X`: a dense matrix of size `n × p`;
* `stats`: statistics collected on the run in a [`SimpleStats`](@ref) structure.

#### Reference

* D. P. O'Leary, [*The block conjugate gradient algorithm and related methods*](https://doi.org/10.1016/0024-3795(80)90247-5), Linear Algebra and its Applications, 29, pp. 293--322, 1980.
"""
function block_cg end

"""
    workspace = block_cg!(workspace::BlockCgWorkspace, A, B; kwargs...)
    workspace = block_cg!(workspace::BlockCgWorkspace, A, B, X0; kwargs...)

In these calls, `kwargs` are keyword arguments of [`block_cg`](@ref), except `nullspace` and
`grounding`, which belong to the workspace.

See [`BlockCgWorkspace`](@ref) for instructions on how to create the `workspace`.

For a more generic interface, you can use [`krylov_workspace`](@ref) with `method = :block_cg` to allocate the workspace,
and [`krylov_solve!`](@ref) to run the block Krylov method in-place.
"""
function block_cg! end

"""
    kgram!(G, X, Y)

`G ← Xᴴ Y` for tall and skinny blocks `X` and `Y` (`n × r` and `n × s` with `n ≫ r, s`), the
Gram products of [`block_cg`](@ref). In double precision with 2 to 8 columns it computes one
matrix-vector product per column, otherwise it calls `mul!`. Array backends can specialise it
where another kernel is faster for such shapes.
"""
function kgram!(G, X, Y)
  # GEMM kernels for such shapes often do not split the long reduction dimension: in double
  # precision one GEMV per column was 2–30× faster with cuBLAS and up to 2× with OpenBLAS.
  # Single precision GEMM is fast already.
  if eltype(G) <: Union{Float64, ComplexF64} && 1 < size(Y, 2) ≤ 8
    for j in axes(Y, 2)
      mul!(view(G, :, j), X', view(Y, :, j))
    end
    return G
  end
  return mul!(G, X', Y)
end

# All small dense work (Gram matrices of the search block, its orthonormalization, the r × r
# solves) happens on the host in buffers of the workspace, with the plain-Julia kernels below:
# no allocation after the workspace is built, and any floating-point type (BigFloat, Float16)
# works. All device operations act on whole n × p and p × p arrays (no views, which GPU array
# packages may hand to slow generic kernels): after an orthonormalization with rank r < p the
# trailing columns of P are zero, and the coefficient matrices copied back to the device have
# zero rows r+1:p, so the full products equal the products with the leading r columns.

# X ← (I - V Vᴴ) X: orthogonal projection onto the range of A (V orthonormal)
function _bcg_project!(X, workspace)
  size(workspace.V, 2) == 0 && return X
  kgram!(workspace.K, workspace.V, X)
  mul!(X, workspace.V, workspace.K, -one(eltype(X)), one(eltype(X)))
  return X
end

# X ← X - V (WᴴV)⁻¹ Wᴴ X: oblique projection onto {Wᴴx = 0} along the null space
function _bcg_ground!(X, workspace)
  size(workspace.V, 2) == 0 && return X
  kgram!(workspace.K, workspace.W, X)
  mul!(X, workspace.F, workspace.K, -one(eltype(X)), one(eltype(X)))
  return X
end

# Euclidean column norms of X: one reduction on the device, p values to the host.
function _bcg_colnorms!(out::Vector{T}, X, workspace) where T
  sum!(abs2, workspace.colsums, X)
  copyto!(workspace.colsumsh, workspace.colsums)
  for j in eachindex(out)
    out[j] = sqrt(max(real(workspace.colsumsh[j]), zero(T)))
  end
  return out
end

# Cyclic Jacobi method for the Hermitian matrix in the leading n × n block of A: eigenvalues in
# λ[1:n], eigenvectors in the columns of the leading n × n block of E. A is overwritten. Each
# rotation is a unitary G = diag(1, conj(φ)) R, where φ = a_pq/|a_pq| makes the 2 × 2 block real
# and R is the real Jacobi rotation that annihilates it.
function _bcg_jacobi!(A::Matrix{FC}, E::Matrix{FC}, λ::Vector{T}, n::Int) where {T, FC}
  for j = 1:n, i = 1:n
    E[i,j] = i == j ? one(FC) : zero(FC)
  end
  for sweep = 1:60
    off = zero(T)
    nrm = zero(T)
    for j = 1:n, i = 1:n
      a = abs2(A[i,j])
      nrm += a
      i == j || (off += a)
    end
    off ≤ eps(T)^2 * nrm && break
    for p = 1:n-1, q = p+1:n
      apq = A[p,q]
      aa = abs(apq)
      aa == 0 && continue
      φ = apq / aa
      θ = (real(A[q,q]) - real(A[p,p])) / (2 * aa)
      t = θ == 0 ? one(T) : sign(θ) / (abs(θ) + sqrt(θ^2 + one(T)))
      c = inv(sqrt(t^2 + one(T)))
      s = t * c
      g11, g12, g21, g22 = FC(c), FC(s), -s * conj(φ), c * conj(φ)
      for k = 1:n  # columns: A ← A G, E ← E G
        akp, akq = A[k,p], A[k,q]
        A[k,p] = akp * g11 + akq * g21
        A[k,q] = akp * g12 + akq * g22
        ekp, ekq = E[k,p], E[k,q]
        E[k,p] = ekp * g11 + ekq * g21
        E[k,q] = ekp * g12 + ekq * g22
      end
      for k = 1:n  # rows: A ← Gᴴ A
        apk, aqk = A[p,k], A[q,k]
        A[p,k] = conj(g11) * apk + conj(g21) * aqk
        A[q,k] = conj(g12) * apk + conj(g22) * aqk
      end
      A[p,q] = A[q,p] = zero(FC)
      A[p,p] = real(A[p,p])
      A[q,q] = real(A[q,q])
    end
  end
  for i = 1:n
    λ[i] = real(A[i,i])
  end
  return λ
end

# perm[1:n] ← indices of λ[1:n] in decreasing order (insertion sort, n is the block size)
function _bcg_sortperm!(perm::Vector{Int}, λ::Vector, n::Int)
  for i = 1:n
    perm[i] = i
  end
  for i = 2:n
    k = perm[i]
    j = i - 1
    while j ≥ 1 && λ[perm[j]] < λ[k]
      perm[j+1] = perm[j]
      j -= 1
    end
    perm[j+1] = k
  end
  return perm
end

# Replace the leading columns of P by an orthonormal basis of span(P) and return its dimension
# r (SVQB). The columns are scaled to unit norm first, so that only (near) linear dependence,
# not scale, decides which directions are dropped. Q is used as scratch.
function _bcg_orthonormalize!(workspace, rank_tol::T) where T
  P, Q, d = workspace.P, workspace.Q, workspace.d
  Gh, Eh, Th, λ, perm = workspace.Gh, workspace.Eh, workspace.Th, workspace.λh, workspace.perm
  p = size(P, 2)
  kgram!(workspace.G, P, P)
  copyto!(Gh, workspace.G)
  tiny = floatmin(T) / eps(T)
  for j = 1:p
    d[j] = real(Gh[j,j]) > tiny ? inv(sqrt(real(Gh[j,j]))) : zero(T)
  end
  for j = 1:p, i = 1:p
    Gh[i,j] *= d[i] * d[j]
  end
  _bcg_jacobi!(Gh, Eh, λ, p)
  _bcg_sortperm!(perm, λ, p)
  λmax = max(λ[perm[1]], zero(T))
  λmax > 0 || return 0
  r = 0
  while r < p && λ[perm[r+1]] > rank_tol * λmax
    r += 1
  end
  r == 0 && return 0
  FC = eltype(Th)
  for c = 1:p  # the r largest eigenvalues; columns r+1:p zero
    j = perm[c]
    s = c ≤ r ? inv(sqrt(λ[j])) : zero(T)
    for i = 1:p
      Th[i,c] = c ≤ r ? d[i] * Eh[i,j] * s : zero(FC)
    end
  end
  copyto!(workspace.C, Th)
  mul!(Q, P, workspace.C)
  P .= Q
  return r
end

# Factorize the Hermitian positive (semi)definite matrix in the leading r × r block of Gh for
# _bcg_spd_solve!: Cholesky L Lᴴ in Lh, or, if that fails, its eigendecomposition in Eh, λh (for
# a pseudo-inverse). Gh is kept.
function _bcg_spd_factor!(workspace, r::Int)
  Gh, L = workspace.Gh, workspace.Lh
  FC = eltype(Gh)
  T = real(FC)
  ok = true
  for j = 1:r
    s = real(Gh[j,j])
    for k = 1:j-1
      s -= abs2(L[j,k])
    end
    if !(s > 0)
      ok = false
      break
    end
    L[j,j] = sqrt(s)
    for i = j+1:r
      v = Gh[i,j]
      for k = 1:j-1
        v -= L[i,k] * conj(L[j,k])
      end
      L[i,j] = v / L[j,j]
    end
  end
  if !ok
    W = workspace.Wh
    for j = 1:r, i = 1:r
      W[i,j] = Gh[i,j]
    end
    _bcg_jacobi!(W, workspace.Eh, workspace.λh, r)
  end
  workspace.chol_ok = ok
  return workspace
end

# Ch[1:r, 1:ncols] ← G⁻¹ Ch[1:r, 1:ncols] with the factorization of _bcg_spd_factor!, times
# `scale`; the rows r+1:p of Ch are set to zero.
function _bcg_spd_solve!(workspace, r::Int, ncols::Int, scale = true)
  C = workspace.Ch
  FC = eltype(C)
  T = real(FC)
  if workspace.chol_ok
    L = workspace.Lh
    for c = 1:ncols
      for i = 1:r  # L z = c
        v = C[i,c]
        for k = 1:i-1
          v -= L[i,k] * C[k,c]
        end
        C[i,c] = v / L[i,i]
      end
      for i = r:-1:1  # Lᴴ y = z
        v = C[i,c]
        for k = i+1:r
          v -= conj(L[k,i]) * C[k,c]
        end
        C[i,c] = v / conj(L[i,i])
      end
    end
  else
    E, λ, w = workspace.Eh, workspace.λh, view(workspace.Wh, :, 1)
    λmax = zero(T)
    for i = 1:r
      λmax = max(λmax, abs(λ[i]))
    end
    for c = 1:ncols
      for j = 1:r  # w = diag(λ⁺) Eᴴ c
        v = zero(FC)
        for i = 1:r
          v += conj(E[i,j]) * C[i,c]
        end
        w[j] = λ[j] > sqrt(eps(T)) * λmax ? v / λ[j] : zero(FC)
      end
      for i = 1:r  # c = E w
        v = zero(FC)
        for j = 1:r
          v += E[i,j] * w[j]
        end
        C[i,c] = v
      end
    end
  end
  for c = 1:ncols
    for i = 1:r
      C[i,c] *= scale
    end
    for i = r+1:size(C, 1)
      C[i,c] = zero(FC)
    end
  end
  return C
end

# Z ← Π M⁻¹ R, with the columns of converged right-hand sides set to zero (deflation)
function _bcg_precondition!(workspace, M, MisI::Bool, ldiv::Bool)
  Z, R = workspace.Z, workspace.R
  if MisI
    copyto!(Z, R)
  else
    ldiv ? ldiv!(Z, M, R) : mul!(Z, M, R)
  end
  _bcg_project!(Z, workspace)
  if !all(workspace.active)
    FC = eltype(Z)
    for j in eachindex(workspace.active)
      workspace.maskh[1,j] = workspace.active[j] ? one(FC) : zero(FC)
    end
    copyto!(workspace.mask, workspace.maskh)
    Z .*= workspace.mask
  end
  return Z
end

def_args_block_cg = (:(A                    ),
                     :(B::AbstractMatrix{FC}))

def_optargs_block_cg = (:(X0::AbstractMatrix),)

def_kwargs_block_cg = (:(; M = I                        ),
                       :(; ldiv::Bool = false           ),
                       :(; atol::T = √eps(T)            ),
                       :(; rtol::T = √eps(T)            ),
                       :(; itmax::Int = 0               ),
                       :(; rank_tol::T = √eps(T)        ),
                       :(; recompute_every::Int = 50    ),
                       :(; timemax::Float64 = Inf       ),
                       :(; verbose::Int = 0             ),
                       :(; history::Bool = false        ),
                       :(; callback = workspace -> false),
                       :(; iostream::IO = kstdout       ))

def_kwargs_workspace_block_cg = (:(; nullspace = nothing),
                                 :(; grounding = nothing))

def_kwargs_block_cg = mapreduce(extract_parameters, vcat, def_kwargs_block_cg)
def_kwargs_workspace_block_cg = extract_parameters.(def_kwargs_workspace_block_cg)

args_block_cg = (:A, :B)
optargs_block_cg = (:X0,)
kwargs_block_cg = (:M, :ldiv, :atol, :rtol, :itmax, :rank_tol, :recompute_every, :timemax, :verbose, :history, :callback, :iostream)
kwargs_workspace_block_cg = (:nullspace, :grounding)

@eval begin
  function block_cg!(workspace :: BlockCgWorkspace{T,FC,SV,SM}, $(def_args_block_cg...); $(def_kwargs_block_cg...)) where {T <: AbstractFloat, FC <: FloatOrComplex{T}, SV <: AbstractVector{FC}, SM <: AbstractMatrix{FC}}

    # Timer
    start_time = time_ns()
    timemax_ns = 1e9 * timemax

    n, m = size(A)
    s, p = size(B)
    m == n || error("System must be square")
    n == s || error("Inconsistent problem size")
    p == workspace.p || error("The workspace was built for $(workspace.p) right-hand sides")
    (verbose > 0) && @printf(iostream, "BLOCK-CG: system of size %d with %d right-hand sides\n", n, p)

    # Check M = Iₙ
    MisI = (M === I)

    # Check type consistency
    eltype(A) == FC || @warn "eltype(A) ≠ $FC. This could lead to errors or additional allocations in operator-matrix products."
    ktypeof(B) == SM || error("ktypeof(B) must be equal to $SM")

    # Set up workspace.
    X, R, Q = workspace.X, workspace.R, workspace.Q
    bnorm, rnorm, tol, active = workspace.bnorm, workspace.rnorm, workspace.tol, workspace.active
    warm_start = workspace.warm_start
    stats = workspace.stats
    RNorms = stats.residuals
    reset!(stats)

    # Compatibility of the right-hand sides: ‖B‖² = ‖ΠB‖² + ‖(I - Π)B‖².
    _bcg_colnorms!(rnorm, B, workspace)
    nB² = sum(abs2, rnorm)
    copyto!(R, B)
    _bcg_project!(R, workspace)
    _bcg_colnorms!(bnorm, R, workspace)
    defect = nB² > 0 ? sqrt(max(nB² - sum(abs2, bnorm), zero(T)) / nB²) : zero(T)
    workspace.defect = defect
    for j = 1:p
      tol[j] = atol + rtol * bnorm[j]
    end

    # Initial guess X₀ ∈ range(A) and residual R₀ = Π(B - AX₀).
    if warm_start
      copyto!(X, workspace.ΔX)
      _bcg_project!(X, workspace)
      mul!(R, A, X)
      R .= B .- R
      _bcg_project!(R, workspace)
    else
      fill!(X, zero(FC))
    end
    _bcg_colnorms!(rnorm, R, workspace)
    for j = 1:p
      active[j] = rnorm[j] > tol[j]
    end
    RNorm = sqrt(sum(abs2, rnorm))  # ‖R₀‖_F
    history && push!(RNorms, RNorm)

    iter = 0
    itmax == 0 && (itmax = 2 * n)

    (verbose > 0) && @printf(iostream, "%5s  %7s  %5s  %5s\n", "k", "‖Rₖ‖", "rank", "timer")
    kdisplay(iter, verbose) && @printf(iostream, "%5d  %7.1e  %5s  %.2fs\n", iter, RNorm, "-", start_time |> ktimer)

    # Stopping criterion
    status = "unknown"
    solved = !any(active)
    tired = iter ≥ itmax
    exhausted = false
    user_requested_exit = false
    overtimed = false

    if !solved
      _bcg_precondition!(workspace, M, MisI, ldiv)
      copyto!(workspace.P, workspace.Z)
    end

    while !(solved || tired || exhausted || user_requested_exit || overtimed)
      # Update iteration index.
      iter = iter + 1

      # Orthonormal basis of the search block.
      r = _bcg_orthonormalize!(workspace, rank_tol)
      if r == 0
        exhausted = true
        break
      end
      P = workspace.P
      mul!(Q, A, P)  # Q ← AP

      # Gₖ = PᴴAP (leading r × r block) and αₖ = Gₖ⁻¹ PᴴR (r × p), both solved on the host.
      kgram!(workspace.G, P, Q)
      Gh = workspace.Gh
      copyto!(Gh, workspace.G)
      for j = 1:r, i = 1:j
        Gh[i,j] = (Gh[i,j] + conj(Gh[j,i])) / 2
        Gh[j,i] = conj(Gh[i,j])
      end
      _bcg_spd_factor!(workspace, r)
      Cd = workspace.C
      kgram!(Cd, P, R)
      copyto!(workspace.Ch, Cd)
      _bcg_spd_solve!(workspace, r, p)
      copyto!(Cd, workspace.Ch)

      # Xₖ = Xₖ₋₁ + Pαₖ and Rₖ = Rₖ₋₁ - APαₖ (or the true residual every `recompute_every` iterations).
      mul!(X, P, Cd, one(FC), one(FC))
      if iter % recompute_every == 0
        mul!(R, A, X)
        R .= B .- R
        _bcg_project!(R, workspace)
      else
        mul!(R, Q, Cd, -one(FC), one(FC))
      end

      _bcg_colnorms!(rnorm, R, workspace)
      for j = 1:p
        active[j] = rnorm[j] > tol[j]
      end
      RNorm = sqrt(sum(abs2, rnorm))
      history && push!(RNorms, RNorm)

      # Update stopping criterion.
      user_requested_exit = callback(workspace) :: Bool
      solved = !any(active)
      tired = iter ≥ itmax
      timer = time_ns() - start_time
      overtimed = timer > timemax_ns
      kdisplay(iter, verbose) && @printf(iostream, "%5d  %7.1e  %5d  %.2fs\n", iter, RNorm, r, start_time |> ktimer)

      # Next search block: P ← Z + Pβₖ with βₖ = -Gₖ⁻¹ (AP)ᴴZ, A-conjugate to the current P.
      if !(solved || tired || user_requested_exit || overtimed)
        Z = _bcg_precondition!(workspace, M, MisI, ldiv)
        kgram!(Cd, Q, Z)
        copyto!(workspace.Ch, Cd)
        _bcg_spd_solve!(workspace, r, p, -one(FC))
        copyto!(Cd, workspace.Ch)
        mul!(Z, P, Cd, one(FC), one(FC))
        @kswap!(workspace.P, workspace.Z)
      end
    end
    (verbose > 0) && @printf(iostream, "\n")

    # Fix the component in the null space: Wᴴ X = 0.
    _bcg_ground!(X, workspace)

    # Termination status
    tired               && (status = "maximum number of iterations exceeded")
    exhausted           && (status = "no search direction left (rank-deficient search block)")
    solved              && (status = "solution good enough given atol and rtol")
    overtimed           && (status = "time limit exceeded")
    user_requested_exit && (status = "user-requested exit")

    workspace.warm_start = false

    # Update stats
    stats.niter = iter
    stats.solved = solved
    stats.inconsistent = defect > √eps(T)
    stats.timer = start_time |> ktimer
    stats.status = status
    return workspace
  end
end
