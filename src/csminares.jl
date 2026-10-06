# An implementation of CS-MinAres.
#
# This method is described in
#
# S.-C. T. Choi and A. Montoison
# CS-MinAres: Normal-Residual Minimization for Complex Symmetric Linear Systems
# Manuscript in preparation.
#
# Sou-Cheng T. Choi
# Alexis Montoison, <alexis.montoison@polymtl.ca>

export csminares, csminares!

"""
    (x, stats) = csminares(A, b::AbstractVector{FC};
                           M=I, ldiv::Bool=false,
                           λ::FC = zero(FC), atol::T=√eps(T),
                           rtol::T=√eps(T), Artol::T = √eps(T),
                           itmax::Int=0, timemax::Float64=Inf,
                           verbose::Int=0, history::Bool=false,
                           callback=workspace->false, iostream::IO=kstdout)

`T` is an `AbstractFloat` such as `Float32` or `Float64`. `FC` is `Complex{T}`.
CS-MinAres is for complex symmetric systems; a real symmetric matrix is also
Hermitian, so use [`minares`](@ref) for it instead.

    (x, stats) = csminares(A, b, x0::AbstractVector; kwargs...)

CS-MinAres can be warm-started from an initial guess `x0` where `kwargs` are the same keyword arguments as above.

CS-MinAres solves the complex symmetric linear system Ax = b of size n, where
A satisfies `transpose(A) == A` (A may be singular and the system may be
inconsistent). It minimizes the normal residual `‖A'rₖ‖₂`, where
`A' = conj(A)` since `transpose(A) = A`, over the conjugate Saunders trial
space generated from `b` — the normal-residual counterpart of MINARES for
complex symmetric matrices. The estimates computed every iteration are
`‖rₖ‖₂` and `‖A'rₖ‖₂`.

CS-MinAres uses two incremental Givens QR factorizations of the nested,
banded projected normal-residual matrices. Each triangular factor has upper
bandwidth 2, so each staged solution direction depends on two previous ones,
and storage is a fixed number of length-n vectors (no growing basis, unlike
the full-basis research reference `Krylov.SpecialSymmetric.csminares_oracle`).

An exactly stationary iterate found before the trial space closes need not
be the Moore-Penrose solution: for an inconsistent system, CS-MinAres stops
as soon as the normal residual is small, which can happen before the
nullspace component of `b` has been removed. See
`Krylov.SpecialSymmetric.minimum_norm_refinement` for the lift that removes
a leftover nullspace component from a stationary iterate.

#### Interface

To easily switch between Krylov methods, use the generic interface [`krylov_solve`](@ref) with `method = :csminares`.

For an in-place variant that reuses memory across solves, see [`csminares!`](@ref).

#### Input arguments

* `A`: a linear operator that models a complex symmetric (possibly
  singular) matrix of dimension `n`, satisfying `transpose(A) == A`;
* `b`: a vector of length `n`.

#### Optional argument

* `x0`: a vector of length `n` that represents an initial guess of the solution `x`.

#### Keyword arguments

* `M`: preconditioning is not yet supported for CS-MinAres; `M` must be `I`;
* `ldiv`: unused while `M` is restricted to `I`;
* `λ`: shift parameter; CS-MinAres then solves `(A + λI)x = b`, which remains
  complex symmetric for any `λ`;
* `atol`: absolute stopping tolerance based on the residual norm;
* `rtol`: relative stopping tolerance based on the residual norm;
* `Artol`: relative stopping tolerance based on the A'-residual norm;
* `itmax`: the maximum number of iterations. If `itmax=0`, the default number of iterations is set to `4n`;
* `timemax`: the time limit in seconds;
* `verbose`: additional details can be displayed if verbose mode is enabled (verbose > 0). Information will be displayed every `verbose` iterations;
* `history`: collect additional statistics on the run such as residual norms, or A'-residual norms;
* `callback`: function or functor called as `callback(workspace)` that returns `true` if the Krylov method should terminate, and `false` otherwise;
* `iostream`: stream to which output is logged.

#### Output arguments

* `x`: a dense vector of length `n`;
* `stats`: statistics collected on the run in a [`SimpleStats`](@ref) structure.

#### Reference

* S.-C. T. Choi and A. Montoison, *CS-MinAres: Normal-Residual Minimization
  for Complex Symmetric Linear Systems*, manuscript in preparation.
"""
function csminares end

"""
    workspace = csminares!(workspace::CsMinaresWorkspace, A, b; kwargs...)
    workspace = csminares!(workspace::CsMinaresWorkspace, A, b, x0; kwargs...)

In these calls, `kwargs` are keyword arguments of [`csminares`](@ref).

See [`CsMinaresWorkspace`](@ref) for instructions on how to create the `workspace`.

For a more generic interface, you can use [`krylov_workspace`](@ref) with `method = :csminares` to allocate the workspace,
and [`krylov_solve!`](@ref) to run the Krylov method in-place.
"""
function csminares! end

def_args_csminares = (:(A                    ),
                      :(b::AbstractVector{FC}))

def_optargs_csminares = (:(x0::AbstractVector),)

def_kwargs_csminares = (:(; M = I                        ),
                        :(; ldiv::Bool = false           ),
                        :(; λ::FC = zero(FC)             ),
                        :(; atol::T = √eps(T)            ),
                        :(; rtol::T = √eps(T)            ),
                        :(; Artol::T = √eps(T)            ),
                        :(; itmax::Int = 0               ),
                        :(; timemax::Float64 = Inf       ),
                        :(; verbose::Int = 0             ),
                        :(; history::Bool = false        ),
                        :(; callback = workspace -> false),
                        :(; iostream::IO = kstdout       ))

def_kwargs_csminares = extract_parameters.(def_kwargs_csminares)

args_csminares = (:A, :b)
optargs_csminares = (:x0,)
kwargs_csminares = (:M, :ldiv, :λ, :atol, :rtol, :Artol, :itmax, :timemax, :verbose, :history, :callback, :iostream)

# Apply the Hermitian reflection [c s; conj(s) -c] to the pair (x, y), c real.
@inline cs_reflect(c, s, x, y) = (c * x + s * y, conj(s) * x - c * y)

@eval begin
  function csminares!(workspace :: CsMinaresWorkspace{T,FC,S}, $(def_args_csminares...); $(def_kwargs_csminares...)) where {T <: AbstractFloat, FC <: Complex{T}, S <: AbstractVector{FC}}

    # Timer
    start_time = time_ns()
    timemax_ns = 1e9 * timemax

    n, m = size(A)
    (m == workspace.m && n == workspace.n) || error("(workspace.m, workspace.n) = ($(workspace.m), $(workspace.n)) is inconsistent with size(A) = ($m, $n)")
    m == n || error("System must be square")
    length(b) == m || error("Inconsistent problem size")
    (verbose > 0) && @printf(iostream, "CS-MINARES: system of size %d\n", n)

    # Tests M = Iₙ
    MisI = (M === I)
    !MisI && error("Preconditioners are not yet supported")

    # Check type consistency
    eltype(A) == FC || error("eltype(A) ≠ $FC")
    ktypeof(b) == S || error("ktypeof(b) must be equal to $S")

    # Set up workspace: the conjugate Saunders basis cycles through 3 slots
    # (v_{j-1}, v_j are both needed to extend to v_{j+1}). The two staged
    # triangular solves each cycle through 2 solution directions. α, β and
    # the stored 2x2 reflections also cycle through small fixed buffers
    # instead of growing with the iteration.
    Δx, x = workspace.Δx, workspace.x
    vbuf = (workspace.v1, workspace.v2, workspace.v3)
    wbuf = (workspace.w1, workspace.w2)
    dbuf = (workspace.w3, workspace.w4)
    v̄, q, u, Au = workspace.v̄, workspace.q, workspace.u, workspace.Au
    col = workspace.col
    αbuf, βbuf = workspace.αbuf, workspace.βbuf
    c1buf, s1buf, c2buf, s2buf = workspace.c1buf, workspace.s1buf, workspace.c2buf, workspace.s2buf
    warm_start = workspace.warm_start
    stats = workspace.stats
    rNorms, ArNorms = stats.residuals, stats.Aresiduals
    reset!(stats)

    iter = 0
    itmax == 0 && (itmax = 4*n)

    kfill!(x, zero(FC))  # x₀

    getv(j) = vbuf[mod1(j, 3)]
    getw(j) = wbuf[mod1(j, 2)]
    getd(j) = dbuf[mod1(j, 2)]
    α(j) = (j ≥ 1) ? αbuf[mod1(j, 4)] : zero(FC)
    # β₁ is the formal zero preceding the first projected off-diagonal;
    # the residual norm β₁ is stored separately above.
    β(j) = (j ≥ 2) ? βbuf[mod1(j, 4)] : zero(T)
    Tsup(j) = β(j)
    Tsub(j) = β(j + 1)

    # r₀ = b - (A + λI)x₀
    if warm_start
      kmul!(q, A, Δx)
      (λ ≠ 0) && kaxpy!(n, λ, Δx, q)
      kaxpby!(n, one(FC), b, -one(FC), q)
      kcopy!(n, getv(1), q)
    else
      kcopy!(n, getv(1), b)
    end
    β₁ = knorm(n, getv(1))
    (β₁ ≠ 0) && kdiv!(n, getv(1), β₁)

    # γ₁ = ‖A'v₁‖ = ‖conj(A*conj(v₁))‖ (A' = conj(A) since transpose(A) = A)
    v̄ .= conj.(getv(1))
    kmul!(q, A, v̄)
    (λ ≠ 0) && kaxpy!(n, λ, v̄, q)
    γ₁ = knorm(n, q)

    rNorm = β₁
    ε = atol + rtol * rNorm
    history && push!(rNorms, rNorm)
    ArNorm = γ₁
    κ = atol + Artol * ArNorm
    history && push!(ArNorms, ArNorm)

    if rNorm == 0 || ArNorm == 0
      stats.niter = 0
      stats.solved, stats.inconsistent = true, false
      stats.timer = start_time |> ktimer
      stats.status = (rNorm == 0) ? "x is a zero-residual solution" : "x = x0 is already optimal"
      warm_start && kaxpy!(n, one(FC), Δx, x)
      workspace.warm_start = false
      return workspace
    end

    (verbose > 0) && @printf(iostream, "%5s  %7s  %7s  %.2fs\n", "k", "‖rₖ‖", "‖A'rₖ‖", 0.0)
    kdisplay(0, verbose) && @printf(iostream, "%5d  %7.1e  %7.1e  %.2fs\n", 0, rNorm, γ₁, start_time |> ktimer)

    btol = eps(T)^(3//4)
    nsteps = 0
    closed = false
    opscale = zero(T)
    ρ1 = zero(FC); ρ2 = zero(FC)
    Bscale = zero(T)
    λbar = γbar = γprev = ϵprev = ϵprev2 = zero(FC)
    k = 0
    status = "unknown"
    solved = false
    tired = false
    breakdown = false
    truncated = false
    user_requested_exit = false
    overtimed = false

    while !(solved || tired || breakdown || user_requested_exit || overtimed)
      k = k + 1

      # Extend the conjugate Saunders basis until step k+1 is available: one
      # look-ahead step is needed to form column k of the projected
      # normal-residual matrix B_k = conj(Tbar_{k+1}) * Tbar_k.
      while !closed && nsteps < k + 1
        j = nsteps + 1
        vⱼ = getv(j)
        v̄ .= conj.(vⱼ)
        kmul!(q, A, v̄)
        (λ ≠ 0) && kaxpy!(n, λ, v̄, q)
        scale = knorm(n, q)
        opscale = max(opscale, scale)
        (j > 1) && kaxpy!(n, -β(j), getv(j - 1), q)
        αⱼ = kdot(n, vⱼ, q)
        kaxpy!(n, -αⱼ, vⱼ, q)
        tail = knorm(n, q)
        is_closed = (tail ≤ btol * scale)
        αbuf[mod1(j, 4)] = αⱼ
        βbuf[mod1(j + 1, 4)] = is_closed ? zero(T) : tail
        !is_closed && kdivcopy!(n, getv(j + 1), q, tail)
        nsteps = j
        closed = is_closed
      end

      final = closed && k == nsteps

      if k == 1
        ρ1 = β₁ * conj(α(1))
        ρ2 = β₁ * conj(β(2))
      end

      # First stage: Tbar_k = Q_k[R_k; 0], where R_k has upper bandwidth 2.
      k == 1 && (λbar = α(1); γbar = β(2))
      c, s, λₖ = sym_givens(λbar, FC(β(k + 1)))
      αnext = (k < nsteps) ? α(k + 1) : zero(FC)
      βnext2 = (k < nsteps) ? β(k + 2) : zero(T)
      γₖ = c * γbar + s * αnext
      λbar_next = conj(s) * γbar - c * αnext
      ϵₖ = s * βnext2
      γbar_next = -c * βnext2

      # Retain the direct projected-column norm for the existing small-pivot
      # safeguard, without factorizing the bandwidth-4 product directly.
      bkm2 = Tsup(k) * conj(Tsup(k - 1))
      bkm1 = Tsup(k) * conj(α(k - 1)) + α(k) * conj(Tsup(k))
      bk = Tsup(k) * conj(Tsub(k - 1)) + α(k) * conj(α(k)) + Tsub(k) * conj(Tsup(k + 1))
      bkp1 = α(k) * conj(Tsub(k)) + Tsub(k) * conj(αnext)
      bkp2 = Tsub(k) * conj(βnext2)
      Bscale = max(Bscale, sqrt(abs2(bkm2) + abs2(bkm1) + abs2(bk) + abs2(bkp1) + abs2(bkp2)))

      # Second stage: B_k = N_k R_k. N_k is lower banded and its kth
      # column is (conj(λₖ), conj(γₖ), conj(ϵₖ)) in rows k:k+2.
      col[1] = zero(FC); col[2] = zero(FC)
      col[3] = conj(λₖ); col[4] = conj(γₖ); col[5] = conj(ϵₖ)
      col[6] = zero(FC); col[7] = zero(FC)
      # Apply the two preceding pairs of second-stage reflections.
      for j in max(1, k - 2):k-1
        c2ⱼ, s2ⱼ, c1ⱼ, s1ⱼ = c2buf[mod1(j, 2)], s2buf[mod1(j, 2)], c1buf[mod1(j, 2)], s1buf[mod1(j, 2)]
        i = j - k + 3
        col[i+1], col[i+2] = cs_reflect(c2ⱼ, s2ⱼ, col[i+1], col[i+2])
        col[i], col[i+1] = cs_reflect(c1ⱼ, s1ⱼ, col[i], col[i+1])
      end
      c2, s2, r2 = sym_givens(col[4], col[5])
      col[4], col[5] = r2, zero(FC)
      c1, s1, μₖ = sym_givens(col[3], col[4])
      col[3], col[4] = μₖ, zero(FC)
      c2buf[mod1(k, 2)] = c2; s2buf[mod1(k, 2)] = s2
      c1buf[mod1(k, 2)] = c1; s1buf[mod1(k, 2)] = s1

      ρ2, ρ3 = cs_reflect(c2, s2, ρ2, zero(FC))
      ζₖ, ρ2 = cs_reflect(c1, s1, ρ1, ρ2)
      ArNorm = hypot(abs(ρ2), abs(ρ3))
      history && push!(ArNorms, ArNorm)
      ρ1, ρ2 = ρ2, ρ3

      pivot = λₖ * μₖ
      u .= conj.(getv(k))
      (k ≥ 2) && kaxpy!(n, -γprev, getw(k - 1), u)
      (k ≥ 3) && kaxpy!(n, -ϵprev2, getw(k - 2), u)
      nu = knorm(n, u)
      truncated = iszero(λₖ) || iszero(nu)
      if !truncated && abs(pivot) ≤ sqrt(eps(T)) * Bscale
        kmul!(Au, A, u)
        (λ ≠ 0) && kaxpy!(n, λ, u, Au)
        truncated = knorm(n, Au) ≤ sqrt(eps(T)) * opscale * nu
      end
      if !truncated
        kdivcopy!(n, getw(k), u, λₖ)
        kcopy!(n, u, getw(k))
        (k ≥ 2) && kaxpy!(n, -col[2], getd(k - 1), u)
        (k ≥ 3) && kaxpy!(n, -col[1], getd(k - 2), u)
        nu = knorm(n, u)
        truncated = iszero(μₖ) || iszero(nu)
        if !truncated && abs(pivot) ≤ sqrt(eps(T)) * Bscale
          kmul!(Au, A, u)
          (λ ≠ 0) && kaxpy!(n, λ, u, Au)
          truncated = knorm(n, Au) ≤ sqrt(eps(T)) * opscale * nu
        end
      end
      if truncated
        if nu > 0
          coeff = kdot(n, u, x) / nu^2
          kaxpy!(n, -coeff, u, x)
        end
      else
        kdivcopy!(n, getd(k), u, μₖ)
        kaxpy!(n, ζₖ, getd(k), x)
      end

      λbar, γbar = λbar_next, γbar_next
      γprev = γₖ
      ϵprev2, ϵprev = ϵprev, ϵₖ

      iter = k
      breakdown = false
      tired = iter ≥ itmax
      timer = time_ns() - start_time
      overtimed = timer > timemax_ns
      user_requested_exit = callback(workspace) :: Bool
      solved = final || truncated || (ArNorm ≤ κ)
      kdisplay(iter, verbose) && @printf(iostream, "%5d  %7s  %7.1e  %.2fs\n", iter, "-", ArNorm, start_time |> ktimer)
    end
    (verbose > 0) && @printf(iostream, "\n")

    # Recompute explicit residuals for the returned iterate.
    kmul!(q, A, x)
    (λ ≠ 0) && kaxpy!(n, λ, x, q)
    kcopy!(n, v̄, b)
    kaxpy!(n, -one(FC), q, v̄)
    rNorm = knorm(n, v̄)
    q .= conj.(v̄)
    kmul!(v̄, A, q)
    (λ ≠ 0) && kaxpy!(n, λ, q, v̄)
    ArNorm = knorm(n, v̄)
    history && push!(rNorms, rNorm)

    solved = (rNorm ≤ ε) || (ArNorm ≤ κ)
    final_closed = closed && iter == nsteps
    # As in minares!, later conditions take priority: a forced exit
    # (iteration limit, callback, or time limit) overrides a status that
    # would otherwise be reported from the subspace/convergence state alone.
    status = if final_closed
      solved ? "solution good enough given atol, rtol and Artol" : "closed subspace but solution not within tolerance"
    elseif truncated
      solved ? "solution good enough given atol, rtol and Artol" : "rank-deficient subspace but solution not within tolerance"
    elseif solved
      "solution good enough given atol, rtol and Artol"
    else
      "unknown"
    end
    tired               && (status = "maximum number of iterations exceeded")
    user_requested_exit && (status = "user-requested exit")
    overtimed           && (status = "time limit exceeded")

    warm_start && kaxpy!(n, one(FC), Δx, x)
    workspace.warm_start = false

    stats.niter = iter
    stats.solved = solved
    stats.timer = start_time |> ktimer
    stats.status = status
    return workspace
  end
end
