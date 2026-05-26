# An implementation of SQMR for the solution of Hermitian (self-adjoint)
# square linear systems Ax = b.
#
# SQMR is based on the *symmetric* (one-sided) Lanczos process.
# Unlike QMR, which solves unsymmetric systems via the two-sided biorthogonal
# Lanczos process (requiring products with both A and Aᴴ), SQMR exploits the
# symmetry A = Aᴴ: the left and right Lanczos vectors collapse to one sequence,
# halving storage and eliminating all Aᴴ products.
#
# The key structural consequence is that the projected matrix Tₖ is
# *symmetric tridiagonal* (not the general banded matrix in QMR), and the
# QMR quasi-minimisation step reduces to a simple two-rotation Givens update
# on this tridiagonal — identical in form to MINRES.
# Without preconditioning, SQMR and MINRES are mathematically equivalent.
# The advantage of SQMR over MINRES is that it accommodates symmetric
# *indefinite* preconditioners, for which MINRES breaks down.
#
# The preconditioned variant uses split (centred) preconditioning: given a
# symmetric preconditioner M (not necessarily positive definite), the Lanczos
# inner products are taken with respect to the M-inner product
#   ⟨u, v⟩_M  =  uᵀ (M⁻¹ v),
# so we maintain alongside each Lanczos vector vₖ its M-image zₖ = M⁻¹ vₖ.
# Breakdowns (⟨v̂, M⁻¹v̂⟩ = 0) are detected exactly; the sign of this inner
# product is allowed to be negative.
#
# This implementation follows Algorithm 7.9 in:
#
#   Y. Saad, Iterative Methods for Sparse Linear Systems, 2nd ed.,
#   SIAM, Philadelphia, 2003.
#
# and the original reference:
#
#   R. W. Freund and N. M. Nachtigal,
#   A new Krylov-subspace method for symmetric indefinite linear systems.
#   Proc. 14th IMACS World Congress on Computational and Applied Mathematics
#   (W. F. Ames, ed.), pp. 1253--1256, 1994.
#
# Alexis Montoison, <alexis.montoison@polymtl.ca>  (workspace / macro pattern)
# Arnav Kapoor, <arnavkapoor23@iiserb.ac.in>        (SQMR implementation)
# Bhopal / Montréal, 2026.

export sqmr, sqmr!

"""
    (x, stats) = sqmr(A, b::AbstractVector{FC};
                      M=I, ldiv::Bool=false,
                      atol::T=√eps(T), rtol::T=√eps(T),
                      itmax::Int=0, timemax::Float64=Inf,
                      verbose::Int=0, history::Bool=false,
                      callback=workspace->false, iostream::IO=kstdout)

`T` is an `AbstractFloat` such as `Float32`, `Float64` or `BigFloat`.
`FC` is `T` or `Complex{T}`.

    (x, stats) = sqmr(A, b, x0::AbstractVector; kwargs...)

SQMR can be warm-started from an initial guess `x0` where `kwargs` are the same keyword arguments as above.

Solve the square Hermitian (self-adjoint) linear system `Ax = b` of size `n`
using the Symmetric QMR method.

SQMR uses the one-sided symmetric Lanczos process and does not require
products with `Aᴴ`.  Unlike MINRES, SQMR supports symmetric *indefinite*
preconditioners; without preconditioning the two methods are mathematically
equivalent.

#### Interface

To easily switch between Krylov methods, use the generic interface [`krylov_solve`](@ref) with `method = :sqmr`.

For an in-place variant that reuses memory across solves, see [`sqmr!`](@ref).

#### Input arguments

* `A`: a linear operator that models a Hermitian matrix of dimension `n`;
* `b`: a vector of length `n`.

#### Optional argument

* `x0`: a vector of length `n` that represents an initial guess of the solution `x`.

#### Keyword arguments

* `M`: linear operator that models a Hermitian (possibly indefinite) nonsingular matrix of size `n` used for centred preconditioning;
* `ldiv`: define whether the preconditioner uses `ldiv!` or `mul!`;
* `atol`: absolute stopping tolerance based on the residual norm;
* `rtol`: relative stopping tolerance based on the residual norm;
* `itmax`: the maximum number of iterations. If `itmax=0`, the default number of iterations is set to `2n`;
* `timemax`: the time limit in seconds;
* `verbose`: additional details can be displayed if verbose mode is enabled (verbose > 0). Information will be displayed every `verbose` iterations;
* `history`: collect additional statistics on the run such as residual norms;
* `callback`: function or functor called as `callback(workspace)` that returns `true` if the Krylov method should terminate, and `false` otherwise;
* `iostream`: stream to which output is logged.

#### Output arguments

* `x`: a dense vector of length `n`;
* `stats`: statistics collected on the run in a [`SimpleStats`](@ref) structure.

#### References

* R. W. Freund and N. M. Nachtigal, [*A new Krylov-subspace method for symmetric indefinite linear systems*](https://www.osti.gov/biblio/36034), Proc. 14th IMACS World Congress, pp. 1253--1256, 1994.
* Y. Saad, *Iterative Methods for Sparse Linear Systems*, 2nd ed., SIAM, 2003.
"""
function sqmr end

"""
    workspace = sqmr!(workspace::SqmrWorkspace, A, b; kwargs...)
    workspace = sqmr!(workspace::SqmrWorkspace, A, b, x0; kwargs...)

In these calls, `kwargs` are keyword arguments of [`sqmr`](@ref).

See [`SqmrWorkspace`](@ref) for instructions on how to create the `workspace`.

For a more generic interface, you can use [`krylov_workspace`](@ref) with `method = :sqmr` to allocate the workspace,
and [`krylov_solve!`](@ref) to run the Krylov method in-place.
"""
function sqmr! end

def_args_sqmr = (:(A                          ),
                 :(b::AbstractVector{FC}       ))

def_optargs_sqmr = (:(x0::AbstractVector),)

def_kwargs_sqmr = (:(; M          = I                ),
                   :(; ldiv::Bool = false             ),
                   :(; atol::T    = √eps(T)           ),
                   :(; rtol::T    = √eps(T)           ),
                   :(; itmax::Int = 0                 ),
                   :(; timemax::Float64 = Inf         ),
                   :(; verbose::Int     = 0           ),
                   :(; history::Bool    = false       ),
                   :(; callback         = workspace -> false),
                   :(; iostream::IO     = kstdout     ))

def_kwargs_sqmr = extract_parameters.(def_kwargs_sqmr)

args_sqmr    = (:A, :b)
optargs_sqmr = (:x0,)
kwargs_sqmr  = (:M, :ldiv, :atol, :rtol, :itmax, :timemax, :verbose, :history, :callback, :iostream)

@eval begin
  function sqmr!(workspace :: SqmrWorkspace{T,FC,S}, $(def_args_sqmr...); $(def_kwargs_sqmr...)) where {T <: AbstractFloat, FC <: FloatOrComplex{T}, S <: AbstractVector{FC}}
    # Timer
    start_time = time_ns()
    timemax_ns = 1e9 * timemax

    m, n = size(A)
    (m == workspace.m && n == workspace.n) || error("(workspace.m, workspace.n) = ($(workspace.m), $(workspace.n)) is inconsistent with size(A) = ($m, $n)")
    m == n || error("System must be square")
    length(b) == n || error("Inconsistent problem size")
    (verbose > 0) && @printf(iostream, "SQMR: system of size %d\n", n)

    # Check M = Iₙ
    MisI = (M === I)

    # Check type consistency
    eltype(A) == FC || @warn "eltype(A) ≠ $FC. This could lead to errors or additional allocations in operator-vector products."
    ktypeof(b) == S || error("ktypeof(b) must be equal to $S")

    # Set up workspace.
    # z = M⁻¹vₖ is only needed when M ≠ I; when M = I it aliases r2.
    allocate_if(!MisI, workspace, :z, S, workspace.x)

    Δx, x, r1, r2, w1, w2, stats = workspace.Δx, workspace.x, workspace.r1, workspace.r2, workspace.w1, workspace.w2, workspace.stats
    warm_start = workspace.warm_start
    rNorms = stats.residuals
    reset!(stats)

    z = MisI ? r2 : workspace.z

    # Initial solution x₀ and residual r₀ = b - Ax₀.
    kfill!(x, zero(FC))
    if warm_start
      kmul!(r1, A, Δx)
      kaxpby!(n, one(FC), b, -one(FC), r1)   # r1 ← b - A*Δx
    else
      kcopy!(n, r1, b)                        # r1 ← b
    end

    # Initialize the symmetric Lanczos process.
    # β₁ M v₁ = b  ⟹  v₁ = b / β₁,  β₁ = √|⟨b, M⁻¹b⟩|
    kcopy!(n, r2, r1)                         # r2 ← r1  (holds v₁, unnormalised)
    MisI || mulorldiv!(z, M, r1, ldiv)        # z ← M⁻¹ r1  (= M⁻¹ v₁ unnorm.)

    β₁ = kdotr(n, r1, z)                      # β₁² = ⟨v₁, M⁻¹v₁⟩  (can be < 0 for indef. M)

    if β₁ == 0
      stats.niter        = 0
      stats.solved       = true
      stats.inconsistent = false
      stats.timer        = start_time |> ktimer
      stats.status       = "x is a zero-residual solution"
      history && push!(rNorms, zero(T))
      warm_start && kaxpy!(n, one(FC), Δx, x)
      workspace.warm_start = false
      return workspace
    end

    β₁  = sqrt(abs(β₁))
    β   = β₁
    oldβ = zero(T)

    rNorm = β₁
    history && push!(rNorms, rNorm)

    # Scalars for the QMR update on the symmetric tridiagonal Tₖ.
    # We maintain the running QR factorisation via two successive Givens rotations,
    # exactly as in MINRES.  The notation follows Saad §6.7 + §7.4.
    #
    #   δbar   — pending (unfinalised) diagonal entry before the new rotation
    #   ε      — fill-in from the rotation applied two steps ago
    #   ϕbar   — residual factor; ‖rₖ‖ ≈ |ϕbar| in exact arithmetic
    #
    δbar = zero(T)
    ε    = zero(T)
    ϕbar = β₁
    cs   = -one(T)     # cosine of the previous Givens rotation
    sn   = zero(T)     # sine  of the previous Givens rotation

    kfill!(w1, zero(FC))
    kfill!(w2, zero(FC))

    iter   = 0
    itmax == 0 && (itmax = 2*n)
    ε_tol  = atol + rtol * β₁

    (verbose > 0) && @printf(iostream, "%5s  %8s  %7s  %5s\n", "k", "αₖ", "‖rₖ‖", "timer")
    kdisplay(iter, verbose) && @printf(iostream, "%5d  %8.1e  %7.1e  %.2fs\n", iter, zero(T), rNorm, start_time |> ktimer)

    solved             = rNorm ≤ ε_tol
    tired              = iter  ≥ itmax
    breakdown          = false
    user_requested_exit = false
    overtimed          = false
    status             = "unknown"

    while !(solved || tired || breakdown || user_requested_exit || overtimed)
      # Update iteration index.
      iter = iter + 1

      # Symmetric Lanczos step.
      # Three-term recurrence:
      #   y ← A zₖ                          (matvec; zₖ = M⁻¹vₖ stored in z)
      #   y ← y / βₖ
      #   y ← y − (βₖ/βₖ₋₁) vₖ₋₁           (deflate previous direction)
      #   αₖ = ⟨vₖ, y⟩ / βₖ                 (Ritz value; real by Hermitian symmetry)
      #   y ← y − (αₖ/βₖ) vₖ               (deflate current direction → v̂ₖ₊₁)
      #   zₖ₊₁ ← M⁻¹ v̂ₖ₊₁
      #   βₖ₊₁ = √|⟨v̂ₖ₊₁, zₖ₊₁⟩|

      kmul!(w1, A, z)                         # w1 ← A zₖ  (scratch; safe to overwrite w1 here)
      kdiv!(n, w1, β)                         # w1 ← w1 / βₖ
      iter ≥ 2 && kaxpy!(n, -β / oldβ, r1, w1)  # w1 ← w1 − (βₖ/βₖ₋₁) vₖ₋₁

      αₖ = real(kdot(n, r2, w1)) / β         # αₖ = Re⟨vₖ, w1⟩ / βₖ  (real for Hermitian A)

      kaxpy!(n, -αₖ / β, r2, w1)             # w1 ← w1 − (αₖ/βₖ) vₖ  →  v̂ₖ₊₁ in w1

      # Apply M to v̂ₖ₊₁ to get M-image for the next β.
      MisI || mulorldiv!(z, M, w1, ldiv)      # z ← M⁻¹ v̂ₖ₊₁

      pq = kdotr(n, w1, MisI ? w1 : z)       # ⟨v̂ₖ₊₁, M⁻¹v̂ₖ₊₁⟩  (real for Hermitian M)

      if pq == 0
        breakdown = !solved
        break
      end

      β_next = sqrt(abs(pq))                  # βₖ₊₁

      # QMR update on the symmetric tridiagonal Tₖ₊₁,ₖ.
      #
      # Apply the previous Givens rotation Qₖ₋₁ to the new column [0; αₖ; βₖ₊₁]:
      #   [  cs   sn ] [ δbar ] = [  γbar ]
      #   [ -sn   cs ] [  αₖ  ] = [  ...  ]
      # (the second row gives the new pending diagonal δbar for the next step)
      #
      #    [ γbar ]   [ cs    sn ] [ δbar ]
      #    [ δbar']   [-sn    cs ] [  αₖ  ]
      #
      γbar  =  cs * δbar + sn * αₖ            # partial diagonal after previous rotation
      δbar  = -sn * δbar + cs * αₖ            # new pending diagonal
      ε_new =  sn * β_next                    # fill-in from this rotation
      δbar2 = -cs * β_next                    # δbar updated once more for next iter

      # Compute the direction vector wₖ = (vₖ − ε wₖ₋₂ − γbar wₖ₋₁) / γ
      # where γ is determined by the *new* rotation below.
      # Build the unnormalised direction in w2 first, then divide by γ after.
      if iter == 1
        kcopy!(n, w2, r2)                     # w2 ← v₁  (no previous directions)
        kdiv!(n, w2, γbar)                    # w2 ← v₁ / γbar₁  (= w₁, but γbar₁ = α₁ here
                                              #   before the new rotation; we divide by γ below)
        # Undo the divide; we redo it properly after computing γ.
        kscal!(n, γbar, w2)                   # w2 ← v₁  (restored)
      elseif iter == 2
        # w₂ = (v₂ − ε₁ w₀ − γbar₁ w₁) / γ₂;  w₀ = 0, so just:
        # w₂ = (v₂ − γbar₁ w₁) / γ₂
        kscal!(n, -γbar, w1)                  # w1 currently holds w₁ (from iter 1 after swap);
                                              # temporarily: w1 ← −γbar₁ w₁
        kcopy!(n, w2, r2)                     # w2 ← v₂
        kaxpy!(n, one(FC), w1, w2)            # w2 ← v₂ − γbar₁ w₁  (unnorm. w₂)
        # restore w1 (needed for ε in iter≥3); divide by γ after computing it below
      else
        # wₖ = (vₖ − ε wₖ₋₂ − γbar wₖ₋₁) / γ
        # After the @kswap! below, w1 = wₖ₋₂ and w2 = wₖ₋₁.
        kscal!(n, -ε, w1)                     # w1 ← −ε wₖ₋₂
        kaxpy!(n, -γbar, w2, w1)              # w1 ← −ε wₖ₋₂ − γbar wₖ₋₁
        kaxpy!(n, one(FC), r2, w1)            # w1 ← vₖ − ε wₖ₋₂ − γbar wₖ₋₁  (unnorm. wₖ)
        kcopy!(n, w2, w1)                     # w2 ← unnorm. wₖ  (will be normalised by γ)
      end

      # New Givens rotation to annihilate βₖ₊₁:
      #   [ cs'  sn' ] [ γbar    ] = [ γ ]
      #   [-sn'  cs' ] [ βₖ₊₁   ]   [ 0 ]
      (cs, sn, γ) = sym_givens(γbar, β_next)
      γ = max(γ, eps(T))                      # guard against exact zero

      # Normalise the direction vector.
      kdiv!(n, w2, γ)                         # w2 ← wₖ = (unnorm. wₖ) / γ

      # Update the right-hand side of the QMR least-squares system:
      #   ϕₖ    = cs * ϕbar
      #   ϕbar  = sn * ϕbar
      ϕₖ    = cs * ϕbar
      ϕbar  = sn * ϕbar

      # Update solution: xₖ ← xₖ₋₁ + ϕₖ wₖ
      kaxpy!(n, FC(ϕₖ), w2, x)

      # Residual norm estimate: ‖rₖ‖ ≈ |ϕbar|  (exact in exact arithmetic)
      rNorm = abs(ϕbar)
      history && push!(rNorms, rNorm)

      # Advance Lanczos vectors.
      kcopy!(n, r1, r2)                       # r1 ← vₖ  (becomes vₖ₋₁)
      kcopy!(n, r2, w1)                       # r2 ← v̂ₖ₊₁ (unnormalised; stored in w1)

      # Normalise vₖ₊₁ and its M-image.
      kdiv!(n, r2, β_next)                    # r2 ← vₖ₊₁ = v̂ₖ₊₁ / βₖ₊₁
      MisI || kdiv!(n, z, β_next)             # z  ← M⁻¹vₖ₊₁ = M⁻¹v̂ₖ₊₁ / βₖ₊₁

      # Swap direction vectors for next iteration.
      iter ≥ 2 && @kswap!(w1, w2)

      # Advance scalars.
      oldβ = β
      β    = β_next
      ε    = ε_new
      δbar = δbar2

      kdisplay(iter, verbose) && @printf(iostream, "%5d  %8.1e  %7.1e  %.2fs\n", iter, αₖ, rNorm, start_time |> ktimer)

      # Stopping conditions.
      resid_decrease_mach = (rNorm + one(T) ≤ one(T))
      user_requested_exit = callback(workspace) :: Bool
      solved    = (rNorm ≤ ε_tol) || resid_decrease_mach
      tired     = iter ≥ itmax
      timer     = time_ns() - start_time
      overtimed = timer > timemax_ns
    end

    (verbose > 0) && @printf(iostream, "\n")

    # Termination status
    tired               && (status = "maximum number of iterations exceeded")
    breakdown           && (status = "Lanczos breakdown ⟨v̂ₖ₊₁, M⁻¹v̂ₖ₊₁⟩ = 0")
    solved              && (status = "solution good enough given atol and rtol")
    user_requested_exit && (status = "user-requested exit")
    overtimed           && (status = "time limit exceeded")

    # Update x
    warm_start && kaxpy!(n, one(FC), Δx, x)
    workspace.warm_start = false

    # Update stats
    stats.niter        = iter
    stats.solved       = solved
    stats.inconsistent = false
    stats.timer        = start_time |> ktimer
    stats.status       = status

    return workspace
  end

  function sqmr($(def_args_sqmr...); $(def_kwargs_sqmr...)) where {T <: AbstractFloat, FC <: FloatOrComplex{T}}
    start_time = time_ns()
    workspace  = SqmrWorkspace(A, b)
    workspace.stats.timer = start_time |> ktimer
    return sqmr!(workspace, $(args_sqmr...); $(kwargs_sqmr...))
  end

  function sqmr($(def_args_sqmr...), $(def_optargs_sqmr...); $(def_kwargs_sqmr...)) where {T <: AbstractFloat, FC <: FloatOrComplex{T}}
    start_time = time_ns()
    workspace  = SqmrWorkspace(A, b)
    workspace.stats.timer = start_time |> ktimer
    return sqmr!(workspace, $(args_sqmr...), $(optargs_sqmr...); $(kwargs_sqmr...))
  end

  krylov_solve(::Val{:sqmr}, $(def_args_sqmr...); $(def_kwargs_sqmr...)) where {T <: AbstractFloat, FC <: FloatOrComplex{T}} =
    sqmr($(args_sqmr...); $(kwargs_sqmr...))

  krylov_solve(::Val{:sqmr}, $(def_args_sqmr...), $(def_optargs_sqmr...); $(def_kwargs_sqmr...)) where {T <: AbstractFloat, FC <: FloatOrComplex{T}} =
    sqmr($(args_sqmr...), $(optargs_sqmr...); $(kwargs_sqmr...))
end
