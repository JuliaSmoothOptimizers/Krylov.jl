# An implementation of SQMR for the solution of Hermitian (self-adjoint)
# square linear systems Ax = b.
#
# SQMR is based on the symmetric (one-sided) Lanczos process.
# Unlike QMR, which solves unsymmetric systems via the two-sided biorthogonal
# Lanczos process (requiring products with both A and Aᴴ), SQMR exploits the
# symmetry A = Aᴴ: the left and right Lanczos vectors collapse to one sequence,
# halving storage and eliminating all Aᴴ products.
#
# The key structural consequence is that the projected matrix Tₖ is
# symmetric tridiagonal (not the general banded matrix in QMR), and the
# QMR quasi-minimisation step reduces to a simple two-rotation Givens update
# on this tridiagonal — identical in form to MINRES.
# Without preconditioning, SQMR and MINRES are mathematically equivalent.
# The advantage of SQMR over MINRES is that it accommodates symmetric
# *indefinite* preconditioners, for which MINRES breaks down.
#
# The preconditioned variant uses centred preconditioning: given a
# symmetric preconditioner M (not necessarily positive definite), the Lanczos
# inner products are taken with respect to the M-inner product
#   ⟨u, v⟩_M  =  uᵀ M v,
# and we maintain alongside each normalized Lanczos vector vₖ its M-image
# zₖ = M⁻¹ vₖ.
# Breakdowns (⟨v̂, M⁻¹v̂⟩ = 0) are detected exactly; the sign of this inner
# product is allowed to be negative.
#
# This implementation follows the SQMR description in:
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
* C. C. Paige and M. A. Saunders, [*Solution of Sparse Indefinite Systems of Linear Equations*](https://doi.org/10.1137/0712047), SIAM Journal on Numerical Analysis, 12(4), pp. 617--629, 1975.
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
    allocate_if(true, workspace, :z, S, workspace.x)

    Δx, x, r1, r2, w1, w2, y, stats = workspace.Δx, workspace.x, workspace.r1, workspace.r2, workspace.w1, workspace.w2, workspace.y, workspace.stats
    warm_start = workspace.warm_start
    rNorms = stats.residuals
    reset!(stats)

    z = workspace.z

    # Initial solution x₀ and residual r₀ = b - Ax₀.
    kfill!(x, zero(FC))
    if warm_start
      kmul!(r1, A, Δx)
      kaxpby!(n, one(FC), b, -one(FC), r1)   # r1 ← b - A*Δx
    else
      kcopy!(n, r1, b)                        # r1 ← b
    end
    kcopy!(n, r2, r1)                         # r2 ← r1  (copy of initial residual)

    # Initialize the symmetric Lanczos process.
    # Solve M z₁ = v̂₁, where v̂₁ = r₀ is the unnormalized initial Lanczos vector.
    if MisI
        kcopy!(n, z, r1)                      # z = r₁ (M⁻¹ = I)
    else
        mulorldiv!(z, M, r1, ldiv)            # z ← M⁻¹ r₁ = ẑ₁
    end

    δ₁ = kdotr(n, r1, z)                      # δ₁ = ⟨v̂₁, M⁻¹v̂₁⟩

    if δ₁ == 0
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

    η   = sqrt(abs(δ₁))                       # η₁ = √|δ₁|
    σ   = δ₁ >= 0 ? one(T) : -one(T)          # σ₁ = sign(δ₁) (sign(0)=1)

    rNorm = η
    history && push!(rNorms, rNorm)

    # Normalize the initial Lanczos and preconditioned vectors.
    kdiv!(n, r2, η)                           # r₂ = v₁ = v̂₁ / η₁
    kdiv!(n, z, η)                            # z  = z₁ = M⁻¹ v₁

    # Scalars for the Givens-based QR factorization of the symmetric tridiagonal Tₖ.
    δbar = zero(T)
    ϵ    = zero(T)
    ϕbar = η
    cs   = -one(T)
    sn   = zero(T)

    kfill!(w1, zero(FC))
    kfill!(w2, zero(FC))

    iter   = 0
    itmax == 0 && (itmax = 2*n)
    ε_tol  = atol + rtol * η

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

      # Symmetric Lanczos step (Freund & Nachtigal 1994).
      # The Lanczos vectors vₖ are normalized in the M⁻¹-inner product:
      #   ⟨vᵢ, M⁻¹vⱼ⟩ = δᵢⱼ
      # The recurrence for the unnormalized next vector v̂ₖ₊₁:
      #   v̂ₖ₊₁ = A vₖ - αₖ vₖ - σₖ ηₖ vₖ₋₁
      # where αₖ = vₖᵀ A vₖ and the off-diagonal entries of the
      # symmetric tridiagonal Tₖ are σₖ ηₖ.
      #
      # r₁ = vₖ₋₁, r₂ = vₖ, z = zₖ = M⁻¹ vₖ.

      kmul!(y, A, r2)                         # y ← A vₖ
      α = kdotr(n, z, y)                      # αₖ = ⟨vₖ, A vₖ⟩ = vₖᵀ M⁻¹ A vₖ

      # Direction update for the solution x.
      # The update follows the MINRES pattern, using the Lanczos
      # vectors vₖ (stored in r₂).  The direction vectors are the
      # columns of Wₖ = Vₖ Rₖ⁻¹, which satisfy
      #   w₁  = v₁ / γ₁
      #   w₂  = (v₂ - δ₂ w₁) / γ₂
      #   wₖ  = (vₖ - δₖ wₖ₋₁ - ϵₖ wₖ₋₂) / γₖ   for k ≥ 3
      δ = cs * δbar + sn * α
      if iter == 1
        w = w2
        kcopy!(n, w, r2)                       # w₁ = v₁
      else
        w = w1
        iter ≥ 3 && kscal!(n, -ϵ, w)          # w ← -ϵ * wₖ₋₂
        kaxpy!(n, -δ, w2, w)                  # w ← w - δ * wₖ₋₁
        kaxpy!(n, one(FC), r2, w)              # w ← w + vₖ
      end

      # Advance the Lanczos recurrence.
      # v̂ₖ₊₁ = A vₖ - αₖ vₖ - σₖ ηₖ vₖ₋₁
      kaxpy!(n, -α, r2, y)                    # y ← y - αₖ vₖ = A vₖ - αₖ vₖ
      iter ≥ 2 && kaxpy!(n, -σ * η, r1, y)    # y ← y - σₖ ηₖ vₖ₋₁

      # Compute the next preconditioned vector: ẑₖ₊₁ = M⁻¹ v̂ₖ₊₁.
      if MisI
        kcopy!(n, z, y)                       # z ← v̂ₖ₊₁  (M⁻¹ = I)
      else
        mulorldiv!(z, M, y, ldiv)             # z ← M⁻¹ v̂ₖ₊₁ = ẑₖ₊₁
      end

      δ_next = kdotr(n, y, z)                 # δₖ₊₁ = ⟨v̂ₖ₊₁, M⁻¹v̂ₖ₊₁⟩
      if δ_next == 0
        breakdown = true
        break
      end

      η_next = sqrt(abs(δ_next))              # ηₖ₊₁ = √|δₖ₊₁|
      σ_next = δ_next >= 0 ? one(T) : -one(T) # σₖ₊₁ = sign(δₖ₊₁)

      # Normalize the new Lanczos and preconditioned vectors.
      kdiv!(n, y, η_next)                     # y ← vₖ₊₁ = v̂ₖ₊₁ / ηₖ₊₁
      kdiv!(n, z, η_next)                     # z ← zₖ₊₁ = ẑₖ₊₁ / ηₖ₊₁

      # Apply the previous Givens rotation to the current column.
      # Convention (as in MINRES):
      #   [cs  sn] [δbar  0      ] = [γbar    ϵₖ₊₁    ]
      #   [sn -cs] [α     σₖ₊₁ηₖ₊₁]   [δₖ₊₁  δbarₖ₊₁]
      offdiag = σ_next * η_next               # signed off-diagonal entry of Tₖ
      γbar = sn * δbar - cs * α
      ϵ    = sn * offdiag
      δbar = -cs * offdiag

      # New Givens rotation to annihilate the subdiagonal element.
      γ = sqrt(γbar * γbar + offdiag * offdiag)
      γ = max(γ, eps(T))

      kdiv!(n, w, γ)                          # wₖ ← wₖ / γₖ

      cs = γbar / γ
      sn = offdiag / γ                        # signed Givens sine

      # Update the right-hand side of the projected least-squares system.
      ϕ    = cs * ϕbar
      ϕbar = sn * ϕbar

      # Update solution: xₖ ← xₖ₋₁ + ϕₖ wₖ
      kaxpy!(n, ϕ, w, x)

      # Residual norm estimate: ‖rₖ‖ = |ϕbar|
      rNorm = abs(ϕbar)
      history && push!(rNorms, rNorm)

      # Swap direction vectors for the next iteration.
      if iter ≥ 2
        @kswap!(w1, w2)
      end

      # Shift Lanczos vectors for the next iteration.
      kcopy!(n, r1, r2)                       # r₁ ← vₖ
      kcopy!(n, r2, y)                        # r₂ ← vₖ₊₁

      # Update σ and η for the next iteration.
      σ = σ_next
      η = η_next

      kdisplay(iter, verbose) && @printf(iostream, "%5d  %8.1e  %7.1e  %.2fs\n", iter, α, rNorm, start_time |> ktimer)

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
end
