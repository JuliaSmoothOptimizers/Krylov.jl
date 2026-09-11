# An implementation of SQMR (also known as PSQMR) for the solution of
# Hermitian (self-adjoint) square linear systems Ax = b, where A may be
# indefinite and the preconditioner M may be indefinite as well.
#
# SQMR generates the same directions as a CG-type recurrence driven by the
# possibly-indefinite bilinear pivot ρₖ = rₖᴴM⁻¹rₖ (no square root of ρₖ is
# ever taken, so ρₖ < 0 causes no breakdown of its own), and then applies a
# quasi-minimization smoothing -- in the style of Freund's transpose-free
# QMR -- to the sequence of CG-type iterates using only the (always
# well-defined) Euclidean norm of the underlying residual. Consequently a
# single product with A and a single application of the preconditioner are
# required per iteration, and no product with Aᴴ or Mᴴ is required.
#
# References:
#
#   R. W. Freund and N. M. Nachtigal,
#   A new Krylov-subspace method for symmetric indefinite linear systems.
#   Proc. 14th IMACS World Congress, pp. 1253--1256, 1994.
#
#   R. W. Freund,
#   A transpose-free quasi-minimal residual algorithm for non-Hermitian linear systems.
#   SIAM Journal on Scientific Computing, Vol. 14(2), pp. 470--482, 1993.
#
#   K.-C. Toh, M. J. Todd and R. H. Tütüncü,
#   On the implementation and usage of SDPT3 -- a Matlab software package for
#   semidefinite-quadratic-linear programming, version 4.0, section on PSQMR.
#
#   Y. Saad, Iterative Methods for Sparse Linear Systems, 2nd ed.,
#   SIAM, Philadelphia, 2003.
#
# Alexis Montoison, <alexis.montoison@polymtl.ca>
# Arnav Kapoor, <arnavkapoor23@iiserb.ac.in>
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
using the Symmetric Quasi-Minimal Residual (SQMR) method.

Unlike MINRES, SQMR supports Hermitian preconditioners `M` that are
*indefinite*: at each iteration SQMR only ever divides by the Euclidean norm
of a residual-like vector, and never by the square root of the (possibly
negative) pivot `rₖᴴM⁻¹rₖ`. Without preconditioning, SQMR and MINRES solve
the same sequence of Galerkin subproblems.

SQMR does not require products with `Aᴴ` or `Mᴴ`.

#### Interface

To easily switch between Krylov methods, use the generic interface [`krylov_solve`](@ref) with `method = :sqmr`.

For an in-place variant that reuses memory across solves, see [`sqmr!`](@ref).

#### Input arguments

* `A`: a linear operator that models a Hermitian matrix of dimension `n`;
* `b`: a vector of length `n`.

#### Optional argument

* `x0`: a vector of length `n` that represents an initial guess of the solution `x`.

#### Keyword arguments

* `M`: linear operator that models a Hermitian (possibly indefinite) nonsingular matrix of size `n` used for centered preconditioning;
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
* R. W. Freund, [*A transpose-free quasi-minimal residual algorithm for non-Hermitian linear systems*](https://doi.org/10.1137/0914029), SIAM Journal on Scientific Computing, Vol. 14(2), pp. 470--482, 1993.
* Y. Saad, [*Iterative Methods for Sparse Linear Systems*](https://www-users.cse.umn.edu/~saad/books.html), 2nd edition, SIAM, Philadelphia, 2003.
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
    r, z, p, w, d = workspace.r, workspace.z, workspace.p, workspace.w, workspace.d
    Δx, x, stats = workspace.Δx, workspace.x, workspace.stats
    warm_start = workspace.warm_start
    rNorms = stats.residuals
    reset!(stats)

    # Initial solution x₀ and residual r₀ = b - Ax₀.
    kfill!(x, zero(FC))
    if warm_start
      kmul!(r, A, Δx)
      kaxpby!(n, one(FC), b, -one(FC), r)
    else
      kcopy!(n, r, b)  # r ← b
    end

    τ = knorm(n, r)  # τ₀ = ‖r₀‖
    history && push!(rNorms, τ)

    if τ == 0
      stats.niter = 0
      stats.solved = true
      stats.inconsistent = false
      stats.timer = start_time |> ktimer
      stats.status = "x is a zero-residual solution"
      warm_start && kaxpy!(n, one(FC), Δx, x)
      workspace.warm_start = false
      return workspace
    end

    # z₁ = M⁻¹r₀ and the initial (possibly negative) pivot ρ₀ = ⟨r₀, z₁⟩.
    MisI ? kcopy!(n, z, r) : mulorldiv!(z, M, r, ldiv)
    ρ = kdotr(n, r, z)

    if ρ == 0
      stats.niter = 0
      stats.solved = false
      stats.inconsistent = false
      stats.timer = start_time |> ktimer
      stats.status = "Breakdown ⟨r₀,M⁻¹r₀⟩ = 0"
      warm_start && kaxpy!(n, one(FC), Δx, x)
      workspace.warm_start = false
      return workspace
    end

    iter = 0
    itmax == 0 && (itmax = 2*n)

    θ = zero(T)   # θₖ₋₁
    kfill!(d, zero(FC))
    kfill!(p, zero(FC))

    ε = atol + rtol * τ
    (verbose > 0) && @printf(iostream, "%5s  %8s  %7s  %5s\n", "k", "ρₖ", "τₖ", "timer")
    kdisplay(iter, verbose) && @printf(iostream, "%5d  %8.1e  %7.1e  %.2fs\n", iter, ρ, τ, start_time |> ktimer)

    solved    = τ ≤ ε
    breakdown = false
    tired     = iter ≥ itmax
    status    = "unknown"
    user_requested_exit = false
    overtimed = false

    ρ_prev  = ρ
    ρ_prev2 = zero(T)

    while !(solved || tired || breakdown || user_requested_exit || overtimed)
      iter = iter + 1

      # Fletcher-Reeves-type direction pₖ = zₖ + (ρₖ₋₁/ρₖ₋₂) pₖ₋₁.
      if iter == 1
        kcopy!(n, p, z)
      else
        β = ρ_prev / ρ_prev2
        kscal!(n, β, p)
        kaxpy!(n, one(FC), z, p)
      end

      kmul!(w, A, p)          # w ← Apₖ
      σ = kdotr(n, p, w)      # σₖ = ⟨pₖ, Apₖ⟩ (real since A is Hermitian)

      if σ == 0
        breakdown = true
        continue
      end

      α = ρ_prev / σ
      kaxpy!(n, -α, w, r)     # rₖ ← rₖ₋₁ - αₖApₖ

      τ_new = knorm(n, r)
      θ_new = τ_new / τ
      c = one(T) / √(one(T) + θ_new^2)
      τ = τ * θ_new * c

      # Quasi-minimization smoothing of the search direction and the iterate.
      kscal!(n, c^2 * θ^2, d)
      kaxpy!(n, c^2 * α, p, d)
      kaxpy!(n, one(FC), d, x)

      θ = θ_new
      history && push!(rNorms, τ)

      # Stopping conditions that do not depend on user input.
      # This is to guard against tolerances that are unreasonably small.
      resid_decrease_mach = (τ + one(T) ≤ one(T))

      user_requested_exit = callback(workspace) :: Bool
      resid_decrease_lim = τ ≤ ε
      solved = resid_decrease_lim || resid_decrease_mach
      tired = iter ≥ itmax
      timer = time_ns() - start_time
      overtimed = timer > timemax_ns

      kdisplay(iter, verbose) && @printf(iostream, "%5d  %8.1e  %7.1e  %.2fs\n", iter, ρ_prev, τ, start_time |> ktimer)

      (solved || tired || user_requested_exit || overtimed) && continue

      # Prepare the next iteration: zₖ₊₁ = M⁻¹rₖ and ρₖ = ⟨rₖ, zₖ₊₁⟩.
      MisI ? kcopy!(n, z, r) : mulorldiv!(z, M, r, ldiv)
      ρ_prev2 = ρ_prev
      ρ_prev = kdotr(n, r, z)

      breakdown = (ρ_prev == 0)
    end
    (verbose > 0) && @printf(iostream, "\n")

    # Termination status
    tired               && (status = "maximum number of iterations exceeded")
    breakdown           && (status = "Breakdown ⟨rₖ,M⁻¹rₖ⟩ = 0 or ⟨pₖ,Apₖ⟩ = 0")
    solved              && (status = "solution good enough given atol and rtol")
    user_requested_exit && (status = "user-requested exit")
    overtimed           && (status = "time limit exceeded")

    # Update x
    warm_start && kaxpy!(n, one(FC), Δx, x)
    workspace.warm_start = false

    # Update stats
    stats.niter = iter
    stats.solved = solved
    stats.inconsistent = false
    stats.timer = start_time |> ktimer
    stats.status = status
    return workspace
  end
end
