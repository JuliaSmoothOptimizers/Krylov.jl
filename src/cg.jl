# A standard implementation of the Conjugate Gradient method.
# The only non-standard point about it is that it does not check
# that the operator is definite.
# It is possible to check that the system is inconsistent by
# monitoring ‖p‖, which would cost an extra norm computation per
# iteration.
#
# This method is described in
#
# M. R. Hestenes and E. Stiefel. Methods of conjugate gradients for solving linear systems.
# Journal of Research of the National Bureau of Standards, 49(6), pp. 409--436, 1952.
#
# Dominique Orban, <dominique.orban@gerad.ca>
# Salt Lake City, UT, March 2015.

export cg, cg!

"""
    (x, stats) = cg(A, b::AbstractVector{FC};
                    M=I, ldiv::Bool=false, radius::T=zero(T),
                    linesearch::Bool=false, atol::T=√eps(T),
                    rtol::T=√eps(T), itmax::Int=0,
                    timemax::Float64=Inf, verbose::Int=0, history::Bool=false,
                    callback=workspace->false, iostream::IO=kstdout)

`T` is an `AbstractFloat` such as `Float32`, `Float64` or `BigFloat`.
`FC` is `T` or `Complex{T}`.

    (x, stats) = cg(A, b, x0::AbstractVector; kwargs...)

CG can be warm-started from an initial guess `x0` where `kwargs` are the same keyword arguments as above.

The conjugate gradient method to solve the Hermitian linear system Ax = b of size n.

The method does _not_ abort if A is not definite.
M also indicates the weighted norm in which residuals are measured.

#### Interface

To easily switch between Krylov methods, use the generic interface [`krylov_solve`](@ref) with `method = :cg`.

For an in-place variant that reuses memory across solves, see [`cg!`](@ref).

#### Input arguments

* `A`: a linear operator that models a Hermitian positive definite matrix of dimension `n`;
* `b`: a vector of length `n`.

#### Optional argument

* `x0`: a vector of length `n` that represents an initial guess of the solution `x`.

#### Keyword arguments

* `M`: linear operator that models a Hermitian positive-definite matrix of size `n` used for centered preconditioning;
* `ldiv`: define whether the preconditioner uses `ldiv!` or `mul!`;
* `radius`: add the trust-region constraint ‖x‖ ≤ `radius` if `radius > 0`. Useful to compute a step in a trust-region method for optimization
  - If 'radius' > 0, and negative curvature is detected, we set the step size to the maximum step size that keeps us on the trust-region boundary, and the search direction is stored in `stats.npc_dir` and `stats.npcCount` is set to 1.
  - If negative curvature occurs right away at k = 0, the solver instead takes the right-hand side (i.e., the negative gradient) as the current solution. The corresponding preconditioned search direction is returned in `workspace.npc_dir`, and `stats.npcCount` is set to 1.;
* `linesearch`: when `true`, the solver is used within an inexact Newton method with line search.
	- If negative curvature is detected at iteration k > 0, the method rolls back to the solution from iteration k – 1. The search direction computed at iteration k is stored in `stats.npc_dir`, and `stats.npcCount` is set to 1.
	-	If negative curvature occurs right away at k = 0, the solver instead takes the right-hand side (i.e., the negative gradient) as the current solution. The corresponding preconditioned search direction is returned in `workspace.npc_dir`, and `stats.npcCount` is set to 1.;
* `atol`: absolute stopping tolerance based on the residual norm;
* `rtol`: relative stopping tolerance based on the residual norm;
* `itmax`: the maximum number of iterations. If `itmax=0`, the default number of iterations is set to `2n`;
* `timemax`: the time limit in seconds;
* `verbose`: additional details can be displayed if verbose mode is enabled (verbose > 0). Information will be displayed every `verbose` iterations;
* `history`: collect additional statistics on the run such as residual norms, or Aᴴ-residual norms;
* `callback`: function or functor called as `callback(workspace)` that returns `true` if the Krylov method should terminate, and `false` otherwise;
* `iostream`: stream to which output is logged.

#### Output arguments

* `x`: a dense vector of length `n`;
* `stats`: statistics collected on the run in a [`SimpleStats`](@ref) structure.

#### Reference

* M. R. Hestenes and E. Stiefel, [*Methods of conjugate gradients for solving linear systems*](https://doi.org/10.6028/jres.049.044), Journal of Research of the National Bureau of Standards, 49(6), pp. 409--436, 1952.
"""
function cg end

"""
    workspace = cg!(workspace::CgWorkspace, A, b; kwargs...)
    workspace = cg!(workspace::CgWorkspace, A, b, x0; kwargs...)

In these calls, `kwargs` are keyword arguments of [`cg`](@ref).

See [`CgWorkspace`](@ref) for instructions on how to create the `workspace`.

For a more generic interface, you can use [`krylov_workspace`](@ref) with `method = :cg` to allocate the workspace,
and [`krylov_solve!`](@ref) to run the Krylov method in-place.
"""
function cg! end

def_args_cg = (:(A                    ),
               :(b::AbstractVector{FC}))

def_optargs_cg = (:(x0::AbstractVector),)

def_kwargs_cg = (:(; M = I                        ),
                 :(; ldiv::Bool = false           ),
                 :(; radius::T = zero(T)          ),
                 :(; linesearch::Bool = false     ),
                 :(; atol::T = √eps(T)            ),
                 :(; rtol::T = √eps(T)            ),
                 :(; itmax::Int = 0               ),
                 :(; timemax::Float64 = Inf       ),
                 :(; verbose::Int = 0             ),
                 :(; history::Bool = false        ),
                 :(; callback = workspace -> false),
                 :(; iostream::IO = kstdout       ))

def_kwargs_cg = extract_parameters.(def_kwargs_cg)

args_cg = (:A, :b)
optargs_cg = (:x0,)
kwargs_cg = (:M, :ldiv, :radius, :linesearch, :atol, :rtol, :itmax, :timemax, :verbose, :history, :callback, :iostream)

@eval begin
  function cg!(workspace :: CgWorkspace{T,FC,S}, $(def_args_cg...); $(def_kwargs_cg...)) where {T <: AbstractFloat, FC <: FloatOrComplex{T}, S <: AbstractVector{FC}}

    # Timer
    start_time = time_ns()
    timemax_ns = 1e9 * timemax

    m, n = size(A)
    (m == workspace.m && n == workspace.n) || error("(workspace.m, workspace.n) = ($(workspace.m), $(workspace.n)) is inconsistent with size(A) = ($m, $n)")
    m == n || error("System must be square")
    length(b) == n || error("Inconsistent problem size")
    linesearch && (radius > 0) && error("`linesearch` set to `true` but trust-region radius > 0")
    (workspace.warm_start && linesearch) && error("warm_start and linesearch cannot be used together")
    (verbose > 0) && @printf(iostream, "CG: system of %d equations in %d variables\n", n, n)

    # Tests M = Iₙ
    MisI = (M === I)

    # Check type consistency
    eltype(A) == FC || @warn "eltype(A) ≠ $FC. This could lead to errors or additional allocations in operator-vector products."
    ktypeof(b) == S || error("ktypeof(b) must be equal to $S")

    # Set up workspace.
    allocate_if(!MisI, workspace, :z, S, workspace.x)  # The length of z is n
    allocate_if(linesearch || (radius > 0), workspace, :npc_dir , S, workspace.x)  # The length of npc_dir is n
    Δx, x, r, p, Ap, stats = workspace.Δx, workspace.x, workspace.r, workspace.p, workspace.Ap, workspace.stats
    warm_start = workspace.warm_start
    rNorms = stats.residuals
    reset!(stats)
    z = MisI ? r : workspace.z
    if linesearch || (radius > 0)
      npc_dir = workspace.npc_dir
    end

    kfill!(x, zero(FC))
    if warm_start
      mul!(r, A, Δx)
      kaxpby!(n, one(FC), b, -one(FC), r)
    else
      kcopy!(n, r, b)  # r ← b
    end
    MisI || mulorldiv!(z, M, r, ldiv)
    kcopy!(n, p, z)  # p ← z
    γ = kdotr(n, r, z)
    γ ≥ 0 || error("The preconditioner `M` is not symmetric positive definite.")
    rNorm = sqrt(γ)
    history && push!(rNorms, rNorm)
    if γ == 0
      stats.niter = 0
      stats.solved, stats.inconsistent = true, false
      stats.timer = start_time |> ktimer
      stats.status = "x is a zero-residual solution"
      warm_start && kaxpy!(n, one(FC), Δx, x)
      workspace.warm_start = false
      if linesearch || (radius > 0)
        kcopy!(n, npc_dir, p)
        stats.npcCount = 1
        stats.indefinite = true
      end
      return workspace
    end

    iter = 0
    itmax == 0 && (itmax = 2 * n)

    pAp = zero(T)
    pNorm² = γ
    ε = atol + rtol * rNorm
    (verbose > 0) && @printf(iostream, "%5s  %7s  %8s  %8s  %8s  %5s\n", "k", "‖r‖", "pAp", "α", "σ", "timer")
    kdisplay(iter, verbose) && @printf(iostream, "%5d  %7.1e", iter, rNorm)

    solved = rNorm ≤ ε
    tired = iter ≥ itmax
    inconsistent = false
    on_boundary = false
    zero_curvature = false
    user_requested_exit = false
    overtimed = false

    status = "unknown"

    while !(solved || tired || zero_curvature || user_requested_exit || overtimed)
      mul!(Ap, A, p)
      pAp = kdotr(n, p, Ap)
      if (pAp ≤ eps(T) * pNorm²) && (radius == 0)
        if abs(pAp) ≤ eps(T) * pNorm²
          zero_curvature = true
          inconsistent = !linesearch
        end
        if linesearch
          iter == 0 && kcopy!(n, x, b)  # x ← b
          kcopy!(n, npc_dir, p)  # npc_dir ← p
          stats.npcCount = 1
          stats.indefinite = true
          solved = true
        end
      end
      (zero_curvature || solved) && continue

      α = γ / pAp
 
      # Compute step size to boundary if applicable.
      if radius == 0
         σ = α
      elseif MisI 
         σ = maximum(to_boundary(n, x, p, z, radius, dNorm2=pNorm²))
      else
         σ = maximum(to_boundary(n, x, p, z, radius, M=M, ldiv=!ldiv)) 
      end

      kdisplay(iter, verbose) && @printf(iostream, "  %8.1e  %8.1e  %8.1e  %.2fs\n", pAp, α, σ, start_time |> ktimer)

      # Move along p from x to the boundary if either
      # the next step leads outside the trust region or
      # we have nonpositive curvature.
      if (radius > 0) && ((pAp ≤ 0) || (α > σ))
        α = σ
        if pAp ≤ 0
          kcopy!(n, npc_dir, p)  # npc_dir ← p
          stats.npcCount = 1
          stats.indefinite = true
        end
        on_boundary = true
      end

      kaxpy!(n,  α,  p, x)
      kaxpy!(n, -α, Ap, r)
      MisI || mulorldiv!(z, M, r, ldiv)
      γ_next = kdotr(n, r, z)
      γ_next ≥ 0 || error("The preconditioner `M` is not symmetric positive definite.")
      rNorm = sqrt(γ_next)
      history && push!(rNorms, rNorm)

      # Stopping conditions that do not depend on user input.
      # This is to guard against tolerances that are unreasonably small.
      resid_decrease_mach = (rNorm + one(T) ≤ one(T))

      resid_decrease_lim = rNorm ≤ ε
      resid_decrease = resid_decrease_lim || resid_decrease_mach
      solved = resid_decrease || on_boundary

      if !solved
        β = γ_next / γ
        pNorm² = γ_next + β^2 * pNorm²
        γ = γ_next
        kaxpby!(n, one(FC), z, β, p)
      end

      iter = iter + 1
      tired = iter ≥ itmax
      user_requested_exit = callback(workspace) :: Bool
      timer = time_ns() - start_time
      overtimed = timer > timemax_ns
      kdisplay(iter, verbose) && @printf(iostream, "%5d  %7.1e", iter, rNorm)
    end
    (verbose > 0) && @printf(iostream, "\n\n")

    # Termination status
    solved && on_boundary             && (status = "on trust-region boundary")
    solved && on_boundary && stats.indefinite      && (status = "on trust-region boundary and indefinite")
    solved && linesearch && (pAp ≤ 0) && (status = "nonpositive curvature detected")
    solved && (status == "unknown")   && (status = "solution good enough given atol and rtol")
    zero_curvature                    && (status = "zero curvature detected")
    tired                             && (status = "maximum number of iterations exceeded")
    user_requested_exit               && (status = "user-requested exit")
    overtimed                         && (status = "time limit exceeded")

    # Update x
    warm_start && kaxpy!(n, one(FC), Δx, x)
    workspace.warm_start = false

    # Update stats
    stats.niter = iter
    stats.solved = solved
    stats.inconsistent = inconsistent
    stats.timer = start_time |> ktimer
    stats.status = status
    return workspace
  end
end
