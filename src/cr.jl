# A truncated version of Stiefel’s Conjugate Residual method described in
#
# M. R. Hestenes and E. Stiefel, Methods of conjugate gradients for solving linear systems.
# Journal of Research of the National Bureau of Standards, 49(6), pp. 409--436, 1952.
#
# E. Stiefel, Relaxationsmethoden bester Strategie zur Losung linearer Gleichungssysteme.
# Commentarii Mathematici Helvetici, 29(1), pp. 157--179, 1955.
#
# D. G. Luenberger, The conjugate residual method for constrained minimization problems.
# SIAM Journal on Numerical Analysis, 7(3), pp. 390--398, 1970.
#
# M-A. Dahito and D. Orban, The Conjugate Residual Method in Linesearch and Trust-Region Methods.
# SIAM Journal on Optimization, 29(3), pp. 1988--2025, 2019.
#
# Marie-Ange Dahito, <marie-ange.dahito@polymtl.ca>
# Montreal, QC, June 2017

export cr, cr!

"""
    (x, stats) = cr(A, b::AbstractVector{FC};
                    M=I, ldiv::Bool=false, radius::T=zero(T),
                    linesearch::Bool=false, γ::T=√eps(T),
                    atol::T=√eps(T), rtol::T=√eps(T), itmax::Int=0,
                    timemax::Float64=Inf, verbose::Int=0, history::Bool=false,
                    callback=workspace->false, iostream::IO=kstdout)

`T` is an `AbstractFloat` such as `Float32`, `Float64` or `BigFloat`.
`FC` is `T` or `Complex{T}`.

    (x, stats) = cr(A, b, x0::AbstractVector; kwargs...)

CR can be warm-started from an initial guess `x0` where `kwargs` are the same keyword arguments as above.

A truncated version of Stiefel’s Conjugate Residual method to solve the Hermitian linear system Ax = b
of size n or the least-squares problem min ‖b - Ax‖ if A is singular.
The matrix A must be Hermitian semi-definite.
M also indicates the weighted norm in which residuals are measured.

#### Interface

To easily switch between Krylov methods, use the generic interface [`krylov_solve`](@ref) with `method = :cr`.

For an in-place variant that reuses memory across solves, see [`cr!`](@ref).

#### Input arguments

* `A`: a linear operator that models a Hermitian positive definite matrix of dimension `n`;
* `b`: a vector of length `n`.

#### Optional argument

* `x0`: a vector of length `n` that represents an initial guess of the solution `x`.

#### Keyword arguments

* `M`: linear operator that models a Hermitian positive-definite matrix of size `n` used for centered preconditioning;
* `ldiv`: define whether the preconditioner uses `ldiv!` or `mul!`;
* `radius`: add the trust-region constraint ‖x‖ ≤ `radius` if `radius > 0`. Useful to compute a step in a trust-region method for optimization; If `radius > 0` and nonpositive curvature is detected, the behavior depends on the iteration and follow similar logic as linesearch;
* `linesearch`: if `true` and nonpositive curvature is detected, the behavior depends on the iteration:
 – at iteration k = 0, return the preconditioned initial search direction in `workspace.npc_dir`;
 – at iteration k > 0,
   - if the residual from iteration k-1 is a nonpositive curvature direction but `workspace.p`, the search direction at iteration k, is not, the residual is stored in `stats.npc_dir` and `stats.npcCount` is set to 1;
   - if `workspace.p` is a nonpositive curvature direction but the residual is not, `workspace.p` is copied into `stats.npc_dir` and `stats.npcCount` is set to 1;
   - if both are nonpositive curvature directions, the residual is stored in `stats.npc_dir` and `stats.npcCount` is set to 2.
* `γ`: tolerance to determine that the curvature of the quadratic model is nonpositive;
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

#### References

* M. R. Hestenes and E. Stiefel, [*Methods of conjugate gradients for solving linear systems*](https://doi.org/10.6028/jres.049.044), Journal of Research of the National Bureau of Standards, 49(6), pp. 409--436, 1952.
* E. Stiefel, [*Relaxationsmethoden bester Strategie zur Losung linearer Gleichungssysteme*](https://doi.org/10.1007/BF02564277), Commentarii Mathematici Helvetici, 29(1), pp. 157--179, 1955.
* D. G. Luenberger, [*The conjugate residual method for constrained minimization problems*](https://doi.org/10.1137/0707032), SIAM Journal on Numerical Analysis, 7(3), pp. 390--398, 1970.
* M-A. Dahito and D. Orban, [*The Conjugate Residual Method in Linesearch and Trust-Region Methods*](https://doi.org/10.1137/18M1204255), SIAM Journal on Optimization, 29(3), pp. 1988--2025, 2019.
"""
function cr end

"""
    workspace = cr!(workspace::CrWorkspace, A, b; kwargs...)
    workspace = cr!(workspace::CrWorkspace, A, b, x0; kwargs...)

In these calls, `kwargs` are keyword arguments of [`cr`](@ref).

See [`CrWorkspace`](@ref) for instructions on how to create the `workspace`.

For a more generic interface, you can use [`krylov_workspace`](@ref) with `method = :cr` to allocate the workspace,
and [`krylov_solve!`](@ref) to run the Krylov method in-place.
"""
function cr! end

def_args_cr = (:(A                    ),
               :(b::AbstractVector{FC}))

def_optargs_cr = (:(x0::AbstractVector),)

def_kwargs_cr = (:(; M = I                        ),
                 :(; ldiv::Bool = false           ),
                 :(; radius::T = zero(T)          ),
                 :(; linesearch::Bool = false     ),
                 :(; γ::T = √eps(T)               ),
                 :(; atol::T = √eps(T)            ),
                 :(; rtol::T = √eps(T)            ),
                 :(; itmax::Int = 0               ),
                 :(; timemax::Float64 = Inf       ),
                 :(; verbose::Int = 0             ),
                 :(; history::Bool = false        ),
                 :(; callback = workspace -> false),
                 :(; iostream::IO = kstdout       ))

def_kwargs_cr = extract_parameters.(def_kwargs_cr)

args_cr = (:A, :b)
optargs_cr = (:x0,)
kwargs_cr = (:M, :ldiv, :radius, :linesearch, :γ, :atol, :rtol, :itmax, :timemax, :verbose, :history, :callback, :iostream)

@eval begin
  function cr!(workspace :: CrWorkspace{T,FC,S}, $(def_args_cr...); $(def_kwargs_cr...)) where {T <: AbstractFloat, FC <: FloatOrComplex{T}, S <: AbstractVector{FC}}

    # Timer
    start_time = time_ns()
    timemax_ns = 1e9 * timemax

    m, n = size(A)
    (m == workspace.m && n == workspace.n) || error("(workspace.m, workspace.n) = ($(workspace.m), $(workspace.n)) is inconsistent with size(A) = ($m, $n)")
    m == n || error("System must be square")
    length(b) == n || error("Inconsistent problem size")
    linesearch && (radius > 0) && error("'linesearch' set to 'true' but radius > 0")
    (verbose > 0) && @printf(iostream, "CR: system of %d equations in %d variables\n", n, n)
    (workspace.warm_start && linesearch) && error("warm_start and linesearch cannot be used together")

    # Tests M = Iₙ
    MisI = (M === I)

    # Check type consistency
    eltype(A) == FC || @warn "eltype(A) ≠ $FC. This could lead to errors or additional allocations in operator-vector products."
    ktypeof(b) == S || error("ktypeof(b) must be equal to $S")

    # Set up workspace
    allocate_if(!MisI, workspace, :Mq, S, workspace.x)  # The length of Mq is n
    allocate_if(linesearch || (radius > 0), workspace, :npc_dir , S, workspace.x)  # The length of npc_dir is n
    Δx, x, r, p, q, Ar, stats = workspace.Δx, workspace.x, workspace.r, workspace.p, workspace.q, workspace.Ar, workspace.stats
    warm_start = workspace.warm_start
    rNorms, ArNorms = stats.residuals, stats.Aresiduals
    reset!(stats)
    Mq = MisI ? q : workspace.Mq
    if linesearch || (radius > 0)
      npc_dir = workspace.npc_dir
    end

    # Initial state.
    kfill!(x, zero(FC))
    if warm_start
      mul!(p, A, Δx)
      kaxpby!(n, one(FC), b, -one(FC), p)
    else
      kcopy!(n, p, b)  # p ← b
    end
    MisI && kcopy!(n, r, p)  # r ← p
    MisI || mulorldiv!(r, M, p, ldiv)
    
    rNorm = knorm_elliptic(n, r, p)  # ‖r‖
    history && push!(rNorms, rNorm)  # Values of ‖r‖

    if rNorm == 0
      stats.niter = 0
      stats.solved, stats.inconsistent = true, false
      stats.timer = start_time |> ktimer
      stats.status = "x is a zero-residual solution"
      history && push!(ArNorms, zero(T))
      warm_start && kaxpy!(n, one(FC), Δx, x)
      workspace.warm_start = false
      return workspace
    end

    mul!(Ar, A, r)
    ρ = kdotr(n, r, Ar)
    
    if ρ == 0
      stats.niter = 0
      stats.solved, stats.inconsistent = true, false
      stats.timer = start_time |> ktimer
      stats.status = "b is a zero-curvature direction"
      history && push!(ArNorms, zero(T))
      workspace.warm_start = false
      if linesearch || (radius > 0)
        kcopy!(n, x, p)  # x ← M⁻¹ b
        kcopy!(n, npc_dir, p)
        stats.npcCount = 1
        stats.indefinite = true
      end
      return workspace
    end

    kcopy!(n, p, r)   # p ← r
    kcopy!(n, q, Ar)  # q ← Ar
    (verbose > 0) && (m = zero(T))  # quadratic model

    iter = 0
    itmax == 0 && (itmax = 2 * n)

    rNorm² = rNorm * rNorm
    pNorm = rNorm
    pNorm² = rNorm²
    pr = rNorm²
    abspr = pr
    pAp = ρ
    abspAp = abs(pAp)
    xNorm = zero(T)
    ArNorm = knorm(n, Ar) # ‖Ar‖
    history && push!(ArNorms, ArNorm)
    ε = atol + rtol * rNorm
    (verbose > 0) && @printf(iostream, "%5s  %8s  %8s  %8s  %5s\n", "k", "‖x‖", "‖r‖", "quad", "timer")
    kdisplay(iter, verbose) && @printf(iostream, "%5d  %8.1e  %8.1e  %8.1e  %.2fs\n", iter, xNorm, rNorm, m, start_time |> ktimer)

    descent = pr > 0 # pᴴr > 0 means p is a descent direction
    solved = rNorm ≤ ε
    tired = iter ≥ itmax
    on_boundary = false
    npcurv = false
    status = "unknown"
    user_requested_exit = false
    overtimed = false

    while ! (solved || tired || user_requested_exit || overtimed)
      if linesearch
        p_curv = pAp ≤ γ * pNorm^2
        r_curv = ρ   ≤ γ * rNorm^2
        if p_curv || r_curv
          npcurv = true
          (verbose > 0) && @printf(iostream, "nonpositive curvature detected: pᴴAp = %8.1e and rᴴAr = %8.1e\n", pAp, ρ)
          stats.solved = true
          stats.niter = iter
          stats.inconsistent = false
          stats.timer = start_time |> ktimer
          stats.status = "nonpositive curvature"
          workspace.warm_start = false
          stats.indefinite = true
          if iter == 0
            kcopy!(n, npc_dir, p)
            kcopy!(n, x, p)  # x ← M⁻¹ b
            stats.npcCount = 1
          else
            if r_curv
              kcopy!(n, npc_dir, r)
              stats.npcCount += 1
            end
            if p_curv
              stats.npcCount += 1
              r_curv || kcopy!(n, npc_dir, p)
            end
          end
          return workspace
        end
      elseif pAp ≤ 0 && radius == 0
        error("Indefinite system and no trust region")
      end
      MisI || mulorldiv!(Mq, M, q, ldiv)

      if radius > 0
        (verbose > 0) && @printf(iostream, "radius = %8.1e > 0 and ‖x‖ = %8.1e\n", radius, xNorm)
        # find t1 > 0 and t2 < 0 such that ‖x + ti * p‖² = radius²  (i = 1, 2)
        xNorm² = xNorm * xNorm
        t = to_boundary(n, x, p, Mq, radius; flip = false, xNorm2 = xNorm², dNorm2 = pNorm²)
        t1 = maximum(t) # > 0
        t2 = minimum(t) # < 0
        tr = maximum(to_boundary(n, x, r, Mq, radius; flip = false, xNorm2 = xNorm², dNorm2 = rNorm²))
        (verbose > 0) && @printf(iostream, "t1 = %8.1e, t2 = %8.1e and tr = %8.1e\n", t1, t2, tr)

        if abspAp ≤ γ * pNorm * knorm(n, q)  # pᴴAp ≃ 0
          npcurv = true  # nonpositive curvature
          stats.indefinite = true
          stats.npcCount = 1
          kcopy!(n, npc_dir, p)
          (verbose > 0) && @printf(iostream, "pᴴAp = %8.1e ≃ 0\n", pAp)
          if abspr ≤ γ * pNorm * rNorm  # pᴴr ≃ 0
            (verbose > 0) && @printf(iostream, "pᴴr = %8.1e ≃ 0, redefining p := r\n", pr)
            p = r  # - ∇q(x)
            q = Ar
            # q(x + αr) = q(x) - α ‖r‖² + ½ α² rᴴAr
            # 1) if rᴴAr > 0, the quadratic decreases from α = 0 to α = ‖r‖² / rᴴAr
            # 2) if rᴴAr ≤ 0, the quadratic decreases to -∞ in the direction r
            if ρ > 0 # case 1
              (verbose > 0) && @printf(iostream, "quadratic is convex in direction r, curv = %8.1e\n", ρ)
              α = min(tr, rNorm² / ρ)
            else # case 2
              (verbose > 0) && @printf(iostream, "r is a direction of nonpositive curvature: %8.1e\n", ρ)
              α = tr
              if iter > 0
                stats.npcCount = 2
                kcopy!(n, npc_dir, r)
              end
            end
          else
            # q_p = q(x + α_p * p) - q(x) = -α_p * rᴴp + ½ (α_p)² * pᴴAp
            # q_r = q(x + α_r * r) - q(x) = -α_r * ‖r‖² + ½ (α_r)² * rᴴAr
            # Δ = q_p - q_r. If Δ > 0, r is followed, else p is followed
            α = descent ? t1 : t2
            ρ > 0 && (tr = min(tr, rNorm² / ρ))
            Δ = -α * pr + tr * rNorm² - (tr)^2 * ρ / 2 # as pᴴAp = 0
            if Δ > 0  # direction r engenders a better decrease
              (verbose > 0) && @printf(iostream, "direction r engenders a bigger decrease. q_p - q_r = %8.1e > 0\n", Δ)
              (verbose > 0) && @printf(iostream, "redefining p := r\n")
              p = r
              q = Ar
              α = tr
            else
              (verbose > 0) && @printf(iostream, "direction p engenders an equal or a bigger decrease. q_p - q_r = %8.1e ≤ 0\n", Δ)
            end
          end

        elseif pAp > 0 && ρ > 0  # no negative curvature
          (verbose > 0) && @printf(iostream, "positive curvature along p and r. pᴴAp = %8.1e and rᴴAr = %8.1e\n", pAp, ρ)
          α = ρ / kdotr(n, q, Mq)
          if α ≥ t1
            α = t1
            on_boundary = true
          end

        elseif pAp > 0 && ρ < 0
          npcurv = true
          stats.indefinite = true
          stats.npcCount = 1
          kcopy!(n, npc_dir, r)
          (verbose > 0) && @printf(iostream, "pᴴAp = %8.1e > 0 and rᴴAr = %8.1e < 0\n", pAp, ρ)
          # q_p is minimal for α_p = rᴴp / pᴴAp
          α = descent ?  min(t1, pr / pAp) : max(t2, pr / pAp)
          Δ = -α * pr + tr * rNorm² + (α^2 * pAp - (tr)^2 * ρ) / 2
          if Δ > 0
            (verbose > 0) && @printf(iostream, "direction r engenders a bigger decrease. q_p - q_r = %8.1e > 0\n", Δ)
            (verbose > 0) && @printf(iostream, "redefining p := r\n")
            p = r
            q = Ar
            α = tr
          else
            (verbose > 0) && @printf(iostream, "direction p engenders an equal or a bigger decrease. q_p - q_r = %8.1e ≤ 0\n", Δ)
          end

        elseif pAp < 0 && ρ > 0
          npcurv = true
          stats.indefinite = true
          stats.npcCount = 1
          kcopy!(n, npc_dir, p)
          (verbose > 0) && @printf(iostream, "pᴴAp = %8.1e < 0 and rᴴAr = %8.1e > 0\n", pAp, ρ)
          α = descent ? t1 : t2
          tr = min(tr, rNorm² / ρ)
          Δ = -α * pr + tr * rNorm² + (α^2 * pAp - (tr)^2 * ρ) / 2
          if Δ > 0
            (verbose > 0) && @printf(iostream, "direction r engenders a bigger decrease. q_p - q_r = %8.1e > 0\n", Δ)
            (verbose > 0) && @printf(iostream, "redefining p := r\n")
            p = r
            q = Ar
            α = tr
          else
            (verbose > 0) && @printf(iostream, "direction p engenders an equal or a bigger decrease. q_p - q_r = %8.1e ≤ 0\n", Δ)
          end

        elseif pAp < 0 && ρ < 0
          npcurv = true
          stats.indefinite = true
          stats.npcCount = 2
          kcopy!(n, npc_dir, r)
          (verbose > 0) && @printf(iostream, "negative curvature along p and r. pᴴAp = %8.1e and rᴴAr = %8.1e\n", pAp, ρ)
          α = descent ? t1 : t2
          Δ = -α * pr + tr * rNorm² + (α^2 * pAp - (tr)^2 * ρ) / 2
          if Δ > 0
            (verbose > 0) && @printf(iostream, "direction r engenders a bigger decrease. q_p - q_r = %8.1e > 0\n", Δ)
            (verbose > 0) && @printf(iostream, "redefining p := r\n")
            p = r
            q = Ar
            α = tr
          else
            (verbose > 0) && @printf(iostream, "direction p engenders an equal or a bigger decrease. q_p - q_r = %8.1e ≤ 0\n", Δ)
          end
        end

      elseif radius == 0
        α = ρ / kdotr(n, q, Mq)  # step
      end

      kaxpy!(n, α, p, x)
      xNorm = knorm(n, x)
      xNorm ≈ radius > 0 && (on_boundary = true)
      kaxpy!(n, -α, Mq, r)  # residual
      if MisI
        rNorm² = kdotr(n, r, r)
        rNorm = sqrt(rNorm²)
      else
        ω = sqrt(α) * sqrt(ρ)
        rNorm = sqrt(abs(rNorm + ω)) * sqrt(abs(rNorm - ω))
        rNorm² = rNorm * rNorm  # rNorm² = rNorm² - α * ρ
      end
      history && push!(rNorms, rNorm)
      mul!(Ar, A, r)
      ArNorm = knorm(n, Ar)
      history && push!(ArNorms, ArNorm)

      iter = iter + 1
      if kdisplay(iter, verbose)
        m = m - α * pr + α^2 * pAp / 2
        @printf(iostream, "%5d  %8.1e  %8.1e  %8.1e  %.2fs\n", iter, xNorm, rNorm, m, start_time |> ktimer)
      end

      # Stopping conditions that do not depend on user input.
      # This is to guard against tolerances that are unreasonably small.
      resid_decrease_mach = (rNorm + one(T) ≤ one(T))

      user_requested_exit = callback(workspace) :: Bool
      resid_decrease_lim = rNorm ≤ ε
      resid_decrease = resid_decrease_lim || resid_decrease_mach
      solved = resid_decrease || npcurv || on_boundary
      tired = iter ≥ itmax
      timer = time_ns() - start_time
      overtimed = timer > timemax_ns

      (solved || tired || user_requested_exit || overtimed) && continue
      ρbar = ρ
      ρ = kdotr(n, r, Ar)
      β = ρ / ρbar # step for the direction computation
      kaxpby!(n, one(FC), r, β, p)
      kaxpby!(n, one(FC), Ar, β, q)

      pNorm² = rNorm² + 2 * β * pr - 2 * β * α * pAp + β^2 * pNorm²
      if pNorm² > sqrt(eps(T))
        pNorm = sqrt(pNorm²)
      elseif abs(pNorm²) ≤ sqrt(eps(T))
        pNorm = zero(T)
      else
        stats.niter = iter
        stats.solved = solved
        stats.inconsistent = false
        stats.timer = start_time |> ktimer
        stats.status = "solver encountered numerical issues"
        warm_start && kaxpy!(n, one(FC), Δx, x)
        workspace.warm_start = false
        return workspace
      end
      pr = rNorm² + β * pr - β * α * pAp  # pᴴr
      abspr = abs(pr)
      pAp = ρ + β^2 * pAp  # pᴴq
      abspAp = abs(pAp)
      descent = pr > 0

    end
    (verbose > 0) && @printf(iostream, "\n")

    # Termination status
    tired               && (status = "maximum number of iterations exceeded")
    solved              && (status = "solution good enough given atol and rtol")
    user_requested_exit && (status = "user-requested exit")
    overtimed           && (status = "time limit exceeded")
    npcurv              && (status = "nonpositive curvature")
    on_boundary         && (status = "on trust-region boundary")

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
