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
                    callback=solver->false, iostream::IO=kstdout)

`T` is an `AbstractFloat` such as `Float32`, `Float64` or `BigFloat`.
`FC` is `T` or `Complex{T}`.

    (x, stats) = cr(A, b, x0::AbstractVector; kwargs...)

CR can be warm-started from an initial guess `x0` where `kwargs` are the same keyword arguments as above.

A truncated version of Stiefel’s Conjugate Residual method to solve the Hermitian linear system Ax = b
of size n or the least-squares problem min ‖b - Ax‖ if A is singular.
The matrix A must be Hermitian semi-definite.
M also indicates the weighted norm in which residuals are measured.

#### Input arguments

* `A`: a linear operator that models a Hermitian positive definite matrix of dimension n;
* `b`: a vector of length n.

#### Optional argument

* `x0`: a vector of length n that represents an initial guess of the solution x.

#### Keyword arguments

* `M`: linear operator that models a Hermitian positive-definite matrix of size `n` used for centered preconditioning;
* `ldiv`: define whether the preconditioner uses `ldiv!` or `mul!`;
* `radius`: add the trust-region constraint ‖x‖ ≤ `radius` if `radius > 0`. Useful to compute a step in a trust-region method for optimization;
* `linesearch`: if `true`, indicate that the solution is to be used in an inexact Newton method with linesearch. If negative curvature is detected at iteration k > 0, the solution of iteration k-1 is returned. If negative curvature is detected at iteration 0, the right-hand side is returned (i.e., the negative gradient);
* `γ`: tolerance to determine that the curvature of the quadratic model is nonpositive;
* `atol`: absolute stopping tolerance based on the residual norm;
* `rtol`: relative stopping tolerance based on the residual norm;
* `itmax`: the maximum number of iterations. If `itmax=0`, the default number of iterations is set to `2n`;
* `timemax`: the time limit in seconds;
* `verbose`: additional details can be displayed if verbose mode is enabled (verbose > 0). Information will be displayed every `verbose` iterations;
* `history`: collect additional statistics on the run such as residual norms, or Aᴴ-residual norms;
* `callback`: function or functor called as `callback(solver)` that returns `true` if the Krylov method should terminate, and `false` otherwise;
* `iostream`: stream to which output is logged.

#### Output arguments

* `x`: a dense vector of length n;
* `stats`: statistics collected on the run in a [`SimpleStats`](@ref) structure.

#### References

* M. R. Hestenes and E. Stiefel, [*Methods of conjugate gradients for solving linear systems*](https://doi.org/10.6028/jres.049.044), Journal of Research of the National Bureau of Standards, 49(6), pp. 409--436, 1952.
* E. Stiefel, [*Relaxationsmethoden bester Strategie zur Losung linearer Gleichungssysteme*](https://doi.org/10.1007/BF02564277), Commentarii Mathematici Helvetici, 29(1), pp. 157--179, 1955.
* D. G. Luenberger, [*The conjugate residual method for constrained minimization problems*](https://doi.org/10.1137/0707032), SIAM Journal on Numerical Analysis, 7(3), pp. 390--398, 1970.
* M-A. Dahito and D. Orban, [*The Conjugate Residual Method in Linesearch and Trust-Region Methods*](https://doi.org/10.1137/18M1204255), SIAM Journal on Optimization, 29(3), pp. 1988--2025, 2019.
"""
function cr end

"""
    solver = cr!(solver::CrSolver, A, b; kwargs...)
    solver = cr!(solver::CrSolver, A, b, x0; kwargs...)

where `kwargs` are keyword arguments of [`cr`](@ref).

See [`CrSolver`](@ref) for more details about the `solver`.
"""
function cr! end

def_args_cr = (:(A                    ),
               :(b::AbstractVector{FC}))

def_optargs_cr = (:(x0::AbstractVector),)

def_kwargs_cr = (:(; M = I                     ),
                 :(; ldiv::Bool = false        ),
                 :(; radius::T = zero(T)       ),
                 :(; linesearch::Bool = false  ),
                 :(; γ::T = √eps(T)            ),
                 :(; atol::T = √eps(T)         ),
                 :(; rtol::T = √eps(T)         ),
                 :(; itmax::Int = 0            ),
                 :(; timemax::Float64 = Inf    ),
                 :(; verbose::Int = 0          ),
                 :(; history::Bool = false     ),
                 :(; callback = solver -> false),
                 :(; iostream::IO = kstdout    ))

def_kwargs_cr = mapreduce(extract_parameters, vcat, def_kwargs_cr)

args_cr = (:A, :b)
optargs_cr = (:x0,)
kwargs_cr = (:M, :ldiv, :radius, :linesearch, :γ, :atol, :rtol, :itmax, :timemax, :verbose, :history, :callback, :iostream)

@eval begin
  function cr($(def_args_cr...), $(def_optargs_cr...); $(def_kwargs_cr...)) where {T <: AbstractFloat, FC <: FloatOrComplex{T}}
    start_time = time_ns()
    solver = CrSolver(A, b)
    warm_start!(solver, $(optargs_cr...))
    elapsed_time = ktimer(start_time)
    timemax -= elapsed_time
    cr!(solver, $(args_cr...); $(kwargs_cr...))
    solver.stats.timer += elapsed_time
    return (solver.x, solver.stats)
  end

  function cr($(def_args_cr...); $(def_kwargs_cr...)) where {T <: AbstractFloat, FC <: FloatOrComplex{T}}
    start_time = time_ns()
    solver = CrSolver(A, b)
    elapsed_time = ktimer(start_time)
    timemax -= elapsed_time
    cr!(solver, $(args_cr...); $(kwargs_cr...))
    solver.stats.timer += elapsed_time
    return (solver.x, solver.stats)
  end

  function cr!(solver :: CrSolver{T,FC,S}, $(def_args_cr...); $(def_kwargs_cr...)) where {T <: AbstractFloat, FC <: FloatOrComplex{T}, S <: AbstractVector{FC}}

    # Timer
    start_time = time_ns()
    timemax_ns = 1e9 * timemax

    m, n = size(A)
    (m == solver.m && n == solver.n) || error("(solver.m, solver.n) = ($(solver.m), $(solver.n)) is inconsistent with size(A) = ($m, $n)")
    m == n || error("System must be square")
    length(b) == n || error("Inconsistent problem size")
    linesearch && (radius > 0) && error("'linesearch' set to 'true' but radius > 0")
    (verbose > 0) && @printf(iostream, "CR: system of %d equations in %d variables\n", n, n)

    # Tests M = Iₙ
    MisI = (M === I)

    # Check type consistency
    eltype(A) == FC || @warn "eltype(A) ≠ $FC. This could lead to errors or additional allocations in operator-vector products."
    ktypeof(b) <: S || error("ktypeof(b) is not a subtype of $S")

    # Set up workspace
    allocate_if(!MisI, solver, :Mq, S, n)
    Δx, x, r, p, q, Ar, stats = solver.Δx, solver.x, solver.r, solver.p, solver.q, solver.Ar, solver.stats
    warm_start = solver.warm_start
    rNorms, ArNorms = stats.residuals, stats.Aresiduals
    reset!(stats)
    Mq = MisI ? q : solver.Mq

    # Initial state.
    x .= zero(FC)
    if warm_start
      mul!(p, A, Δx)
      @kaxpby!(n, one(FC), b, -one(FC), p)
    else
      p .= b
    end
    MisI && (r .= p)
    MisI || mulorldiv!(r, M, p, ldiv)
    mul!(Ar, A, r)
    ρ = @kdotr(n, r, Ar)

    rNorm = sqrt(@kdotr(n, r, p))   # ‖r‖
    history && push!(rNorms, rNorm) # Values of ‖r‖

    if ρ == 0
      stats.niter = 0
      stats.solved, stats.inconsistent = true, false
      stats.timer = ktimer(start_time)
      stats.status = "x = 0 is a zero-residual solution"
      history && push!(ArNorms, zero(T))
      solver.warm_start = false
      return solver
    end
    p .= r
    q .= Ar
    (verbose > 0) && (m = zero(T)) # quadratic model

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
    ArNorm = @knrm2(n, Ar) # ‖Ar‖
    history && push!(ArNorms, ArNorm)
    ε = atol + rtol * rNorm
    (verbose > 0) && @printf(iostream, "%5s  %8s  %8s  %8s  %5s\n", "k", "‖x‖", "‖r‖", "quad", "timer")
    kdisplay(iter, verbose) && @printf(iostream, "%5d  %8.1e  %8.1e  %8.1e  %.2fs\n", iter, xNorm, rNorm, m, ktimer(start_time))

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
        if (pAp ≤ γ * pNorm²) || (ρ ≤ γ * rNorm²)
          npcurv = true
          (verbose > 0) && @printf(iostream, "nonpositive curvature detected: pᴴAp = %8.1e and rᴴAr = %8.1e\n", pAp, ρ)
          stats.solved = solved
          stats.inconsistent = false
          stats.timer = ktimer(start_time)
          stats.status = "nonpositive curvature"
          return solver
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

        if abspAp ≤ γ * pNorm * @knrm2(n, q) # pᴴAp ≃ 0
          npcurv = true # nonpositive curvature
          (verbose > 0) && @printf(iostream, "pᴴAp = %8.1e ≃ 0\n", pAp)
          if abspr ≤ γ * pNorm * rNorm # pᴴr ≃ 0
            (verbose > 0) && @printf(iostream, "pᴴr = %8.1e ≃ 0, redefining p := r\n", pr)
            p = r # - ∇q(x)
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
            end
          else
            # q_p = q(x + α_p * p) - q(x) = -α_p * rᴴp + ½ (α_p)² * pᴴAp
            # q_r = q(x + α_r * r) - q(x) = -α_r * ‖r‖² + ½ (α_r)² * rᴴAr
            # Δ = q_p - q_r. If Δ > 0, r is followed, else p is followed
            α = descent ? t1 : t2
            ρ > 0 && (tr = min(tr, rNorm² / ρ))
            Δ = -α * pr + tr * rNorm² - (tr)^2 * ρ / 2 # as pᴴAp = 0
            if Δ > 0 # direction r engenders a better decrease
              (verbose > 0) && @printf(iostream, "direction r engenders a bigger decrease. q_p - q_r = %8.1e > 0\n", Δ)
              (verbose > 0) && @printf(iostream, "redefining p := r\n")
              p = r
              q = Ar
              α = tr
            else
              (verbose > 0) && @printf(iostream, "direction p engenders an equal or a bigger decrease. q_p - q_r = %8.1e ≤ 0\n", Δ)
            end
          end

        elseif pAp > 0 && ρ > 0 # no negative curvature
          (verbose > 0) && @printf(iostream, "positive curvatures along p and r. pᴴAp = %8.1e and rᴴAr = %8.1e\n", pAp, ρ)
          α = ρ / @kdotr(n, q, Mq)
          if α ≥ t1
            α = t1
            on_boundary = true
          end

        elseif pAp > 0 && ρ < 0
          npcurv = true
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
          (verbose > 0) && @printf(iostream, "negative curvatures along p and r. pᴴAp = %8.1e and rᴴAr = %8.1e\n", pAp, ρ)
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
        α = ρ / @kdotr(n, q, Mq) # step
      end

      @kaxpy!(n, α, p, x)
      xNorm = @knrm2(n, x)
      xNorm ≈ radius && (on_boundary = true)
      @kaxpy!(n, -α, Mq, r) # residual
      if MisI
        rNorm² = @kdotr(n, r, r)
        rNorm = sqrt(rNorm²)
      else
        ω = sqrt(α) * sqrt(ρ)
        rNorm = sqrt(abs(rNorm + ω)) * sqrt(abs(rNorm - ω))
        rNorm² = rNorm * rNorm  # rNorm² = rNorm² - α * ρ
      end
      history && push!(rNorms, rNorm)
      mul!(Ar, A, r)
      ArNorm = @knrm2(n, Ar)
      history && push!(ArNorms, ArNorm)

      iter = iter + 1
      if kdisplay(iter, verbose)
        m = m - α * pr + α^2 * pAp / 2
        @printf(iostream, "%5d  %8.1e  %8.1e  %8.1e  %.2fs\n", iter, xNorm, rNorm, m, ktimer(start_time))
      end

      # Stopping conditions that do not depend on user input.
      # This is to guard against tolerances that are unreasonably small.
      resid_decrease_mach = (rNorm + one(T) ≤ one(T))

      user_requested_exit = callback(solver) :: Bool
      resid_decrease_lim = rNorm ≤ ε
      resid_decrease = resid_decrease_lim || resid_decrease_mach
      solved = resid_decrease || npcurv || on_boundary
      tired = iter ≥ itmax
      timer = time_ns() - start_time
      overtimed = timer > timemax_ns

      (solved || tired || user_requested_exit || overtimed) && continue
      ρbar = ρ
      ρ = @kdotr(n, r, Ar)
      β = ρ / ρbar # step for the direction computation
      @kaxpby!(n, one(FC), r, β, p)
      @kaxpby!(n, one(FC), Ar, β, q)

      pNorm² = rNorm² + 2 * β * pr - 2 * β * α * pAp + β^2 * pNorm²
      if pNorm² > sqrt(eps(T))
        pNorm = sqrt(pNorm²)
      elseif abs(pNorm²) ≤ sqrt(eps(T))
        pNorm = zero(T)
      else
        stats.niter = iter
        stats.solved = solved
        stats.inconsistent = false
        stats.timer = ktimer(start_time)
        stats.status = "solver encountered numerical issues"
        solver.warm_start = false
        return solver
      end
      pr = rNorm² + β * pr - β * α * pAp # pᴴr
      abspr = abs(pr)
      pAp = ρ + β^2 * pAp # pᴴq
      abspAp = abs(pAp)
      descent = pr > 0

    end
    (verbose > 0) && @printf(iostream, "\n")

    # Termination status
    tired               && (status = "maximum number of iterations exceeded")
    on_boundary         && (status = "on trust-region boundary")
    npcurv              && (status = "nonpositive curvature")
    solved              && (status = "solution good enough given atol and rtol")
    user_requested_exit && (status = "user-requested exit")
    overtimed           && (status = "time limit exceeded")

    # Update x
    warm_start && @kaxpy!(n, one(FC), Δx, x)
    solver.warm_start = false

    # Update stats
    stats.niter = iter
    stats.solved = solved
    stats.inconsistent = false
    stats.timer = ktimer(start_time)
    stats.status = status
    return solver
  end
end
