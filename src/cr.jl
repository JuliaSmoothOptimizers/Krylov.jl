# A truncated version of Stiefel’s Conjugate Residual method described in
#
# M. R. Hestenes and E. Stiefel, Methods of conjugate gradients for solving linear systems.
# Journal of Research of the National Bureau of Standards, 49(6), pp. 409--436, 1952.
#
# E. Stiefel, Relaxationsmethoden bester Strategie zur Losung linearer Gleichungssysteme.
# Commentarii Mathematici Helvetici, 29(1), pp. 157--179, 1955.
#
# M-A. Dahito and D. Orban, The Conjugate Residual Method in Linesearch and Trust-Region Methods.
# SIAM Journal on Optimization, 29(3), pp. 1988--2025, 2019.
#
# Marie-Ange Dahito, <marie-ange.dahito@polymtl.ca>
# Montreal, QC, June 2017

export cr, cr!

"""
    (x, stats) = cr(A, b::AbstractVector{FC};
                    M=I, atol::T=√eps(T), rtol::T=√eps(T), γ::T=√eps(T), itmax::Int=0,
                    radius::T=zero(T), verbose::Int=0, linesearch::Bool=false, history::Bool=false,
                    callback :: Function = (args...) -> false)

`T` is an `AbstractFloat` such as `Float32`, `Float64` or `BigFloat`.
`FC` is `T` or `Complex{T}`.

A truncated version of Stiefel’s Conjugate Residual method to solve the symmetric linear system Ax = b or the least-squares problem min ‖b - Ax‖.
The matrix A must be positive semi-definite.

A preconditioner M may be provided in the form of a linear operator and is assumed to be symmetric and positive definite.
M also indicates the weighted norm in which residuals are measured.

In a linesearch context, 'linesearch' must be set to 'true'.

If `itmax=0`, the default number of iterations is set to `2 * n`,
with `n = length(b)`.

CR can be warm-started from an initial guess `x0` with the method

    (x, stats) = cr(A, b, x0; kwargs...)

where `kwargs` are the same keyword arguments as above.

The callback is called as `callback(solver, A, b)` and should return `true` if the main loop should terminate,
and `false` otherwise.

#### References

* M. R. Hestenes and E. Stiefel, [*Methods of conjugate gradients for solving linear systems*](https://doi.org/10.6028/jres.049.044), Journal of Research of the National Bureau of Standards, 49(6), pp. 409--436, 1952.
* E. Stiefel, [*Relaxationsmethoden bester Strategie zur Losung linearer Gleichungssysteme*](https://doi.org/10.1007/BF02564277), Commentarii Mathematici Helvetici, 29(1), pp. 157--179, 1955.
* M-A. Dahito and D. Orban, [*The Conjugate Residual Method in Linesearch and Trust-Region Methods*](https://doi.org/10.1137/18M1204255), SIAM Journal on Optimization, 29(3), pp. 1988--2025, 2019.
"""
function cr end

function cr(A, b :: AbstractVector{FC}, x0 :: AbstractVector; kwargs...) where FC <: FloatOrComplex
  solver = CrSolver(A, b)
  cr!(solver, A, b, x0; kwargs...)
  return (solver.x, solver.stats)
end

function cr(A, b :: AbstractVector{FC}; kwargs...) where FC <: FloatOrComplex
  solver = CrSolver(A, b)
  cr!(solver, A, b; kwargs...)
  return (solver.x, solver.stats)
end

"""
    solver = cr!(solver::CrSolver, A, b; kwargs...)
    solver = cr!(solver::CrSolver, A, b, x0; kwargs...)

where `kwargs` are keyword arguments of [`cr`](@ref).

See [`CrSolver`](@ref) for more details about the `solver`.
"""
function cr! end

function cr!(solver :: CrSolver{T,FC,S}, A, b :: AbstractVector{FC}, x0 :: AbstractVector; kwargs...) where {T <: AbstractFloat, FC <: FloatOrComplex{T}, S <: DenseVector{FC}}
  warm_start!(solver, x0)
  cr!(solver, A, b; kwargs...)
  return solver
end

function cr!(solver :: CrSolver{T,FC,S}, A, b :: AbstractVector{FC};
             M=I, atol :: T=√eps(T), rtol :: T=√eps(T), γ :: T=√eps(T), itmax :: Int=0,
             radius :: T=zero(T), verbose :: Int=0, linesearch :: Bool=false, history :: Bool=false,
             callback :: Function = (args...) -> false) where {T <: AbstractFloat, FC <: FloatOrComplex{T}, S <: DenseVector{FC}}

  linesearch && (radius > 0) && error("'linesearch' set to 'true' but radius > 0")
  n, m = size(A)
  m == n || error("System must be square")
  length(b) == n || error("Inconsistent problem size")
  (verbose > 0) && @printf("CR: system of %d equations in %d variables\n", n, n)

  # Tests M = Iₙ
  MisI = (M === I)

  # Check type consistency
  eltype(A) == FC || error("eltype(A) ≠ $FC")
  ktypeof(b) == S || error("ktypeof(b) ≠ $S")

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
  mul!(r, M, p)
  mul!(Ar, A, r)
  ρ = @kdotr(n, r, Ar)

  rNorm = sqrt(@kdotr(n, r, p))   # ‖r‖
  history && push!(rNorms, rNorm) # Values of ‖r‖

  if ρ == 0
    stats.niter = 0
    stats.solved, stats.inconsistent = true, false
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
  (verbose > 0) && @printf("%5s %8s %8s %8s\n", "k", "‖x‖", "‖r‖", "quad")
  kdisplay(iter, verbose) && @printf("    %d  %8.1e %8.1e %8.1e\n", iter, xNorm, rNorm, m)

  descent = pr > 0 # pᵀr > 0 means p is a descent direction
  solved = rNorm ≤ ε
  tired = iter ≥ itmax
  on_boundary = false
  npcurv = false
  status = "unknown"
  user_requested_exit = false

  while ! (solved || tired || user_requested_exit)
    if linesearch
      if (pAp ≤ γ * pNorm²) || (ρ ≤ γ * rNorm²)
        npcurv = true
        (verbose > 0) && @printf("nonpositive curvature detected: pᵀAp = %8.1e and rᵀAr = %8.1e\n", pAp, ρ)
        stats.solved = solved
        stats.inconsistent = false
        stats.status = "nonpositive curvature"
        return solver
      end
    elseif pAp ≤ 0 && radius == 0
      error("Indefinite system and no trust region")
    end
    MisI || mul!(Mq, M, q)

    if radius > 0
      (verbose > 0) && @printf("radius = %8.1e > 0 and ‖x‖ = %8.1e\n", radius, xNorm)
      # find t1 > 0 and t2 < 0 such that ‖x + ti * p‖² = radius²  (i = 1, 2)
      xNorm² = xNorm * xNorm
      t = to_boundary(x, p, radius; flip = false, xNorm2 = xNorm², dNorm2 = pNorm²)
      t1 = maximum(t) # > 0
      t2 = minimum(t) # < 0
      tr = maximum(to_boundary(x, r, radius; flip = false, xNorm2 = xNorm², dNorm2 = rNorm²))
      (verbose > 0) && @printf("t1 = %8.1e, t2 = %8.1e and tr = %8.1e\n", t1, t2, tr)

      if abspAp ≤ γ * pNorm * @knrm2(n, q) # pᵀAp ≃ 0
        npcurv = true # nonpositive curvature
        (verbose > 0) && @printf("pᵀAp = %8.1e ≃ 0\n", pAp)
        if abspr ≤ γ * pNorm * rNorm # pᵀr ≃ 0
          (verbose > 0) && @printf("pᵀr = %8.1e ≃ 0, redefining p := r\n", pr)
          p = r # - ∇q(x)
          q = Ar
          # q(x + αr) = q(x) - α ‖r‖² + ½ α² rᵀAr
          # 1) if rᵀAr > 0, the quadratic decreases from α = 0 to α = ‖r‖² / rᵀAr
          # 2) if rᵀAr ≤ 0, the quadratic decreases to -∞ in the direction r
          if ρ > 0 # case 1
            (verbose > 0) && @printf("quadratic is convex in direction r, curv = %8.1e\n", ρ)
            α = min(tr, rNorm² / ρ)
          else # case 2
            (verbose > 0) && @printf("r is a direction of nonpositive curvature: %8.1e\n", ρ)
            α = tr
          end
        else
          # q_p = q(x + α_p * p) - q(x) = -α_p * rᵀp + ½ (α_p)² * pᵀAp
          # q_r = q(x + α_r * r) - q(x) = -α_r * ‖r‖² + ½ (α_r)² * rᵀAr
          # Δ = q_p - q_r. If Δ > 0, r is followed, else p is followed
          α = descent ? t1 : t2
          ρ > 0 && (tr = min(tr, rNorm² / ρ))
          Δ = -α * pr + tr * rNorm² - (tr)^2 * ρ / 2 # as pᵀAp = 0
          if Δ > 0 # direction r engenders a better decrease
            (verbose > 0) && @printf("direction r engenders a bigger decrease. q_p - q_r = %8.1e > 0\n", Δ)
            (verbose > 0) && @printf("redefining p := r\n")
            p = r
            q = Ar
            α = tr
          else
            (verbose > 0) && @printf("direction p engenders an equal or a bigger decrease. q_p - q_r = %8.1e ≤ 0\n", Δ)
          end
        end

      elseif pAp > 0 && ρ > 0 # no negative curvature
        (verbose > 0) && @printf("positive curvatures along p and r. pᵀAp = %8.1e and rᵀAr = %8.1e\n", pAp, ρ)
        α = ρ / @kdotr(n, q, Mq)
        if α ≥ t1
          α = t1
          on_boundary = true
        end

      elseif pAp > 0 && ρ < 0
        npcurv = true
        (verbose > 0) && @printf("pᵀAp = %8.1e > 0 and rᵀAr = %8.1e < 0\n", pAp, ρ)
        # q_p is minimal for α_p = rᵀp / pᵀAp
        α = descent ?  min(t1, pr / pAp) : max(t2, pr / pAp)
        Δ = -α * pr + tr * rNorm² + (α^2 * pAp - (tr)^2 * ρ) / 2
        if Δ > 0
          (verbose > 0) && @printf("direction r engenders a bigger decrease. q_p - q_r = %8.1e > 0\n", Δ)
          (verbose > 0) && @printf("redefining p := r\n")
          p = r
          q = Ar
          α = tr
        else
          (verbose > 0) && @printf("direction p engenders an equal or a bigger decrease. q_p - q_r = %8.1e ≤ 0\n", Δ)
        end

      elseif pAp < 0 && ρ > 0
        npcurv = true
        (verbose > 0) && @printf("pᵀAp = %8.1e < 0 and rᵀAr = %8.1e > 0\n", pAp, ρ)
        α = descent ? t1 : t2
        tr = min(tr, rNorm² / ρ)
        Δ = -α * pr + tr * rNorm² + (α^2 * pAp - (tr)^2 * ρ) / 2
        if Δ > 0
          (verbose > 0) && @printf("direction r engenders a bigger decrease. q_p - q_r = %8.1e > 0\n", Δ)
          (verbose > 0) && @printf("redefining p := r\n")
          p = r
          q = Ar
          α = tr
        else
          (verbose > 0) && @printf("direction p engenders an equal or a bigger decrease. q_p - q_r = %8.1e ≤ 0\n", Δ)
        end

      elseif pAp < 0 && ρ < 0
        npcurv = true
        (verbose > 0) && @printf("negative curvatures along p and r. pᵀAp = %8.1e and rᵀAr = %8.1e\n", pAp, ρ)
        α = descent ? t1 : t2
        Δ = -α * pr + tr * rNorm² + (α^2 * pAp - (tr)^2 * ρ) / 2
        if Δ > 0
          (verbose > 0) && @printf("direction r engenders a bigger decrease. q_p - q_r = %8.1e > 0\n", Δ)
          (verbose > 0) && @printf("redefining p := r\n")
          p = r
          q = Ar
          α = tr
        else
          (verbose > 0) && @printf("direction p engenders an equal or a bigger decrease. q_p - q_r = %8.1e ≤ 0\n", Δ)
        end
      end

    elseif radius == 0
      α = ρ / @kdotr(n, q, Mq) # step
    end

    @kaxpy!(n, α, p, x)
    xNorm = @knrm2(n, x)
    xNorm ≈ radius && (on_boundary = true)
    @kaxpy!(n, -α, Mq, r) # residual
    rNorm² = abs(rNorm² - α * ρ)
    rNorm = sqrt(rNorm²)
    history && push!(rNorms, rNorm)
    mul!(Ar, A, r)
    ArNorm = @knrm2(n, Ar)
    history && push!(ArNorms, ArNorm)

    iter = iter + 1
    if kdisplay(iter, verbose)
      m = m - α * pr + α^2 * pAp / 2
      @printf("    %d  %8.1e %8.1e %8.1e\n", iter, xNorm, rNorm, m)
    end

    user_requested_exit = callback(solver, A, b) :: Bool
    solved = (rNorm ≤ ε) || npcurv || on_boundary
    tired = iter ≥ itmax

    (solved || tired || user_requested_exit) && continue
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
      stats.status = "solver encountered numerical issues"
      solver.warm_start = false
      return solver
    end
    pr = rNorm² + β * pr - β * α * pAp # pᵀr
    abspr = abs(pr)
    pAp = ρ + β^2 * pAp # pᵀq
    abspAp = abs(pAp)
    descent = pr > 0

  end
  (verbose > 0) && @printf("\n")

  tired               && (status = "maximum number of iterations exceeded")
  on_boundary         && (status = "on trust-region boundary")
  npcurv              && (status = "nonpositive curvature")
  solved              && (status = "solution good enough given atol and rtol")
  user_requested_exit && (status = "user-requested exit")

  # Update x
  warm_start && @kaxpy!(n, one(FC), Δx, x)
  solver.warm_start = false

  # Update stats
  stats.niter = iter
  stats.solved = solved
  stats.inconsistent = false
  stats.status = status
  return solver
end
