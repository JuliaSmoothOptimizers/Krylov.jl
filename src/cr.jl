# A truncated version of Stiefel’s Conjugate Residual method
# cr(A, b, M, atol, rtol, γ, itmax, radius, verbose, linesearch) solves the linear system 'A * x = b' or the least-squares problem :
# 'min ‖b - A * x‖²' within a region of fixed radius.
#
# Marie-Ange Dahito, <marie-ange.dahito@polymtl.ca>
# Montreal, QC, June 2017

export cr

"""
    (x, stats) = cr(A, b; M, atol, rtol, γ, itmax, radius, verbose, linesearch)

A truncated version of Stiefel’s Conjugate Residual method to solve the symmetric linear system Ax=b.
The matrix A must be positive semi-definite.

A preconditioner M may be provided in the form of a linear operator and is
assumed to be symmetric and positive definite.
M also indicates the weighted norm in which residuals are measured.

In a linesearch context, 'linesearch' must be set to 'true'.
"""
function cr(A, b :: AbstractVector{T};
            M=opEye(), atol :: T=√eps(T), rtol :: T=√eps(T), γ :: T=√eps(T), itmax :: Int=0,
            radius :: T=zero(T), verbose :: Bool=false, linesearch :: Bool=false) where T <: AbstractFloat

  if linesearch && (radius > 0)
    error("'linesearch' set to 'true' but radius > 0")
  end
  n = size(b, 1) # size of the problem
  (size(A, 1) == n & size(A, 2) == n) || error("Inconsistent problem size")
  verbose && @printf("CR: system of %d equations in %d variables\n", n, n)

  # Check type consistency
  eltype(A) == T || error("eltype(A) ≠ $T")
  isa(M, opEye) || (eltype(M) == T) || error("eltype(M) ≠ $T")

  # Determine the storage type of b
  S = typeof(b)

  # Initial state.
  x = kzeros(S, n)  # initial estimation x = 0
  xNorm = zero(T)
  r = copy(M * b)  # initial residual r = M * (b - Ax) = M * b
  Ar = A * r
  ρ = @kdot(n, r, Ar)
  ρ == 0 && return (x, SimpleStats(true, false, [zero(T)], T[], "x = 0 is a zero-residual solution"))
  p = copy(r)
  q = copy(Ar)
  if verbose
    m = zero(T) # quadratic model
  end

  iter = 0
  itmax == 0 && (itmax = 2 * n)

  rNorm = sqrt(@kdot(n, r, b)) # ‖r‖
  rNorms = [rNorm] # Values of ‖r‖
  rNorm² = rNorm * rNorm
  pNorm = rNorm
  pNorm² = rNorm²
  pr = rNorm²
  abspr = pr
  pAp = ρ
  abspAp = abs(pAp)
  ArNorm = @knrm2(n, Ar) # ‖Ar‖
  ArNorms = [ArNorm]
  ε = atol + rtol * rNorm
  verbose && @printf("%5s %8s %8s %8s\n", "Iter", "‖x‖", "‖r‖", "quad")
  verbose && @printf("    %d  %8.1e %8.1e %8.1e\n", iter, xNorm, rNorm, m)

  descent = pr > 0 # pᵀr > 0 means p is a descent direction
  solved = rNorm ≤ ε
  tired = iter ≥ itmax
  on_boundary = false
  npcurv = false
  status = "unknown"

  while ! (solved || tired)
    if linesearch
      if (pAp ≤ γ * pNorm²) || (ρ ≤ γ * rNorm²)
        npcurv = true
        verbose && @printf("nonpositive curvature detected: pᵀAp = %8.1e and rᵀAr = %8.1e\n", pAp, ρ)
        stats = SimpleStats(solved, false, rNorms, ArNorms, "nonpositive curvature")
        iter == 0 && return (b, stats)
        return (x, stats)
      end
    elseif pAp ≤ 0 && radius == 0
      error("Indefinite system and no trust region")
    end
    Mq = M * q

    if radius > 0
      verbose && @printf("radius = %8.1e > 0 and ‖x‖ = %8.1e\n", radius, xNorm)
      # find t1 > 0 and t2 < 0 such that ‖x + ti * p‖² = radius²  (i = 1, 2)
      xNorm² = xNorm * xNorm
      t = to_boundary(x, p, radius; flip = false, xNorm2 = xNorm², dNorm2 = pNorm²)
      t1 = maximum(t) # > 0
      t2 = minimum(t) # < 0
      tr = maximum(to_boundary(x, r, radius; flip = false, xNorm2 = xNorm², dNorm2 = rNorm²))
      verbose && @printf("t1 = %8.1e, t2 = %8.1e and tr = %8.1e\n", t1, t2, tr)

      if abspAp ≤ γ * pNorm * @knrm2(n, q) # pᵀAp ≃ 0
        npcurv = true # nonpositive curvature
        verbose && @printf("pᵀAp = %8.1e ≃ 0\n", pAp)
        if abspr ≤ γ * pNorm * rNorm # pᵀr ≃ 0
          verbose && @printf("pᵀr = %8.1e ≃ 0, redefining p := r\n", pr)
          p = r # - ∇q(x)
          q = Ar
          # q(x + αr) = q(x) - α ‖r‖² + ½ α² rᵀAr
          # 1) if rᵀAr > 0, the quadratic decreases from α = 0 to α = ‖r‖² / rᵀAr
          # 2) if rᵀAr ≤ 0, the quadratic decreases to -∞ in the direction r
          if ρ > 0 # case 1
            verbose && @printf("quadratic is convex in direction r, curv = %8.1e\n", ρ)
            α = min(tr, rNorm² / ρ)
          else # case 2
            verbose && @printf("r is a direction of nonpositive curvature: %8.1e\n", ρ)
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
            verbose && @printf("direction r engenders a bigger decrease. q_p - q_r = %8.1e > 0\n", Δ)
            verbose && @printf("redefining p := r\n")
            p = r
            q = Ar
            α = tr
          else
            verbose && @printf("direction p engenders an equal or a bigger decrease. q_p - q_r = %8.1e ≤ 0\n", Δ)
          end
        end

      elseif pAp > 0 && ρ > 0 # no negative curvature
        verbose && @printf("positive curvatures along p and r. pᵀAp = %8.1e and rᵀAr = %8.1e\n", pAp, ρ)
        α = ρ / @kdot(n, q, Mq)
        if α ≥ t1
          α = t1
          on_boundary = true
        end

      elseif pAp > 0 && ρ < 0
        npcurv = true
        verbose && @printf("pᵀAp = %8.1e > 0 and rᵀAr = %8.1e < 0\n", pAp, ρ)
        # q_p is minimal for α_p = rᵀp / pᵀAp
        α = descent ?  min(t1, pr / pAp) : max(t2, pr / pAp)
        Δ = -α * pr + tr * rNorm² + (α^2 * pAp - (tr)^2 * ρ) / 2
        if Δ > 0
          verbose && @printf("direction r engenders a bigger decrease. q_p - q_r = %8.1e > 0\n", Δ)
          verbose && @printf("redefining p := r\n")
          p = r
          q = Ar
          α = tr
        else
          verbose && @printf("direction p engenders an equal or a bigger decrease. q_p - q_r = %8.1e ≤ 0\n", Δ)
        end

      elseif pAp < 0 && ρ > 0
        npcurv = true
        verbose && @printf("pᵀAp = %8.1e < 0 and rᵀAr = %8.1e > 0\n", pAp, ρ)
        α = descent ? t1 : t2
        tr = min(tr, rNorm² / ρ)
        Δ = -α * pr + tr * rNorm² + (α^2 * pAp - (tr)^2 * ρ) / 2
        if Δ > 0
          verbose && @printf("direction r engenders a bigger decrease. q_p - q_r = %8.1e > 0\n", Δ)
          verbose && @printf("redefining p := r\n")
          p = r
          q = Ar
          α = tr
        else
          verbose && @printf("direction p engenders an equal or a bigger decrease. q_p - q_r = %8.1e ≤ 0\n", Δ)
        end

      elseif pAp < 0 && ρ < 0
        npcurv = true
        verbose && @printf("negative curvatures along p and r. pᵀAp = %8.1e and rᵀAr = %8.1e\n", pAp, ρ)
        α = descent ? t1 : t2
        Δ = -α * pr + tr * rNorm² + (α^2 * pAp - (tr)^2 * ρ) / 2
        if Δ > 0
          verbose && @printf("direction r engenders a bigger decrease. q_p - q_r = %8.1e > 0\n", Δ)
          verbose && @printf("redefining p := r\n")
          p = r
          q = Ar
          α = tr
        else
          verbose && @printf("direction p engenders an equal or a bigger decrease. q_p - q_r = %8.1e ≤ 0\n", Δ)
        end
      end

    elseif radius == 0
      α = ρ / @kdot(n, q, Mq) # step
    end

    @kaxpy!(n, α, p, x)
    xNorm = @knrm2(n, x)
    xNorm ≈ radius && (on_boundary = true)
    @kaxpy!(n, -α, Mq, r) # residual
    rNorm² = abs(rNorm² - α * ρ)
    rNorm = sqrt(rNorm²)
    push!(rNorms, rNorm)
    Ar = A * r
    ArNorm = @knrm2(n, Ar)
    push!(ArNorms, ArNorm)

    iter = iter + 1
    if verbose
      m = m - α * pr + α^2 * pAp / 2
      @printf("    %d  %8.1e %8.1e %8.1e\n", iter, xNorm, rNorm, m)
    end

    solved = (rNorm ≤ ε) || npcurv || on_boundary
    tired = iter ≥ itmax

    (solved || tired) && continue
    ρbar = ρ
    ρ = @kdot(n, r, Ar)
    β = ρ / ρbar # step for the direction computation
    @kaxpby!(n, one(T), r, β, p)
    @kaxpby!(n, one(T), Ar, β, q)

    pNorm² = rNorm² + 2 * β * pr - 2 * β * α * pAp + β^2 * pNorm²
    pNorm = sqrt(pNorm²)
    pr = rNorm² + β * pr - β * α * pAp # pᵀr
    abspr = abs(pr)
    pAp = ρ + β^2 * pAp # pᵀq
    abspAp = abs(pAp)
    descent = pr > 0

  end
  verbose && @printf("\n")

  status = npcurv ? "nonpositive curvature" : (on_boundary ? "on trust-region boundary" : (tired ? "maximum number of iterations exceeded" : "solution good enough given atol and rtol"))
  stats = SimpleStats(solved, false, rNorms, ArNorms, status)
  return (x, stats)
end
