# An implementation of CRLS for the solution of the
# over-determined linear least-squares problem
#
#  minimize ‖Ax - b‖₂
#
# equivalently, of the linear system
#
#  AᵀAx = Aᵀb.
#
# This implementation follows the formulation given in
#
# D. C.-L. Fong, Minimum-Residual Methods for Sparse
# Least-Squares using Golubg-Kahan Bidiagonalization,
# Ph.D. Thesis, Stanford University, 2011
#
# with the difference that it also recurs r = b - Ax.
#
# Dominique Orban, <dominique.orban@gerad.ca>
# Princeton, NJ, March 2015.

export crls


"""
    (x, stats) = crls(A, b; M, λ, atol, rtol, radius, itmax, verbose)

Solve the linear least-squares problem

    minimize ‖b - Ax‖₂² + λ ‖x‖₂²

using the Conjugate Residuals (CR) method. This method is equivalent to
applying MINRES to the normal equations

    (AᵀA + λI) x = Aᵀb.

This implementation recurs the residual r := b - Ax.

CRLS produces monotonic residuals ‖r‖₂ and optimality residuals ‖Aᵀr‖₂.
It is formally equivalent to LSMR, though can be substantially less accurate,
but simpler to implement.
"""
function crls(A, b :: AbstractVector{T};
              M=opEye(), λ :: T=zero(T), atol :: T=√eps(T), rtol :: T=√eps(T),
              radius :: T=zero(T), itmax :: Int=0, verbose :: Bool=false) where T <: AbstractFloat

  m, n = size(A)
  size(b, 1) == m || error("Inconsistent problem size")
  verbose && @printf("CRLS: system of %d equations in %d variables\n", m, n)

  # Check type consistency
  eltype(A) == T || error("eltype(A) ≠ $T")
  isa(M, opEye) || (eltype(M) == T) || error("eltype(M) ≠ $T")

  # Compute the adjoint of A
  Aᵀ = A'

  # Determine the storage type of b
  S = typeof(b)

  x = kzeros(S, n)
  r  = copy(b)
  bNorm = @knrm2(m, r)  # norm(b - A * x0) if x0 ≠ 0.
  bNorm == 0 && return x, SimpleStats(true, false, [zero(T)], [zero(T)], "x = 0 is a zero-residual solution")

  Mr = M * r
  Ar = copy(Aᵀ * Mr)  # - λ * x0 if x0 ≠ 0.
  s  = A * Ar
  Ms = M * s

  p  = copy(Ar)
  Ap = copy(s)
  q  = Aᵀ * Ms # Ap
  λ > 0 && @kaxpy!(n, λ, p, q)  # q = q + λ * p
  γ  = @kdot(m, s, Ms)  # Faster than γ = dot(s, Ms)
  iter = 0
  itmax == 0 && (itmax = m + n)

  rNorm = bNorm  # + λ * ‖x0‖ if x0 ≠ 0 and λ > 0.
  ArNorm = @knrm2(n, Ar)  # Marginally faster than norm(Ar)
  λ > 0 && (γ += λ * ArNorm * ArNorm)
  rNorms = [rNorm;]
  ArNorms = [ArNorm;]
  ε = atol + rtol * ArNorm
  verbose && @printf("%5s  %8s  %8s\n", "Aprod", "‖Aᵀr‖", "‖r‖")
  verbose && @printf("%5d  %8.2e  %8.2e\n", 3, ArNorm, rNorm)

  status = "unknown"
  on_boundary = false
  solved = ArNorm ≤ ε
  tired = iter ≥ itmax
  psd = false

  while ! (solved || tired)
    qNorm² = @kdot(n, q, q) # dot(q, q)
    α = γ / qNorm²

    # if a trust-region constraint is give, compute step to the boundary
    # (note that α > 0 in CRLS)
    if radius > 0
      pNorm = @knrm2(n, p)
      if @knrm2(m, Ap)^2 ≤ ε * sqrt(qNorm²) * pNorm # the quadratic is constant in the direction p
        psd = true # det(AᵀA) = 0
        p = Ar # p = Aᵀr
        pNorm² = ArNorm * ArNorm
        q = Aᵀ * s
        α = min(ArNorm^2 / γ, maximum(to_boundary(x, p, radius, flip = false, dNorm2 = pNorm²))) # the quadratic is minimal in the direction Aᵀr for α = ‖Ar‖²/γ
      else
        pNorm² = pNorm * pNorm
        σ = maximum(to_boundary(x, p, radius, flip = false, dNorm2 = pNorm²))
        if α ≥ σ
          α = σ
          on_boundary = true
        end
      end
    end

    @kaxpy!(n,  α, p,   x)     # Faster than  x =  x + α *  p
    @kaxpy!(n, -α, q,  Ar)     # Faster than Ar = Ar - α *  q
    ArNorm = @knrm2(n, Ar)
    solved = psd || on_boundary
    solved && continue
    @kaxpy!(m, -α, Ap,  r)     # Faster than  r =  r - α * Ap
    s = A * Ar
    Ms = M * s
    γ_next = @kdot(m, s, Ms)   # Faster than γ_next = dot(s, s)
    λ > 0 && (γ_next += λ * ArNorm * ArNorm)
    β = γ_next / γ

    @kaxpby!(n, one(T), Ar, β, p)    # Faster than  p = Ar + β *  p
    @kaxpby!(m, one(T), s, β, Ap)    # Faster than Ap =  s + β * Ap
    MAp = M * Ap
    q = Aᵀ * MAp
    λ > 0 && @kaxpy!(n, λ, p, q)  # q = q + λ * p

    γ = γ_next
    if λ > 0
      rNorm = sqrt(@kdot(m, r, r) + λ * @kdot(n, x, x))
    else
      rNorm = @knrm2(m, r)  # norm(r)
    end
    #     ArNorm = norm(Ar)
    push!(rNorms, rNorm)
    push!(ArNorms, ArNorm)
    iter = iter + 1
    verbose && @printf("%5d  %8.2e  %8.2e\n", 3 + 2 * iter, ArNorm, rNorm)
    solved = (ArNorm ≤ ε) || on_boundary
    tired = iter ≥ itmax
  end

  status = psd ? "zero-curvature encountered" : (on_boundary ? "on trust-region boundary" : (tired ? "maximum number of iterations exceeded" : "solution good enough given atol and rtol"))
  stats = SimpleStats(solved, false, rNorms, ArNorms, status)
  return (x, stats)
end
