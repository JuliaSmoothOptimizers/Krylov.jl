# An implementation of CRMR for the solution of the
# (under/over-determined or square) linear system
#
#  Ax = b.
#
# The method seeks to solve the minimum-norm problem
#
#  min ‖x‖²  s.t. Ax = b,
#
# and is equivalent to applying the conjugate residual method
# to the linear system
#
#  AAᵀy = b.
#
# This method is equivalent to Craig-MR, described in
#
# M. Arioli and D. Orban, Iterative Methods for Symmetric
# Quasi-Definite Linear Systems, Part I: Theory.
# Cahier du GERAD G-2013-32, GERAD, Montreal QC, Canada, 2013.
#
# D. Orban, The Projected Golub-Kahan Process for Constrained
# Linear Least-Squares Problems. Cahier du GERAD G-2014-15,
# GERAD, Montreal QC, Canada, 2014.
#
# Dominique Orban, <dominique.orban@gerad.ca>
# Montreal, QC, April 2015.

export crmr


"""
    (x, stats) = crmr(A, b; M, λ, atol, rtol, itmax, verbose)

Solve the consistent linear system

    Ax + √λs = b

using the Conjugate Residual (CR) method, where λ ≥ 0 is a regularization
parameter. This method is equivalent to applying CR to the normal equations
of the second kind

    (AAᵀ + λI) y = b

but is more stable. When λ = 0, this method solves the minimum-norm problem

    min ‖x‖₂  s.t.  x ∈ argmin ‖Ax - b‖₂.

When λ > 0, this method solves the problem

    min ‖(x,s)‖₂  s.t. Ax + √λs = b.

CGMR produces monotonic residuals ‖r‖₂.
It is formally equivalent to CRAIG-MR, though can be slightly less accurate,
but simpler to implement. Only the x-part of the solution is returned.

A preconditioner M may be provided in the form of a linear operator.
"""
function crmr(A, b :: AbstractVector{T};
              M=opEye(), λ :: T=zero(T), atol :: T=√eps(T),
              rtol :: T=√eps(T), itmax :: Int=0, verbose :: Bool=false) where T <: AbstractFloat

  m, n = size(A)
  size(b, 1) == m || error("Inconsistent problem size")
  verbose && @printf("CRMR: system of %d equations in %d variables\n", m, n)

  # Check type consistency
  eltype(A) == T || error("eltype(A) ≠ $T")
  isa(M, opEye) || (eltype(M) == T) || error("eltype(M) ≠ $T")

  # Compute the adjoint of A
  Aᵀ = A'

  # Determine the storage type of b
  S = typeof(b)

  x = kzeros(S, n)  # initial estimation x = 0
  r = copy(M * b)   # initial residual r = M * (b - Ax) = M * b
  bNorm = @knrm2(m, r)  # norm(b - A * x0) if x0 ≠ 0.
  bNorm == 0 && return x, SimpleStats(true, false, [zero(T)], [zero(T)], "x = 0 is a zero-residual solution")
  rNorm = bNorm  # + λ * ‖x0‖ if x0 ≠ 0 and λ > 0.
  λ > 0 && (s = copy(r))
  Aᵀr = Aᵀ * r # - λ * x0 if x0 ≠ 0.
  p  = copy(Aᵀr)
  γ  = @kdot(n, Aᵀr, Aᵀr)  # Faster than γ = dot(Aᵀr, Aᵀr)
  λ > 0 && (γ += λ * rNorm * rNorm)
  iter = 0
  itmax == 0 && (itmax = m + n)

  ArNorm = sqrt(γ)
  rNorms = [rNorm;]
  ArNorms = [ArNorm;]
  ɛ_c = atol + rtol * rNorm  # Stopping tolerance for consistent systems.
  ɛ_i = atol + rtol * ArNorm  # Stopping tolerance for inconsistent systems.
  verbose && @printf("%5s  %8s  %8s\n", "Aprod", "‖Aᵀr‖", "‖r‖")
  verbose && @printf("%5d  %8.2e  %8.2e\n", 1, ArNorm, rNorm)

  status = "unknown"
  solved = rNorm ≤ ɛ_c
  inconsistent = (rNorm > 100 * ɛ_c) && (ArNorm ≤ ɛ_i)
  tired = iter ≥ itmax

  while ! (solved || inconsistent || tired)
    q = A * p
    λ > 0 && @kaxpy!(m, λ, s, q)  # q = q + λ * s
    Mq = M * q
    α = γ / @kdot(m, q, Mq)    # Compute qᵗ * M * q
    @kaxpy!(n,  α, p, x)       # Faster than  x =  x + α *  p
    @kaxpy!(m, -α, Mq, r)      # Faster than  r =  r - α * Mq
    rNorm = @knrm2(m, r)       # norm(r)
    Aᵀr = Aᵀ * r
    γ_next = @kdot(n, Aᵀr, Aᵀr)  # Faster than γ_next = dot(Aᵀr, Aᵀr)
    λ > 0 && (γ_next += λ * rNorm * rNorm)
    β = γ_next / γ

    @kaxpby!(n, one(T), Aᵀr, β, p)  # Faster than  p = Aᵀr + β * p
    if λ > 0
      @kaxpby!(m, one(T), r, β, s) # s = r + β * s
    end

    γ = γ_next
    ArNorm = sqrt(γ)
    push!(rNorms, rNorm)
    push!(ArNorms, ArNorm)
    iter = iter + 1
    verbose && @printf("%5d  %8.2e  %8.2e\n", 1 + 2 * iter, ArNorm, rNorm)
    solved = rNorm ≤ ɛ_c
    inconsistent = (rNorm > 100 * ɛ_c) && (ArNorm ≤ ɛ_i)
    tired = iter ≥ itmax
  end

  status = tired ? "maximum number of iterations exceeded" : (inconsistent ? "system probably inconsistent but least squares/norm solution found" : "solution good enough given atol and rtol")
  stats = SimpleStats(solved, inconsistent, rNorms, ArNorms, status)
  return (x, stats)
end
