# An implementation of CGLS for the solution of the
# over-determined linear least-squares problem
#
#  minimize ‖Ax - b‖
#
# equivalently, of the normal equations
#
#  AᵀAx = Aᵀb.
#
# CGLS is formally equivalent to applying the conjugate gradient method
# to the normal equations but should be more stable. It is also formally
# equivalent to LSQR though LSQR should be expected to be more stable on
# ill-conditioned or poorly scaled problems.
#
# This method is described in
#
# M. R. Hestenes and E. Stiefel. Methods of conjugate gradients for solving linear systems.
# Journal of Research of the National Bureau of Standards, 49(6), pp. 409--436, 1952.
#
# This implementation is the standard formulation, as recommended by
#
# A. Björck, T. Elfving and Z. Strakos, Stability of Conjugate Gradient
# and Lanczos Methods for Linear Least Squares Problems.
# SIAM Journal on Matrix Analysis and Applications, 19(3), pp. 720--736, 1998.
#
# Dominique Orban, <dominique.orban@gerad.ca>
# Princeton, NJ, March 2015.

export cgls, cgls!


"""
    (x, stats) = cgls(A, b::AbstractVector{T};
                      M=opEye(), λ::T=zero(T), atol::T=√eps(T), rtol::T=√eps(T),
                      radius::T=zero(T), itmax::Int=0, verbose::Int=0, history::Bool=false) where T <: AbstractFloat

Solve the regularized linear least-squares problem

    minimize ‖b - Ax‖₂² + λ ‖x‖₂²

using the Conjugate Gradient (CG) method, where λ ≥ 0 is a regularization
parameter. This method is equivalent to applying CG to the normal equations

    (AᵀA + λI) x = Aᵀb

but is more stable.

CGLS produces monotonic residuals ‖r‖₂ but not optimality residuals ‖Aᵀr‖₂.
It is formally equivalent to LSQR, though can be slightly less accurate,
but simpler to implement.

#### References

* M. R. Hestenes and E. Stiefel. *Methods of conjugate gradients for solving linear systems*, Journal of Research of the National Bureau of Standards, 49(6), pp. 409--436, 1952.
* A. Björck, T. Elfving and Z. Strakos, *Stability of Conjugate Gradient and Lanczos Methods for Linear Least Squares Problems*, SIAM Journal on Matrix Analysis and Applications, 19(3), pp. 720--736, 1998.
"""
function cgls(A, b :: AbstractVector{T}; kwargs...) where T <: AbstractFloat
  solver = CglsSolver(A, b)
  cgls!(solver, A, b; kwargs...)
end

function cgls!(solver :: CglsSolver{T,S}, A, b :: AbstractVector{T};
               M=opEye(), λ :: T=zero(T), atol :: T=√eps(T), rtol :: T=√eps(T),
               radius :: T=zero(T), itmax :: Int=0, verbose :: Int=0, history :: Bool=false) where {T <: AbstractFloat, S <: DenseVector{T}}

  m, n = size(A)
  size(b, 1) == m || error("Inconsistent problem size")
  (verbose > 0) && @printf("CGLS: system of %d equations in %d variables\n", m, n)

  # Check type consistency
  eltype(A) == T || error("eltype(A) ≠ $T")
  ktypeof(b) == S || error("ktypeof(b) ≠ $S")
  isa(M, opEye) || eltype(M) == T || error("eltype(M) ≠ $T")

  # Compute the adjoint of A
  Aᵀ = A'

  # Set up workspace.
  x, p, r = solver.x, solver.p, solver.r

  x .= zero(T)
  r .= b
  bNorm = @knrm2(m, r)   # Marginally faster than norm(b)
  bNorm == 0 && return x, SimpleStats(true, false, [zero(T)], [zero(T)], "x = 0 is a zero-residual solution")
  Mr = M * r
  s = Aᵀ * Mr
  p .= s
  γ = @kdot(n, s, s)  # Faster than γ = dot(s, s)
  iter = 0
  itmax == 0 && (itmax = m + n)

  rNorm  = bNorm
  ArNorm = sqrt(γ)
  rNorms = history ? [rNorm] : T[]
  ArNorms = history ? [ArNorm] : T[]
  ε = atol + rtol * ArNorm
  (verbose > 0) && @printf("%5s  %8s  %8s\n", "Aprod", "‖A'r‖", "‖r‖")
  display(iter, verbose) && @printf("%5d  %8.2e  %8.2e\n", 1, ArNorm, rNorm)

  status = "unknown"
  on_boundary = false
  solved = ArNorm ≤ ε
  tired = iter ≥ itmax

  while ! (solved || tired)
    q = A * p
    Mq = M * q
    δ = @kdot(m, q, Mq)   # Faster than α = γ / dot(q, q)
    λ > 0 && (δ += λ * @kdot(n, p, p))
    α = γ / δ

    # if a trust-region constraint is give, compute step to the boundary
    σ = radius > 0 ? maximum(to_boundary(x, p, radius)) : α
    if (radius > 0) & (α > σ)
      α = σ
      on_boundary = true
    end

    @kaxpy!(n,  α, p, x)     # Faster than x = x + α * p
    @kaxpy!(m, -α, q, r)     # Faster than r = r - α * q
    Mr = M * r
    s = Aᵀ * Mr
    λ > 0 && @kaxpy!(n, -λ, x, s)   # s = A' * r - λ * x
    γ_next = @kdot(n, s, s)  # Faster than γ_next = dot(s, s)
    β = γ_next / γ
    @kaxpby!(n, one(T), s, β, p) # Faster than p = s + β * p
    γ = γ_next
    rNorm = @knrm2(m, r)  # Marginally faster than norm(r)
    ArNorm = sqrt(γ)
    history && push!(rNorms, rNorm)
    history && push!(ArNorms, ArNorm)
    iter = iter + 1
    display(iter, verbose) && @printf("%5d  %8.2e  %8.2e\n", 1 + 2 * iter, ArNorm, rNorm)
    solved = (ArNorm ≤ ε) | on_boundary
    tired = iter ≥ itmax
  end
  (verbose > 0) && @printf("\n")

  status = on_boundary ? "on trust-region boundary" : (tired ? "maximum number of iterations exceeded" : "solution good enough given atol and rtol")
  stats = SimpleStats(solved, false, rNorms, ArNorms, status)
  return (x, stats)
end
