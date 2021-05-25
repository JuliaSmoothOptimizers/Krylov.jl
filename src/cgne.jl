# An implementation of CGNE for the solution of the consistent
# (under/over-determined or square) linear system
#
#  Ax = b.
#
# The method seeks to solve the minimum-norm problem
#
#  min ‖x‖²  s.t. Ax = b,
#
# and is equivalent to applying the conjugate gradient method
# to the linear system
#
#  AAᵀy = b.
#
# This method is also known as Craig's method, CGME, and other
# names, and is described in
#
# J. E. Craig. The N-step iteration procedures.
# Journal of Mathematics and Physics, 34(1), pp. 64--73, 1955.
#
# which is based on Craig's thesis from MIT:
#
# J. E. Craig. Iterations Procedures for Simultaneous Equations.
# Ph.D. Thesis, Department of Electrical Engineering, MIT, 1954.
#
# Dominique Orban, <dominique.orban@gerad.ca>
# Montréal, QC, April 2015.

export cgne, cgne!


"""
    (x, stats) = cgne(A, b::AbstractVector{T};
                      M=opEye(), λ::T=zero(T), atol::T=√eps(T), rtol::T=√eps(T),
                      itmax::Int=0, verbose::Int=0, history::Bool=false) where T <: AbstractFloat

Solve the consistent linear system

    Ax + √λs = b

using the Conjugate Gradient (CG) method, where λ ≥ 0 is a regularization
parameter. This method is equivalent to applying CG to the normal equations
of the second kind

    (AAᵀ + λI) y = b

but is more stable. When λ = 0, this method solves the minimum-norm problem

  min ‖x‖₂  s.t. Ax = b.

When λ > 0, it solves the problem

    min ‖(x,s)‖₂  s.t. Ax + √λs = b.

CGNE produces monotonic errors ‖x-x*‖₂ but not residuals ‖r‖₂.
It is formally equivalent to CRAIG, though can be slightly less accurate,
but simpler to implement. Only the x-part of the solution is returned.

A preconditioner M may be provided in the form of a linear operator.

#### References

* J. E. Craig, *The N-step iteration procedures*, Journal of Mathematics and Physics, 34(1), pp. 64--73, 1955.
* J. E. Craig, *Iterations Procedures for Simultaneous Equations*, Ph.D. Thesis, Department of Electrical Engineering, MIT, 1954.
"""
function cgne(A, b :: AbstractVector{T}; kwargs...) where T <: AbstractFloat
  solver = CgneSolver(A, b)
  cgne!(solver, A, b; kwargs...)
end

function cgne!(solver :: CgneSolver{T,S}, A, b :: AbstractVector{T};
               M=opEye(), λ :: T=zero(T), atol :: T=√eps(T), rtol :: T=√eps(T),
               itmax :: Int=0, verbose :: Int=0, history :: Bool=false) where {T <: AbstractFloat, S <: DenseVector{T}}

  m, n = size(A)
  length(b) == m || error("Inconsistent problem size")
  (verbose > 0) && @printf("CGNE: system of %d equations in %d variables\n", m, n)

  # Tests M == Iₙ
  MisI = isa(M, opEye) || (M == I)

  # Check type consistency
  eltype(A) == T || error("eltype(A) ≠ $T")
  ktypeof(b) == S || error("ktypeof(b) ≠ $S")
  MisI || (eltype(M) == T) || error("eltype(M) ≠ $T")

  # Compute the adjoint of A
  Aᵀ = A'

  # Set up workspace.
  !MisI   && isnothing(solver.z) && (solver.z = S(undef, m))
  (λ > 0) && isnothing(solver.s) && (solver.s = S(undef, m))
  x, p, Aᵀz, r, q, s = solver.x, solver.p, solver.Aᵀz, solver.r, solver.q, solver.s
  z = MisI ? r : solver.z

  x .= zero(T)
  r .= b
  MisI || mul!(z, M, r)
  rNorm = @knrm2(m, r)   # Marginally faster than norm(r)
  rNorm == 0 && return x, SimpleStats(true, false, [rNorm], T[], "x = 0 is a zero-residual solution")
  λ > 0 && (s .= r)

  mul!(p, Aᵀ, z)

  # Use ‖p‖ to detect inconsistent system.
  # An inconsistent system will necessarily have AA' singular.
  # Because CGNE is equivalent to CG applied to AA'y = b, there will be a
  # conjugate direction u such that u'AA'u = 0, i.e., A'u = 0. In this
  # implementation, p is a substitute for A'u.
  pNorm = @knrm2(n, p)

  γ = @kdot(m, r, z)  # Faster than γ = dot(r, z)
  iter = 0
  itmax == 0 && (itmax = m + n)

  rNorms = history ? [rNorm] : T[]
  ɛ_c = atol + rtol * rNorm  # Stopping tolerance for consistent systems.
  ɛ_i = atol + rtol * pNorm  # Stopping tolerance for inconsistent systems.
  (verbose > 0) && @printf("%5s  %8s\n", "Aprod", "‖r‖")
  display(iter, verbose) && @printf("%5d  %8.2e\n", 1, rNorm)

  status = "unknown"
  solved = rNorm ≤ ɛ_c
  inconsistent = (rNorm > 100 * ɛ_c) && (pNorm ≤ ɛ_i)
  tired = iter ≥ itmax

  while ! (solved || inconsistent || tired)
    mul!(q, A, p)
    λ > 0 && @kaxpy!(m, λ, s, q)
    δ = @kdot(n, p, p)   # Faster than dot(p, p)
    λ > 0 && (δ += λ * @kdot(m, s, s))
    α = γ / δ
    @kaxpy!(n,  α, p, x)     # Faster than x = x + α * p
    @kaxpy!(m, -α, q, r)     # Faster than r = r - α * q
    MisI || mul!(z, M, r)
    γ_next = @kdot(m, r, z)  # Faster than γ_next = dot(r, z)
    β = γ_next / γ
    mul!(Aᵀz, Aᵀ, z)
    @kaxpby!(n, one(T), Aᵀz, β, p)  # Faster than p = Aᵀz + β * p
    pNorm = @knrm2(n, p)
    if λ > 0
      @kaxpby!(m, one(T), r, β, s)  # s = r + β * s
    end
    γ = γ_next
    rNorm = sqrt(γ_next)
    history && push!(rNorms, rNorm)
    iter = iter + 1
    display(iter, verbose) && @printf("%5d  %8.2e\n", 1 + 2 * iter, rNorm)
    solved = rNorm ≤ ɛ_c
    inconsistent = (rNorm > 100 * ɛ_c) && (pNorm ≤ ɛ_i)
    tired = iter ≥ itmax
  end
  (verbose > 0) && @printf("\n")

  status = tired ? "maximum number of iterations exceeded" : (inconsistent ? "system probably inconsistent" : "solution good enough given atol and rtol")
  stats = SimpleStats(solved, inconsistent, rNorms, T[], status)
  return (x, stats)
end
