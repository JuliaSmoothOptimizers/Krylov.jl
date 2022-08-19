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
    (x, stats) = cgne(A, b::AbstractVector{FC};
                      N=I, λ::T=zero(T), atol::T=√eps(T), rtol::T=√eps(T),
                      itmax::Int=0, verbose::Int=0, history::Bool=false,
                      ldiv::Bool=false, callback=solver->false)

`T` is an `AbstractFloat` such as `Float32`, `Float64` or `BigFloat`.
`FC` is `T` or `Complex{T}`.

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

A preconditioner N may be provided in the form of a linear operator.

The callback is called as `callback(solver)` and should return `true` if the main loop should terminate,
and `false` otherwise.

#### References

* J. E. Craig, [*The N-step iteration procedures*](https://doi.org/10.1002/sapm195534164), Journal of Mathematics and Physics, 34(1), pp. 64--73, 1955.
* J. E. Craig, *Iterations Procedures for Simultaneous Equations*, Ph.D. Thesis, Department of Electrical Engineering, MIT, 1954.
"""
function cgne end

function cgne(A, b :: AbstractVector{FC}; kwargs...) where FC <: FloatOrComplex
  solver = CgneSolver(A, b)
  cgne!(solver, A, b; kwargs...)
  return (solver.x, solver.stats)
end

"""
    solver = cgne!(solver::CgneSolver, A, b; kwargs...)

where `kwargs` are keyword arguments of [`cgne`](@ref).

See [`CgneSolver`](@ref) for more details about the `solver`.
"""
function cgne! end

function cgne!(solver :: CgneSolver{T,FC,S}, A, b :: AbstractVector{FC};
               N=I, λ :: T=zero(T), atol :: T=√eps(T), rtol :: T=√eps(T),
               itmax :: Int=0, verbose :: Int=0, history :: Bool=false,
               ldiv :: Bool=false, callback = solver -> false) where {T <: AbstractFloat, FC <: FloatOrComplex{T}, S <: DenseVector{FC}}

  m, n = size(A)
  length(b) == m || error("Inconsistent problem size")
  (verbose > 0) && @printf("CGNE: system of %d equations in %d variables\n", m, n)

  # Tests N = Iₙ
  NisI = (N === I)

  # Check type consistency
  eltype(A) == FC || error("eltype(A) ≠ $FC")
  ktypeof(b) == S || error("ktypeof(b) ≠ $S")

  # Compute the adjoint of A
  Aᵀ = A'

  # Set up workspace.
  allocate_if(!NisI, solver, :z, S, m)
  allocate_if(λ > 0, solver, :s, S, m)
  x, p, Aᵀz, r, q, s, stats = solver.x, solver.p, solver.Aᵀz, solver.r, solver.q, solver.s, solver.stats
  rNorms = stats.residuals
  reset!(stats)
  z = NisI ? r : solver.z

  x .= zero(FC)
  r .= b
  NisI || mulorldiv!(z, N, r, ldiv)
  rNorm = @knrm2(m, r)   # Marginally faster than norm(r)
  history && push!(rNorms, rNorm)
  if rNorm == 0
    stats.niter = 0
    stats.solved, stats.inconsistent = true, false
    stats.status = "x = 0 is a zero-residual solution"
    return solver
  end
  λ > 0 && (s .= r)
  mul!(p, Aᵀ, z)

  # Use ‖p‖ to detect inconsistent system.
  # An inconsistent system will necessarily have AA' singular.
  # Because CGNE is equivalent to CG applied to AA'y = b, there will be a
  # conjugate direction u such that u'AA'u = 0, i.e., A'u = 0. In this
  # implementation, p is a substitute for A'u.
  pNorm = @knrm2(n, p)

  γ = @kdotr(m, r, z)  # Faster than γ = dot(r, z)
  iter = 0
  itmax == 0 && (itmax = m + n)

  ɛ_c = atol + rtol * rNorm  # Stopping tolerance for consistent systems.
  ɛ_i = atol + rtol * pNorm  # Stopping tolerance for inconsistent systems.
  (verbose > 0) && @printf("%5s  %8s\n", "k", "‖r‖")
  kdisplay(iter, verbose) && @printf("%5d  %8.2e\n", iter, rNorm)

  status = "unknown"
  solved = rNorm ≤ ɛ_c
  inconsistent = (rNorm > 100 * ɛ_c) && (pNorm ≤ ɛ_i)
  tired = iter ≥ itmax
  user_requested_exit = false

  while ! (solved || inconsistent || tired || user_requested_exit)
    mul!(q, A, p)
    λ > 0 && @kaxpy!(m, λ, s, q)
    δ = @kdotr(n, p, p)   # Faster than dot(p, p)
    λ > 0 && (δ += λ * @kdotr(m, s, s))
    α = γ / δ
    @kaxpy!(n,  α, p, x)     # Faster than x = x + α * p
    @kaxpy!(m, -α, q, r)     # Faster than r = r - α * q
    NisI || mulorldiv!(z, N, r, ldiv)
    γ_next = @kdotr(m, r, z)  # Faster than γ_next = dot(r, z)
    β = γ_next / γ
    mul!(Aᵀz, Aᵀ, z)
    @kaxpby!(n, one(FC), Aᵀz, β, p)  # Faster than p = Aᵀz + β * p
    pNorm = @knrm2(n, p)
    if λ > 0
      @kaxpby!(m, one(FC), r, β, s)  # s = r + β * s
    end
    γ = γ_next
    rNorm = sqrt(γ_next)
    history && push!(rNorms, rNorm)
    iter = iter + 1
    kdisplay(iter, verbose) && @printf("%5d  %8.2e\n", iter, rNorm)

    # Stopping conditions that do not depend on user input.
    # This is to guard against tolerances that are unreasonably small.
    resid_decrease_mach = (rNorm + one(T) ≤ one(T))

    user_requested_exit = callback(solver) :: Bool
    resid_decrease_lim = rNorm ≤ ɛ_c
    solved = resid_decrease_lim || resid_decrease_mach
    inconsistent = (rNorm > 100 * ɛ_c) && (pNorm ≤ ɛ_i)
    tired = iter ≥ itmax
  end
  (verbose > 0) && @printf("\n")

  tired               && (status = "maximum number of iterations exceeded")
  inconsistent        && (status = "system probably inconsistent")
  solved              && (status = "solution good enough given atol and rtol")
  user_requested_exit && (status = "user-requested exit")

  # Update stats
  stats.niter = iter
  stats.solved = solved
  stats.inconsistent = inconsistent
  stats.status = status
  return solver
end
