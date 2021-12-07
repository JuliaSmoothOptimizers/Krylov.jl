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
# D. Orban and M. Arioli. Iterative Solution of Symmetric Quasi-Definite Linear Systems,
# Volume 3 of Spotlights. SIAM, Philadelphia, PA, 2017.
#
# D. Orban, The Projected Golub-Kahan Process for Constrained
# Linear Least-Squares Problems. Cahier du GERAD G-2014-15,
# GERAD, Montreal QC, Canada, 2014.
#
# Dominique Orban, <dominique.orban@gerad.ca>
# Montreal, QC, April 2015.

export crmr, crmr!


"""
    (x, stats) = crmr(A, b::AbstractVector{FC};
                      M=I, λ::T=zero(T), atol::T=√eps(T),
                      rtol::T=√eps(T), itmax::Int=0, verbose::Int=0, history::Bool=false)

`T` is an `AbstractFloat` such as `Float32`, `Float64` or `BigFloat`.
`FC` is `T` or `Complex{T}`.

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

CRMR produces monotonic residuals ‖r‖₂.
It is formally equivalent to CRAIG-MR, though can be slightly less accurate,
but simpler to implement. Only the x-part of the solution is returned.

A preconditioner M may be provided in the form of a linear operator.

#### References

* D. Orban and M. Arioli, [*Iterative Solution of Symmetric Quasi-Definite Linear Systems*](https://doi.org/10.1137/1.9781611974737), Volume 3 of Spotlights. SIAM, Philadelphia, PA, 2017.
* D. Orban, [*The Projected Golub-Kahan Process for Constrained Linear Least-Squares Problems*](https://dx.doi.org/10.13140/RG.2.2.17443.99360). Cahier du GERAD G-2014-15, 2014.
"""
function crmr(A, b :: AbstractVector{FC}; kwargs...) where FC <: FloatOrComplex
  solver = CrmrSolver(A, b)
  crmr!(solver, A, b; kwargs...)
  return (solver.x, solver.stats)
end

function crmr!(solver :: CrmrSolver{T,FC,S}, A, b :: AbstractVector{FC};
               M=I, λ :: T=zero(T), atol :: T=√eps(T),
               rtol :: T=√eps(T), itmax :: Int=0, verbose :: Int=0, history :: Bool=false) where {T <: AbstractFloat, FC <: FloatOrComplex{T}, S <: DenseVector{FC}}

  m, n = size(A)
  length(b) == m || error("Inconsistent problem size")
  (verbose > 0) && @printf("CRMR: system of %d equations in %d variables\n", m, n)

  # Tests M == Iₙ
  MisI = (M == I)

  # Check type consistency
  eltype(A) == T || error("eltype(A) ≠ $T")
  ktypeof(b) == S || error("ktypeof(b) ≠ $S")
  MisI || (promote_type(eltype(M), T) == T) || error("eltype(M) can't be promoted to $T")

  # Compute the adjoint of A
  Aᵀ = A'

  # Set up workspace.
  allocate_if(!MisI, solver, :Mq, S, m)
  allocate_if(λ > 0, solver, :s , S, m)
  x, p, Aᵀr, r = solver.x, solver.p, solver.Aᵀr, solver.r
  q, s, stats = solver.q, solver.s, solver.stats
  rNorms, ArNorms = stats.residuals, stats.Aresiduals
  reset!(stats)
  Mq = MisI ? q : solver.Mq

  x .= zero(T)  # initial estimation x = 0
  mul!(r, M, b) # initial residual r = M * (b - Ax) = M * b
  bNorm = @knrm2(m, r)  # norm(b - A * x0) if x0 ≠ 0.
  rNorm = bNorm  # + λ * ‖x0‖ if x0 ≠ 0 and λ > 0.
  history && push!(rNorms, rNorm)
  if bNorm == 0
    stats.solved, stats.inconsistent = true, false
    stats.status = "x = 0 is a zero-residual solution"
    history && push!(ArNorms, zero(T))
    return solver
  end
  λ > 0 && (s .= r)
  mul!(Aᵀr, Aᵀ, r)  # - λ * x0 if x0 ≠ 0.
  p .= Aᵀr
  γ = @kdot(n, Aᵀr, Aᵀr)  # Faster than γ = dot(Aᵀr, Aᵀr)
  λ > 0 && (γ += λ * rNorm * rNorm)
  iter = 0
  itmax == 0 && (itmax = m + n)

  ArNorm = sqrt(γ)
  history && push!(ArNorms, ArNorm)
  ɛ_c = atol + rtol * rNorm  # Stopping tolerance for consistent systems.
  ɛ_i = atol + rtol * ArNorm  # Stopping tolerance for inconsistent systems.
  (verbose > 0) && @printf("%5s  %8s  %8s\n", "Aprod", "‖Aᵀr‖", "‖r‖")
  display(iter, verbose) && @printf("%5d  %8.2e  %8.2e\n", 1, ArNorm, rNorm)

  status = "unknown"
  solved = rNorm ≤ ɛ_c
  inconsistent = (rNorm > 100 * ɛ_c) && (ArNorm ≤ ɛ_i)
  tired = iter ≥ itmax

  while ! (solved || inconsistent || tired)
    mul!(q, A, p)
    λ > 0 && @kaxpy!(m, λ, s, q)  # q = q + λ * s
    MisI || mul!(Mq, M, q)
    α = γ / @kdot(m, q, Mq)    # Compute qᵗ * M * q
    @kaxpy!(n,  α, p, x)       # Faster than  x =  x + α *  p
    @kaxpy!(m, -α, Mq, r)      # Faster than  r =  r - α * Mq
    rNorm = @knrm2(m, r)       # norm(r)
    mul!(Aᵀr, Aᵀ, r)
    γ_next = @kdot(n, Aᵀr, Aᵀr)  # Faster than γ_next = dot(Aᵀr, Aᵀr)
    λ > 0 && (γ_next += λ * rNorm * rNorm)
    β = γ_next / γ

    @kaxpby!(n, one(T), Aᵀr, β, p)  # Faster than  p = Aᵀr + β * p
    if λ > 0
      @kaxpby!(m, one(T), r, β, s) # s = r + β * s
    end

    γ = γ_next
    ArNorm = sqrt(γ)
    history && push!(rNorms, rNorm)
    history && push!(ArNorms, ArNorm)
    iter = iter + 1
    display(iter, verbose) && @printf("%5d  %8.2e  %8.2e\n", 1 + 2 * iter, ArNorm, rNorm)
    solved = rNorm ≤ ɛ_c
    inconsistent = (rNorm > 100 * ɛ_c) && (ArNorm ≤ ɛ_i)
    tired = iter ≥ itmax
  end
  (verbose > 0) && @printf("\n")

  status = tired ? "maximum number of iterations exceeded" : (inconsistent ? "system probably inconsistent but least squares/norm solution found" : "solution good enough given atol and rtol")

  # Update stats
  stats.solved = solved
  stats.inconsistent = inconsistent
  stats.status = status
  return solver
end
