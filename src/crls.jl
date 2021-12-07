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
# Ph.D. Thesis, Stanford University, 2011.
#
# with the difference that it also recurs r = b - Ax.
#
# Dominique Orban, <dominique.orban@gerad.ca>
# Princeton, NJ, March 2015.

export crls, crls!


"""
    (x, stats) = crls(A, b::AbstractVector{FC};
                      M=I, λ::T=zero(T), atol::T=√eps(T), rtol::T=√eps(T),
                      radius::T=zero(T), itmax::Int=0, verbose::Int=0, history::Bool=false)

`T` is an `AbstractFloat` such as `Float32`, `Float64` or `BigFloat`.
`FC` is `T` or `Complex{T}`.

Solve the linear least-squares problem

    minimize ‖b - Ax‖₂² + λ‖x‖₂²

using the Conjugate Residuals (CR) method. This method is equivalent to
applying MINRES to the normal equations

    (AᵀA + λI) x = Aᵀb.

This implementation recurs the residual r := b - Ax.

CRLS produces monotonic residuals ‖r‖₂ and optimality residuals ‖Aᵀr‖₂.
It is formally equivalent to LSMR, though can be substantially less accurate,
but simpler to implement.

#### Reference

* D. C.-L. Fong, *Minimum-Residual Methods for Sparse, Least-Squares using Golubg-Kahan Bidiagonalization*, Ph.D. Thesis, Stanford University, 2011.
"""
function crls(A, b :: AbstractVector{FC}; kwargs...) where FC <: FloatOrComplex
  solver = CrlsSolver(A, b)
  crls!(solver, A, b; kwargs...)
  return (solver.x, solver.stats)
end

function crls!(solver :: CrlsSolver{T,FC,S}, A, b :: AbstractVector{FC};
               M=I, λ :: T=zero(T), atol :: T=√eps(T), rtol :: T=√eps(T),
               radius :: T=zero(T), itmax :: Int=0, verbose :: Int=0, history :: Bool=false) where {T <: AbstractFloat, FC <: FloatOrComplex{T}, S <: DenseVector{FC}}

  m, n = size(A)
  length(b) == m || error("Inconsistent problem size")
  (verbose > 0) && @printf("CRLS: system of %d equations in %d variables\n", m, n)

  # Tests M == Iₙ
  MisI = (M == I)

  # Check type consistency
  eltype(A) == T || error("eltype(A) ≠ $T")
  ktypeof(b) == S || error("ktypeof(b) ≠ $S")
  MisI || (promote_type(eltype(M), T) == T) || error("eltype(M) can't be promoted to $T")

  # Compute the adjoint of A
  Aᵀ = A'

  # Set up workspace.
  allocate_if(!MisI, solver, :Ms, S, m)
  x, p, Ar, q = solver.x, solver.p, solver.Ar, solver.q
  r, Ap, s, stats = solver.r, solver.Ap, solver.s, solver.stats
  rNorms, ArNorms = stats.residuals, stats.Aresiduals
  reset!(stats)
  Ms  = MisI ? s  : solver.Ms
  Mr  = MisI ? r  : solver.Ms
  MAp = MisI ? Ap : solver.Ms

  x .= zero(T)
  r .= b
  bNorm = @knrm2(m, r)  # norm(b - A * x0) if x0 ≠ 0.
  rNorm = bNorm  # + λ * ‖x0‖ if x0 ≠ 0 and λ > 0.
  history && push!(rNorms, rNorm)
  if bNorm == 0
    stats.solved, stats.inconsistent = true, false
    stats.status = "x = 0 is a zero-residual solution"
    history && push!(ArNorms, zero(T))
    return solver
  end

  MisI || mul!(Mr, M, r)
  mul!(Ar, Aᵀ, Mr)  # - λ * x0 if x0 ≠ 0.
  mul!(s, A, Ar)
  MisI || mul!(Ms, M, s)

  p  .= Ar
  Ap .= s
  mul!(q, Aᵀ, Ms)  # Ap
  λ > 0 && @kaxpy!(n, λ, p, q)  # q = q + λ * p
  γ  = @kdot(m, s, Ms)  # Faster than γ = dot(s, Ms)
  iter = 0
  itmax == 0 && (itmax = m + n)

  ArNorm = @knrm2(n, Ar)  # Marginally faster than norm(Ar)
  λ > 0 && (γ += λ * ArNorm * ArNorm)
  history && push!(ArNorms, ArNorm)
  ε = atol + rtol * ArNorm
  (verbose > 0) && @printf("%5s  %8s  %8s\n", "Aprod", "‖Aᵀr‖", "‖r‖")
  display(iter, verbose) && @printf("%5d  %8.2e  %8.2e\n", 3, ArNorm, rNorm)

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
        mul!(q, Aᵀ, s)
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
    mul!(s, A, Ar)
    MisI || mul!(Ms, M, s)
    γ_next = @kdot(m, s, Ms)   # Faster than γ_next = dot(s, s)
    λ > 0 && (γ_next += λ * ArNorm * ArNorm)
    β = γ_next / γ

    @kaxpby!(n, one(T), Ar, β, p)    # Faster than  p = Ar + β *  p
    @kaxpby!(m, one(T), s, β, Ap)    # Faster than Ap =  s + β * Ap
    MisI || mul!(MAp, M, Ap)
    mul!(q, Aᵀ, MAp)
    λ > 0 && @kaxpy!(n, λ, p, q)  # q = q + λ * p

    γ = γ_next
    if λ > 0
      rNorm = sqrt(@kdot(m, r, r) + λ * @kdot(n, x, x))
    else
      rNorm = @knrm2(m, r)  # norm(r)
    end
    #     ArNorm = norm(Ar)
    history && push!(rNorms, rNorm)
    history && push!(ArNorms, ArNorm)
    iter = iter + 1
    display(iter, verbose) && @printf("%5d  %8.2e  %8.2e\n", 3 + 2 * iter, ArNorm, rNorm)
    solved = (ArNorm ≤ ε) || on_boundary
    tired = iter ≥ itmax
  end
  (verbose > 0) && @printf("\n")

  status = psd ? "zero-curvature encountered" : (on_boundary ? "on trust-region boundary" : (tired ? "maximum number of iterations exceeded" : "solution good enough given atol and rtol"))

  # Update stats
  stats.solved = solved
  stats.inconsistent = false
  stats.status = status
  return solver
end
