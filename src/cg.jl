# A standard implementation of the Conjugate Gradient method.
# The only non-standard point about it is that it does not check
# that the operator is definite.
# It is possible to check that the system is inconsistent by
# monitoring ‖p‖, which would cost an extra norm computation per
# iteration.
#
# This method is described in
#
# M. R. Hestenes and E. Stiefel. Methods of conjugate gradients for solving linear systems.
# Journal of Research of the National Bureau of Standards, 49(6), pp. 409--436, 1952.
#
# Dominique Orban, <dominique.orban@gerad.ca>
# Salt Lake City, UT, March 2015.

export cg, cg!


"""
    (x, stats) = cg(A, b::AbstractVector{FC};
                    M=I, atol::T=√eps(T), rtol::T=√eps(T), restart::Bool=false,
                    itmax::Int=0, radius::T=zero(T), linesearch::Bool=false,
                    verbose::Int=0, history::Bool=false)

`T` is an `AbstractFloat` such as `Float32`, `Float64` or `BigFloat`.
`FC` is `T` or `Complex{T}`.

The conjugate gradient method to solve the symmetric linear system Ax=b.

The method does _not_ abort if A is not definite.

A preconditioner M may be provided in the form of a linear operator and is
assumed to be symmetric and positive definite.
M also indicates the weighted norm in which residuals are measured.

If `itmax=0`, the default number of iterations is set to `2 * n`,
with `n = length(b)`.

#### Reference

* M. R. Hestenes and E. Stiefel, [*Methods of conjugate gradients for solving linear systems*](https://doi.org/10.6028/jres.049.044), Journal of Research of the National Bureau of Standards, 49(6), pp. 409--436, 1952.
"""
function cg(A, b :: AbstractVector{FC}; kwargs...) where FC <: FloatOrComplex
  solver = CgSolver(A, b)
  cg!(solver, A, b; kwargs...)
  return (solver.x, solver.stats)
end

function cg!(solver :: CgSolver{T,FC,S}, A, b :: AbstractVector{FC};
             M=I, atol :: T=√eps(T), rtol :: T=√eps(T), restart :: Bool=false,
             itmax :: Int=0, radius :: T=zero(T), linesearch :: Bool=false,
             verbose :: Int=0, history :: Bool=false) where {T <: AbstractFloat, FC <: FloatOrComplex{T}, S <: DenseVector{FC}}

  linesearch && (radius > 0) && error("`linesearch` set to `true` but trust-region radius > 0")

  n, m = size(A)
  m == n || error("System must be square")
  length(b) == n || error("Inconsistent problem size")
  (verbose > 0) && @printf("CG: system of %d equations in %d variables\n", n, n)

  # Tests M == Iₙ
  MisI = (M == I)

  # Check type consistency
  eltype(A) == T || error("eltype(A) ≠ $T")
  ktypeof(b) == S || error("ktypeof(b) ≠ $S")
  MisI || (promote_type(eltype(M), T) == T) || error("eltype(M) can't be promoted to $T")

  # Set up workspace.
  allocate_if(!MisI  , solver, :z , S, n)
  allocate_if(restart, solver, :Δx, S, n)
  Δx, x, r, p, Ap, stats = solver.Δx, solver.x, solver.r, solver.p, solver.Ap, solver.stats
  rNorms = stats.residuals
  reset!(stats)
  z = MisI ? r : solver.z

  restart && (Δx .= x)
  x .= zero(T)
  if restart
    mul!(r, A, Δx)
    @kaxpby!(n, one(T), b, -one(T), r)
  else
    r .= b
  end
  MisI || mul!(z, M, r)
  p .= z
  γ = @kdot(n, r, z)
  rNorm = sqrt(γ)
  history && push!(rNorms, rNorm)
  if γ == 0 
    stats.solved, stats.inconsistent = true, false
    stats.status = "x = 0 is a zero-residual solution"
    return solver
  end

  iter = 0
  itmax == 0 && (itmax = 2 * n)

  pAp = zero(T)
  pNorm² = γ
  ε = atol + rtol * rNorm
  (verbose > 0) && @printf("%5s  %7s  %8s  %8s  %8s\n", "k", "‖r‖", "pAp", "α", "σ")
  display(iter, verbose) && @printf("%5d  %7.1e  ", iter, rNorm)

  solved = rNorm ≤ ε
  tired = iter ≥ itmax
  inconsistent = false
  on_boundary = false
  zero_curvature = false

  status = "unknown"

  while !(solved || tired || zero_curvature)
    mul!(Ap, A, p)
    pAp = @kdot(n, p, Ap)
    if (pAp ≤ eps(T) * pNorm²) && (radius == 0)
      if abs(pAp) ≤ eps(T) * pNorm²
        zero_curvature = true
        inconsistent = !linesearch
      end
      if linesearch
        iter == 0 && (x .= b)
        solved = true
      end
    end
    (zero_curvature || solved) && continue

    α = γ / pAp

    # Compute step size to boundary if applicable.
    σ = radius > 0 ? maximum(to_boundary(x, p, radius, dNorm2=pNorm²)) : α

    display(iter, verbose) && @printf("%8.1e  %8.1e  %8.1e\n", pAp, α, σ)

    # Move along p from x to the boundary if either
    # the next step leads outside the trust region or
    # we have nonpositive curvature.
    if (radius > 0) && ((pAp ≤ 0) || (α > σ))
      α = σ
      on_boundary = true
    end

    @kaxpy!(n,  α,  p, x)
    @kaxpy!(n, -α, Ap, r)
    MisI || mul!(z, M, r)
    γ_next = @kdot(n, r, z)
    rNorm = sqrt(γ_next)
    history && push!(rNorms, rNorm)

    solved = (rNorm ≤ ε) || on_boundary

    if !solved
      β = γ_next / γ
      pNorm² = γ_next + β^2 * pNorm²
      γ = γ_next
      @kaxpby!(n, one(T), z, β, p)
    end

    iter = iter + 1
    tired = iter ≥ itmax
    display(iter, verbose) && @printf("%5d  %7.1e  ", iter, rNorm)
  end
  (verbose > 0) && @printf("\n")

  solved && on_boundary && (status = "on trust-region boundary")
  solved && linesearch && (pAp ≤ 0) && (status = "nonpositive curvature detected")
  solved && (status == "unknown") && (status = "solution good enough given atol and rtol")
  zero_curvature && (status = "zero curvature detected")
  tired && (status = "maximum number of iterations exceeded")

  # Update x
  restart && @kaxpy!(n, one(T), Δx, x)

  # Update stats
  stats.solved = solved
  stats.inconsistent = inconsistent
  stats.status = status
  return solver
end
