# An implementation of the Lanczos version of the conjugate gradient method
# for a family of shifted systems of the form (AᵀA + λI) x = b.
#
# The implementation follows
# A. Frommer and P. Maass, Fast CG-Based Methods for Tikhonov-Phillips Regularization,
# SIAM Journal on Scientific Computing, 20(5), pp. 1831--1850, 1999.
#
# Tangi Migot, <tangi.migot@polymtl.ca>
# Montreal, July 2022.

export cgls_lanczos_shift, cgls_lanczos_shift!


"""
    (x, stats) = cgls_lanczos_shift(A, b::AbstractVector{FC};
                      M=I, λ::T=zero(T), atol::T=√eps(T), rtol::T=√eps(T),
                      radius::T=zero(T), itmax::Int=0, verbose::Int=0, history::Bool=false,
                      callback=solver->false)

`T` is an `AbstractFloat` such as `Float32`, `Float64` or `BigFloat`.
`FC` is `T` or `Complex{T}`.

Solve the regularized linear least-squares problem

    minimize ‖b - Ax‖₂² + λ‖x‖₂²

using the Conjugate Gradient (CG) method, where λ ≥ 0 is a regularization
parameter. This method is equivalent to applying CG to the normal equations

    (AᵀA + λI) x = Aᵀb

but is more stable.

CGLS produces monotonic residuals ‖r‖₂ but not optimality residuals ‖Aᵀr‖₂.
It is formally equivalent to LSQR, though can be slightly less accurate,
but simpler to implement.

The callback is called as `callback(solver)` and should return `true` if the main loop should terminate,
and `false` otherwise.

#### References

* M. R. Hestenes and E. Stiefel. [*Methods of conjugate gradients for solving linear systems*](https://doi.org/10.6028/jres.049.044), Journal of Research of the National Bureau of Standards, 49(6), pp. 409--436, 1952.
* A. Björck, T. Elfving and Z. Strakos, [*Stability of Conjugate Gradient and Lanczos Methods for Linear Least Squares Problems*](https://doi.org/10.1137/S089547989631202X), SIAM Journal on Matrix Analysis and Applications, 19(3), pp. 720--736, 1998.
"""
function cgls_lanczos_shift end

function cgls_lanczos_shift(A, b :: AbstractVector{FC}, shifts :: AbstractVector{T}; kwargs...) where {T <: AbstractFloat, FC <: FloatOrComplex{T}}
  nshifts = length(shifts)
  solver = CglsLanczosShiftSolver(A, b, nshifts)
  cgls_lanczos_shift!(solver, A, b, shifts; kwargs...)
  return (solver.x, solver.stats)
end

"""
    solver = cgls_lanczos_shift!(solver::CglsLanczosShiftSolver, A, b; kwargs...)

where `kwargs` are keyword arguments of [`cgls_lanczos_shift`](@ref).

See [`CglsLanczosShiftSolver`](@ref) for more details about the `solver`.
"""
function cgls_lanczos_shift! end

function cgls_lanczos_shift!(solver :: CglsLanczosShiftSolver{T,FC,S}, A, b :: AbstractVector{FC}, shifts :: AbstractVector{T};
               M=I, atol :: T=√eps(T), rtol :: T=√eps(T),
               itmax :: Int=0, verbose :: Int=0, history :: Bool=false,
               callback = solver -> false) where {T <: AbstractFloat, FC <: FloatOrComplex{T}, S <: DenseVector{FC}}

  m, n = size(A)
  length(b) == m || error("Inconsistent problem size")

  nshifts = length(shifts)
  (verbose > 0) && @printf("CGLS Shifts: system of %d equations in %d variables with %d shifts\n", m, n, nshifts)


  # Tests M = Iₙ
  MisI = (M === I)
  if !MisI
    @warn "Preconditioner not implemented"
  end

  # Check type consistency
  eltype(A) == FC || error("eltype(A) ≠ $FC")
  ktypeof(b) == S || error("ktypeof(b) ≠ $S")

  # Compute the adjoint of A
  Aᵀ = A'

  # Set up workspace.
  allocate_if(!MisI, solver, :v, S, n)
  u_prev, utilde = solver.Mv_prev, solver.Mv_next
  u = solver.u
  x, p, σ, δhat = solver.x, solver.p, solver.σ, solver.δhat
  ω, γ, rNorms, converged = solver.ω, solver.γ, solver.rNorms, solver.converged
  not_cv, stats = solver.not_cv, solver.stats
  rNorms_history, indefinite, status = stats.residuals, stats.indefinite, stats.status
  reset!(stats)
  v = solver.v # v = MisI ? Mv : solver.v

  for i = 1 : nshifts
    x[i] .= zero(FC)                       # x₀
  end

  u .= b
  u_prev .= zero(T)
  mul!(v, A', u)                          # v₁ ← A' * b
  β = sqrt(@kdotr(m, v, v))                # β₁ = v₁ᵀ M v₁

  rNorms .= β
  if history
    for i = 1 : nshifts
      push!(rNorms_history[i], rNorms[i])
    end
  end

  # Keep track of shifted systems with negative curvature if required.
  indefinite .= false

  if β == 0
    stats.niter = 0
    stats.solved = true
    status .= "x = 0 is a zero-residual solution"
    return solver
  end

  # Initialize each p to v.
  for i = 1 : nshifts
    p[i] .= v
  end

  # Initialize Lanczos process.
  # β₁v₁ = b
  @kscal!(m, one(FC) / β, v)          # v₁  ←  v₁ / β₁
  # MisI || @kscal!(n, one(FC) / β, Mv)  # Mv₁ ← Mv₁ / β₁
  # Mv_prev .= Mv
  @kscal!(n, one(FC) / β, u)

  # Initialize some constants used in recursions below.
  ρ = one(T)
  σ .= β
  δhat .= zero(T)
  ω .= zero(T)
  γ .= one(T)

  # Define stopping tolerance.
  ε = atol + rtol * β

  # Keep track of shifted systems that have converged.
  for i = 1 : nshifts
    converged[i] = rNorms[i] ≤ ε
    not_cv[i] = !converged[i]
  end
  iter = 0
  itmax == 0 && (itmax = 2 * max(m, n))

  #(verbose > 0) && @printf("%5s  %8s  %8s\n", "k", "‖Aᵀr‖", "‖r‖") REMOVE
  #kdisplay(iter, verbose) && @printf("%5d  %8.2e  %8.2e\n", iter, ArNorm, rNorm) REMOVE
  # Build format strings for printing.
  if kdisplay(iter, verbose)
    fmt = "%5d" * repeat("  %8.1e", nshifts) * "\n"
    # precompile printf for our particular format
    local_printf(data...) = Core.eval(Main, :(@printf($fmt, $(data)...)))
    local_printf(iter, rNorms...)
  end

  solved = sum(not_cv) == 0 # ArNorm ≤ ε
  tired = iter ≥ itmax
  status .= "unknown"
  user_requested_exit = false

  # Main loop.
  while ! (solved || tired)

    # Form next Lanczos vector.
    mul!(utilde, A, v)                 # utildeₖ ← Avₖ
    δ = @kdotr(n, utilde, utilde)       # δₖ = vₖᵀAᵀAvₖ
    @kaxpy!(n, -δ, u, utilde)          # uₖ₊₁ = utildeₖ - δₖuₖ - βₖuₖ₋₁
    @kaxpy!(n, -β, u_prev, utilde)
    mul!(v, A', utilde)                # vₖ₊₁ = Aᵀuₖ₊₁
    β = sqrt(@kdotr(m, v, v))           # βₖ₊₁ = vₖ₊₁ᵀ M vₖ₊₁
    @kscal!(m, one(FC) / β, v)            # vₖ₊₁  ←  vₖ₊₁ / βₖ₊₁
    @kscal!(n, one(FC) / β, utilde)       # uₖ₊₁ = uₖ₊₁ / βₖ₊₁
    u_prev .= u
    u .= utilde

    MisI || (ρ = @kdotr(m, v, v))
    for i = 1 : nshifts
      δhat[i] = δ + ρ * shifts[i]
      γ[i] = 1 / (δhat[i] - ω[i] / γ[i])
    end

    # Compute next CG iterate for each shifted system that has not yet converged.
    for i = 1 : nshifts
      not_cv[i] = !converged[i]
      if not_cv[i]
        @kaxpy!(m, γ[i], p[i], x[i])
        ω[i] = β * γ[i]
        σ[i] *= -ω[i]
        ω[i] *= ω[i]
        @kaxpby!(m, σ[i], v, ω[i], p[i])

        # Update list of systems that have not converged.
        rNorms[i] = abs(σ[i])
        ##############################################################
        #rtheo = A' * M * A * x[i] + shifts[i] * x[i] - A' * M * b
        #@show i, iter, rNorms[i], norm(rtheo), norm(x), norm(v)
        converged[i] = rNorms[i] ≤ ε
      end
    end

    if length(not_cv) > 0 && history
      for i = 1 : nshifts
        not_cv[i] && push!(rNorms_history[i], rNorms[i])
      end
    end

    # Is there a better way than to update this array twice per iteration?
    for i = 1 : nshifts
      not_cv[i] = !converged[i]
    end
    iter = iter + 1
    kdisplay(iter, verbose) && local_printf(iter, rNorms...)

    user_requested_exit = callback(solver) :: Bool
    solved = sum(not_cv) == 0
    tired = iter ≥ itmax
  end
  (verbose > 0) && @printf("\n")

  for i = 1 : nshifts
    tired  && (stats.status[i] = "maximum number of iterations exceeded")
    converged[i] && (stats.status[i] = "solution good enough given atol and rtol")
  end
  user_requested_exit && (status .= "user-requested exit")


  # Update stats
  stats.niter = iter
  stats.solved = solved
  stats.inconsistent .= false
  return solver
end
