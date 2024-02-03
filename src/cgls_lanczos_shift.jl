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

#### Input arguments

* `A`: a linear operator that models a Hermitian matrix of dimension n;
* `b`: a vector of length n;
* `shifts`: a vector of length p.

#### Keyword arguments

* `M`: linear operator that models a Hermitian positive-definite matrix of size `n` used for centered preconditioning;
* `ldiv`: define whether the preconditioner uses `ldiv!` or `mul!`;
* `check_curvature`: if `true`, check that the curvature of the quadratic along the search direction is positive, and abort if not, unless `linesearch` is also `true`;
* `atol`: absolute stopping tolerance based on the residual norm;
* `rtol`: relative stopping tolerance based on the residual norm;
* `itmax`: the maximum number of iterations. If `itmax=0`, the default number of iterations is set to `2n`;
* `timemax`: the time limit in seconds;
* `verbose`: additional details can be displayed if verbose mode is enabled (verbose > 0). Information will be displayed every `verbose` iterations;
* `history`: collect additional statistics on the run such as residual norms, or Aᴴ-residual norms;
* `callback`: function or functor called as `callback(solver)` that returns `true` if the Krylov method should terminate, and `false` otherwise;
* `iostream`: stream to which output is logged.

#### Output arguments

* `x`: a vector of p dense vectors, each one of length n;
* `stats`: statistics collected on the run in a [`LanczosShiftStats`](@ref) structure.

#### References
* M. R. Hestenes and E. Stiefel. [*Methods of conjugate gradients for solving linear systems*](https://doi.org/10.6028/jres.049.044), Journal of Research of the National Bureau of Standards, 49(6), pp. 409--436, 1952.
* A. Björck, T. Elfving and Z. Strakos, [*Stability of Conjugate Gradient and Lanczos Methods for Linear Least Squares Problems*](https://doi.org/10.1137/S089547989631202X), SIAM Journal on Matrix Analysis and Applications, 19(3), pp. 720--736, 1998.
"""
function cgls_lanczos_shift end

"""
    solver = cgls_lanczos_shift!(solver::CglsLanczosShiftSolver, A, b; kwargs...)
where `kwargs` are keyword arguments of [`cgls_lanczos_shift`](@ref).
See [`CglsLanczosShiftSolver`](@ref) for more details about the `solver`.
"""
function cgls_lanczos_shift! end

def_args_cg_lanczos_shift = (:(A                        ),
                             :(b::AbstractVector{FC}    ),
                             :(shifts::AbstractVector{T}))

def_kwargs_cg_lanczos_shift = (:(; M = I                        ),
                               :(; ldiv::Bool = false           ),
                               :(; check_curvature::Bool = false),
                               :(; atol::T = √eps(T)            ),
                               :(; rtol::T = √eps(T)            ),
                               :(; itmax::Int = 0               ),
                               :(; timemax::Float64 = Inf       ),
                               :(; verbose::Int = 0             ),
                               :(; history::Bool = false        ),
                               :(; callback = solver -> false   ),
                               :(; iostream::IO = kstdout       ))

def_kwargs_cg_lanczos_shift = mapreduce(extract_parameters, vcat, def_kwargs_cg_lanczos_shift)

args_cg_lanczos_shift = (:A, :b, :shifts)
kwargs_cg_lanczos_shift = (:M, :ldiv, :check_curvature, :atol, :rtol, :itmax, :timemax, :verbose, :history, :callback, :iostream)

solver :: CglsLanczosShiftSolver{T,FC,S}, A, b :: AbstractVector{FC}, shifts :: AbstractVector{T};
        M=I, atol :: T=√eps(T), rtol :: T=√eps(T),
        itmax :: Int=0, verbose :: Int=0, history :: Bool=false,
        callback = solver -> false) where {T <: AbstractFloat, FC <: FloatOrComplex{T}, S <: DenseVector{FC}}

@eval begin
  function cgls_lanczos_shift($(def_args_cg_lanczos_shift...); $(def_kwargs_cg_lanczos_shift...)) where {T <: AbstractFloat, FC <: FloatOrComplex{T}}
    start_time = time_ns()
    nshifts = length(shifts)
    solver = CglsLanczosShiftSolver(A, b, nshifts)
    elapsed_time = ktimer(start_time)
    timemax -= elapsed_time
    cgls_lanczos_shift!(solver, $(args_cg_lanczos_shift...); $(kwargs_cg_lanczos_shift...))
    solver.stats.timer += elapsed_time
    return (solver.x, solver.stats)
  end

  function cgls_lanczos_shift!(solver :: CgLanczosShiftSolver{T,FC,S}, $(def_args_cgls_lanczos_shift...); $(def_kwargs_cgls_lanczos_shift...)) where {T <: AbstractFloat, FC <: FloatOrComplex{T}, S <: AbstractVector{FC}}

    # Timer
    start_time = time_ns()
    timemax_ns = 1e9 * timemax

    m, n = size(A)
    length(b) == m || error("Inconsistent problem size")

    nshifts = length(shifts)
    nshifts == solver.nshifts || error("solver.nshifts = $(solver.nshifts) is inconsistent with length(shifts) = $nshifts")
    (verbose > 0) && @printf(iostream, "CGLS-LANCZOS-SHIFT: system of %d equations in %d variables with %d shifts\n", m, n, nshifts)

    # Tests M = Iₙ
    MisI = (M === I)
    if !MisI
      @warn "Preconditioner not implemented"
    end

    # Check type consistency
    eltype(A) == FC || @warn "eltype(A) ≠ $FC. This could lead to errors or additional allocations in operator-vector products."
    ktypeof(b) <: S || error("ktypeof(b) is not a subtype of $S")

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

    # Initial state.
    ## Distribute x similarly to shifts.
    for i = 1 : nshifts
      x[i] .= zero(FC) # x₀
    end

    u .= b
    u_prev .= zero(T)
    mul!(v, A', u)                      # v₁ ← A' * b
    β = sqrt(@kdotr(n, v, v))           # β₁ = v₁ᵀ M v₁
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
      stats.timer = ktimer(start_time)
      status .= "x = 0 is a zero-residual solution"
      return solver
    end

    # Initialize each p to v.
    for i = 1 : nshifts
      p[i] .= v
    end

    # Initialize Lanczos process.
    # β₁v₁ = b
    @kscal!(n, one(FC) / β, v)          # v₁  ←  v₁ / β₁
    # MisI || @kscal!(n, one(FC) / β, Mv)  # Mv₁ ← Mv₁ / β₁
    # Mv_prev .= Mv
    @kscal!(m, one(FC) / β, u)

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

    # Build format strings for printing.
    (verbose > 0) && (fmt = Printf.Format("%5d" * repeat("  %8.1e", nshifts) * "  %.2fs\n"))
    kdisplay(iter, verbose) && Printf.format(iostream, fmt, iter, rNorms..., ktimer(start_time))

    solved = !reduce(|, not_cv) # ArNorm ≤ ε
    tired = iter ≥ itmax
    status .= "unknown"
    user_requested_exit = false
    overtimed = false

    # Main loop.
    while ! (solved || tired || user_requested_exit || overtimed)

      # Form next Lanczos vector.
      mul!(utilde, A, v)                 # utildeₖ ← Avₖ
      δ = @kdotr(m, utilde, utilde)       # δₖ = vₖᵀAᵀAvₖ
      @kaxpy!(m, -δ, u, utilde)          # uₖ₊₁ = utildeₖ - δₖuₖ - βₖuₖ₋₁
      @kaxpy!(m, -β, u_prev, utilde)
      mul!(v, A', utilde)                # vₖ₊₁ = Aᵀuₖ₊₁
      β = sqrt(@kdotr(n, v, v))           # βₖ₊₁ = vₖ₊₁ᵀ M vₖ₊₁
      @kscal!(n, one(FC) / β, v)            # vₖ₊₁  ←  vₖ₊₁ / βₖ₊₁
      @kscal!(m, one(FC) / β, utilde)       # uₖ₊₁ = uₖ₊₁ / βₖ₊₁
      u_prev .= u
      u .= utilde

      MisI || (ρ = @kdotr(n, v, v))
      for i = 1 : nshifts
        δhat[i] = δ + ρ * shifts[i]
        γ[i] = 1 / (δhat[i] - ω[i] / γ[i])
      end

      # Compute next CG iterate for each shifted system that has not yet converged.
      for i = 1 : nshifts
        not_cv[i] = !converged[i]
        if not_cv[i]
          @kaxpy!(n, γ[i], p[i], x[i])
          ω[i] = β * γ[i]
          σ[i] *= -ω[i]
          ω[i] *= ω[i]
          @kaxpby!(n, σ[i], v, ω[i], p[i])

          # Update list of systems that have not converged.
          rNorms[i] = abs(σ[i])
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
      kdisplay(iter, verbose) && Printf.format(iostream, fmt, iter, rNorms..., ktimer(start_time))

      user_requested_exit = callback(solver) :: Bool
      solved = !reduce(|, not_cv)
      tired = iter ≥ itmax
      timer = time_ns() - start_time
      overtimed = timer > timemax_ns
    end
    (verbose > 0) && @printf(iostream, "\n")

    # Termination status
    overtimed && (status = "time limit exceeded")
    for i = 1 : nshifts
      tired  && (stats.status[i] = "maximum number of iterations exceeded")
      converged[i] && (stats.status[i] = "solution good enough given atol and rtol")
    end
    user_requested_exit && (status .= "user-requested exit")

      # Update stats
    stats.niter = iter
    stats.solved = solved
    stats.timer = ktimer(start_time)
    stats.inconsistent .= false
    return solver
  end
end
