# An implementation of the Lanczos version of the conjugate gradient method
# for a family of shifted systems of the form (AᴴA + λI) x = Aᴴb.
#
# The implementation follows
# A. Frommer and P. Maass, Fast CG-Based Methods for Tikhonov-Phillips Regularization,
# SIAM Journal on Scientific Computing, 20(5), pp. 1831--1850, 1999.
#
# Tangi Migot, <tangi.migot@polymtl.ca>
# Montreal, July 2022.

export cgls_lanczos_shift, cgls_lanczos_shift!


"""
    (x, stats) = cgls_lanczos_shift(A, b::AbstractVector{FC}, shifts::AbstractVector{T};
                                    M=I, λ::T=zero(T), atol::T=√eps(T), rtol::T=√eps(T),
                                    radius::T=zero(T), itmax::Int=0, verbose::Int=0,
                                    history::Bool=false, callback=workspace->false,
                                    iostream::IO=kstdout)

`T` is an `AbstractFloat` such as `Float32`, `Float64` or `BigFloat`.
`FC` is `T` or `Complex{T}`.

Solve the regularized linear least-squares problem

    minimize ‖b - Ax‖₂² + λ‖x‖₂²

using the Conjugate Gradient (CG) method, where λ ≥ 0 is a regularization
parameter. This method is equivalent to applying CG to the normal equations

    (AᴴA + λI) x = Aᴴb

but is more stable.

CGLS produces monotonic residuals ‖r‖₂ but not optimality residuals ‖Aᴴr‖₂.
It is formally equivalent to LSQR, though can be slightly less accurate,
but simpler to implement.

#### Interface

To easily switch between Krylov methods, use the generic interface [`krylov_solve`](@ref) with `method = :cgls_lanczos_shift`.

For an in-place variant that reuses memory across solves, see [`cgls_lanczos_shift!`](@ref).

#### Input arguments

* `A`: a linear operator that models a matrix of dimension `m × n`;
* `b`: a vector of length `m`;
* `shifts`: a vector of length `p`.

#### Keyword arguments

* `M`: linear operator that models a Hermitian positive-definite matrix of size `n` used for preconditioning;
* `ldiv`: define whether the preconditioner uses `ldiv!` or `mul!`;
* `atol`: absolute stopping tolerance based on the residual norm;
* `rtol`: relative stopping tolerance based on the residual norm;
* `itmax`: the maximum number of iterations. If `itmax=0`, the default number of iterations is set to `m+n`;
* `timemax`: the time limit in seconds;
* `verbose`: additional details can be displayed if verbose mode is enabled (verbose > 0). Information will be displayed every `verbose` iterations;
* `history`: collect additional statistics on the run such as residual norms, or Aᴴ-residual norms;
* `callback`: function or functor called as `callback(workspace)` that returns `true` if the Krylov method should terminate, and `false` otherwise;
* `iostream`: stream to which output is logged.

#### Output arguments

* `x`: a vector of `p` dense vectors, each one of length `n`;
* `stats`: statistics collected on the run in a [`LanczosShiftStats`](@ref) structure.

#### References
* M. R. Hestenes and E. Stiefel. [*Methods of conjugate gradients for solving linear systems*](https://doi.org/10.6028/jres.049.044), Journal of Research of the National Bureau of Standards, 49(6), pp. 409--436, 1952.
* A. Björck, T. Elfving and Z. Strakos, [*Stability of Conjugate Gradient and Lanczos Methods for Linear Least Squares Problems*](https://doi.org/10.1137/S089547989631202X), SIAM Journal on Matrix Analysis and Applications, 19(3), pp. 720--736, 1998.
"""
function cgls_lanczos_shift end

"""
    workspace = cgls_lanczos_shift!(workspace::CglsLanczosShiftWorkspace, A, b, shifts; kwargs...)

In this call, `kwargs` are keyword arguments of [`cgls_lanczos_shift`](@ref).

See [`CglsLanczosShiftWorkspace`](@ref) for instructions on how to create the `workspace`.

For a more generic interface, you can use [`krylov_workspace`](@ref) with `method = :cgls_lanczos_shift` to allocate the workspace,
and [`krylov_solve!`](@ref) to run the Krylov method in-place.
"""
function cgls_lanczos_shift! end

def_args_cgls_lanczos_shift = (:(A                        ),
                               :(b::AbstractVector{FC}    ),
                               :(shifts::AbstractVector{T}))

def_kwargs_cgls_lanczos_shift = (:(; M = I                        ),
                                 :(; ldiv::Bool = false           ),
                                 :(; atol::T = √eps(T)            ),
                                 :(; rtol::T = √eps(T)            ),
                                 :(; itmax::Int = 0               ),
                                 :(; timemax::Float64 = Inf       ),
                                 :(; verbose::Int = 0             ),
                                 :(; history::Bool = false        ),
                                 :(; callback = workspace -> false),
                                 :(; iostream::IO = kstdout       ))

def_kwargs_cgls_lanczos_shift = extract_parameters.(def_kwargs_cgls_lanczos_shift)

args_cgls_lanczos_shift = (:A, :b, :shifts)
kwargs_cgls_lanczos_shift = (:M, :ldiv, :atol, :rtol, :itmax, :timemax, :verbose, :history, :callback, :iostream)

@eval begin
  function cgls_lanczos_shift!(workspace :: CglsLanczosShiftWorkspace{T,FC,S}, $(def_args_cgls_lanczos_shift...); $(def_kwargs_cgls_lanczos_shift...)) where {T <: AbstractFloat, FC <: FloatOrComplex{T}, S <: AbstractVector{FC}}

    # Timer
    start_time = time_ns()
    timemax_ns = 1e9 * timemax

    m, n = size(A)
    (m == workspace.m && n == workspace.n) || error("(workspace.m, workspace.n) = ($(workspace.m), $(workspace.n)) is inconsistent with size(A) = ($m, $n)")
    length(b) == m || error("Inconsistent problem size")

    nshifts = length(shifts)
    nshifts == workspace.nshifts || error("workspace.nshifts = $(workspace.nshifts) is inconsistent with length(shifts) = $nshifts")
    (verbose > 0) && @printf(iostream, "CGLS-LANCZOS-SHIFT: system of %d equations in %d variables with %d shifts\n", m, n, nshifts)

    # Tests M = Iₙ
    MisI = (M === I)
    !MisI && error("Preconditioner `M` is not supported.")

    # Check type consistency
    eltype(A) == FC || @warn "eltype(A) ≠ $FC. This could lead to errors or additional allocations in operator-vector products."
    ktypeof(b) == S || error("ktypeof(b) must be equal to $S")

    # Compute the adjoint of A
    Aᴴ = A'

    # Set up workspace.
    allocate_if(!MisI, workspace, :v, S, workspace.Mv)  # The length of v is n
    v, u_prev, u, u_next = workspace.Mv, workspace.u_prev, workspace.u, workspace.u_next
    x, p, σ, δhat = workspace.x, workspace.p, workspace.σ, workspace.δhat
    ω, γ, rNorms, converged = workspace.ω, workspace.γ, workspace.rNorms, workspace.converged
    not_cv, stats = workspace.not_cv, workspace.stats
    rNorms_history, status = stats.residuals, stats.status
    reset!(stats)

    # Initial state.
    ## Distribute x similarly to shifts.
    for i = 1 : nshifts
      kfill!(x[i], zero(FC))  # x₀
    end

    kcopy!(m, u, b)              # u ← b
    kfill!(u_prev, zero(FC))
    kmul!(v, Aᴴ, u)              # v₁ ← Aᴴ * b
    β = knorm_elliptic(n, v, v)  # β₁ = v₁ᵀ M v₁
    kfill!(rNorms, β)
    if history
      for i = 1 : nshifts
        push!(rNorms_history[i], rNorms[i])
      end
    end

    if β == 0
      stats.niter = 0
      stats.solved = true
      stats.timer = start_time |> ktimer
      stats.status = "x is a zero-residual solution"
      return workspace
    end

    # Initialize each p to v.
    for i = 1 : nshifts
      kcopy!(n, p[i], v)  # pᵢ ← v
    end

    # Initialize Lanczos process.
    # β₁v₁ = b
    kdiv!(n, v, β)  # v₁ ← v₁ / β₁
    kdiv!(m, u, β)

    # Initialize some constants used in recursions below.
    ρ = one(T)
    kfill!(σ, β)
    kfill!(δhat, zero(T))
    kfill!(ω, zero(T))
    kfill!(γ, one(T))

    # Define stopping tolerance.
    ε = atol + rtol * β

    # Keep track of shifted systems that have converged.
    for i = 1 : nshifts
      converged[i] = rNorms[i] ≤ ε
      not_cv[i] = !converged[i]
    end
    iter = 0
    itmax == 0 && (itmax = m + n)

    # Build format strings for printing.
    (verbose > 0) && (fmt = Printf.Format("%5d" * repeat("  %8.1e", nshifts) * "  %.2fs\n"))
    kdisplay(iter, verbose) && Printf.format(iostream, fmt, iter, rNorms..., start_time |> ktimer)

    solved = !reduce(|, not_cv) # ArNorm ≤ ε
    tired = iter ≥ itmax
    status = "unknown"
    user_requested_exit = false
    overtimed = false

    # Main loop.
    while ! (solved || tired || user_requested_exit || overtimed)

      # Form next Lanczos vector.
      kmul!(u_next, A, v)           # u_nextₖ ← Avₖ
      δ = kdotr(m, u_next, u_next)  # δₖ = vₖᵀAᴴAvₖ
      kaxpy!(m, -δ, u, u_next)      # uₖ₊₁ = u_nextₖ - δₖuₖ - βₖuₖ₋₁
      kaxpy!(m, -β, u_prev, u_next)
      kmul!(v, Aᴴ, u_next)          # vₖ₊₁ = Aᴴuₖ₊₁
      β = knorm_elliptic(n, v, v)   # βₖ₊₁ = vₖ₊₁ᵀ M vₖ₊₁
      kdiv!(n, v, β)                # vₖ₊₁ = vₖ₊₁ / βₖ₊₁
      kdiv!(m, u_next, β)           # uₖ₊₁ = uₖ₊₁ / βₖ₊₁
      kcopy!(m, u_prev, u)          # u_prev ← u
      kcopy!(m, u, u_next)          # u ← u_next

      MisI || (ρ = kdotr(n, v, v))
      for i = 1 : nshifts
        δhat[i] = δ + ρ * shifts[i]
        γ[i] = inv(δhat[i] - ω[i] / γ[i])
      end

      # Compute next CG iterate for each shifted system that has not yet converged.
      for i = 1 : nshifts
        not_cv[i] = !converged[i]
        if not_cv[i]
          kaxpy!(n, γ[i], p[i], x[i])
          ω[i] = β * γ[i]
          σ[i] *= -ω[i]
          ω[i] *= ω[i]
          kaxpby!(n, σ[i], v, ω[i], p[i])

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
      kdisplay(iter, verbose) && Printf.format(iostream, fmt, iter, rNorms..., start_time |> ktimer)

      user_requested_exit = callback(workspace) :: Bool
      solved = !reduce(|, not_cv)
      tired = iter ≥ itmax
      timer = time_ns() - start_time
      overtimed = timer > timemax_ns
    end
    (verbose > 0) && @printf(iostream, "\n")

    # Termination status
    tired               && (status = "maximum number of iterations exceeded")
    solved              && (status = "solution good enough given atol and rtol")
    user_requested_exit && (status = "user-requested exit")
    overtimed           && (status = "time limit exceeded")

    # Update stats
    stats.niter = iter
    stats.solved = solved
    stats.timer = start_time |> ktimer
    stats.status = status
    return workspace
  end
end
