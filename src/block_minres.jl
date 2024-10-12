# An implementation of block-MINRES for the solution of the square linear system AX = B.
#
# Alexis Montoison, <alexis.montoison@polymtl.ca> -- <amontoison@anl.gov>
# Argonne National Laboratory -- Chicago, October 2024.

export block_minres, block_minres!

"""
    (X, stats) = block_minres(A, b::AbstractMatrix{FC};
                              M=I, ldiv::Bool=false,
                              atol::T=√eps(T), rtol::T=√eps(T), itmax::Int=0,
                              timemax::Float64=Inf, verbose::Int=0, history::Bool=false,
                              callback=solver->false, iostream::IO=kstdout)

`T` is an `AbstractFloat` such as `Float32`, `Float64` or `BigFloat`.
`FC` is `T` or `Complex{T}`.

    (X, stats) = block_minres(A, B, X0::AbstractMatrix; kwargs...)

Block-MINRES can be warm-started from an initial guess `X0` where `kwargs` are the same keyword arguments as above.

Solve the Hermitian linear system AX = B of size n with p right-hand sides using block-MINRES.

#### Input arguments

* `A`: a linear operator that models a Hermitian matrix of dimension n;
* `B`: a matrix of size n × p.

#### Optional argument

* `X0`: a matrix of size n × p that represents an initial guess of the solution X.

#### Keyword arguments

* `M`: linear operator that models a Hermitian positive-definite matrix of size `n` used for centered preconditioning;
* `ldiv`: define whether the preconditioners use `ldiv!` or `mul!`;
* `atol`: absolute stopping tolerance based on the residual norm;
* `rtol`: relative stopping tolerance based on the residual norm;
* `itmax`: the maximum number of iterations. If `itmax=0`, the default number of iterations is set to `2 * div(n,p)`;
* `timemax`: the time limit in seconds;
* `verbose`: additional details can be displayed if verbose mode is enabled (verbose > 0). Information will be displayed every `verbose` iterations;
* `history`: collect additional statistics on the run such as residual norms;
* `callback`: function or functor called as `callback(solver)` that returns `true` if the block-Krylov method should terminate, and `false` otherwise;
* `iostream`: stream to which output is logged.

#### Output arguments

* `X`: a dense matrix of size n × p;
* `stats`: statistics collected on the run in a [`SimpleStats`](@ref) structure.
"""
function block_minres end

"""
    solver = block_minres!(solver::BlockMinresSolver, B; kwargs...)
    solver = block_minres!(solver::BlockMinresSolver, B, X0; kwargs...)

where `kwargs` are keyword arguments of [`block_minres`](@ref).

See [`BlockMinresSolver`](@ref) for more details about the `solver`.
"""
function block_minres! end

def_args_block_minres = (:(A                    ),
                         :(B::AbstractMatrix{FC}))

def_optargs_block_minres = (:(X0::AbstractMatrix),)

def_kwargs_block_minres = (:(; M = I                            ),
                           :(; ldiv::Bool = false               ),
                           :(; atol::T = √eps(T)                ),
                           :(; rtol::T = √eps(T)                ),
                           :(; itmax::Int = 0                   ),
                           :(; timemax::Float64 = Inf           ),
                           :(; verbose::Int = 0                 ),
                           :(; history::Bool = false            ),
                           :(; callback = solver -> false       ),
                           :(; iostream::IO = kstdout           ))

def_kwargs_block_minres = mapreduce(extract_parameters, vcat, def_kwargs_block_minres)

args_block_minres = (:A, :B)
optargs_block_minres = (:X0,)
kwargs_block_minres = (:M, :ldiv, :atol, :rtol, :itmax, :timemax, :verbose, :history, :callback, :iostream)

@eval begin
  function block_minres($(def_args_block_minres...), $(def_optargs_block_minres...); $(def_kwargs_block_minres...)) where {T <: AbstractFloat, FC <: FloatOrComplex{T}}
    start_time = time_ns()
    solver = BlockMinresSolver(A, B)
    warm_start!(solver, $(optargs_block_minres...))
    elapsed_time = ktimer(start_time)
    timemax -= elapsed_time
    block_minres!(solver, $(args_block_minres...); $(kwargs_block_minres...))
    solver.stats.timer += elapsed_time
    return solver.X, solver.stats
  end

  function block_minres($(def_args_block_minres...); $(def_kwargs_block_minres...)) where {T <: AbstractFloat, FC <: FloatOrComplex{T}}
    start_time = time_ns()
    solver = BlockMinresSolver(A, B)
    elapsed_time = ktimer(start_time)
    timemax -= elapsed_time
    block_minres!(solver, $(args_block_minres...); $(kwargs_block_minres...))
    solver.stats.timer += elapsed_time
    return solver.X, solver.stats
  end

  function block_minres!(solver :: BlockMinresSolver{T,FC,SV,SM}, $(def_args_block_minres...); $(def_kwargs_block_minres...)) where {T <: AbstractFloat, FC <: FloatOrComplex{T}, SV <: AbstractVector{FC}, SM <: AbstractMatrix{FC}}

    # Timer
    start_time = time_ns()
    timemax_ns = 1e9 * timemax

    n, m = size(A)
    s, p = size(B)
    m == n || error("System must be square")
    n == s || error("Inconsistent problem size")
    (verbose > 0) && @printf(iostream, "BLOCK-MINRES: system of size %d with %d right-hand sides\n", n, p)

    # Check M = Iₙ
    MisI = (M === I)
    MisI || error("Block-MINRES doesn't support preconditioning yet.")

    # Check type consistency
    eltype(A) == FC || @warn "eltype(A) ≠ $FC. This could lead to errors or additional allocations in operator-matrix products."
    ktypeof(B) <: SM || error("ktypeof(B) is not a subtype of $SM")

    # Set up workspace.
    ΔX, X, W, V, Z = solver.ΔX, solver.X, solver.W, solver.V, solver.Z
    C, D, R, H, τ, stats = solver.C, solver.D, solver.R, solver.H, solver.τ, solver.stats
    warm_start = solver.warm_start
    RNorms = stats.residuals
    reset!(stats)
    R₀ = warm_start ? Q : B

    # Define the blocks D1 and D2
    D1 = view(D, 1:p, :)
    D2 = view(D, p+1:2p, :)
    trans = FC <: AbstractFloat ? 'T' : 'C'

    # Coefficients for mul!
    α = -one(FC)
    β = one(FC)
    γ = one(FC)

    # Initial solution X₀.
    fill!(X, zero(FC))

    # Initial residual R₀.
    if warm_start
      mul!(Q, A, ΔX)
      Q .= B .- Q
    end
    MisI || mulorldiv!(R₀, M, W, ldiv)  # R₀ = M(B - AX₀)
    RNorm = norm(R₀)                    # ‖R₀‖_F
    history && push!(RNorms, RNorm)

    iter = 0
    itmax == 0 && (itmax = 2*div(n,p))

    ε = atol + rtol * RNorm
    (verbose > 0) && @printf(iostream, "%5s  %7s  %5s\n", "k", "‖Rₖ‖", "timer")
    kdisplay(iter, verbose) && @printf(iostream, "%5d  %7.1e  %.2fs\n", iter, RNorm, ktimer(start_time))

    # Stopping criterion
    status = "unknown"
    solved = RNorm ≤ ε
    tired = iter ≥ itmax
    user_requested_exit = false
    overtimed = false

    while !(solved || tired || user_requested_exit || overtimed)
      # Update iteration index.
      iter = iter + 1

      # Initial Ψ₁ and V₁
      copyto!(V, R₀)
      householder!(V, Z, τ)

      # Continue the block-Lanczos process.
      mul!(W, A, V)  # Q ← AVₖ
      for i = 1 : inner_iter
        mul!(Ω, V', W)       # Ωₖ = Vₖᴴ * Q
        (iter ≥ 2) && mul!(Q, ...)  # Q ← Q - βₖ * Vₖ₋₁ * Ψₖᴴ
        mul!(Q, V, R, α, β)  # Q = Q - Vₖ * Ωₖ
      end

      # Vₖ₊₁ and Ψₖ₊₁ are stored in Q and C.
      householder!(Q, C, τ)

      # Update the QR factorization of Tₖ₊₁.ₖ.
      # Apply previous Householder reflections Ωᵢ.
      for i = 1 : inner_iter-1
        D1 .= R[nr+i]
        D2 .= R[nr+i+1]
        @kormqr!('L', trans, H[i], τ[i], D)
        R[nr+i] .= D1
        R[nr+i+1] .= D2
      end

      # Compute and apply current Householder reflection Ωₖ.
      H[inner_iter][1:p,:] .= R[nr+inner_iter]
      H[inner_iter][p+1:2p,:] .= C
      householder!(H[inner_iter], R[nr+inner_iter], τ[inner_iter], compact=true)

      # Update Zₖ = (Qₖ)ᴴΓE₁ = (Λ₁, ..., Λₖ, Λbarₖ₊₁)
      D1 .= Z[inner_iter]
      D2 .= zero(FC)
      @kormqr!('L', trans, H[inner_iter], τ[inner_iter], D)
      Z[inner_iter] .= D1

      # Update residual norm estimate.
      # ‖ M(B - AXₖ) ‖_F = ‖Λbarₖ₊₁‖_F
      C .= D2
      RNorm = norm(C)
      history && push!(RNorms, RNorm)

      # Update stopping criterion.
      user_requested_exit = callback(solver) :: Bool
      solved = RNorm ≤ ε
      tired = iter ≥ itmax
      timer = time_ns() - start_time
      overtimed = timer > timemax_ns
      kdisplay(iter, verbose) && @printf(iostream, "%5d  %7.1e  %.2fs\n", iter, RNorm, ktimer(start_time))
    end
    (verbose > 0) && @printf(iostream, "\n")

    # Termination status
    tired               && (status = "maximum number of iterations exceeded")
    solved              && (status = "solution good enough given atol and rtol")
    overtimed           && (status = "time limit exceeded")
    user_requested_exit && (status = "user-requested exit")

    # Update Xₖ
    warm_start && (X .+= ΔX)
    solver.warm_start = false

    # Update stats
    stats.niter = iter
    stats.solved = solved
    stats.timer = ktimer(start_time)
    stats.status = status
    return solver
  end
end
