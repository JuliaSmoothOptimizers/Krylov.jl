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
                              callback=workspace->false, iostream::IO=kstdout)

`T` is an `AbstractFloat` such as `Float32`, `Float64` or `BigFloat`.
`FC` is `T` or `Complex{T}`.

    (X, stats) = block_minres(A, B, X0::AbstractMatrix; kwargs...)

Block-MINRES can be warm-started from an initial guess `X0` where `kwargs` are the same keyword arguments as above.

Solve the Hermitian linear system AX = B of size n with p right-hand sides using block-MINRES.

#### Interface

To easily switch between block Krylov methods, use the generic interface [`krylov_solve`](@ref) with `method = :block_minres`.

For an in-place variant that reuses memory across solves, see [`block_minres!`](@ref).

#### Input arguments

* `A`: a linear operator that models a Hermitian matrix of dimension `n`;
* `B`: a matrix of size `n × p`.

#### Optional argument

* `X0`: a matrix of size `n × p` that represents an initial guess of the solution `X`.

#### Keyword arguments

* `M`: linear operator that models a Hermitian positive-definite matrix of size `n` used for centered preconditioning;
* `ldiv`: define whether the preconditioners use `ldiv!` or `mul!`;
* `atol`: absolute stopping tolerance based on the residual norm;
* `rtol`: relative stopping tolerance based on the residual norm;
* `itmax`: the maximum number of iterations. If `itmax=0`, the default number of iterations is set to `2 * div(n,p)`;
* `timemax`: the time limit in seconds;
* `verbose`: additional details can be displayed if verbose mode is enabled (verbose > 0). Information will be displayed every `verbose` iterations;
* `history`: collect additional statistics on the run such as residual norms;
* `callback`: function or functor called as `callback(workspace)` that returns `true` if the block-Krylov method should terminate, and `false` otherwise;
* `iostream`: stream to which output is logged.

#### Output arguments

* `X`: a dense matrix of size `n × p`;
* `stats`: statistics collected on the run in a [`SimpleStats`](@ref) structure.
"""
function block_minres end

"""
    workspace = block_minres!(workspace::BlockMinresWorkspace, B; kwargs...)
    workspace = block_minres!(workspace::BlockMinresWorkspace, B, X0; kwargs...)

In these calls, `kwargs` are keyword arguments of [`block_minres`](@ref).

See [`BlockMinresWorkspace`](@ref) for instructions on how to create the `workspace`.

For a more generic interface, you can use [`krylov_workspace`](@ref) to allocate the workspace,
and [`krylov_solve!`](@ref) to run the block Krylov method in-place.
"""
function block_minres! end

def_args_block_minres = (:(A                    ),
                         :(B::AbstractMatrix{FC}))

def_optargs_block_minres = (:(X0::AbstractMatrix),)

def_kwargs_block_minres = (:(; M = I                        ),
                           :(; ldiv::Bool = false           ),
                           :(; atol::T = √eps(T)            ),
                           :(; rtol::T = √eps(T)            ),
                           :(; itmax::Int = 0               ),
                           :(; timemax::Float64 = Inf       ),
                           :(; verbose::Int = 0             ),
                           :(; history::Bool = false        ),
                           :(; callback = workspace -> false),
                           :(; iostream::IO = kstdout       ))

def_kwargs_block_minres = mapreduce(extract_parameters, vcat, def_kwargs_block_minres)

args_block_minres = (:A, :B)
optargs_block_minres = (:X0,)
kwargs_block_minres = (:M, :ldiv, :atol, :rtol, :itmax, :timemax, :verbose, :history, :callback, :iostream)

@eval begin
  function block_minres!(workspace :: BlockMinresWorkspace{T,FC,SV,SM}, $(def_args_block_minres...); $(def_kwargs_block_minres...)) where {T <: AbstractFloat, FC <: FloatOrComplex{T}, SV <: AbstractVector{FC}, SM <: AbstractMatrix{FC}}

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
    ktypeof(B) == SM || error("ktypeof(B) must be equal to $SM")

    # Set up workspace.
    Vₖ₋₁, Vₖ = workspace.Vₖ₋₁, workspace.Vₖ
    ΔX, X, Q, C = workspace.ΔX, workspace.X, workspace.Q, workspace.C
    D, Φ, stats = workspace.D, workspace.Φ, workspace.stats
    wₖ₋₂, wₖ₋₁ = workspace.wₖ₋₂, workspace.wₖ₋₁
    Hₖ₋₂, Hₖ₋₁ = workspace.Hₖ₋₂, workspace.Hₖ₋₁
    τₖ₋₂, τₖ₋₁ = workspace.τₖ₋₂, workspace.τₖ₋₁
    warm_start = workspace.warm_start
    RNorms = stats.residuals
    reset!(stats)
    R₀ = warm_start ? Q : B

    # Temporary buffers -- should be stored in the workspace
    Ψₖ = similar(B, p, p)
    Ωₖ = similar(B, p, p)
    Ψₖ₊₁ = similar(B, p, p)
    Πₖ₋₂ = similar(B, p, p)
    Γbarₖ₋₁ = similar(B, p, p)
    Γₖ₋₁ = similar(B, p, p)
    Λbarₖ = similar(B, p, p)
    Λₖ = similar(B, p, p)

    # Define the blocks D1 and D2
    D1 = view(D, 1:p, :)
    D2 = view(D, p+1:2p, :)
    trans = FC <: AbstractFloat ? 'T' : 'C'
    Φbarₖ = Φₖ = Φbarₖ₊₁ = Φ

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
    RNorm = norm(R₀)  # ‖R₀‖_F
    history && push!(RNorms, RNorm)

    iter = 0
    itmax == 0 && (itmax = 2*div(n,p))

    ε = atol + rtol * RNorm
    (verbose > 0) && @printf(iostream, "%5s  %7s  %5s\n", "k", "‖Rₖ‖", "timer")
    kdisplay(iter, verbose) && @printf(iostream, "%5d  %7.1e  %.2fs\n", iter, RNorm, start_time |> ktimer)

    # Stopping criterion
    status = "unknown"
    solved = RNorm ≤ ε
    tired = iter ≥ itmax
    user_requested_exit = false
    overtimed = false

    # Initial Ψ₁ and V₁
    τ = τₖ₋₂
    copyto!(Vₖ, R₀)
    householder!(Vₖ, Φbarₖ, τ)

    while !(solved || tired || user_requested_exit || overtimed)
      # Update iteration index.
      iter = iter + 1

      # Continue the block-Lanczos process.
      mul!(Q, A, Vₖ)                          # Q ← AVₖ
      mul!(Ωₖ, Vₖ', Q)                        # Ωₖ = Vₖᴴ * Q
      (iter ≥ 2) && mul!(Q, Vₖ₋₁, Ψₖ', α, β)  # Q ← Q - Vₖ₋₁ * Ψₖᴴ
      mul!(Q, Vₖ, Ωₖ, α, β)                   # Q = Q - Vₖ * Ωₖ

      # Update the QR factorization of Tₖ₊₁.ₖ = Qₖ [ Rₖ ].
      #                                            [ Oᵀ ]
      #
      # [ Ω₁ Ψ₂ᴴ 0  •  •  •  0   ]      [ Λ₁ Γ₁ Π₁ 0  •  •  0  ]
      # [ Ψ₂ Ω₂  •  •        •   ]      [ 0  Λ₂ Γ₂ •  •     •  ]
      # [ 0  •   •  •  •     •   ]      [ •  •  Λ₃ •  •  •  •  ]
      # [ •  •   •  •  •  •  •   ] = Qₖ [ •     •  •  •  •  0  ]
      # [ •      •  •  •  •  0   ]      [ •        •  •  • Πₖ₋₂]
      # [ •         •  •  •  Ψₖᴴ ]      [ •           •  • Γₖ₋₁]
      # [ •            •  Ψₖ Ωₖ  ]      [ 0  •  •  •  •  0  Λₖ ]
      # [ 0  •   •  •  •  0  Ψₖ₊₁]      [ 0  •  •  •  •  •  0  ]
      #
      # If k = 1, we don't have any previous reflection.
      # If k = 2, we apply the last reflection.
      # If k ≥ 3, we only apply the two previous reflections.

      # Apply previous Householder reflections Θₖ₋₂.
      if iter ≥ 3
        D1 .= zero(T)
        D2 .= Ψₖ'
        kormqr!('L', trans, Hₖ₋₂, τₖ₋₂, D)
        Πₖ₋₂ .= D1
        Γbarₖ₋₁ .= D2
      end

      # Apply previous Householder reflections Θₖ₋₁.
      if iter ≥ 2
        (iter == 2) && (Γbarₖ₋₁ .= Ψₖ')
        D1 .= Γbarₖ₋₁
        D2 .= Ωₖ
        kormqr!('L', trans, Hₖ₋₁, τₖ₋₁, D)
        Γₖ₋₁ .= D1
        Λbarₖ .= D2
      end

      # Vₖ₊₁ and Ψₖ₊₁ are stored in Q and Ψₖ₊₁.
      τ = τₖ₋₂
      householder!(Q, Ψₖ₊₁, τ)

      # Compute and apply current Householder reflection θₖ.
      Hₖ = Hₖ₋₂
      τₖ = τₖ₋₂
      (iter == 1) && (Λbarₖ .= Ωₖ)
      Hₖ[1:p,:] .= Λbarₖ
      Hₖ[p+1:2p,:] .= Ψₖ₊₁
      householder!(Hₖ, Λₖ, τₖ, compact=true)

      # Update Zₖ = (Qₖ)ᴴΨ₁E₁ = (Φ₁, ..., Φₖ, Φbarₖ₊₁)
      D1 .= Φbarₖ
      D2 .= zero(FC)
      kormqr!('L', trans, Hₖ, τₖ, D)
      Φₖ .= D1

      # Compute the directions Wₖ, the last columns of Wₖ = Vₖ(Rₖ)⁻¹ ⟷ (Rₖ)ᵀ(Wₖ)ᵀ = (Vₖ)ᵀ
      # w₁Λ₁ = v₁
      if iter == 1
        wₖ = wₖ₋₁
        wₖ .= Vₖ
        rdiv!(wₖ, UpperTriangular(Λₖ))
      end
      # w₂Λ₂ = v₂ - w₁Γ₁
      if iter == 2
        wₖ = wₖ₋₂
        wₖ .= (-wₖ₋₁ * Γₖ₋₁)
        wₖ .+= Vₖ
        rdiv!(wₖ, UpperTriangular(Λₖ))
      end
      # wₖΛₖ = vₖ - wₖ₋₁Γₖ₋₁ - wₖ₋₂Πₖ₋₂
      if iter ≥ 3
        wₖ = wₖ₋₂
        wₖ .= (-wₖ₋₂ * Πₖ₋₂)
        wₖ .= (wₖ - wₖ₋₁ * Γₖ₋₁)
        wₖ .+= Vₖ
        rdiv!(wₖ, UpperTriangular(Λₖ))
      end

      # Update Xₖ = VₖYₖ = WₖZₖ
      # Xₖ = Xₖ₋₁ + wₖ * Φₖ
      R = B - A * X
      mul!(X, wₖ, Φₖ, γ, β)

      # Update residual norm estimate.
      # ‖ M(B - AXₖ) ‖_F = ‖Φbarₖ₊₁‖_F
      Φbarₖ₊₁ .= D2
      RNorm = norm(Φbarₖ₊₁)
      history && push!(RNorms, RNorm)

      # Compute vₖ and vₖ₊₁
      copyto!(Vₖ₋₁, Vₖ)  # vₖ₋₁ ← vₖ
      copyto!(Vₖ, Q)     # vₖ ← vₖ₊₁

      # Update directions for X and other variables...
      if iter ≥ 2
        @kswap!(wₖ₋₂, wₖ₋₁)
        @kswap!(Hₖ₋₂, Hₖ₋₁)
        @kswap!(τₖ₋₂, τₖ₋₁)
      end

      if iter == 1
        copyto!(Hₖ₋₁, Hₖ)
        copyto!(τₖ₋₁, τₖ)
      end
      copyto!(Ψₖ, Ψₖ₊₁)

      # Update stopping criterion.
      user_requested_exit = callback(workspace) :: Bool
      solved = RNorm ≤ ε
      tired = iter ≥ itmax
      timer = time_ns() - start_time
      overtimed = timer > timemax_ns
      kdisplay(iter, verbose) && @printf(iostream, "%5d  %7.1e  %.2fs\n", iter, RNorm, start_time |> ktimer)
    end
    (verbose > 0) && @printf(iostream, "\n")

    # Termination status
    tired               && (status = "maximum number of iterations exceeded")
    solved              && (status = "solution good enough given atol and rtol")
    overtimed           && (status = "time limit exceeded")
    user_requested_exit && (status = "user-requested exit")

    # Update Xₖ
    warm_start && (X .+= ΔX)
    workspace.warm_start = false

    # Update stats
    stats.niter = iter
    stats.solved = solved
    stats.timer = start_time |> ktimer
    stats.status = status
    return workspace
  end
end
