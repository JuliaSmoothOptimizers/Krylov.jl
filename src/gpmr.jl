# An implementation of GPMR for the solution of unsymmetric partitioned linear systems.
#
# This method is described in
#
# A. Montoison and D. Orban
# GPMR: An Iterative Method for Unsymmetric Partitioned Linear Systems.
# SIAM Journal on Matrix Analysis and Applications, 44(1), pp. 293--311, 2023.
#
# Alexis Montoison, <alexis.montoison@polymtl.ca>
# Montréal, August 2021.

export gpmr, gpmr!

"""
    (x, y, stats) = gpmr(A, B, b::AbstractVector{FC}, c::AbstractVector{FC};
                         memory::Int=20, C=I, D=I, E=I, F=I,
                         ldiv::Bool=false, gsp::Bool=false,
                         λ::FC=one(FC), μ::FC=one(FC),
                         reorthogonalization::Bool=false, atol::T=√eps(T),
                         rtol::T=√eps(T), itmax::Int=0,
                         timemax::Float64=Inf, verbose::Int=0, history::Bool=false,
                         callback=solver->false, iostream::IO=kstdout)

`T` is an `AbstractFloat` such as `Float32`, `Float64` or `BigFloat`.
`FC` is `T` or `Complex{T}`.

    (x, y, stats) = gpmr(A, B, b, c, x0::AbstractVector, y0::AbstractVector; kwargs...)

GPMR can be warm-started from initial guesses `x0` and `y0` where `kwargs` are the same keyword arguments as above.

Given matrices `A` of dimension m × n and `B` of dimension n × m,
GPMR solves the non-Hermitian partitioned linear system

    [ λIₘ   A  ] [ x ] = [ b ]
    [  B   μIₙ ] [ y ]   [ c ],

of size (n+m) × (n+m) where λ and μ are real or complex numbers.
`A` can have any shape and `B` has the shape of `Aᴴ`.
`A`, `B`, `b` and `c` must be all nonzero.

This implementation allows left and right block diagonal preconditioners

    [ C    ] [ λM   A ] [ E    ] [ E⁻¹x ] = [ Cb ]
    [    D ] [  B  μN ] [    F ] [ F⁻¹y ]   [ Dc ],

and can solve

    [ λM   A ] [ x ] = [ b ]
    [  B  μN ] [ y ]   [ c ]

when `CE = M⁻¹` and `DF = N⁻¹`.

By default, GPMR solves unsymmetric linear systems with `λ = 1` and `μ = 1`.

GPMR is based on the orthogonal Hessenberg reduction process and its relations with the block-Arnoldi process.
The residual norm ‖rₖ‖ is monotonically decreasing in GPMR.

GPMR stops when `itmax` iterations are reached or when `‖rₖ‖ ≤ atol + ‖r₀‖ * rtol`.
`atol` is an absolute tolerance and `rtol` is a relative tolerance.

#### Input arguments

* `A`: a linear operator that models a matrix of dimension m × n;
* `B`: a linear operator that models a matrix of dimension n × m;
* `b`: a vector of length m;
* `c`: a vector of length n.

#### Optional arguments

* `x0`: a vector of length m that represents an initial guess of the solution x;
* `y0`: a vector of length n that represents an initial guess of the solution y.

#### Keyword arguments

* `memory`: if `restart = true`, the restarted version GPMR(k) is used with `k = memory`. If `restart = false`, the parameter `memory` should be used as a hint of the number of iterations to limit dynamic memory allocations. Additional storage will be allocated if the number of iterations exceeds `memory`;
* `C`: linear operator that models a nonsingular matrix of size `m`, and represents the first term of the block-diagonal left preconditioner;
* `D`: linear operator that models a nonsingular matrix of size `n`, and represents the second term of the block-diagonal left preconditioner;
* `E`: linear operator that models a nonsingular matrix of size `m`, and represents the first term of the block-diagonal right preconditioner;
* `F`: linear operator that models a nonsingular matrix of size `n`, and represents the second term of the block-diagonal right preconditioner;
* `ldiv`: define whether the preconditioners use `ldiv!` or `mul!`;
* `gsp`: if `true`, set `λ = 1` and `μ = 0` for generalized saddle-point systems;
* `λ` and `μ`: diagonal scaling factors of the partitioned linear system;
* `reorthogonalization`: reorthogonalize the new vectors of the Krylov basis against all previous vectors;
* `atol`: absolute stopping tolerance based on the residual norm;
* `rtol`: relative stopping tolerance based on the residual norm;
* `itmax`: the maximum number of iterations. If `itmax=0`, the default number of iterations is set to `m+n`;
* `timemax`: the time limit in seconds;
* `verbose`: additional details can be displayed if verbose mode is enabled (verbose > 0). Information will be displayed every `verbose` iterations;
* `history`: collect additional statistics on the run such as residual norms, or Aᴴ-residual norms;
* `callback`: function or functor called as `callback(solver)` that returns `true` if the Krylov method should terminate, and `false` otherwise;
* `iostream`: stream to which output is logged.

#### Output arguments

* `x`: a dense vector of length m;
* `y`: a dense vector of length n;
* `stats`: statistics collected on the run in a [`SimpleStats`](@ref) structure.

#### Reference

* A. Montoison and D. Orban, [*GPMR: An Iterative Method for Unsymmetric Partitioned Linear Systems*](https://doi.org/10.1137/21M1459265), SIAM Journal on Matrix Analysis and Applications, 44(1), pp. 293--311, 2023.
"""
function gpmr end

"""
    solver = gpmr!(solver::GpmrSolver, A, B, b, c; kwargs...)
    solver = gpmr!(solver::GpmrSolver, A, B, b, c, x0, y0; kwargs...)

where `kwargs` are keyword arguments of [`gpmr`](@ref).

Note that the `memory` keyword argument is the only exception.
It's required to create a `GpmrSolver` and can't be changed later.

See [`GpmrSolver`](@ref) for more details about the `solver`.
"""
function gpmr! end

def_args_gpmr = (:(A                    ),
                 :(B                    ),
                 :(b::AbstractVector{FC}),
                 :(c::AbstractVector{FC}))

def_optargs_gpmr = (:(x0 :: AbstractVector),
                    :(y0 :: AbstractVector))

def_kwargs_gpmr = (:(; C = I                            ),
                   :(; D = I                            ),
                   :(; E = I                            ),
                   :(; F = I                            ),
                   :(; ldiv::Bool = false               ),
                   :(; gsp::Bool = false                ),
                   :(; λ::FC = one(FC)                  ),
                   :(; μ::FC = one(FC)                  ),
                   :(; reorthogonalization::Bool = false),
                   :(; atol::T = √eps(T)                ),
                   :(; rtol::T = √eps(T)                ),
                   :(; itmax::Int = 0                   ),
                   :(; timemax::Float64 = Inf           ),
                   :(; verbose::Int = 0                 ),
                   :(; history::Bool = false            ),
                   :(; callback = solver -> false       ),
                   :(; iostream::IO = kstdout           ))

def_kwargs_gpmr = mapreduce(extract_parameters, vcat, def_kwargs_gpmr)

args_gpmr = (:A, :B, :b, :c)
optargs_gpmr = (:x0, :y0)
kwargs_gpmr = (:C, :D, :E, :F, :ldiv, :gsp, :λ, :μ, :reorthogonalization, :atol, :rtol, :itmax, :timemax, :verbose, :history, :callback, :iostream)

@eval begin
  function gpmr($(def_args_gpmr...), $(def_optargs_gpmr...); memory :: Int=20, $(def_kwargs_gpmr...)) where {T <: AbstractFloat, FC <: FloatOrComplex{T}}
    start_time = time_ns()
    solver = GpmrSolver(A, b, memory)
    warm_start!(solver, $(optargs_gpmr...))
    elapsed_time = ktimer(start_time)
    timemax -= elapsed_time
    gpmr!(solver, $(args_gpmr...); $(kwargs_gpmr...))
    solver.stats.timer += elapsed_time
    return (solver.x, solver.y, solver.stats)
  end

  function gpmr($(def_args_gpmr...); memory :: Int=20, $(def_kwargs_gpmr...)) where {T <: AbstractFloat, FC <: FloatOrComplex{T}}
    start_time = time_ns()
    solver = GpmrSolver(A, b, memory)
    elapsed_time = ktimer(start_time)
    timemax -= elapsed_time
    gpmr!(solver, $(args_gpmr...); $(kwargs_gpmr...))
    solver.stats.timer += elapsed_time
    return (solver.x, solver.y, solver.stats)
  end

  function gpmr!(solver :: GpmrSolver{T,FC,S}, $(def_args_gpmr...); $(def_kwargs_gpmr...)) where {T <: AbstractFloat, FC <: FloatOrComplex{T}, S <: AbstractVector{FC}}

    # Timer
    start_time = time_ns()
    timemax_ns = 1e9 * timemax

    m, n = size(A)
    s, t = size(B)
    (m == solver.m && n == solver.n) || error("(solver.m, solver.n) = ($(solver.m), $(solver.n)) is inconsistent with size(A) = ($m, $n)")
    m == t         || error("Inconsistent problem size")
    s == n         || error("Inconsistent problem size")
    length(b) == m || error("Inconsistent problem size")
    length(c) == n || error("Inconsistent problem size")
    (verbose > 0) && @printf(iostream, "GPMR: system of %d equations in %d variables\n", m+n, m+n)

    # Check C = E = Iₘ and D = F = Iₙ
    CisI = (C === I)
    DisI = (D === I)
    EisI = (E === I)
    FisI = (F === I)

    # Check type consistency
    eltype(A) == FC || @warn "eltype(A) ≠ $FC. This could lead to errors or additional allocations in operator-vector products."
    eltype(B) == FC || @warn "eltype(B) ≠ $FC. This could lead to errors or additional allocations in operator-vector products."
    ktypeof(b) <: S || error("ktypeof(b) is not a subtype of $S")
    ktypeof(c) <: S || error("ktypeof(c) is not a subtype of $S")

    # Determine λ and μ associated to generalized saddle point systems.
    gsp && (λ = one(FC) ; μ = zero(FC))

    warm_start = solver.warm_start
    warm_start && (λ ≠ 0) && !EisI && error("Warm-start with right preconditioners is not supported.")
    warm_start && (μ ≠ 0) && !FisI && error("Warm-start with right preconditioners is not supported.")

    # Set up workspace.
    allocate_if(!CisI, solver, :q , S, m)
    allocate_if(!DisI, solver, :p , S, n)
    allocate_if(!EisI, solver, :wB, S, m)
    allocate_if(!FisI, solver, :wA, S, n)
    wA, wB, dA, dB, Δx, Δy = solver.wA, solver.wB, solver.dA, solver.dB, solver.Δx, solver.Δy
    x, y, V, U, gs, gc = solver.x, solver.y, solver.V, solver.U, solver.gs, solver.gc
    zt, R, stats = solver.zt, solver.R, solver.stats
    rNorms = stats.residuals
    reset!(stats)
    b₀ = warm_start ? dA : b
    c₀ = warm_start ? dB : c
    q  = CisI ? dA : solver.q
    p  = DisI ? dB : solver.p

    # Initial solutions x₀ and y₀.
    x .= zero(FC)
    y .= zero(FC)

    iter = 0
    itmax == 0 && (itmax = m+n)

    # Initialize workspace.
    nr = 0           # Number of coefficients stored in Rₖ
    mem = length(V)  # Memory
    ωₖ = zero(FC)    # Auxiliary variable to store fₖₖ
    for i = 1 : mem
      V[i] .= zero(FC)
      U[i] .= zero(FC)
    end
    gs .= zero(FC)  # Givens sines used for the factorization QₖRₖ = Sₖ₊₁.ₖ.
    gc .= zero(T)   # Givens cosines used for the factorization QₖRₖ = Sₖ₊₁.ₖ.
    R  .= zero(FC)  # Upper triangular matrix Rₖ.
    zt .= zero(FC)  # Rₖzₖ = tₖ with (tₖ, τbar₂ₖ₊₁, τbar₂ₖ₊₂) = (Qₖ)ᴴ(βe₁ + γe₂).

    # Warm-start
    # If λ ≠ 0, Cb₀ = Cb - CAΔy - λΔx because CM = Iₘ and E = Iₘ
    # E ≠ Iₘ is only allowed when λ = 0 because E⁻¹Δx can't be computed to use CME = Iₘ
    # Compute C(b - AΔy) - λΔx
    warm_start && mul!(b₀, A, Δy)
    warm_start && @kaxpby!(m, one(FC), b, -one(FC), b₀)
    !CisI && mulorldiv!(q, C, b₀, ldiv)
    !CisI && (b₀ = q)
    warm_start && (λ ≠ 0) && @kaxpy!(m, -λ, Δx, b₀)

    # If μ ≠ 0, Dc₀ = Dc - DBΔx - μΔy because DN = Iₙ and F = Iₙ
    # F ≠ Iₙ is only allowed when μ = 0 because F⁻¹Δy can't be computed to use DNF = Iₘ
    # Compute D(c - BΔx) - μΔy
    warm_start && mul!(c₀, B, Δx)
    warm_start && @kaxpby!(n, one(FC), c, -one(FC), c₀)
    !DisI && mulorldiv!(p, D, c₀, ldiv)
    !DisI && (c₀ = p)
    warm_start && (μ ≠ 0) && @kaxpy!(n, -μ, Δy, c₀)

    # Initialize the orthogonal Hessenberg reduction process.
    # βv₁ = Cb
    β = @knrm2(m, b₀)
    β ≠ 0 || error("b must be nonzero")
    V[1] .= b₀ ./ β

    # γu₁ = Dc
    γ = @knrm2(n, c₀)
    γ ≠ 0 || error("c must be nonzero")
    U[1] .= c₀ ./ γ

    # Compute ‖r₀‖² = γ² + β²
    rNorm = sqrt(γ^2 + β^2)
    history && push!(rNorms, rNorm)
    ε = atol + rtol * rNorm

    # Initialize t̄₀
    zt[1] = β
    zt[2] = γ

    (verbose > 0) && @printf(iostream, "%5s  %7s  %7s  %7s  %5s\n", "k", "‖rₖ‖", "hₖ₊₁.ₖ", "fₖ₊₁.ₖ", "timer")
    kdisplay(iter, verbose) && @printf(iostream, "%5d  %7.1e  %7s  %7s  %.2fs\n", iter, rNorm, "✗ ✗ ✗ ✗", "✗ ✗ ✗ ✗", ktimer(start_time))

    # Tolerance for breakdown detection.
    btol = eps(T)^(3/4)

    # Stopping criterion.
    breakdown = false
    inconsistent = false
    solved = rNorm ≤ ε
    tired = iter ≥ itmax
    status = "unknown"
    user_requested_exit = false
    overtimed = false

    while !(solved || tired || breakdown || user_requested_exit || overtimed)

      # Update iteration index.
      iter = iter + 1
      k = iter
      nr₂ₖ₋₁ = nr       # Position of the column 2k-1 in Rₖ.
      nr₂ₖ = nr + 2k-1  # Position of the column 2k in Rₖ.

      # Update workspace if more storage is required
      if iter > mem
        for i = 1 : 4k-1
          push!(R, zero(FC))
        end
        for i = 1 : 4
          push!(gs, zero(FC))
          push!(gc, zero(T))
        end
      end

      # Continue the orthogonal Hessenberg reduction process.
      # CAFUₖ = VₖHₖ + hₖ₊₁.ₖ * vₖ₊₁(eₖ)ᵀ = Vₖ₊₁Hₖ₊₁.ₖ
      # DBEVₖ = UₖFₖ + fₖ₊₁.ₖ * uₖ₊₁(eₖ)ᵀ = Uₖ₊₁Fₖ₊₁.ₖ
      wA = FisI ? U[iter] : solver.wA
      wB = EisI ? V[iter] : solver.wB
      FisI || mulorldiv!(wA, F, U[iter], ldiv)  # wA = Fuₖ
      EisI || mulorldiv!(wB, E, V[iter], ldiv)  # wB = Evₖ
      mul!(dA, A, wA)                           # dA = AFuₖ
      mul!(dB, B, wB)                           # dB = BEvₖ
      CisI || mulorldiv!(q, C, dA, ldiv)        # q  = CAFuₖ
      DisI || mulorldiv!(p, D, dB, ldiv)        # p  = DBEvₖ

      for i = 1 : iter
        hᵢₖ = @kdot(m, V[i], q)    # hᵢ.ₖ = (vᵢ)ᴴq
        fᵢₖ = @kdot(n, U[i], p)    # fᵢ.ₖ = (uᵢ)ᴴp
        @kaxpy!(m, -hᵢₖ, V[i], q)  # q ← q - hᵢ.ₖvᵢ
        @kaxpy!(n, -fᵢₖ, U[i], p)  # p ← p - fᵢ.ₖuᵢ
        R[nr₂ₖ + 2i-1] = hᵢₖ
        (i < iter) ? R[nr₂ₖ₋₁ + 2i] = fᵢₖ : ωₖ = fᵢₖ
      end

      # Reorthogonalization of the Krylov basis.
      if reorthogonalization
        for i = 1 : iter
          Htmp = @kdot(m, V[i], q)    # hₜₘₚ = (vᵢ)ᴴq
          Ftmp = @kdot(n, U[i], p)    # fₜₘₚ = (uᵢ)ᴴp
          @kaxpy!(m, -Htmp, V[i], q)  # q ← q - hₜₘₚvᵢ
          @kaxpy!(n, -Ftmp, U[i], p)  # p ← p - fₜₘₚuᵢ
          R[nr₂ₖ + 2i-1] += Htmp                            # hᵢ.ₖ = hᵢ.ₖ + hₜₘₚ
          (i < iter) ? R[nr₂ₖ₋₁ + 2i] += Ftmp : ωₖ += Ftmp  # fᵢ.ₖ = fᵢ.ₖ + fₜₘₚ
        end
      end

      Haux = @knrm2(m, q)   # hₖ₊₁.ₖ = ‖q‖₂
      Faux = @knrm2(n, p)   # fₖ₊₁.ₖ = ‖p‖₂

      # Add regularization terms.
      R[nr₂ₖ₋₁ + 2k-1] = λ  # S₂ₖ₋₁.₂ₖ₋₁ = λ
      R[nr₂ₖ + 2k]     = μ  # S₂ₖ.₂ₖ = μ

      # Notations : Wₖ = [w₁ ••• wₖ] = [v₁ 0  ••• vₖ 0 ]
      #                                [0  u₁ ••• 0  uₖ]
      #
      # rₖ = [ b ] - [ λI   A ] [ xₖ ] = [ b ] - [ λI   A ] Wₖzₖ
      #      [ c ]   [  B  μI ] [ yₖ ]   [ c ]   [  B  μI ]
      #
      # block-Arnoldi formulation : [ λI   A ] Wₖ = Wₖ₊₁Sₖ₊₁.ₖ
      #                             [  B  μI ]
      #
      # GPMR subproblem : min ‖ rₖ ‖ ↔ min ‖ Sₖ₊₁.ₖzₖ - βe₁ - γe₂ ‖
      #
      # Update the QR factorization of Sₖ₊₁.ₖ = Qₖ [ Rₖ ].
      #                                            [ Oᵀ ]
      #
      # Apply previous givens reflections when k ≥ 2
      # [ 1                ][ 1                ][ c₂.ᵢ  s₂.ᵢ       ][ c₁.ᵢ        s₁.ᵢ ] [ r̄₂ᵢ₋₁.₂ₖ₋₁  r̄₂ᵢ₋₁.₂ₖ ]   [ r₂ᵢ₋₁.₂ₖ₋₁  r₂ᵢ₋₁.₂ₖ ]
      # [    c₄.ᵢ  s₄.ᵢ    ][    c₃.ᵢ     s₃.ᵢ ][ s̄₂.ᵢ -c₂.ᵢ       ][       1          ] [ r̄₂ᵢ.₂ₖ₋₁    r̄₂ᵢ.₂ₖ   ] = [ r₂ᵢ.₂ₖ₋₁    r₂ᵢ.₂ₖ   ]
      # [    s̄₄.ᵢ -c₄.ᵢ    ][          1       ][             1    ][          1       ] [ ρ           hᵢ₊₁.ₖ   ]   [ r̄₂ᵢ₊₁.₂ₖ₋₁  r̄₂ᵢ₊₁.₂ₖ ]
      # [                1 ][    s̄₃.ᵢ    -c₃.ᵢ ][                1 ][ s̄₁.ᵢ       -c₁.ᵢ ] [ fᵢ₊₁.ₖ      δ        ]   [ r̄₂ᵢ₊₂.₂ₖ₋₁  r̄₂ᵢ₊₂.₂ₖ ]
      #
      # r̄₁.₂ₖ₋₁ = 0, r̄₁.₂ₖ = h₁.ₖ, r̄₂.₂ₖ₋₁ = f₁.ₖ and r̄₂.₂ₖ = 0.
      # (ρ, δ) = (λ, μ) if i == k-1, (ρ, δ) = (0, 0) otherwise.
      for i = 1 : iter-1
        for nrcol ∈ (nr₂ₖ₋₁, nr₂ₖ)
          flag = (i == iter-1 && nrcol == nr₂ₖ₋₁)
          αₖ = flag ? ωₖ : R[nrcol + 2i+2]

          c₁ᵢ = gc[4i-3]
          s₁ᵢ = gs[4i-3]
          rtmp            =      c₁ᵢ  * R[nrcol + 2i-1] + s₁ᵢ * αₖ
          αₖ              = conj(s₁ᵢ) * R[nrcol + 2i-1] - c₁ᵢ * αₖ
          R[nrcol + 2i-1] = rtmp

          c₂ᵢ = gc[4i-2]
          s₂ᵢ = gs[4i-2]
          rtmp            =      c₂ᵢ  * R[nrcol + 2i-1] + s₂ᵢ * R[nrcol + 2i]
          R[nrcol + 2i]   = conj(s₂ᵢ) * R[nrcol + 2i-1] - c₂ᵢ * R[nrcol + 2i]
          R[nrcol + 2i-1] = rtmp

          c₃ᵢ = gc[4i-1]
          s₃ᵢ = gs[4i-1]
          rtmp          =      c₃ᵢ  * R[nrcol + 2i] + s₃ᵢ * αₖ
          αₖ            = conj(s₃ᵢ) * R[nrcol + 2i] - c₃ᵢ * αₖ
          R[nrcol + 2i] = rtmp

          c₄ᵢ = gc[4i]
          s₄ᵢ = gs[4i]
          rtmp            =      c₄ᵢ  * R[nrcol + 2i] + s₄ᵢ * R[nrcol + 2i+1]
          R[nrcol + 2i+1] = conj(s₄ᵢ) * R[nrcol + 2i] - c₄ᵢ * R[nrcol + 2i+1]
          R[nrcol + 2i]   = rtmp

          flag ? ωₖ = αₖ : R[nrcol + 2i+2] = αₖ
        end
      end

      # Compute and apply current givens reflections
      # [ 1                ][ 1                ][ c₂.ₖ  s₂.ₖ       ][ c₁.ₖ        s₁.ₖ ] [ r̄₂ₖ₋₁.₂ₖ₋₁  r̄₂ₖ₋₁.₂ₖ ]    [ r₂ₖ₋₁.₂ₖ₋₁  r₂ₖ₋₁.₂ₖ ]
      # [    c₄.ₖ  s₄.ₖ    ][    c₃.ₖ     s₃.ₖ ][ s̄₂.ₖ -c₂.ₖ       ][       1          ] [ r̄₂ₖ.₂ₖ₋₁    r̄₂ₖ.₂ₖ   ] =  [             r₂ₖ.₂ₖ   ]
      # [    s̄₄.ₖ -c₄.ₖ    ][          1       ][             1    ][          1       ] [             hₖ₊₁.ₖ   ]    [                      ]
      # [                1 ][    s̄₃.ₖ    -c₃.ₖ ][                1 ][ s̄₁.ₖ       -c₁.ₖ ] [ fₖ₊₁.ₖ               ]    [                      ]
      (c₁ₖ, s₁ₖ, R[nr₂ₖ₋₁ + 2k-1]) = sym_givens(R[nr₂ₖ₋₁ + 2k-1], Faux)  # annihilate fₖ₊₁.ₖ
      θₖ             = conj(s₁ₖ) * R[nr₂ₖ + 2k-1]
      R[nr₂ₖ + 2k-1] =      c₁ₖ  * R[nr₂ₖ + 2k-1]

      (c₂ₖ, s₂ₖ, R[nr₂ₖ₋₁ + 2k-1]) = sym_givens(R[nr₂ₖ₋₁ + 2k-1], ωₖ)  # annihilate ωₖ = r̄₂ₖ.₂ₖ₋₁
      rtmp           =      c₂ₖ  * R[nr₂ₖ + 2k-1] + s₂ₖ * R[nr₂ₖ + 2k]
      R[nr₂ₖ + 2k]   = conj(s₂ₖ) * R[nr₂ₖ + 2k-1] - c₂ₖ * R[nr₂ₖ + 2k]
      R[nr₂ₖ + 2k-1] = rtmp

      (c₃ₖ, s₃ₖ, R[nr₂ₖ + 2k]) = sym_givens(R[nr₂ₖ + 2k], θₖ)  # annihilate Θₖ = r̄₂ₖ₊₂.₂ₖ

      (c₄ₖ, s₄ₖ, R[nr₂ₖ + 2k]) = sym_givens(R[nr₂ₖ + 2k], Haux)  # annihilate hₖ₊₁.ₖ

      # Update t̄ₖ = (τ₁, ..., τ₂ₖ, τbar₂ₖ₊₁, τbar₂ₖ₊₂).
      #
      # [ 1                ][ 1                ][ c₂.ₖ  s₂.ₖ       ][ c₁.ₖ        s₁.ₖ ] [ τbar₂ₖ₋₁ ]   [ τ₂ₖ₋₁    ]
      # [    c₄.ₖ  s₄.ₖ    ][    c₃.ₖ     s₃.ₖ ][ s̄₂.ₖ -c₂.ₖ       ][       1          ] [ τbar₂ₖ   ] = [ τ₂ₖ      ]
      # [    s̄₄.ₖ -c₄.ₖ    ][          1       ][             1    ][          1       ] [          ]   [ τbar₂ₖ₊₁ ]
      # [                1 ][    s̄₃.ₖ    -c₃.ₖ ][                1 ][ s̄₁.ₖ       -c₁.ₖ ] [          ]   [ τbar₂ₖ₊₂ ]
      τbar₂ₖ₊₂ = conj(s₁ₖ) * zt[2k-1]
      zt[2k-1] =      c₁ₖ  * zt[2k-1]

      τtmp     =      c₂ₖ  * zt[2k-1] + s₂ₖ * zt[2k]
      zt[2k]   = conj(s₂ₖ) * zt[2k-1] - c₂ₖ * zt[2k]
      zt[2k-1] = τtmp

      τtmp     =      c₃ₖ  * zt[2k] + s₃ₖ * τbar₂ₖ₊₂
      τbar₂ₖ₊₂ = conj(s₃ₖ) * zt[2k] - c₃ₖ * τbar₂ₖ₊₂
      zt[2k]   = τtmp

      τbar₂ₖ₊₁ = conj(s₄ₖ) * zt[2k]
      zt[2k]   =      c₄ₖ  * zt[2k]

      # Update gc and gs vectors
      gc[4k-3], gc[4k-2], gc[4k-1], gc[4k] = c₁ₖ, c₂ₖ, c₃ₖ, c₄ₖ
      gs[4k-3], gs[4k-2], gs[4k-1], gs[4k] = s₁ₖ, s₂ₖ, s₃ₖ, s₄ₖ

      # Compute ‖rₖ‖² = |τbar₂ₖ₊₁|² + |τbar₂ₖ₊₂|²
      rNorm = sqrt(abs2(τbar₂ₖ₊₁) + abs2(τbar₂ₖ₊₂))
      history && push!(rNorms, rNorm)

      # Update the number of coefficients in Rₖ.
      nr = nr + 4k-1

      # Stopping conditions that do not depend on user input.
      # This is to guard against tolerances that are unreasonably small.
      resid_decrease_mach = (rNorm + one(T) ≤ one(T))

      # Update stopping criterion.
      user_requested_exit = callback(solver) :: Bool
      resid_decrease_lim = rNorm ≤ ε
      breakdown = Faux ≤ btol && Haux ≤ btol
      solved = resid_decrease_lim || resid_decrease_mach
      tired = iter ≥ itmax
      timer = time_ns() - start_time
      overtimed = timer > timemax_ns
      kdisplay(iter, verbose) && @printf(iostream, "%5d  %7.1e  %7.1e  %7.1e  %.2fs\n", iter, rNorm, Haux, Faux, ktimer(start_time))

      # Compute vₖ₊₁ and uₖ₊₁
      if !(solved || tired || breakdown || user_requested_exit || overtimed)
        if iter ≥ mem
          push!(V, S(undef, m))
          push!(U, S(undef, n))
          push!(zt, zero(FC), zero(FC))
        end

        # hₖ₊₁.ₖ ≠ 0
        if Haux > btol
          V[k+1] .= q ./ Haux  # hₖ₊₁.ₖvₖ₊₁ = q
        else
          # Breakdown -- hₖ₊₁.ₖ = ‖q‖₂ = 0 and Auₖ ∈ Span{v₁, ..., vₖ}
          V[k+1] .= zero(FC)  # vₖ₊₁ = 0 such that vₖ₊₁ ⊥ Span{v₁, ..., vₖ}
        end

        # fₖ₊₁.ₖ ≠ 0
        if Faux > btol
          U[k+1] .= p ./ Faux  # fₖ₊₁.ₖuₖ₊₁ = p
        else
          # Breakdown -- fₖ₊₁.ₖ = ‖p‖₂ = 0 and Bvₖ ∈ Span{u₁, ..., uₖ}
          U[k+1] .= zero(FC)  # uₖ₊₁ = 0 such that uₖ₊₁ ⊥ Span{u₁, ..., uₖ}
        end

        zt[2k+1] = τbar₂ₖ₊₁
        zt[2k+2] = τbar₂ₖ₊₂
      end
    end
    (verbose > 0) && @printf(iostream, "\n")

    # Compute zₖ = (ζ₁, ..., ζ₂ₖ) by solving Rₖzₖ = tₖ with backward substitution.
    for i = 2iter : -1 : 1
      pos = nr + i - 2iter              # position of rᵢ.ₖ
      for j = 2iter : -1 : i+1
        zt[i] = zt[i] - R[pos] * zt[j]  # ζᵢ ← ζᵢ - rᵢ.ⱼζⱼ
        pos = pos - j + 1               # position of rᵢ.ⱼ₋₁
      end
      # Rₖ can be singular if the system is inconsistent
      if abs(R[pos]) ≤ btol
        zt[i] = zero(FC)
        inconsistent = true
      else
        zt[i] = zt[i] / R[pos]          # ζᵢ ← ζᵢ / rᵢ.ᵢ
      end
    end

    # Compute xₖ and yₖ
    for i = 1 : iter
      @kaxpy!(m, zt[2i-1], V[i], x)  # xₖ = ζ₁v₁ + ζ₃v₂ + ••• + ζ₂ₖ₋₁vₖ
      @kaxpy!(n, zt[2i]  , U[i], y)  # xₖ = ζ₂u₁ + ζ₄u₂ + ••• + ζ₂ₖuₖ
    end
    if !EisI
      wB .= x
      mulorldiv!(x, E, wB, ldiv)
    end
    if !FisI
      wA .= y
      mulorldiv!(y, F, wA, ldiv)
    end
    warm_start && @kaxpy!(m, one(FC), Δx, x)
    warm_start && @kaxpy!(n, one(FC), Δy, y)
    solver.warm_start = false

    # Termination status
    tired               && (status = "maximum number of iterations exceeded")
    solved              && (status = "solution good enough given atol and rtol")
    inconsistent        && (status = "found approximate least-squares solution")
    user_requested_exit && (status = "user-requested exit")
    overtimed           && (status = "time limit exceeded")

    # Update stats
    stats.niter = iter
    stats.solved = solved
    stats.inconsistent = inconsistent
    stats.storage = sizeof(solver)
    stats.timer = ktimer(start_time)
    stats.status = status
    return solver
  end
end
