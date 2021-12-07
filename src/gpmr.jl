# An implementation of GPMR for the solution of unsymmetric partitioned linear systems.
#
# This method is described in
#
# A. Montoison and D. Orban
# GPMR: An Iterative Method for Unsymmetric Partitioned Linear Systems
# Cahier du GERAD G-2021-62.
#
# Alexis Montoison, <alexis.montoison@polymtl.ca>
# Montréal, August 2021.

export gpmr, gpmr!

"""
    (x, y, stats) = gpmr(A, B, b::AbstractVector{FC}, c::AbstractVector{FC};
                         C=I, D=I, E=I, F=I, atol::T=√eps(T), rtol::T=√eps(T),
                         gsp::Bool=false, reorthogonalization::Bool=false, itmax::Int=0,
                         restart::Bool=false, λ::T=one(T), μ::T=one(T),
                         verbose::Int=0, history::Bool=false)

`T` is an `AbstractFloat` such as `Float32`, `Float64` or `BigFloat`.
`FC` is `T` or `Complex{T}`.

GPMR solves the unsymmetric partitioned linear system

    [ λI   A ] [ x ] = [ b ]
    [  B  μI ] [ y ]   [ c ],

where λ and μ are real numbers.
`A` can have any shape and `B` has the shape of `Aᵀ`.
`A`, `B`, `b` and `c` must be all nonzero.

This implementation allows left and right block diagonal preconditioners

    [ C    ] [ λM   A ] [ E    ] [ E⁻¹x ] = [ Cb ]
    [    D ] [  B  μN ] [    F ] [ F⁻¹y ]   [ Dc ],

and can solve

    [ λM   A ] [ x ] = [ b ]
    [  B  μN ] [ y ]   [ c ]

when `CE = M⁻¹` and `DF = N⁻¹`.

By default, GPMR solves unsymmetric linear systems with `λ = 1` and `μ = 1`.
If `gsp = true`, `λ = 1`, `μ = 0` and the associated generalized saddle point system is solved.
`λ` and `μ` are also keyword arguments that can be directly modified for more specific problems.

GPMR is based on the orthogonal Hessenberg reduction process and its relations with the block-Arnoldi process.
The residual norm ‖rₖ‖ is monotonically decreasing in GPMR.

GPMR stops when `itmax` iterations are reached or when `‖rₖ‖ ≤ atol + ‖r₀‖ * rtol`.
`atol` is an absolute tolerance and `rtol` is a relative tolerance.

Additional details can be displayed if verbose mode is enabled (verbose > 0).
Information will be displayed every `verbose` iterations.

#### Reference

* A. Montoison and D. Orban, [*GPMR: An Iterative Method for Unsymmetric Partitioned Linear Systems*](https://dx.doi.org/10.13140/RG.2.2.24069.68326), Cahier du GERAD G-2021-62, GERAD, Montréal, 2021.
"""
function gpmr(A, B, b :: AbstractVector{FC}, c :: AbstractVector{FC}; memory :: Int=20, kwargs...) where FC <: FloatOrComplex
  solver = GpmrSolver(A, b, memory)
  gpmr!(solver, A, B, b, c; kwargs...)
  return (solver.x, solver.y, solver.stats)
end

"""
    solver = gpmr!(solver::GpmrSolver, args...; kwargs...)

where `args` and `kwargs` are arguments and keyword arguments of [`gpmr`](@ref).

See [`GpmrSolver`](@ref) for more details about the `solver`.
"""
function gpmr!(solver :: GpmrSolver{T,FC,S}, A, B, b :: AbstractVector{FC}, c :: AbstractVector{FC};
               C=I, D=I, E=I, F=I, atol :: T=√eps(T), rtol :: T=√eps(T),
               gsp :: Bool=false, reorthogonalization :: Bool=false, itmax :: Int=0,
               restart :: Bool=false, λ :: T=one(T), μ :: T=one(T),
               verbose :: Int=0, history::Bool=false) where {T <: AbstractFloat, FC <: FloatOrComplex{T}, S <: DenseVector{FC}}

  m, n = size(A)
  s, t = size(B)
  m == t         || error("Inconsistent problem size")
  s == n         || error("Inconsistent problem size")
  length(b) == m || error("Inconsistent problem size")
  length(c) == n || error("Inconsistent problem size")
  (verbose > 0) && @printf("GPMR: system of %d equations in %d variables\n", m+n, m+n)

  # Check C = E = Iₘ and D = F = Iₙ
  CisI = (C === I)
  DisI = (D === I)
  EisI = (E === I)
  FisI = (F === I)

  # Check type consistency
  eltype(A) == T || error("eltype(A) ≠ $T")
  eltype(B) == T || error("eltype(B) ≠ $T")
  ktypeof(b) == S || error("ktypeof(b) ≠ $S")
  ktypeof(c) == S || error("ktypeof(c) ≠ $S")
  restart && (λ ≠ 0) && (!CisI || !EisI) && error("Restart with preconditioners is not supported.")
  restart && (μ ≠ 0) && (!DisI || !FisI) && error("Restart with preconditioners is not supported.")

  # Set up workspace.
  allocate_if(!CisI  , solver, :q , S, m)
  allocate_if(!DisI  , solver, :p , S, n)
  allocate_if(!EisI  , solver, :wB, S, m)
  allocate_if(!FisI  , solver, :wA, S, n)
  allocate_if(restart, solver, :Δx, S, m)
  allocate_if(restart, solver, :Δy, S, n)
  wA, wB, dA, dB, Δx, Δy = solver.wA, solver.wB, solver.dA, solver.dB, solver.Δx, solver.Δy
  x, y, V, U, gs, gc = solver.x, solver.y, solver.V, solver.U, solver.gs, solver.gc
  zt, R, stats = solver.zt, solver.R, solver.stats
  rNorms = stats.residuals
  reset!(stats)
  b₀ = restart ? dA : b
  c₀ = restart ? dB : c
  q  = CisI ? dA : solver.q
  p  = DisI ? dB : solver.p

  # Initial solutions x₀ and y₀.
  restart && (Δx .= x)
  restart && (Δy .= y)
  x .= zero(T)
  y .= zero(T)

  iter = 0
  itmax == 0 && (itmax = m+n)

  # Initialize workspace.
  nr = 0           # Number of coefficients stored in Rₖ
  mem = length(V)  # Memory
  ωₖ = zero(T)     # Auxiliary variable to store fₖₖ
  for i = 1 : mem
    V[i] .= zero(T)
    U[i] .= zero(T)
  end
  gs .= zero(T)  # Givens sines used for the factorization QₖRₖ = Sₖ₊₁.ₖ.
  gc .= zero(T)  # Givens cosines used for the factorization QₖRₖ = Sₖ₊₁.ₖ.
  R  .= zero(T)  # Upper triangular matrix Rₖ.
  zt .= zero(T)  # Rₖzₖ = tₖ with (tₖ, τbar₂ₖ₊₁, τbar₂ₖ₊₂) = (Qₖ)ᵀ(βe₁ + γe₂).

  # [ λI   A ] [ xₖ ] = [ b - λΔx - AΔy ] = [ b₀ ]
  # [  B  μI ] [ yₖ ]   [ c - BΔx - μΔy ]   [ c₀ ]
  if restart
    mul!(b₀, A, Δy)
    (λ ≠ 0) && @kaxpy!(m, λ, Δx, b₀)
    @kaxpby!(m, one(T), b, -one(T), b₀)
    mul!(c₀, B, Δx)
    (μ ≠ 0) && @kaxpy!(n, μ, Δy, c₀)
    @kaxpby!(n, one(T), c, -one(T), c₀)
  end

  # Initialize the orthogonal Hessenberg reduction process.
  # βv₁ = Cb
  if !CisI
    mul!(q, C, b₀)
    b₀ = q
  end
  β = @knrm2(m, b₀)
  β ≠ 0 || error("b must be nonzero")
  @. V[1] = b₀ / β

  # γu₁ = Dc
  if !DisI
    mul!(p, D, c₀)
    c₀ = p
  end
  γ = @knrm2(n, c₀)
  γ ≠ 0 || error("c must be nonzero")
  @. U[1] = c₀ / γ

  # Compute ‖r₀‖² = γ² + β²
  rNorm = sqrt(γ^2 + β^2)
  history && push!(rNorms, rNorm)
  ε = atol + rtol * rNorm

  # Initialize t̄₀
  zt[1] = β
  zt[2] = γ

  (verbose > 0) && @printf("%5s  %7s  %7s  %7s\n", "k", "‖rₖ‖", "hₖ₊₁.ₖ", "fₖ₊₁.ₖ")
  display(iter, verbose) && @printf("%5d  %7.1e  %7s  %7s\n", iter, rNorm, "✗ ✗ ✗ ✗", "✗ ✗ ✗ ✗")

  # Determine λ and μ associated to generalized saddle point systems.
  gsp && (λ = one(T) ; μ = zero(T))

  # Stopping criterion.
  breakdown = false
  solved = rNorm ≤ ε
  tired = iter ≥ itmax
  status = "unknown"

  while !(solved || tired || breakdown)

    # Update iteration index.
    iter = iter + 1
    k = iter
    nr₂ₖ₋₁ = nr       # Position of the column 2k-1 in Rₖ.
    nr₂ₖ = nr + 2k-1  # Position of the column 2k in Rₖ.

    # Update workspace if more storage is required
    if iter > mem
      for i = 1 : 4k-1
        push!(R, zero(T))
      end
      for i = 1 : 4
        push!(gs, zero(T))
        push!(gc, zero(T))
      end
    end

    # Continue the orthogonal Hessenberg reduction process.
    # CAFUₖ = VₖHₖ + hₖ₊₁.ₖ * vₖ₊₁(eₖ)ᵀ = Vₖ₊₁Hₖ₊₁.ₖ
    # DBEVₖ = UₖFₖ + fₖ₊₁.ₖ * uₖ₊₁(eₖ)ᵀ = Uₖ₊₁Fₖ₊₁.ₖ
    wA = FisI ? U[iter] : solver.wA
    wB = EisI ? V[iter] : solver.wB
    FisI || mul!(wA, F, U[iter])  # wA = Fuₖ
    EisI || mul!(wB, E, V[iter])  # wB = Evₖ
    mul!(dA, A, wA)               # dA = AFuₖ
    mul!(dB, B, wB)               # dB = BEvₖ
    CisI || mul!(q, C, dA)        # q  = CAFuₖ
    DisI || mul!(p, D, dB)        # p  = DBEvₖ

    for i = 1 : iter
      hᵢₖ = @kdot(m, V[i], q)    # hᵢ.ₖ = vᵢAuₖ
      fᵢₖ = @kdot(n, U[i], p)    # fᵢ.ₖ = uᵢBvₖ
      @kaxpy!(m, -hᵢₖ, V[i], q)  # q ← q - hᵢ.ₖvᵢ
      @kaxpy!(n, -fᵢₖ, U[i], p)  # p ← p - fᵢ.ₖuᵢ
      R[nr₂ₖ + 2i-1] = hᵢₖ
      (i < iter) ? R[nr₂ₖ₋₁ + 2i] = fᵢₖ : ωₖ = fᵢₖ
    end

    # Reorthogonalization of the Krylov basis.
    if reorthogonalization
      for i = 1 : iter
        Htmp = @kdot(m, V[i], q)    # hₜₘₚ = qᵀvᵢ
        Ftmp = @kdot(n, U[i], p)    # fₜₘₚ = pᵀuᵢ
        @kaxpy!(m, -Htmp, V[i], q)  # q ← q - hₜₘₚvᵢ
        @kaxpy!(n, -Ftmp, U[i], p)  # p ← p - fₜₘₚuᵢ
        R[nr₂ₖ + 2i-1] += Htmp                               # hᵢ.ₖ = hᵢ.ₖ + hₜₘₚ
        (iter < iter) ? R[nr₂ₖ₋₁ + 2i] += Ftmp : ωₖ += Ftmp  # fᵢ.ₖ = fᵢ.ₖ + fₜₘₚ
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
    # [    c₄.ᵢ  s₄.ᵢ    ][    c₃.ᵢ     s₃.ᵢ ][ s₂.ᵢ -c₂.ᵢ       ][       1          ] [ r̄₂ᵢ.₂ₖ₋₁    r̄₂ᵢ.₂ₖ   ] = [ r₂ᵢ.₂ₖ₋₁    r₂ᵢ.₂ₖ   ]
    # [    s₄.ᵢ -c₄.ᵢ    ][          1       ][             1    ][          1       ] [ ρ           hᵢ₊₁.ₖ   ]   [ r̄₂ᵢ₊₁.₂ₖ₋₁  r̄₂ᵢ₊₁.₂ₖ ]
    # [                1 ][    s₃.ᵢ    -c₃.ᵢ ][                1 ][ s₁.ᵢ       -c₁.ᵢ ] [ fᵢ₊₁.ₖ      δ        ]   [ r̄₂ᵢ₊₂.₂ₖ₋₁  r̄₂ᵢ₊₂.₂ₖ ]
    #
    # r̄₁.₂ₖ₋₁ = 0, r̄₁.₂ₖ = h₁.ₖ, r̄₂.₂ₖ₋₁ = f₁.ₖ and r̄₂.₂ₖ = 0.
    # (ρ, δ) = (λ, μ) if i == k-1, (ρ, δ) = (0, 0) otherwise.
    for i = 1 : iter-1
      for nrcol ∈ (nr₂ₖ₋₁, nr₂ₖ)
        flag = (i == iter-1 && nrcol == nr₂ₖ₋₁)
        αₖ = flag ? ωₖ : R[nrcol + 2i+2]

        c₁ᵢ = gc[4i-3]
        s₁ᵢ = gs[4i-3]
        rtmp            = c₁ᵢ * R[nrcol + 2i-1] + s₁ᵢ * αₖ
        αₖ              = s₁ᵢ * R[nrcol + 2i-1] - c₁ᵢ * αₖ
        R[nrcol + 2i-1] = rtmp

        c₂ᵢ = gc[4i-2]
        s₂ᵢ = gs[4i-2]
        rtmp            = c₂ᵢ * R[nrcol + 2i-1] + s₂ᵢ * R[nrcol + 2i]
        R[nrcol + 2i]   = s₂ᵢ * R[nrcol + 2i-1] - c₂ᵢ * R[nrcol + 2i]
        R[nrcol + 2i-1] = rtmp

        c₃ᵢ = gc[4i-1]
        s₃ᵢ = gs[4i-1]
        rtmp          = c₃ᵢ * R[nrcol + 2i] + s₃ᵢ * αₖ
        αₖ            = s₃ᵢ * R[nrcol + 2i] - c₃ᵢ * αₖ
        R[nrcol + 2i] = rtmp

        c₄ᵢ = gc[4i]
        s₄ᵢ = gs[4i]
        rtmp            = c₄ᵢ * R[nrcol + 2i] + s₄ᵢ * R[nrcol + 2i+1]
        R[nrcol + 2i+1] = s₄ᵢ * R[nrcol + 2i] - c₄ᵢ * R[nrcol + 2i+1]
        R[nrcol + 2i]   = rtmp

        flag ? ωₖ = αₖ : R[nrcol + 2i+2] = αₖ
      end
    end

    # Compute and apply current givens reflections
    # [ 1                ][ 1                ][ c₂.ₖ  s₂.ₖ       ][ c₁.ₖ        s₁.ₖ ] [ r̄₂ₖ₋₁.₂ₖ₋₁  r̄₂ₖ₋₁.₂ₖ ]    [ r₂ₖ₋₁.₂ₖ₋₁  r₂ₖ₋₁.₂ₖ ]
    # [    c₄.ₖ  s₄.ₖ    ][    c₃.ₖ     s₃.ₖ ][ s₂.ₖ -c₂.ₖ       ][       1          ] [ r̄₂ₖ.₂ₖ₋₁    r̄₂ₖ.₂ₖ   ] =  [             r₂ₖ.₂ₖ   ]
    # [    s₄.ₖ -c₄.ₖ    ][          1       ][             1    ][          1       ] [             hₖ₊₁.ₖ   ]    [                      ]
    # [                1 ][    s₃.ₖ    -c₃.ₖ ][                1 ][ s₁.ₖ       -c₁.ₖ ] [ fₖ₊₁.ₖ               ]    [                      ]
    (c₁ₖ, s₁ₖ, R[nr₂ₖ₋₁ + 2k-1]) = sym_givens(R[nr₂ₖ₋₁ + 2k-1], Faux)  # annihilate fₖ₊₁.ₖ
    θₖ             = s₁ₖ * R[nr₂ₖ + 2k-1]
    R[nr₂ₖ + 2k-1] = c₁ₖ * R[nr₂ₖ + 2k-1]

    (c₂ₖ, s₂ₖ, R[nr₂ₖ₋₁ + 2k-1]) = sym_givens(R[nr₂ₖ₋₁ + 2k-1], ωₖ)  # annihilate ωₖ = r̄₂ₖ.₂ₖ₋₁
    rtmp           = c₂ₖ * R[nr₂ₖ + 2k-1] + s₂ₖ * R[nr₂ₖ + 2k]
    R[nr₂ₖ + 2k]   = s₂ₖ * R[nr₂ₖ + 2k-1] - c₂ₖ * R[nr₂ₖ + 2k]
    R[nr₂ₖ + 2k-1] = rtmp

    (c₃ₖ, s₃ₖ, R[nr₂ₖ + 2k]) = sym_givens(R[nr₂ₖ + 2k], θₖ)  # annihilate Θₖ = r̄₂ₖ₊₂.₂ₖ

    (c₄ₖ, s₄ₖ, R[nr₂ₖ + 2k]) = sym_givens(R[nr₂ₖ + 2k], Haux)  # annihilate hₖ₊₁.ₖ

    # Update t̄ₖ = (τ₁, ..., τ₂ₖ, τbar₂ₖ₊₁, τbar₂ₖ₊₂).
    #
    # [ 1                ][ 1                ][ c₂.ₖ  s₂.ₖ       ][ c₁.ₖ        s₁.ₖ ] [ τbar₂ₖ₋₁ ]   [ τ₂ₖ₋₁    ]
    # [    c₄.ₖ  s₄.ₖ    ][    c₃.ₖ     s₃.ₖ ][ s₂.ₖ -c₂.ₖ       ][       1          ] [ τbar₂ₖ   ] = [ τ₂ₖ      ]
    # [    s₄.ₖ -c₄.ₖ    ][          1       ][             1    ][          1       ] [          ]   [ τbar₂ₖ₊₁ ]
    # [                1 ][    s₃.ₖ    -c₃.ₖ ][                1 ][ s₁.ₖ       -c₁.ₖ ] [          ]   [ τbar₂ₖ₊₂ ]
    τbar₂ₖ₊₂ = s₁ₖ * zt[2k-1]
    zt[2k-1] = c₁ₖ * zt[2k-1]

    τtmp     = c₂ₖ * zt[2k-1] + s₂ₖ * zt[2k]
    zt[2k]   = s₂ₖ * zt[2k-1] - c₂ₖ * zt[2k]
    zt[2k-1] = τtmp

    τtmp     = c₃ₖ * zt[2k] + s₃ₖ * τbar₂ₖ₊₂
    τbar₂ₖ₊₂ = s₃ₖ * zt[2k] - c₃ₖ * τbar₂ₖ₊₂
    zt[2k]   = τtmp

    τbar₂ₖ₊₁ = s₄ₖ * zt[2k]
    zt[2k]   = c₄ₖ * zt[2k]

    # Update gc and gs vectors
    gc[4k-3], gc[4k-2], gc[4k-1], gc[4k] = c₁ₖ, c₂ₖ, c₃ₖ, c₄ₖ
    gs[4k-3], gs[4k-2], gs[4k-1], gs[4k] = s₁ₖ, s₂ₖ, s₃ₖ, s₄ₖ

    # Compute ‖rₖ‖² = (τbar₂ₖ₊₁)² + (τbar₂ₖ₊₂)²
    rNorm = sqrt(τbar₂ₖ₊₁^2 + τbar₂ₖ₊₂^2)
    history && push!(rNorms, rNorm)

    # Update the number of coefficients in Rₖ.
    nr = nr + 4k-1

    # Update stopping criterion.
    breakdown = Faux ≤ eps(T) && Haux ≤ eps(T)
    solved = rNorm ≤ ε
    tired = iter ≥ itmax
    display(iter, verbose) && @printf("%5d  %7.1e  %7.1e  %7.1e\n", iter, rNorm, Haux, Faux)

    # Compute vₖ₊₁ and uₖ₊₁
    if !(solved || tired || breakdown)
      if iter ≥ mem
        push!(V, S(undef, m))
        push!(U, S(undef, n))
        push!(zt, zero(T), zero(T))
      end

      # hₖ₊₁.ₖ ≠ 0
      if Haux > eps(T)
        @. V[k+1] = q / Haux  # hₖ₊₁.ₖvₖ₊₁ = q
      else
        # Breakdown -- hₖ₊₁.ₖ = ‖q‖₂ = 0 and Auₖ ∈ Span{v₁, ..., vₖ}
        V[k+1] .= zero(T)  # vₖ₊₁ = 0 such that vₖ₊₁ ⊥ Span{v₁, ..., vₖ}
      end

      # fₖ₊₁.ₖ ≠ 0
      if Faux > eps(T)
        @. U[k+1] = p / Faux  # fₖ₊₁.ₖuₖ₊₁ = p
      else
        # Breakdown -- fₖ₊₁.ₖ = ‖p‖₂ = 0 and Bvₖ ∈ Span{u₁, ..., uₖ}
        U[k+1] .= zero(T)  # uₖ₊₁ = 0 such that uₖ₊₁ ⊥ Span{u₁, ..., uₖ}
      end

      zt[2k+1] = τbar₂ₖ₊₁
      zt[2k+2] = τbar₂ₖ₊₂
    end
  end
  (verbose > 0) && @printf("\n")

  # Compute zₖ = (ζ₁, ..., ζ₂ₖ) by solving Rₖzₖ = tₖ with backward substitution.
  for i = 2iter : -1 : 1
    pos = nr + i - 2iter              # position of rᵢ.ₖ
    for j = 2iter : -1 : i+1
      zt[i] = zt[i] - R[pos] * zt[j]  # ζᵢ ← ζᵢ - rᵢ.ⱼζⱼ
      pos = pos - j + 1               # position of rᵢ.ⱼ₋₁
    end
    # Rₖ can be singular is the system is inconsistent
    if R[pos] ≤ eps(T)
      zt[i] = zero(T)
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
    mul!(x, E, wB)
  end
  if !FisI
    wA .= y
    mul!(y, F, wA)
  end
  restart && @kaxpy!(m, one(T), Δx, x)
  restart && @kaxpy!(n, one(T), Δy, y)

  tired     && (status = "maximum number of iterations exceeded")
  breakdown && (status = "found approximate least-squares solution")
  solved    && (status = "solution good enough given atol and rtol")

  # Update stats
  stats.solved = solved
  stats.inconsistent = !solved && breakdown
  stats.status = status
  return solver
end
