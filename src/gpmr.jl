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
    (x, y, stats) = gpmr(A, B, b::AbstractVector{FC}, c::AbstractVector{FC}; memory::Int=20,
                         C=I, D=I, E=I, F=I, atol::T=√eps(T), rtol::T=√eps(T),
                         gsp::Bool=false, reorthogonalization::Bool=false,
                         itmax::Int=0, λ::FC=one(FC), μ::FC=one(FC),
                         restart::Bool=false, verbose::Int=0, history::Bool=false)

`T` is an `AbstractFloat` such as `Float32`, `Float64` or `BigFloat`.
`FC` is `T` or `Complex{T}`.

GPMR solves the unsymmetric partitioned linear system

    [ λI   A ] [ x ] = [ b ]
    [  B  μI ] [ y ]   [ c ],

where λ and μ are real or complex numbers.
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

Full reorthogonalization is available with the `reorthogonalization` option.

If `restart = true`, the restarted version GPMR(k) is used with `k = memory`.
If `restart = false`, the parameter `memory` should be used as a hint of the number of iterations to limit dynamic allocations.
More storage will be allocated only if the number of iterations exceed `memory`.

Additional details can be displayed if verbose mode is enabled (verbose > 0).
Information will be displayed every `verbose` iterations.

GPMR can be warm-started from initial guesses `x0` and `y0` with the method

    (x, y, stats) = gpmr(A, B, b, c, x0, y0; kwargs...)

where `kwargs` are the same keyword arguments as above.

#### Reference

* A. Montoison and D. Orban, [*GPMR: An Iterative Method for Unsymmetric Partitioned Linear Systems*](https://dx.doi.org/10.13140/RG.2.2.24069.68326), Cahier du GERAD G-2021-62, GERAD, Montréal, 2021.
"""
function gpmr end

function gpmr(A, B, b :: AbstractVector{FC}, c :: AbstractVector{FC}, x0 :: AbstractVector, y0 :: AbstractVector; memory :: Int=20, kwargs...) where FC <: FloatOrComplex
  solver = GpmrSolver(A, b, memory)
  gpmr!(solver, A, B, b, c, x0, y0; kwargs...)
  return (solver.x, solver.y, solver.stats)
end

function gpmr(A, B, b :: AbstractVector{FC}, c :: AbstractVector{FC}; memory :: Int=20, kwargs...) where FC <: FloatOrComplex
  solver = GpmrSolver(A, b, memory)
  gpmr!(solver, A, B, b, c; kwargs...)
  return (solver.x, solver.y, solver.stats)
end

"""
    solver = gpmr!(solver::GpmrSolver, A, B, b, c; kwargs...)
    solver = gpmr!(solver::GpmrSolver, A, B, b, c, x0, y0; kwargs...)

where `kwargs` are keyword arguments of [`gpmr`](@ref).

Note that the `memory` keyword argument is the only exception.
It's required to create a `GpmrSolver` and can't be changed later.

See [`GpmrSolver`](@ref) for more details about the `solver`.
"""
function gpmr! end

function gpmr!(solver :: GpmrSolver{T,FC,S}, A, B, b :: AbstractVector{FC}, c :: AbstractVector{FC},
                x0 :: AbstractVector, y0 :: AbstractVector; kwargs...) where {T <: AbstractFloat, FC <: FloatOrComplex{T}, S <: DenseVector{FC}}
  warm_start!(solver, x0, y0)
  gpmr!(solver, A, B, b, c; kwargs...)
  return solver
end

function gpmr!(solver :: GpmrSolver{T,FC,S}, A, B, b :: AbstractVector{FC}, c :: AbstractVector{FC};
               C=I, D=I, E=I, F=I, atol :: T=√eps(T), rtol :: T=√eps(T),
               gsp :: Bool=false, reorthogonalization :: Bool=false,
               itmax :: Int=0, λ :: FC=one(FC), μ :: FC=one(FC),
               restart :: Bool=false, verbose :: Int=0, history::Bool=false) where {T <: AbstractFloat, FC <: FloatOrComplex{T}, S <: DenseVector{FC}}

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
  eltype(A) == FC || error("eltype(A) ≠ $FC")
  eltype(B) == FC || error("eltype(B) ≠ $FC")
  ktypeof(b) == S || error("ktypeof(b) ≠ $S")
  ktypeof(c) == S || error("ktypeof(c) ≠ $S")

  # Determine λ and μ associated to generalized saddle point systems.
  gsp && (λ = one(FC) ; μ = zero(FC))

  warm_start = solver.warm_start
  warm_start && (λ ≠ 0) && !EisI && error("Warm-start with right preconditioners is not supported.")
  warm_start && (μ ≠ 0) && !FisI && error("Warm-start with right preconditioners is not supported.")

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
  b₀ = warm_start ? dA : b
  c₀ = warm_start ? dB : c
  q  = CisI ? dA : solver.q
  p  = DisI ? dB : solver.p
  xr = restart ? Δx : x
  yr = restart ? Δy : y

  # Initial solutions x₀ and y₀.
  x .= zero(FC)
  y .= zero(FC)

  # Warm-start
  # If λ ≠ 0, Cb₀ = Cb - CAΔy - λΔx because CM = Iₘ and E = Iₘ
  # E ≠ Iₘ is only allowed when λ = 0 because E⁻¹Δx can't be computed to use CME = Iₘ
  # Compute C(b - AΔy) - λΔx
  warm_start && mul!(b₀, A, Δy)
  warm_start && @kaxpby!(m, one(FC), b, -one(FC), b₀)
  !CisI && mul!(q, C, b₀)
  !CisI && (b₀ = q)
  warm_start && (λ ≠ 0) && @kaxpy!(m, -λ, Δx, b₀)

  # If μ ≠ 0, Dc₀ = Dc - DBΔx - μΔy because DN = Iₙ and F = Iₙ
  # F ≠ Iₙ is only allowed when μ = 0 because F⁻¹Δy can't be computed to use DNF = Iₘ
  # Compute D(c - BΔx) - μΔy
  warm_start && mul!(c₀, B, Δx)
  warm_start && @kaxpby!(n, one(FC), c, -one(FC), c₀)
  !DisI && mul!(p, D, c₀)
  !DisI && (c₀ = p)
  warm_start && (μ ≠ 0) && @kaxpy!(n, -μ, Δy, c₀)

  warm_start && restart && @kaxpy!(m, one(FC), Δx, x)
  warm_start && restart && @kaxpy!(n, one(FC), Δy, y)

  # Compute ‖r₀‖² = γ² + β²
  β = @knrm2(m, b₀)
  β ≠ 0 || error("b must be nonzero")
  γ = @knrm2(n, c₀)
  γ ≠ 0 || error("c must be nonzero")
  rNorm = sqrt(γ^2 + β^2)
  history && push!(rNorms, rNorm)
  ε = atol + rtol * rNorm

  mem = length(V)  # Memory
  npass = 0        # Number of pass

  iter = 0        # Cumulative number of iterations
  inner_iter = 0  # Number of iterations in a pass

  itmax == 0 && (itmax = m+n)
  inner_itmax = itmax

  (verbose > 0) && @printf("%5s  %5s  %7s  %7s  %7s\n", "pass", "k", "‖rₖ‖", "hₖ₊₁.ₖ", "fₖ₊₁.ₖ")
  kdisplay(iter, verbose) && @printf("%5d  %5d  %7.1e  %7s  %7s\n", npass, iter, rNorm, "✗ ✗ ✗ ✗", "✗ ✗ ✗ ✗")

  # Tolerance for breakdown detection.
  btol = eps(T)^(3/4)

  # Stopping criterion.
  breakdown = false
  inconsistent = false
  solved = rNorm ≤ ε
  tired = iter ≥ itmax
  inner_tired = inner_iter ≥ inner_itmax
  status = "unknown"

  # Ajouter br et cr au GpmrSolver
  br = restart ? copy(b₀) : b₀
  cr = restart ? copy(c₀) : c₀

  while !(solved || tired || breakdown)

    # Initialize workspace.
    nr = 0           # Number of coefficients stored in Rₖ
    ωₖ = zero(FC)    # Auxiliary variable to store fₖₖ
    for i = 1 : mem
      V[i] .= zero(FC)
      U[i] .= zero(FC)
    end
    gs .= zero(FC)  # Givens sines used for the factorization QₖRₖ = Sₖ₊₁.ₖ.
    gc .= zero(T)   # Givens cosines used for the factorization QₖRₖ = Sₖ₊₁.ₖ.
    R  .= zero(FC)  # Upper triangular matrix Rₖ.
    zt .= zero(FC)  # Rₖzₖ = tₖ with (tₖ, τbar₂ₖ₊₁, τbar₂ₖ₊₂) = (Qₖ)ᵀ(βe₁ + γe₂).

    if restart
      if npass ≥ 1
        # br = br - λE⁻¹x - CAy
        (λ ≠ 0) && @kaxpy!(m, -λ, xr, br)
        ytmp = FisI ? yr : solver.wA
        !FisI && mul!(ytmp, F, yr)
        mul!(dA, A, ytmp)
        !CisI && mul!(q, C, dA)
        @kaxpy!(m, -one(FC), q, br)

        # cr = cr - μF⁻¹y - DBx
        (μ ≠ 0) && @kaxpy!(n, -μ, yr, cr)
        xtmp = EisI ? xr : solver.wB
        !EisI && mul!(xtmp, E, xr)
        mul!(dB, B, xtmp)
        !DisI && mul!(p, D, dB)
        @kaxpy!(n, -one(FC), p, cr)

        # Update x and y
        @kaxpy!(m, one(FC), xtmp, x)
        @kaxpy!(n, one(FC), ytmp, y)
      end
      xr .= zero(FC)  # xr === Δx when restart is set to true
      yr .= zero(FC)  # yr === Δy when restart is set to true
    end

    # Initialize the orthogonal Hessenberg reduction process.
    # βv₁ = b₀
    β = @knrm2(m, br)
    @. V[1] = br / β

    # γu₁ = c₀
    γ = @knrm2(n, cr)
    @. U[1] = cr / γ

    # Initialize t̄₀
    zt[1] = β
    zt[2] = γ

    npass = npass + 1
    inner_iter = 0
    inner_tired = false

    while !(solved || inner_tired || breakdown)

      # Update iteration index.
      inner_iter = inner_iter + 1
      k = inner_iter
      nr₂ₖ₋₁ = nr       # Position of the column 2k-1 in Rₖ.
      nr₂ₖ = nr + 2k-1  # Position of the column 2k in Rₖ.

      # Update workspace if more storage is required
      if !restart && (inner_iter > mem)
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
      wA = FisI ? U[inner_iter] : solver.wA
      wB = EisI ? V[inner_iter] : solver.wB
      FisI || mul!(wA, F, U[inner_iter])  # wA = Fuₖ
      EisI || mul!(wB, E, V[inner_iter])  # wB = Evₖ
      mul!(dA, A, wA)                     # dA = AFuₖ
      mul!(dB, B, wB)                     # dB = BEvₖ
      CisI || mul!(q, C, dA)              # q  = CAFuₖ
      DisI || mul!(p, D, dB)              # p  = DBEvₖ

      for i = 1 : inner_iter
        hᵢₖ = @kdot(m, V[i], q)    # hᵢ.ₖ = vᵢAuₖ
        fᵢₖ = @kdot(n, U[i], p)    # fᵢ.ₖ = uᵢBvₖ
        @kaxpy!(m, -hᵢₖ, V[i], q)  # q ← q - hᵢ.ₖvᵢ
        @kaxpy!(n, -fᵢₖ, U[i], p)  # p ← p - fᵢ.ₖuᵢ
        R[nr₂ₖ + 2i-1] = hᵢₖ
        (i < inner_iter) ? R[nr₂ₖ₋₁ + 2i] = fᵢₖ : ωₖ = fᵢₖ
      end

      # Reorthogonalization of the Krylov basis.
      if reorthogonalization
        for i = 1 : inner_iter
          Htmp = @kdot(m, V[i], q)    # hₜₘₚ = qᵀvᵢ
          Ftmp = @kdot(n, U[i], p)    # fₜₘₚ = pᵀuᵢ
          @kaxpy!(m, -Htmp, V[i], q)  # q ← q - hₜₘₚvᵢ
          @kaxpy!(n, -Ftmp, U[i], p)  # p ← p - fₜₘₚuᵢ
          R[nr₂ₖ + 2i-1] += Htmp                                  # hᵢ.ₖ = hᵢ.ₖ + hₜₘₚ
          (i < inner_iter) ? R[nr₂ₖ₋₁ + 2i] += Ftmp : ωₖ += Ftmp  # fᵢ.ₖ = fᵢ.ₖ + fₜₘₚ
        end
      end

      Haux = @knrm2(m, q)  # hₖ₊₁.ₖ = ‖q‖₂
      Faux = @knrm2(n, p)  # fₖ₊₁.ₖ = ‖p‖₂

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
      for i = 1 : inner_iter-1
        for nrcol ∈ (nr₂ₖ₋₁, nr₂ₖ)
          flag = (i == inner_iter-1 && nrcol == nr₂ₖ₋₁)
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

      # Update stopping criterion.
      breakdown = Faux ≤ btol && Haux ≤ btol
      solved = rNorm ≤ ε
      inner_tired = restart ? inner_iter ≥ min(mem, inner_itmax) : inner_iter ≥ inner_itmax
      kdisplay(iter+inner_iter, verbose) && @printf("%5d  %5d  %7.1e  %7.1e  %7.1e\n", npass, iter+inner_iter, rNorm, Haux, Faux)

      # Compute vₖ₊₁ and uₖ₊₁
      if !(solved || inner_tired || breakdown)
        if !restart && (inner_iter ≥ mem)
          push!(V, S(undef, m))
          push!(U, S(undef, n))
          push!(zt, zero(FC), zero(FC))
        end

        # hₖ₊₁.ₖ ≠ 0
        if Haux > btol
          @. V[k+1] = q / Haux  # hₖ₊₁.ₖvₖ₊₁ = q
        else
          # Breakdown -- hₖ₊₁.ₖ = ‖q‖₂ = 0 and Auₖ ∈ Span{v₁, ..., vₖ}
          V[k+1] .= zero(FC)  # vₖ₊₁ = 0 such that vₖ₊₁ ⊥ Span{v₁, ..., vₖ}
        end

        # fₖ₊₁.ₖ ≠ 0
        if Faux > btol
          @. U[k+1] = p / Faux  # fₖ₊₁.ₖuₖ₊₁ = p
        else
          # Breakdown -- fₖ₊₁.ₖ = ‖p‖₂ = 0 and Bvₖ ∈ Span{u₁, ..., uₖ}
          U[k+1] .= zero(FC)  # uₖ₊₁ = 0 such that uₖ₊₁ ⊥ Span{u₁, ..., uₖ}
        end

        zt[2k+1] = τbar₂ₖ₊₁
        zt[2k+2] = τbar₂ₖ₊₂
      end
    end

    # Compute zₖ = (ζ₁, ..., ζ₂ₖ) by solving Rₖzₖ = tₖ with backward substitution.
    for i = 2*inner_iter : -1 : 1
      pos = nr + i - 2*inner_iter       # position of rᵢ.ₖ
      for j = 2*inner_iter : -1 : i+1
        zt[i] = zt[i] - R[pos] * zt[j]  # ζᵢ ← ζᵢ - rᵢ.ⱼζⱼ
        pos = pos - j + 1               # position of rᵢ.ⱼ₋₁
      end
      # Rₖ can be singular if the system is inconsistent
      if abs(R[pos]) ≤ btol
        zt[i] = zero(FC)
        inconsistent = true
      else
        zt[i] = zt[i] / R[pos]  # ζᵢ ← ζᵢ / rᵢ.ᵢ
      end
    end

    # Compute E⁻¹xₖ and F⁻¹yₖ
    for i = 1 : inner_iter
      @kaxpy!(m, zt[2i-1], V[i], xr)  # xₖ = ζ₁v₁ + ζ₃v₂ + ••• + ζ₂ₖ₋₁vₖ
      @kaxpy!(n, zt[2i]  , U[i], yr)  # xₖ = ζ₂u₁ + ζ₄u₂ + ••• + ζ₂ₖuₖ
    end

    # Update inner_itmax, iter and tired variables.
    inner_itmax = inner_itmax - inner_iter
    iter = iter + inner_iter
    tired = iter ≥ itmax
  end
  (verbose > 0) && @printf("\n")

  xtmp = EisI ? xr : solver.wB
  !EisI && mul!(xtmp, E, xr)
  !EisI && !restart && (x .= xtmp)
  restart && @kaxpy!(m, one(FC), xtmp, x)

  ytmp = FisI ? yr : solver.wA
  !FisI && mul!(ytmp, F, yr)
  !FisI && !restart && (y .= ytmp)
  restart && @kaxpy!(n, one(FC), ytmp, y)

  warm_start && !restart && @kaxpy!(m, one(FC), Δx, x)
  warm_start && !restart && @kaxpy!(n, one(FC), Δy, y)
  solver.warm_start = false

  tired        && (status = "maximum number of iterations exceeded")
  solved       && (status = "solution good enough given atol and rtol")
  inconsistent && (status = "found approximate least-squares solution")

  # Update stats
  stats.niter = iter
  stats.solved = solved
  stats.inconsistent = inconsistent
  stats.status = status
  return solver
end
