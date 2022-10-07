# An implementation of BILQR for the solution of square
# consistent linear adjoint systems Ax = b and Aᴴy = c.
#
# This method is described in
#
# A. Montoison and D. Orban
# BiLQ: An Iterative Method for Nonsymmetric Linear Systems with a Quasi-Minimum Error Property.
# SIAM Journal on Matrix Analysis and Applications, 41(3), pp. 1145--1166, 2020.
#
# Alexis Montoison, <alexis.montoison@polymtl.ca>
# Montreal, July 2019.

export bilqr, bilqr!

"""
    (x, y, stats) = bilqr(A, b::AbstractVector{FC}, c::AbstractVector{FC};
                          atol::T=√eps(T), rtol::T=√eps(T), transfer_to_bicg::Bool=true,
                          itmax::Int=0, verbose::Int=0, history::Bool=false,
                          callback=solver->false)

`T` is an `AbstractFloat` such as `Float32`, `Float64` or `BigFloat`.
`FC` is `T` or `Complex{T}`.

Combine BiLQ and QMR to solve adjoint systems.

    [0  A] [y] = [b]
    [Aᴴ 0] [x]   [c]

The relation `bᴴc ≠ 0` must be satisfied.
BiLQ is used for solving primal system `Ax = b` of size n.
QMR is used for solving dual system `Aᴴy = c` of size n.

An option gives the possibility of transferring from the BiLQ point to the
BiCG point, when it exists. The transfer is based on the residual norm.

BiLQR can be warm-started from initial guesses `x0` and `y0` with

    (x, y, stats) = bilqr(A, b, c, x0, y0; kwargs...)

where `kwargs` are the same keyword arguments as above.

The callback is called as `callback(solver)` and should return `true` if the main loop should terminate,
and `false` otherwise.

#### Input arguments

* `A`: a linear operator that models a matrix of dimension n;
* `b`: a vector of length n;
* `c`: a vector of length n.

#### Output arguments

* `x`: a dense vector of length n;
* `y`: a dense vector of length n;
* `stats`: statistics collected on the run in a [`AdjointStats`](@ref) structure.

#### Reference

* A. Montoison and D. Orban, [*BiLQ: An Iterative Method for Nonsymmetric Linear Systems with a Quasi-Minimum Error Property*](https://doi.org/10.1137/19M1290991), SIAM Journal on Matrix Analysis and Applications, 41(3), pp. 1145--1166, 2020.
"""
function bilqr end

function bilqr(A, b :: AbstractVector{FC}, c :: AbstractVector{FC}, x0 :: AbstractVector, y0 :: AbstractVector; kwargs...) where FC <: FloatOrComplex
  solver = BilqrSolver(A, b)
  bilqr!(solver, A, b, c, x0, y0; kwargs...)
  return (solver.x, solver.y, solver.stats)
end

function bilqr(A, b :: AbstractVector{FC}, c :: AbstractVector{FC}; kwargs...) where FC <: FloatOrComplex
  solver = BilqrSolver(A, b)
  bilqr!(solver, A, b, c; kwargs...)
  return (solver.x, solver.y, solver.stats)
end

"""
    solver = bilqr!(solver::BilqrSolver, A, b, c; kwargs...)
    solver = bilqr!(solver::BilqrSolver, A, b, c, x0, y0; kwargs...)

where `kwargs` are keyword arguments of [`bilqr`](@ref).

See [`BilqrSolver`](@ref) for more details about the `solver`.
"""
function bilqr! end

function bilqr!(solver :: BilqrSolver{T,FC,S}, A, b :: AbstractVector{FC}, c :: AbstractVector{FC},
                x0 :: AbstractVector, y0 :: AbstractVector; kwargs...) where {T <: AbstractFloat, FC <: FloatOrComplex{T}, S <: DenseVector{FC}}
  warm_start!(solver, x0, y0)
  bilqr!(solver, A, b, c; kwargs...)
  return solver
end

function bilqr!(solver :: BilqrSolver{T,FC,S}, A, b :: AbstractVector{FC}, c :: AbstractVector{FC};
                atol :: T=√eps(T), rtol :: T=√eps(T), transfer_to_bicg :: Bool=true,
                itmax :: Int=0, verbose :: Int=0, history :: Bool=false,
                callback = solver -> false) where {T <: AbstractFloat, FC <: FloatOrComplex{T}, S <: DenseVector{FC}}

  m, n = size(A)
  m == n || error("Systems must be square")
  length(b) == m || error("Inconsistent problem size")
  length(c) == n || error("Inconsistent problem size")
  (verbose > 0) && @printf("BILQR: systems of size %d\n", n)

  # Check type consistency
  eltype(A) == FC || error("eltype(A) ≠ $FC")
  ktypeof(b) <: S || error("ktypeof(b) ≠ $S")
  ktypeof(c) <: S || error("ktypeof(c) ≠ $S")

  # Compute the adjoint of A
  Aᴴ = A'

  # Set up workspace.
  uₖ₋₁, uₖ, q, vₖ₋₁, vₖ = solver.uₖ₋₁, solver.uₖ, solver.q, solver.vₖ₋₁, solver.vₖ
  p, Δx, Δy, x, t = solver.p, solver.Δx, solver.Δy, solver.x, solver.y
  d̅, wₖ₋₃, wₖ₋₂, stats = solver.d̅, solver.wₖ₋₃, solver.wₖ₋₂, solver.stats
  warm_start = solver.warm_start
  rNorms, sNorms = stats.residuals_primal, stats.residuals_dual
  reset!(stats)
  r₀ = warm_start ? q : b
  s₀ = warm_start ? p : c

  if warm_start
    mul!(r₀, A, Δx)
    @kaxpby!(n, one(FC), b, -one(FC), r₀)
    mul!(s₀, Aᴴ, Δy)
    @kaxpby!(n, one(FC), c, -one(FC), s₀)
  end

  # Initial solution x₀ and residual norm ‖r₀‖ = ‖b - Ax₀‖.
  x .= zero(FC)          # x₀
  bNorm = @knrm2(n, r₀)  # rNorm = ‖r₀‖

  # Initial solution t₀ and residual norm ‖s₀‖ = ‖c - Aᴴy₀‖.
  t .= zero(FC)          # t₀
  cNorm = @knrm2(n, s₀)  # sNorm = ‖s₀‖

  iter = 0
  itmax == 0 && (itmax = 2*n)

  history && push!(rNorms, bNorm)
  history && push!(sNorms, cNorm)
  εL = atol + rtol * bNorm
  εQ = atol + rtol * cNorm
  (verbose > 0) && @printf("%5s  %7s  %7s\n", "k", "‖rₖ‖", "‖sₖ‖")
  kdisplay(iter, verbose) && @printf("%5d  %7.1e  %7.1e\n", iter, bNorm, cNorm)

  # Initialize the Lanczos biorthogonalization process.
  cᴴb = @kdot(n, s₀, r₀)  # ⟨s₀,r₀⟩ = ⟨c - Aᴴy₀,b - Ax₀⟩
  if cᴴb == 0
    stats.niter = 0
    stats.solved_primal = false
    stats.solved_dual = false
    stats.status = "Breakdown bᴴc = 0"
    solver.warm_start = false
    return solver
  end

  # Set up workspace.
  βₖ = √(abs(cᴴb))            # β₁γ₁ = (c - Aᴴy₀)ᴴ(b - Ax₀)
  γₖ = cᴴb / βₖ               # β₁γ₁ = (c - Aᴴy₀)ᴴ(b - Ax₀)
  vₖ₋₁ .= zero(FC)            # v₀ = 0
  uₖ₋₁ .= zero(FC)            # u₀ = 0
  vₖ .= r₀ ./ βₖ              # v₁ = (b - Ax₀) / β₁
  uₖ .= s₀ ./ conj(γₖ)        # u₁ = (c - Aᴴy₀) / γ̄₁
  cₖ₋₁ = cₖ = -one(T)         # Givens cosines used for the LQ factorization of Tₖ
  sₖ₋₁ = sₖ = zero(FC)        # Givens sines used for the LQ factorization of Tₖ
  d̅ .= zero(FC)               # Last column of D̅ₖ = Vₖ(Qₖ)ᴴ
  ζₖ₋₁ = ζbarₖ = zero(FC)     # ζₖ₋₁ and ζbarₖ are the last components of z̅ₖ = (L̅ₖ)⁻¹β₁e₁
  ζₖ₋₂ = ηₖ = zero(FC)        # ζₖ₋₂ and ηₖ are used to update ζₖ₋₁ and ζbarₖ
  δbarₖ₋₁ = δbarₖ = zero(FC)  # Coefficients of Lₖ₋₁ and L̅ₖ modified over the course of two iterations
  ψbarₖ₋₁ = ψₖ₋₁ = zero(FC)   # ψₖ₋₁ and ψbarₖ are the last components of h̅ₖ = Qₖγ̄₁e₁
  norm_vₖ = bNorm / βₖ        # ‖vₖ‖ is used for residual norm estimates
  ϵₖ₋₃ = λₖ₋₂ = zero(FC)      # Components of Lₖ₋₁
  wₖ₋₃ .= zero(FC)            # Column k-3 of Wₖ = Uₖ(Lₖ)⁻ᴴ
  wₖ₋₂ .= zero(FC)            # Column k-2 of Wₖ = Uₖ(Lₖ)⁻ᴴ
  τₖ = zero(T)                # τₖ is used for the dual residual norm estimate

  # Stopping criterion.
  solved_lq = bNorm == 0
  solved_lq_tol = solved_lq_mach = false
  solved_cg = solved_cg_tol = solved_cg_mach = false
  solved_primal = solved_lq || solved_cg
  solved_qr_tol = solved_qr_mach = false
  solved_dual = cNorm == 0
  tired = iter ≥ itmax
  breakdown = false
  status = "unknown"
  user_requested_exit = false

  while !((solved_primal && solved_dual) || tired || breakdown || user_requested_exit)
    # Update iteration index.
    iter = iter + 1

    # Continue the Lanczos biorthogonalization process.
    # AVₖ  = VₖTₖ    + βₖ₊₁vₖ₊₁(eₖ)ᵀ = Vₖ₊₁Tₖ₊₁.ₖ
    # AᴴUₖ = Uₖ(Tₖ)ᴴ + γ̄ₖ₊₁uₖ₊₁(eₖ)ᵀ = Uₖ₊₁(Tₖ.ₖ₊₁)ᴴ

    mul!(q, A , vₖ)  # Forms vₖ₊₁ : q ← Avₖ
    mul!(p, Aᴴ, uₖ)  # Forms uₖ₊₁ : p ← Aᴴuₖ

    @kaxpy!(n, -γₖ, vₖ₋₁, q)  # q ← q - γₖ * vₖ₋₁
    @kaxpy!(n, -βₖ, uₖ₋₁, p)  # p ← p - β̄ₖ * uₖ₋₁

    αₖ = @kdot(n, uₖ, q)  # αₖ = ⟨uₖ,q⟩

    @kaxpy!(n, -     αₖ , vₖ, q)  # q ← q - αₖ * vₖ
    @kaxpy!(n, -conj(αₖ), uₖ, p)  # p ← p - ᾱₖ * uₖ

    pᴴq = @kdot(n, p, q)  # pᴴq  = ⟨p,q⟩
    βₖ₊₁ = √(abs(pᴴq))    # βₖ₊₁ = √(|pᴴq|)
    γₖ₊₁ = pᴴq / βₖ₊₁     # γₖ₊₁ = pᴴq / βₖ₊₁

    # Update the LQ factorization of Tₖ = L̅ₖQₖ.
    # [ α₁ γ₂ 0  •  •  •  0 ]   [ δ₁   0    •   •   •    •    0   ]
    # [ β₂ α₂ γ₃ •        • ]   [ λ₁   δ₂   •                 •   ]
    # [ 0  •  •  •  •     • ]   [ ϵ₁   λ₂   δ₃  •             •   ]
    # [ •  •  •  •  •  •  • ] = [ 0    •    •   •   •         •   ] Qₖ
    # [ •     •  •  •  •  0 ]   [ •    •    •   •   •    •    •   ]
    # [ •        •  •  •  γₖ]   [ •         •   •  λₖ₋₂ δₖ₋₁  0   ]
    # [ 0  •  •  •  0  βₖ αₖ]   [ •    •    •   0  ϵₖ₋₂ λₖ₋₁ δbarₖ]

    if iter == 1
      δbarₖ = αₖ
    elseif iter == 2
      # [δbar₁ γ₂] [c₂  s̄₂] = [δ₁   0  ]
      # [ β₂   α₂] [s₂ -c₂]   [λ₁ δbar₂]
      (cₖ, sₖ, δₖ₋₁) = sym_givens(δbarₖ₋₁, γₖ)
      λₖ₋₁  =      cₖ  * βₖ + sₖ * αₖ
      δbarₖ = conj(sₖ) * βₖ - cₖ * αₖ
    else
      # [0  βₖ  αₖ] [cₖ₋₁   s̄ₖ₋₁   0] = [sₖ₋₁βₖ  -cₖ₋₁βₖ  αₖ]
      #             [sₖ₋₁  -cₖ₋₁   0]
      #             [ 0      0     1]
      #
      # [ λₖ₋₂   δbarₖ₋₁  γₖ] [1   0   0 ] = [λₖ₋₂  δₖ₋₁    0  ]
      # [sₖ₋₁βₖ  -cₖ₋₁βₖ  αₖ] [0   cₖ  s̄ₖ]   [ϵₖ₋₂  λₖ₋₁  δbarₖ]
      #                       [0   sₖ -cₖ]
      (cₖ, sₖ, δₖ₋₁) = sym_givens(δbarₖ₋₁, γₖ)
      ϵₖ₋₂  =  sₖ₋₁ * βₖ
      λₖ₋₁  = -cₖ₋₁ *      cₖ  * βₖ + sₖ * αₖ
      δbarₖ = -cₖ₋₁ * conj(sₖ) * βₖ - cₖ * αₖ
    end

    if !solved_primal
      # Compute ζₖ₋₁ and ζbarₖ, last components of the solution of L̅ₖz̅ₖ = β₁e₁
      # [δbar₁] [ζbar₁] = [β₁]
      if iter == 1
        ηₖ = βₖ
      end
      # [δ₁    0  ] [  ζ₁ ] = [β₁]
      # [λ₁  δbar₂] [ζbar₂]   [0 ]
      if iter == 2
        ηₖ₋₁ = ηₖ
        ζₖ₋₁ = ηₖ₋₁ / δₖ₋₁
        ηₖ   = -λₖ₋₁ * ζₖ₋₁
      end
      # [λₖ₋₂  δₖ₋₁    0  ] [ζₖ₋₂ ] = [0]
      # [ϵₖ₋₂  λₖ₋₁  δbarₖ] [ζₖ₋₁ ]   [0]
      #                     [ζbarₖ]
      if iter ≥ 3
        ζₖ₋₂ = ζₖ₋₁
        ηₖ₋₁ = ηₖ
        ζₖ₋₁ = ηₖ₋₁ / δₖ₋₁
        ηₖ   = -ϵₖ₋₂ * ζₖ₋₂ - λₖ₋₁ * ζₖ₋₁
      end

      # Relations for the directions dₖ₋₁ and d̅ₖ, the last two columns of D̅ₖ = Vₖ(Qₖ)ᴴ.
      # [d̅ₖ₋₁ vₖ] [cₖ  s̄ₖ] = [dₖ₋₁ d̅ₖ] ⟷ dₖ₋₁ = cₖ * d̅ₖ₋₁ + sₖ * vₖ
      #           [sₖ -cₖ]             ⟷ d̅ₖ   = s̄ₖ * d̅ₖ₋₁ - cₖ * vₖ
      if iter ≥ 2
        # Compute solution xₖ.
        # (xᴸ)ₖ ← (xᴸ)ₖ₋₁ + ζₖ₋₁ * dₖ₋₁
        @kaxpy!(n, ζₖ₋₁ * cₖ,  d̅, x)
        @kaxpy!(n, ζₖ₋₁ * sₖ, vₖ, x)
      end

      # Compute d̅ₖ.
      if iter == 1
        # d̅₁ = v₁
        @. d̅ = vₖ
      else
        # d̅ₖ = s̄ₖ * d̅ₖ₋₁ - cₖ * vₖ
        @kaxpby!(n, -cₖ, vₖ, conj(sₖ), d̅)
      end

      # Compute ⟨vₖ,vₖ₊₁⟩ and ‖vₖ₊₁‖
      vₖᴴvₖ₊₁ = @kdot(n, vₖ, q) / βₖ₊₁
      norm_vₖ₊₁ = @knrm2(n, q) / βₖ₊₁

      # Compute BiLQ residual norm
      # ‖rₖ‖ = √(|μₖ|²‖vₖ‖² + |ωₖ|²‖vₖ₊₁‖² + μ̄ₖωₖ⟨vₖ,vₖ₊₁⟩ + μₖω̄ₖ⟨vₖ₊₁,vₖ⟩)
      if iter == 1
        rNorm_lq = bNorm
      else
        μₖ = βₖ * (sₖ₋₁ * ζₖ₋₂ - cₖ₋₁ * cₖ * ζₖ₋₁) + αₖ * sₖ * ζₖ₋₁
        ωₖ = βₖ₊₁ * sₖ * ζₖ₋₁
        θₖ = conj(μₖ) * ωₖ * vₖᴴvₖ₊₁
        rNorm_lq = sqrt(abs2(μₖ) * norm_vₖ^2 + abs2(ωₖ) * norm_vₖ₊₁^2 + 2 * real(θₖ))
      end
      history && push!(rNorms, rNorm_lq)

      # Update ‖vₖ‖
      norm_vₖ = norm_vₖ₊₁

      # Compute BiCG residual norm
      # ‖rₖ‖ = |ρₖ| * ‖vₖ₊₁‖
      if transfer_to_bicg && (abs(δbarₖ) > eps(T))
        ζbarₖ = ηₖ / δbarₖ
        ρₖ = βₖ₊₁ * (sₖ * ζₖ₋₁ - cₖ * ζbarₖ)
        rNorm_cg = abs(ρₖ) * norm_vₖ₊₁
      end

      # Update primal stopping criterion
      solved_lq_tol = rNorm_lq ≤ εL
      solved_lq_mach = rNorm_lq + 1 ≤ 1
      solved_lq = solved_lq_tol || solved_lq_mach
      solved_cg_tol = transfer_to_bicg && (abs(δbarₖ) > eps(T)) && (rNorm_cg ≤ εL)
      solved_cg_mach = transfer_to_bicg && (abs(δbarₖ) > eps(T)) && (rNorm_cg + 1 ≤ 1)
      solved_cg = solved_cg_tol || solved_cg_mach
      solved_primal = solved_lq || solved_cg
    end

    if !solved_dual
      # Compute ψₖ₋₁ and ψbarₖ the last coefficients of h̅ₖ = Qₖγ̄₁e₁.
      if iter == 1
        ψbarₖ = conj(γₖ)
      else
        # [cₖ  s̄ₖ] [ψbarₖ₋₁] = [ ψₖ₋₁ ]
        # [sₖ -cₖ] [   0   ]   [ ψbarₖ]
        ψₖ₋₁  = cₖ * ψbarₖ₋₁
        ψbarₖ = sₖ * ψbarₖ₋₁
      end

      # Compute the direction wₖ₋₁, the last column of Wₖ₋₁ = (Uₖ₋₁)(Lₖ₋₁)⁻ᴴ ⟷ (L̄ₖ₋₁)(Wₖ₋₁)ᵀ = (Uₖ₋₁)ᵀ.
      # w₁ = u₁ / δ̄₁
      if iter == 2
        wₖ₋₁ = wₖ₋₂
        @kaxpy!(n, one(FC), uₖ₋₁, wₖ₋₁)
        @. wₖ₋₁ = uₖ₋₁ / conj(δₖ₋₁)
      end
      # w₂ = (u₂ - λ̄₁w₁) / δ̄₂
      if iter == 3
        wₖ₋₁ = wₖ₋₃
        @kaxpy!(n, one(FC), uₖ₋₁, wₖ₋₁)
        @kaxpy!(n, -conj(λₖ₋₂), wₖ₋₂, wₖ₋₁)
        @. wₖ₋₁ = wₖ₋₁ / conj(δₖ₋₁)
      end
      # wₖ₋₁ = (uₖ₋₁ - λ̄ₖ₋₂wₖ₋₂ - ϵ̄ₖ₋₃wₖ₋₃) / δ̄ₖ₋₁
      if iter ≥ 4
        @kscal!(n, -conj(ϵₖ₋₃), wₖ₋₃)
        wₖ₋₁ = wₖ₋₃
        @kaxpy!(n, one(FC), uₖ₋₁, wₖ₋₁)
        @kaxpy!(n, -conj(λₖ₋₂), wₖ₋₂, wₖ₋₁)
        @. wₖ₋₁ = wₖ₋₁ / conj(δₖ₋₁)
      end

      if iter ≥ 3
        # Swap pointers.
        @kswap(wₖ₋₃, wₖ₋₂)
      end

      if iter ≥ 2
        # Compute solution tₖ₋₁.
        # tₖ₋₁ ← tₖ₋₂ + ψₖ₋₁ * wₖ₋₁
        @kaxpy!(n, ψₖ₋₁, wₖ₋₁, t)
      end

      # Update ψbarₖ₋₁
      ψbarₖ₋₁ = ψbarₖ

      # Compute τₖ = τₖ₋₁ + ‖uₖ‖²
      τₖ += @kdotr(n, uₖ, uₖ)

      # Compute QMR residual norm ‖sₖ₋₁‖ ≤ |ψbarₖ| * √τₖ
      sNorm = abs(ψbarₖ) * √τₖ
      history && push!(sNorms, sNorm)

      # Update dual stopping criterion
      solved_qr_tol = sNorm ≤ εQ
      solved_qr_mach = sNorm + 1 ≤ 1
      solved_dual = solved_qr_tol || solved_qr_mach
    end

    # Compute vₖ₊₁ and uₖ₊₁.
    @. vₖ₋₁ = vₖ  # vₖ₋₁ ← vₖ
    @. uₖ₋₁ = uₖ  # uₖ₋₁ ← uₖ

    if pᴴq ≠ zero(FC)
      @. vₖ = q / βₖ₊₁        # βₖ₊₁vₖ₊₁ = q
      @. uₖ = p / conj(γₖ₊₁)  # γ̄ₖ₊₁uₖ₊₁ = p
    end

    # Update ϵₖ₋₃, λₖ₋₂, δbarₖ₋₁, cₖ₋₁, sₖ₋₁, γₖ and βₖ.
    if iter ≥ 3
      ϵₖ₋₃ = ϵₖ₋₂
    end
    if iter ≥ 2
      λₖ₋₂ = λₖ₋₁
    end
    δbarₖ₋₁ = δbarₖ
    cₖ₋₁    = cₖ
    sₖ₋₁    = sₖ
    γₖ      = γₖ₊₁
    βₖ      = βₖ₊₁

    user_requested_exit = callback(solver) :: Bool
    tired = iter ≥ itmax
    breakdown = !solved_lq && !solved_cg && (pᴴq == 0)

    kdisplay(iter, verbose) &&  solved_primal && !solved_dual && @printf("%5d  %7s  %7.1e\n", iter, "", sNorm)
    kdisplay(iter, verbose) && !solved_primal &&  solved_dual && @printf("%5d  %7.1e  %7s\n", iter, rNorm_lq, "")
    kdisplay(iter, verbose) && !solved_primal && !solved_dual && @printf("%5d  %7.1e  %7.1e\n", iter, rNorm_lq, sNorm)
  end
  (verbose > 0) && @printf("\n")

  # Compute BICG point
  # (xᶜ)ₖ ← (xᴸ)ₖ₋₁ + ζbarₖ * d̅ₖ
  if solved_cg
    @kaxpy!(n, ζbarₖ, d̅, x)
  end

  tired                            && (status = "maximum number of iterations exceeded")
  breakdown                        && (status = "Breakdown ⟨uₖ₊₁,vₖ₊₁⟩ = 0")
  solved_lq_tol  && !solved_dual   && (status = "Only the primal solution xᴸ is good enough given atol and rtol")
  solved_cg_tol  && !solved_dual   && (status = "Only the primal solution xᶜ is good enough given atol and rtol")
  !solved_primal && solved_qr_tol  && (status = "Only the dual solution t is good enough given atol and rtol")
  solved_lq_tol  && solved_qr_tol  && (status = "Both primal and dual solutions (xᴸ, t) are good enough given atol and rtol")
  solved_cg_tol  && solved_qr_tol  && (status = "Both primal and dual solutions (xᶜ, t) are good enough given atol and rtol")
  solved_lq_mach && !solved_dual   && (status = "Only found approximate zero-residual primal solution xᴸ")
  solved_cg_mach && !solved_dual   && (status = "Only found approximate zero-residual primal solution xᶜ")
  !solved_primal && solved_qr_mach && (status = "Only found approximate zero-residual dual solution t")
  solved_lq_mach && solved_qr_mach && (status = "Found approximate zero-residual primal and dual solutions (xᴸ, t)")
  solved_cg_mach && solved_qr_mach && (status = "Found approximate zero-residual primal and dual solutions (xᶜ, t)")
  solved_lq_mach && solved_qr_tol  && (status = "Found approximate zero-residual primal solutions xᴸ and a dual solution t good enough given atol and rtol")
  solved_cg_mach && solved_qr_tol  && (status = "Found approximate zero-residual primal solutions xᶜ and a dual solution t good enough given atol and rtol")
  solved_lq_tol  && solved_qr_mach && (status = "Found a primal solution xᴸ good enough given atol and rtol and an approximate zero-residual dual solutions t")
  solved_cg_tol  && solved_qr_mach && (status = "Found a primal solution xᶜ good enough given atol and rtol and an approximate zero-residual dual solutions t")
  user_requested_exit              && (status = "user-requested exit")

  # Update x and y
  warm_start && @kaxpy!(n, one(FC), Δx, x)
  warm_start && @kaxpy!(n, one(FC), Δy, t)
  solver.warm_start = false

  # Update stats
  stats.niter = iter
  stats.status = status
  stats.solved_primal = solved_primal
  stats.solved_dual = solved_dual
  return solver
end
