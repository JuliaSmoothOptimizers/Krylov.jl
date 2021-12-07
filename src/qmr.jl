# An implementation of QMR for the solution of unsymmetric
# and square linear system Ax = b.
#
# This method is described in
#
# R. W. Freund and N. M. Nachtigal
# QMR : a quasi-minimal residual method for non-Hermitian linear systems.
# Numerische mathematik, Vol. 60(1), pp. 315--339, 1991.
#
# R. W. Freund and N. M. Nachtigal
# An implementation of the QMR method based on coupled two-term recurrences.
# SIAM Journal on Scientific Computing, Vol. 15(2), pp. 313--337, 1994.
#
# A. Montoison and D. Orban
# BiLQ: An Iterative Method for Nonsymmetric Linear Systems with a Quasi-Minimum Error Property.
# SIAM Journal on Matrix Analysis and Applications, 41(3), pp. 1145--1166, 2020.
#
# Alexis Montoison, <alexis.montoison@polymtl.ca>
# Montreal, May 2019.

export qmr, qmr!

"""
    (x, stats) = qmr(A, b::AbstractVector{FC}; c::AbstractVector{FC}=b,
                     atol::T=√eps(T), rtol::T=√eps(T),
                     itmax::Int=0, verbose::Int=0, history::Bool=false)

`T` is an `AbstractFloat` such as `Float32`, `Float64` or `BigFloat`.
`FC` is `T` or `Complex{T}`.

Solve the square linear system Ax = b using the QMR method.

QMR is based on the Lanczos biorthogonalization process and requires two initial vectors `b` and `c`.
The relation `bᵀc ≠ 0` must be satisfied and by default `c = b`.
When `A` is symmetric and `b = c`, QMR is equivalent to MINRES.

#### References

* R. W. Freund and N. M. Nachtigal, [*QMR : a quasi-minimal residual method for non-Hermitian linear systems*](https://doi.org/10.1007/BF01385726), Numerische mathematik, Vol. 60(1), pp. 315--339, 1991.
* R. W. Freund and N. M. Nachtigal, [*An implementation of the QMR method based on coupled two-term recurrences*](https://doi.org/10.1137/0915022), SIAM Journal on Scientific Computing, Vol. 15(2), pp. 313--337, 1994.
* A. Montoison and D. Orban, [*BiLQ: An Iterative Method for Nonsymmetric Linear Systems with a Quasi-Minimum Error Property*](https://doi.org/10.1137/19M1290991), SIAM Journal on Matrix Analysis and Applications, 41(3), pp. 1145--1166, 2020.
"""
function qmr(A, b :: AbstractVector{FC}; kwargs...) where FC <: FloatOrComplex
  solver = QmrSolver(A, b)
  qmr!(solver, A, b; kwargs...)
  return (solver.x, solver.stats)
end

"""
    solver = qmr!(solver::QmrSolver, args...; kwargs...)

where `args` and `kwargs` are arguments and keyword arguments of [`qmr`](@ref).

See [`QmrSolver`](@ref) for more details about the `solver`.
"""
function qmr!(solver :: QmrSolver{T,FC,S}, A, b :: AbstractVector{FC}; c :: AbstractVector{FC}=b,
              atol :: T=√eps(T), rtol :: T=√eps(T),
              itmax :: Int=0, verbose :: Int=0, history :: Bool=false) where {T <: AbstractFloat, FC <: FloatOrComplex{T}, S <: DenseVector{FC}}

  n, m = size(A)
  m == n || error("System must be square")
  length(b) == m || error("Inconsistent problem size")
  (verbose > 0) && @printf("QMR: system of size %d\n", n)

  # Check type consistency
  eltype(A) == T || error("eltype(A) ≠ $T")
  ktypeof(b) == S || error("ktypeof(b) ≠ $S")
  ktypeof(c) == S || error("ktypeof(c) ≠ $S")

  # Compute the adjoint of A
  Aᵀ = A'

  # Set up workspace.
  uₖ₋₁, uₖ, q, vₖ₋₁, vₖ, p = solver.uₖ₋₁, solver.uₖ, solver.q, solver.vₖ₋₁, solver.vₖ, solver.p
  x, wₖ₋₂, wₖ₋₁, stats = solver.x, solver.wₖ₋₂, solver.wₖ₋₁, solver.stats
  rNorms = stats.residuals
  reset!(stats)

  # Initial solution x₀ and residual norm ‖r₀‖.
  x .= zero(T)
  rNorm = @knrm2(n, b)  # ‖r₀‖
  history && push!(rNorms, rNorm)
  if rNorm == 0
    stats.solved = true
    stats.inconsistent = false
    stats.status = "x = 0 is a zero-residual solution"
    return solver
  end

  iter = 0
  itmax == 0 && (itmax = 2*n)

  ε = atol + rtol * rNorm
  (verbose > 0) && @printf("%5s  %7s\n", "k", "‖rₖ‖")
  display(iter, verbose) && @printf("%5d  %7.1e\n", iter, rNorm)

  # Initialize the Lanczos biorthogonalization process.
  bᵗc = @kdot(n, b, c)  # ⟨b,c⟩
  if bᵗc == 0
    stats.solved = false
    stats.inconsistent = false
    stats.status = "Breakdown bᵀc = 0"
    return solver
  end

  βₖ = √(abs(bᵗc))            # β₁γ₁ = bᵀc
  γₖ = bᵗc / βₖ               # β₁γ₁ = bᵀc
  vₖ₋₁ .= zero(T)             # v₀ = 0
  uₖ₋₁ .= zero(T)             # u₀ = 0
  vₖ .= b ./ βₖ               # v₁ = b / β₁
  uₖ .= c ./ γₖ               # u₁ = c / γ₁
  cₖ₋₂ = cₖ₋₁ = cₖ = zero(T)  # Givens cosines used for the QR factorization of Tₖ₊₁.ₖ
  sₖ₋₂ = sₖ₋₁ = sₖ = zero(T)  # Givens sines used for the QR factorization of Tₖ₊₁.ₖ
  wₖ₋₂ .= zero(T)             # Column k-2 of Wₖ = Vₖ(Rₖ)⁻¹
  wₖ₋₁ .= zero(T)             # Column k-1 of Wₖ = Vₖ(Rₖ)⁻¹
  ζbarₖ = βₖ                  # ζbarₖ is the last component of z̅ₖ = (Qₖ)ᵀβ₁e₁
  τₖ = @kdot(n, vₖ, vₖ)       # τₖ is used for the residual norm estimate

  # Stopping criterion.
  solved    = rNorm ≤ ε
  breakdown = false
  tired     = iter ≥ itmax
  status    = "unknown"

  while !(solved || tired || breakdown)
    # Update iteration index.
    iter = iter + 1

    # Continue the Lanczos biorthogonalization process.
    # AVₖ  = VₖTₖ    + βₖ₊₁vₖ₊₁(eₖ)ᵀ = Vₖ₊₁Tₖ₊₁.ₖ
    # AᵀUₖ = Uₖ(Tₖ)ᵀ + γₖ₊₁uₖ₊₁(eₖ)ᵀ = Uₖ₊₁(Tₖ.ₖ₊₁)ᵀ

    mul!(q, A , vₖ)  # Forms vₖ₊₁ : q ← Avₖ
    mul!(p, Aᵀ, uₖ)  # Forms uₖ₊₁ : p ← Aᵀuₖ

    @kaxpy!(n, -γₖ, vₖ₋₁, q)  # q ← q - γₖ * vₖ₋₁
    @kaxpy!(n, -βₖ, uₖ₋₁, p)  # p ← p - βₖ * uₖ₋₁

    αₖ = @kdot(n, q, uₖ)      # αₖ = qᵀuₖ

    @kaxpy!(n, -αₖ, vₖ, q)    # q ← q - αₖ * vₖ
    @kaxpy!(n, -αₖ, uₖ, p)    # p ← p - αₖ * uₖ

    qᵗp = @kdot(n, p, q)      # qᵗp  = ⟨q,p⟩
    βₖ₊₁ = √(abs(qᵗp))        # βₖ₊₁ = √(|qᵗp|)
    γₖ₊₁ = qᵗp / βₖ₊₁         # γₖ₊₁ = qᵗp / βₖ₊₁

    # Update the QR factorization of Tₖ₊₁.ₖ = Qₖ [ Rₖ ].
    #                                            [ Oᵀ ]
    # [ α₁ γ₂ 0  •  •  •   0  ]      [ δ₁ λ₁ ϵ₁ 0  •  •  0  ]
    # [ β₂ α₂ γ₃ •         •  ]      [ 0  δ₂ λ₂ •  •     •  ]
    # [ 0  •  •  •  •      •  ]      [ •  •  δ₃ •  •  •  •  ]
    # [ •  •  •  •  •  •   •  ] = Qₖ [ •     •  •  •  •  0  ]
    # [ •     •  •  •  •   0  ]      [ •        •  •  • ϵₖ₋₂]
    # [ •        •  •  •   γₖ ]      [ •           •  • λₖ₋₁]
    # [ •           •  βₖ  αₖ ]      [ •              •  δₖ ]
    # [ 0  •  •  •  •  0  βₖ₊₁]      [ 0  •  •  •  •  •  0  ]
    #
    # If k = 1, we don't have any previous reflexion.
    # If k = 2, we apply the last reflexion.
    # If k ≥ 3, we only apply the two previous reflexions.

    # Apply previous Givens reflections Qₖ₋₂.ₖ₋₁
    if iter ≥ 3
      # [cₖ₋₂  sₖ₋₂] [0 ] = [  ϵₖ₋₂ ]
      # [sₖ₋₂ -cₖ₋₂] [γₖ]   [λbarₖ₋₁]
      ϵₖ₋₂    =  sₖ₋₂ * γₖ
      λbarₖ₋₁ = -cₖ₋₂ * γₖ
    end

    # Apply previous Givens reflections Qₖ₋₁.ₖ
    if iter ≥ 2
      iter == 2 && (λbarₖ₋₁ = γₖ)
      # [cₖ₋₁  sₖ₋₁] [λbarₖ₋₁] = [λₖ₋₁ ]
      # [sₖ₋₁ -cₖ₋₁] [   αₖ  ]   [δbarₖ]
      λₖ₋₁  = cₖ₋₁ * λbarₖ₋₁ + sₖ₋₁ * αₖ
      δbarₖ = sₖ₋₁ * λbarₖ₋₁ - cₖ₋₁ * αₖ

      # Update sₖ₋₂ and cₖ₋₂.
      sₖ₋₂ = sₖ₋₁
      cₖ₋₂ = cₖ₋₁
    end

    # Compute and apply current Givens reflection Qₖ.ₖ₊₁
    iter == 1 && (δbarₖ = αₖ)
    # [cₖ  sₖ] [δbarₖ] = [δₖ]
    # [sₖ -cₖ] [βₖ₊₁ ]   [0 ]
    (cₖ, sₖ, δₖ) = sym_givens(δbarₖ, βₖ₊₁)

    # Update z̅ₖ₊₁ = Qₖ.ₖ₊₁ [ z̄ₖ ]
    #                      [ 0  ]
    #
    # [cₖ  sₖ] [ζbarₖ] = [   ζₖ  ]
    # [sₖ -cₖ] [  0  ]   [ζbarₖ₊₁]
    ζₖ      = cₖ * ζbarₖ
    ζbarₖ₊₁ = sₖ * ζbarₖ

    # Update sₖ₋₁ and cₖ₋₁.
    sₖ₋₁ = sₖ
    cₖ₋₁ = cₖ

    # Compute the direction wₖ, the last column of Wₖ = Vₖ(Rₖ)⁻¹ ⟷ (Rₖ)ᵀ(Wₖ)ᵀ = (Vₖ)ᵀ.
    # w₁ = v₁ / δ₁
    if iter == 1
      wₖ = wₖ₋₁
      @kaxpy!(n, one(T), vₖ, wₖ)
      @. wₖ = wₖ / δₖ
    end
    # w₂ = (v₂ - λ₁w₁) / δ₂
    if iter == 2
      wₖ = wₖ₋₂
      @kaxpy!(n, -λₖ₋₁, wₖ₋₁, wₖ)
      @kaxpy!(n, one(T), vₖ, wₖ)
      @. wₖ = wₖ / δₖ
    end
    # wₖ = (vₖ - λₖ₋₁wₖ₋₁ - ϵₖ₋₂wₖ₋₂) / δₖ
    if iter ≥ 3
      @kscal!(n, -ϵₖ₋₂, wₖ₋₂)
      wₖ = wₖ₋₂
      @kaxpy!(n, -λₖ₋₁, wₖ₋₁, wₖ)
      @kaxpy!(n, one(T), vₖ, wₖ)
      @. wₖ = wₖ / δₖ
    end

    # Compute solution xₖ.
    # xₖ ← xₖ₋₁ + ζₖ * wₖ
    @kaxpy!(n, ζₖ, wₖ, x)

    # Compute vₖ₊₁ and uₖ₊₁.
    @. vₖ₋₁ = vₖ  # vₖ₋₁ ← vₖ
    @. uₖ₋₁ = uₖ  # uₖ₋₁ ← uₖ

    if qᵗp ≠ zero(T)
      @. vₖ = q / βₖ₊₁  # βₖ₊₁vₖ₊₁ = q
      @. uₖ = p / γₖ₊₁  # γₖ₊₁uₖ₊₁ = p
    end

    # Compute τₖ₊₁ = τₖ + ‖vₖ₊₁‖²
    τₖ₊₁ = τₖ + @kdot(n, vₖ, vₖ)

    # Compute ‖rₖ‖ ≤ |ζbarₖ₊₁|√τₖ₊₁
    rNorm = abs(ζbarₖ₊₁) * √τₖ₊₁
    history && push!(rNorms, rNorm)

    # Update directions for x.
    if iter ≥ 2
      @kswap(wₖ₋₂, wₖ₋₁)
    end

    # Update ζbarₖ, βₖ, γₖ and τₖ.
    ζbarₖ = ζbarₖ₊₁
    βₖ    = βₖ₊₁
    γₖ    = γₖ₊₁
    τₖ    = τₖ₊₁

    # Update stopping criterion.
    solved = rNorm ≤ ε
    tired = iter ≥ itmax
    breakdown = !solved && (qᵗp == 0)
    display(iter, verbose) && @printf("%5d  %7.1e\n", iter, rNorm)
  end
  (verbose > 0) && @printf("\n")

  tired     && (status = "maximum number of iterations exceeded")
  breakdown && (status = "Breakdown ⟨uₖ₊₁,vₖ₊₁⟩ = 0")
  solved    && (status = "solution good enough given atol and rtol")

  # Update stats
  stats.solved = solved
  stats.inconsistent = false
  stats.status = status
  return solver
end
