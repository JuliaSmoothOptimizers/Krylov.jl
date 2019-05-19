# An implementation of USYMQR for the solution of unsymmetric linear system Ax = b.
#
# This method is described in
#
# M. A. Saunders, H. D. Simon, and E. L. Yip
# Two Conjugate-Gradient-Type Methods for Unsymmetric Linear Equations.
# SIAM Journal on Numerical Analysis, 25(4), pp. 927--940, 1988.
#
# A. Buttari, D. Orban, D. Ruiz and D. Titley-Peloquin
# A tridiagonalization method for symmetric saddle-point and quasi-definite systems.
# Cahier du GERAD G-2018-42, GERAD, Montreal, 2018. doi:10.13140/RG.2.2.26337.20328
#
# Alexis Montoison, <alexis.montoison@polymtl.ca>
# Montreal, November 2018.

export usymqr

"""Solve the linear system Ax = b using USYMQR method.
USYMQR can also be applied to under-determined and over-determined problems.

USYMQR algorithm is based on a tridiagonalization process for unsymmetric matrices.
It's considered as a generalization of MINRES.

It can also be applied to under-determined and over-determined problems.
This version of USYMQR works in multiprecision.

The tridiagonalization of A can be done in elliptic norms with preconditioners M⁻¹ and N⁻¹.
Uₖ and Vₖ will be respectively M- and N-orthogonal.
"""
function usymqr(A :: AbstractLinearOperator, b :: AbstractVector{T}, c :: AbstractVector{T};
                M :: AbstractLinearOperator=opEye(),
                N :: AbstractLinearOperator=opEye(),
                atol :: T=√eps(T), rtol :: T=√eps(T),
                itmax :: Int=0, verbose :: Bool=false) where T <: AbstractFloat

  m, n = size(A)
  length(b) == m || error("Inconsistent problem size")
  length(c) == n || error("Inconsistent problem size")
  verbose && @printf("USYMQR: system of %d equations in %d variables\n", m, n)

  # Initial solution x₀ and residual norm ‖r₀‖₂.
  x = zeros(T, n)
  rNorm = @knrm2(m, b)
  rNorm == 0 && return x, SimpleStats(true, false, [rNorm], T[], "x = 0 is a zero-residual solution")

  iter = 0
  itmax == 0 && (itmax = 2*n)

  rNorms = [rNorm;]
  AᵀrNorms = T[]
  ε = atol + rtol * rNorm
  verbose && @printf("%5d  %7.1e\n", iter, rNorm)

  # Set up workspace.
  M⁻¹b = M * b
  N⁻¹c = N * c
  βₖ = sqrt(@kdot(m, b, M⁻¹b)) # β₁ = ‖u₁‖_M
  γₖ = sqrt(@kdot(n, c, N⁻¹c)) # γ₁ = ‖v₁‖_N
  Muₖ₋₁ = zeros(T, m)          # Mu₀ = 0
  Nvₖ₋₁ = zeros(T, n)          # Nv₀ = 0
  Muₖ = b / βₖ                 # β₁Mu₁ = b
  Nvₖ = c / γₖ                 # γ₁Nv₁ = c
  cₖ₋₂ = cₖ₋₁ = cₖ = zero(T)   # Givens cosines used for the QR factorization of Tₖ₊₁.ₖ
  sₖ₋₂ = sₖ₋₁ = sₖ = zero(T)   # Givens sines used for the QR factorization of Tₖ₊₁.ₖ
  wₖ₋₂ = zeros(T, n)           # Penultimate direction for x.
  wₖ₋₁ = zeros(T, n)           # Last direction for x.
  ζbarₖ = βₖ                   # ζₖ and ζbarₖ₊₁ are the last components of zₖ = (Qₖ)ᵀβ₁e₁

  # Stopping criterion.
  solved = rNorm ≤ ε
  tired = iter ≥ itmax
  status = "unknown"

  while !(solved || tired)
    # Update iteration index.
    iter = iter + 1

    # Continue the tridiagonalization.
    uₖ = M * Muₖ    # uₖ = M⁻¹ * Muₖ
    vₖ = N * Nvₖ    # vₖ = N⁻¹ * Nvₖ

    q = A * vₖ      # q ← Avₖ
    p = A.tprod(uₖ) # p ← Aᵀuₖ

    @kaxpy!(m, -γₖ, Muₖ₋₁, q) # q ← q - γₖ * Muₖ₋₁
    @kaxpy!(n, -βₖ, Nvₖ₋₁, p) # p ← p - βₖ * Nvₖ₋₁

    αₖ = @kdot(m, uₖ, q)      # αₖ = qᵀuₖ

    @kaxpy!(m, -αₖ, Muₖ, q)   # q ← q - αₖ * Muₖ
    @kaxpy!(n, -αₖ, Nvₖ, p)   # p ← p - αₖ * Nvₖ

    @. Muₖ₋₁ = Muₖ            # Muₖ₋₁ ← Muₖ
    βₖ₊₁ = @knrm2(m, q)       # βₖ₊₁ = ‖q‖₂

    @. Nvₖ₋₁ = Nvₖ            # Nvₖ₋₁ ← Nvₖ
    γₖ₊₁ = @knrm2(n, p)       # γₖ₊₁ = ‖p‖₂

    # Update the QR factorization of Tₖ₊₁.ₖ = Qₖ₊₁ [ Rₖ ].
    #                                              [ Oᵀ ]
    # [ α₁ γ₂ 0  •  •  •   0  ]        [ δ₁ λ₁ ϵ₁ 0  •  •  0  ]
    # [ β₂ α₂ γ₃ •         •  ]        [ 0  δ₂ λ₂ •  •     •  ]
    # [ 0  •  •  •  •      •  ]        [ •  •  δ₃ •  •  •  •  ]
    # [ •  •  •  •  •  •   •  ] = Qₖ₊₁ [ •     •  •  •  •  0  ]
    # [ •     •  •  •  •   0  ]        [ •        •  •  • ϵₖ₋₂]
    # [ •        •  •  •   γₖ ]        [ •           •  • λₖ₋₁]
    # [ •           •  βₖ  αₖ ]        [ 0  •  •  •  •  0  δₖ ]
    # [ 0  •  •  •  •  0  βₖ₊₁]        [ 0  •  •  •  •  •  0  ]
    #
    # If k = 1, we don't have any previous reflexion.
    # If k = 2, we apply the last reflexion.
    # If k ≥ 3, we only apply the two previous reflexions.

    # Apply previous Givens reflections Ωᵢ.
    if iter ≥ 3
      # [cₖ₋₂  sₖ₋₂] [0 ] = [ϵₖ₋₂]
      # [sₖ₋₂ -cₖ₋₂] [γₖ]   [ƛₖ₋₁]
      ϵₖ₋₂ =  sₖ₋₂ * γₖ
      ƛₖ₋₁ = -cₖ₋₂ * γₖ
    end
    if iter ≥ 2
      iter == 2 && (ƛₖ₋₁ = γₖ)
      # [cₖ₋₁  sₖ₋₁] [ƛₖ₋₁] = [λₖ₋₁ ]
      # [sₖ₋₁ -cₖ₋₁] [ αₖ ]   [δbarₖ]
      δbarₖ = sₖ₋₁ * ƛₖ₋₁ - cₖ₋₁ * αₖ
      λₖ₋₁  = cₖ₋₁ * ƛₖ₋₁ + sₖ₋₁ * αₖ

      # Update sₖ₋₂ and cₖ₋₂.
      sₖ₋₂ = sₖ₋₁
      cₖ₋₂ = cₖ₋₁
    end
    iter == 1 && (δbarₖ = αₖ)

    # Compute and apply current Givens reflection Ωₖ.
    # [cₖ  sₖ] [δbarₖ] = [δₖ]
    # [sₖ -cₖ] [βₖ₊₁ ]   [0 ]
    (cₖ, sₖ, δₖ) = sym_givens(δbarₖ, βₖ₊₁)

    # Update the right-hand zₖ = Ωₖzₖ₋₁.
    # [cₖ  sₖ] [ζbarₖ] = [   ζₖ  ]
    # [sₖ -cₖ] [  0  ]   [ζbarₖ₊₁]
    ζbarₖ₊₁ = sₖ * ζbarₖ
    ζₖ      = cₖ * ζbarₖ

    # Update sₖ₋₁ and cₖ₋₁.
    sₖ₋₁ = sₖ
    cₖ₋₁ = cₖ

    # Compute the direction wₖ, the last column of Wₖ = VₖRₖ⁻¹ ⟷ (Rₖ)ᵀ(Wₖ)ᵀ = (Vₖ)ᵀ.
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
    # wₖ = (vₖ - λₖ₋₁wₖ₋₁ - ϵₖ₋₂wₖ₋₂) / δₖ
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

    # Compute ‖rₖ‖ = |ζbarₖ₊₁|.
    rNorm = abs(ζbarₖ₊₁)
    push!(rNorms, rNorm)

    # Compute ‖Aᵀrₖ₋₁‖ = |ζbarₖ| * √((δbarₖ)² + (ƛₖ)²).
    AᵀrNorm = abs(ζbarₖ) * √(δbarₖ^2 + (cₖ * γₖ₊₁)^2) # ƛₖ = - cₖ * γₖ₊₁
    push!(AᵀrNorms, AᵀrNorm)

    # Compute Muₖ₊₁ and Nuₖ₊₁.
    if βₖ₊₁ ≠ zero(T)
      @. Muₖ = q / βₖ₊₁ # βₖ₊₁Muₖ₊₁ = q
    end
    if γₖ₊₁ ≠ zero(T)
      @. Nvₖ = p / γₖ₊₁ # γₖ₊₁Nvₖ₊₁ = p
    end

    # Update directions for x.
    if iter ≥ 2
      @kswap(wₖ₋₂, wₖ₋₁)
    end

    # Update ζbarₖ, γₖ, βₖ.
    ζbarₖ = ζbarₖ₊₁
    γₖ    = γₖ₊₁
    βₖ    = βₖ₊₁

    # Update stopping criterion.
    solved = rNorm ≤ ε || AᵀrNorm ≤ ε
    tired = iter ≥ itmax
    verbose && @printf("%5d  %7.1e  %7.1e\n", iter, rNorm, AᵀrNorm)
  end
  verbose && @printf("\n")
  status = tired ? "maximum number of iterations exceeded" : "solution good enough given atol and rtol"
  stats = SimpleStats(solved, false, rNorms, AᵀrNorms, status)
  return (x, stats)
end
