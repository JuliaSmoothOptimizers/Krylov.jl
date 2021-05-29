# An implementation of LNLQ for the solution of the consistent linear system
#
#  Ax = b.
#
# The method seeks to solve the minimum-norm problem
#
#  min ‖x‖  s.t.  Ax = b,
#
# and is equivalent to applying the SYMMLQ method
# to the linear system
#
#  AAᵀy = b with x = Aᵀy and can be reformulated as
#
#  [ -I  Aᵀ ][ x ] = [ 0 ]
#  [  A     ][ y ]   [ b ].
#
# This method is based on the Golub-Kahan bidiagonalization process and is described in
#
# R. Estrin, D. Orban, M.A. Saunders, LNLQ: An Iterative Method for Least-Norm Problems with an Error Minimization Property,
# SIAM Journal on Matrix Analysis and Applications, 40(3), pp. 1102--1124, 2019.
#
# Alexis Montoison, <alexis.montoison@polymtl.ca>
# Montréal, March 2019 -- Alès, January 2020.

export lnlq, lnlq!

"""
    (x, y, stats) = lnlq(A, b::AbstractVector{T};
                         M=I, N=I, sqd::Bool=false, λ::T=zero(T),
                         atol::T=√eps(T), rtol::T=√eps(T), itmax::Int=0,
                         transfer_to_craig::Bool=true, verbose::Int=0, history::Bool=false) where T <: AbstractFloat

Find the least-norm solution of the consistent linear system

    Ax + λs = b

using the LNLQ method, where λ ≥ 0 is a regularization parameter.

For a system in the form Ax = b, LNLQ method is equivalent to applying
SYMMLQ to AAᵀy = b and recovering x = Aᵀy but is more stable.
Note that y are the Lagrange multipliers of the least-norm problem

    minimize ‖x‖  s.t.  Ax = b.

If `sqd = true`, LNLQ solves the symmetric and quasi-definite system

    [ -F   Aᵀ ] [ x ]   [ 0 ]
    [  A   E  ] [ y ] = [ b ],

where E and F are symmetric and positive definite.
LNLQ is then equivalent to applying SYMMLQ to `(AF⁻¹Aᵀ + E)y = b` with `Fx = Aᵀy`.
Preconditioners M = E⁻¹ ≻ 0 and N = F⁻¹ ≻ 0 may be provided in the form of linear operators.

If `sqd = false`, LNLQ solves the symmetric and indefinite system

    [ -F   Aᵀ ] [ x ]   [ 0 ]
    [  A   0  ] [ y ] = [ b ].

In this case, M can still be specified and indicates the weighted norm in which residuals are measured.

In this implementation, both the x and y-parts of the solution are returned.

#### Reference

* R. Estrin, D. Orban, M.A. Saunders, *LNLQ: An Iterative Method for Least-Norm Problems with an Error Minimization Property*, SIAM Journal on Matrix Analysis and Applications, 40(3), pp. 1102--1124, 2019.
"""
function lnlq(A, b :: AbstractVector{T}; kwargs...) where T <: AbstractFloat
  solver = LnlqSolver(A, b)
  lnlq!(solver, A, b; kwargs...)
end

function lnlq!(solver :: LnlqSolver{T,S}, A, b :: AbstractVector{T};
               M=I, N=I, sqd :: Bool=false, λ :: T=zero(T),
               atol :: T=√eps(T), rtol :: T=√eps(T), itmax :: Int=0,
               transfer_to_craig :: Bool=true, verbose :: Int=0, history :: Bool=false) where {T <: AbstractFloat, S <: DenseVector{T}}

  m, n = size(A)
  length(b) == m || error("Inconsistent problem size")
  (verbose > 0) && @printf("LNLQ: system of %d equations in %d variables\n", m, n)

  # Tests M == Iₘ and N == Iₙ
  MisI = (M == I)
  NisI = (N == I)

  # Check type consistency
  eltype(A) == T || error("eltype(A) ≠ $T")
  ktypeof(b) == S || error("ktypeof(b) ≠ $S")
  MisI || (eltype(M) == T) || error("eltype(M) ≠ $T")
  NisI || (eltype(N) == T) || error("eltype(N) ≠ $T")

  # Compute the adjoint of A
  Aᵀ = A'

  # When solving a SQD system, set regularization parameter λ = 1.
  sqd && (λ = one(T))

  # Set up workspace.
  allocate_if(!MisI, solver, :u, S, m)
  allocate_if(!NisI, solver, :v, S, n)
  allocate_if(λ > 0, solver, :q, S, n)
  x, Nv, Aᵀu, y, w̄, Mu, Av, q = solver.x, solver.Nv, solver.Aᵀu, solver.y, solver.w̄, solver.Mu, solver.Av, solver.q
  u = MisI ? Mu : solver.u
  v = NisI ? Nv : solver.v

  # Initial solutions (x₀, y₀) and residual norm ‖r₀‖.
  x .= zero(T)
  y .= zero(T)

  bNorm = @knrm2(m, b)
  bNorm == 0 && return x, y, SimpleStats(true, false, [bNorm], T[], "x = 0 is a zero-residual solution")

  rNorms = history ? [bNorm] : T[]
  ε = atol + rtol * bNorm

  iter = 0
  itmax == 0 && (itmax = m + n)

  (verbose > 0) && @printf("%5s  %7s\n", "k", "‖rₖ‖")
  display(iter, verbose) && @printf("%5d  %7.1e\n", iter, bNorm)

  # Update iteration index
  iter = iter + 1

  # Initialize generalized Golub-Kahan bidiagonalization.
  # β₁Mu₁ = b.
  Mu .= b
  MisI || mul!(u, M, Mu)      # u₁ = M⁻¹ * Mu₁
  βₖ = sqrt(@kdot(m, u, Mu))  # β₁ = ‖u₁‖_M
  if βₖ ≠ 0
    @kscal!(m, 1 / βₖ, u)
    MisI || @kscal!(m, 1 / βₖ, Mu)
  end

  # α₁Nv₁ = Aᵀu₁.
  mul!(Aᵀu, Aᵀ, u)
  Nv .= Aᵀu
  NisI || mul!(v, N, Nv)      # v₁ = N⁻¹ * Nv₁
  αₖ = sqrt(@kdot(n, v, Nv))  # α₁ = ‖v₁‖_N
  if αₖ ≠ 0
    @kscal!(n, 1 / αₖ, v)
    NisI || @kscal!(n, 1 / αₖ, Nv)
  end

  w̄ .= u             # Direction w̄₁
  cₖ = sₖ = zero(T)  # Givens sines and cosines used for the LQ factorization of (Lₖ)ᵀ
  ζₖ₋₁ = zero(T)     # ζₖ₋₁ and ζbarₖ are the last components of z̅ₖ
  ηₖ = zero(T)       # Coefficient of M̅ₖ

  # Variable used for the regularization.
  λₖ  = λ             # λ₁ = λ
  cpₖ = spₖ = one(T)  # Givens sines and cosines used to zero out λₖ
  cdₖ = sdₖ = one(T)  # Givens sines and cosines used to define λₖ₊₁
  λ > 0 && (q .= v)   # Additional vector needed to update x, by definition q₀ = 0

  # Initialize the regularization.
  if λ > 0
    #        k    2k      k   2k           k      2k
    # k   [  αₖ   λₖ ] [ cpₖ  spₖ ] = [  αhatₖ    0   ]
    # k+1 [ βₖ₊₁  0  ] [ spₖ -cpₖ ]   [ βhatₖ₊₁  θₖ₊₁ ]
    (cpₖ, spₖ, αhatₖ) = sym_givens(αₖ, λₖ)

    # q̄₁ = sp₁ * v₁
    @kscal!(n, spₖ, q)
  else
    αhatₖ = αₖ
  end

  # Begin the LQ factorization of (Lₖ)ᵀ = M̅ₖQₖ.
  # [ α₁ β₂ 0  •  •  •  0 ]   [ ϵ₁  0   •   •   •   •   0   ]
  # [ 0  α₂ •  •        • ]   [ η₂  ϵ₂  •               •   ]
  # [ •  •  •  •  •     • ]   [ 0   •   •   •           •   ]
  # [ •     •  •  •  •  • ] = [ •   •   •   •   •       •   ] Qₖ
  # [ •        •  •  •  0 ]   [ •       •   •   •   •   •   ]
  # [ •           •  •  βₖ]   [ •           •   •   •   0   ]
  # [ 0  •  •  •  •  0  αₖ]   [ 0   •   •   •   0   ηₖ ϵbarₖ]

  ϵbarₖ = αhatₖ  # ϵbar₁ = αhat₁

  # Hₖ = Bₖ(Lₖ)ᵀ = [   Lₖ(Lₖ)ᵀ   ] ⟹ (Hₖ₋₁)ᵀ = [Lₖ₋₁Mₖ₋₁  0] Qₖ
  #                [ αₖβₖ₊₁(eₖ)ᵀ ]
  #
  # Solve Lₖtₖ = β₁e₁ and M̅ₖz̅ₖ = tₖ
  # tₖ = (τ₁, •••, τₖ)
  # z̅ₖ = (zₖ₋₁, ζbarₖ) = (ζ₁, •••, ζₖ₋₁, ζbarₖ)

  τₖ    = βₖ / αhatₖ  # τ₁ = β₁ / αhat₁
  ζbarₖ = τₖ / ϵbarₖ  # ζbar₁ = τ₁ / ϵbar₁

  # Stopping criterion.
  solved_lq = solved_cg = false
  tired = false
  status = "unknown"

  while !(solved_lq || solved_cg || tired)

    # Update of (xᵃᵘˣ)ₖ = Vₖtₖ
    if λ > 0
      # (xᵃᵘˣ)ₖ ← (xᵃᵘˣ)ₖ₋₁ + τₖ * (cpₖvₖ + spₖqₖ₋₁)
      @kaxpy!(n, τₖ * cpₖ, v, x)
      if iter ≥ 2
        @kaxpy!(n, τₖ * spₖ, q, x)
        # q̄ₖ ← spₖ * vₖ - cpₖ * qₖ₋₁
        @kaxpby!(n, spₖ, v, -cpₖ, q)
      end
    else
      # (xᵃᵘˣ)ₖ ← (xᵃᵘˣ)ₖ₋₁ + τₖ * vₖ
      @kaxpy!(n, τₖ, v, x)
    end

    # Continue the generalized Golub-Kahan bidiagonalization.
    # AVₖ    = MUₖ₊₁Bₖ
    # AᵀUₖ₊₁ = NVₖ(Bₖ)ᵀ + αₖ₊₁Nvₖ₊₁(eₖ₊₁)ᵀ = NVₖ₊₁(Lₖ₊₁)ᵀ
    #
    #      [ α₁ 0  •  •  •  •  0 ]
    #      [ β₂ α₂ •           • ]
    #      [ 0  •  •  •        • ]
    # Lₖ = [ •  •  •  •  •     • ]
    #      [ •     •  •  •  •  • ]
    #      [ •        •  •  •  0 ]
    #      [ 0  •  •  •  0  βₖ αₖ]
    #
    # Bₖ = [    Lₖ     ]
    #      [ βₖ₊₁(eₖ)ᵀ ]

    # βₖ₊₁Muₖ₊₁ = Avₖ - αₖMuₖ
    mul!(Av, A, v)
    @kaxpby!(m, one(T), Av, -αₖ, Mu)
    MisI || mul!(u, M, Mu)        # uₖ₊₁ = M⁻¹ * Muₖ₊₁
    βₖ₊₁ = sqrt(@kdot(m, u, Mu))  # βₖ₊₁ = ‖uₖ₊₁‖_M
    if βₖ₊₁ ≠ 0
      @kscal!(m, 1 / βₖ₊₁, u)
      MisI || @kscal!(m, 1 / βₖ₊₁, Mu)
    end

    # αₖ₊₁Nvₖ₊₁ = Aᵀuₖ₊₁ - βₖ₊₁Nvₖ
    mul!(Aᵀu, Aᵀ, u)
    @kaxpby!(n, one(T), Aᵀu, -βₖ₊₁, Nv)
    NisI || mul!(v, N, Nv)        # vₖ₊₁ = N⁻¹ * Nvₖ₊₁
    αₖ₊₁ = sqrt(@kdot(n, v, Nv))  # αₖ₊₁ = ‖vₖ₊₁‖_N
    if αₖ₊₁ ≠ 0
      @kscal!(n, 1 / αₖ₊₁, v)
      NisI || @kscal!(n, 1 / αₖ₊₁, Nv)
    end

    # Continue the regularization.
    if λ > 0
      #        k    2k      k   2k           k      2k
      # k   [  αₖ   λₖ ] [ cpₖ  spₖ ] = [  αhatₖ    0   ]
      # k+1 [ βₖ₊₁  0  ] [ spₖ -cpₖ ]   [ βhatₖ₊₁  θₖ₊₁ ]
      βhatₖ₊₁ = cpₖ * βₖ₊₁
      θₖ₊₁    = spₖ * βₖ₊₁

      #       2k  2k+1     2k  2k+1       2k  2k+1
      # k   [  0    0 ] [ -cdₖ  sdₖ ] = [ 0    0  ]
      # k+1 [ θₖ₊₁  λ ] [  sdₖ  cdₖ ]   [ 0  λₖ₊₁ ]
      (cdₖ, sdₖ, λₖ₊₁) = sym_givens(λ, θₖ₊₁)

      # qₖ ← sdₖ * q̄ₖ
      @kscal!(n, sdₖ, q)

      #       k+1   2k+1      k+1    2k+1        k+1     2k+1
      # k+1 [ αₖ₊₁  λₖ₊₁ ] [ cpₖ₊₁  spₖ₊₁ ] = [ αhatₖ₊₁   0   ]
      # k+2 [ βₖ₊₂   0   ] [ spₖ₊₁ -cpₖ₊₁ ]   [  γₖ₊₂    θₖ₊₂ ]
      (cpₖ₊₁, spₖ₊₁, αhatₖ₊₁) = sym_givens(αₖ₊₁, λₖ₊₁)
    else
      βhatₖ₊₁ = βₖ₊₁
      αhatₖ₊₁ = αₖ₊₁
    end

    # Continue the LQ factorization of (Lₖ₊₁)ᵀ.
    # [ηₖ ϵbarₖ βₖ₊₁] [1     0     0 ] = [ηₖ  ϵₖ     0    ]
    # [0    0   αₖ₊₁] [0   cₖ₊₁  sₖ₊₁]   [0  ηₖ₊₁  ϵbarₖ₊₁]
    #                 [0   sₖ₊₁ -cₖ₊₁]

    (cₖ₊₁, sₖ₊₁, ϵₖ) = sym_givens(ϵbarₖ, βhatₖ₊₁)
    ηₖ₊₁    =   αhatₖ₊₁ * sₖ₊₁
    ϵbarₖ₊₁ = - αhatₖ₊₁ * cₖ₊₁

    # Update solutions of Lₖ₊₁tₖ₊₁ = β₁e₁ and M̅ₖ₊₁z̅ₖ₊₁ = tₖ₊₁.
    τₖ₊₁    = - βhatₖ₊₁ * τₖ / αhatₖ₊₁
    ζₖ      = cₖ₊₁ * ζbarₖ
    ζbarₖ₊₁ = (τₖ₊₁ - ηₖ₊₁ * ζₖ) / ϵbarₖ₊₁

    # Relations for the directions wₖ and w̄ₖ₊₁
    # [w̄ₖ uₖ₊₁] [cₖ₊₁  sₖ₊₁] = [wₖ w̄ₖ₊₁] → wₖ   = cₖ₊₁ * w̄ₖ + sₖ₊₁ * uₖ₊₁
    #           [sₖ₊₁ -cₖ₊₁]             → w̄ₖ₊₁ = sₖ₊₁ * w̄ₖ - cₖ₊₁ * uₖ₊₁

    # (yᴸ)ₖ₊₁ ← (yᴸ)ₖ + ζₖ * wₖ
    @kaxpy!(m, ζₖ * cₖ₊₁, w̄, y)
    @kaxpy!(m, ζₖ * sₖ₊₁, u, y)

    # Compute w̄ₖ₊₁
    @kaxpby!(m, -cₖ₊₁, u, sₖ₊₁, w̄)

    # Compute residual norm ‖(rᴸ)ₖ‖ = |αₖ| * √((ϵbarₖζbarₖ)² + (βₖ₊₁sₖζₖ₋₁)²)
    if iter == 1
      rNorm_lq = bNorm
    else
      rNorm_lq = abs(αhatₖ) * √((ϵbarₖ * ζbarₖ)^2 + (βhatₖ₊₁ * sₖ * ζₖ₋₁)^2)
    end
    history && push!(rNorms, rNorm_lq)

    # Compute residual norm ‖(rᶜ)ₖ‖ = |βₖ₊₁ * τₖ|
    if transfer_to_craig
      rNorm_cg = abs(βhatₖ₊₁ * τₖ)
    end

    # Update sₖ, cₖ, αₖ, βₖ, ηₖ, ϵbarₖ, τₖ, ζₖ₋₁ and ζbarₖ.
    cₖ    = cₖ₊₁
    sₖ    = sₖ₊₁
    αₖ    = αₖ₊₁
    αhatₖ = αhatₖ₊₁
    βₖ    = βₖ₊₁
    ηₖ    = ηₖ₊₁
    ϵbarₖ = ϵbarₖ₊₁
    τₖ    = τₖ₊₁
    ζₖ₋₁  = ζₖ
    ζbarₖ = ζbarₖ₊₁

    # Update regularization variables.
    if λ > 0
      cpₖ = cpₖ₊₁
      spₖ = spₖ₊₁
    end

    # Update stopping criterion.
    tired = iter ≥ itmax
    solved_lq = rNorm_lq ≤ ε
    solved_cg = transfer_to_craig && rNorm_cg ≤ ε
    display(iter, verbose) && @printf("%5d  %7.1e\n", iter, rNorm_lq)

    # Update iteration index.
    iter = iter + 1
  end
  (verbose > 0) && @printf("\n")

  if solved_cg
    if λ > 0
      # (xᶜ)ₖ ← (xᵃᵘˣ)ₖ₋₁ + τₖ * (cpₖvₖ + spₖqₖ₋₁)
      @kaxpy!(n, τₖ * cpₖ, v, x)
      if iter ≥ 2
        @kaxpy!(n, τₖ * spₖ, q, x)
      end
    else
      # (xᶜ)ₖ ← (xᵃᵘˣ)ₖ₋₁ + τₖ * vₖ
      @kaxpy!(n, τₖ, v, x)
    end
    # (yᶜ)ₖ ← (yᴸ)ₖ₋₁ + ζbarₖ * w̄ₖ
    @kaxpy!(m, ζbarₖ, w̄, y)
  else
    if λ > 0
      # (xᴸ)ₖ ← (xᵃᵘˣ)ₖ₋₁ + ηₖζₖ₋₁ * (cpₖvₖ + spₖqₖ₋₁)
      @kaxpy!(n, ηₖ * ζₖ₋₁ * cpₖ, v, x)
      if iter ≥ 2
        @kaxpy!(n, ηₖ * ζₖ₋₁ * spₖ, q, x)
      end
    else
      # (xᴸ)ₖ ← (xᵃᵘˣ)ₖ₋₁ + ηₖζₖ₋₁ * vₖ
      @kaxpy!(n, ηₖ * ζₖ₋₁, v, x)
    end
  end

  tired     && (status = "maximum number of iterations exceeded")
  solved_lq && (status = "solutions (xᴸ, yᴸ) good enough for the tolerances given")
  solved_cg && (status = "solutions (xᶜ, yᶜ) good enough for the tolerances given")
  stats = SimpleStats(solved_lq || solved_cg, false, rNorms, T[], status)
  return (x, y, stats)
end
