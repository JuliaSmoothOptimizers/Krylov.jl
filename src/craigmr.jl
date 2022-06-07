# An implementation of CRAIG-MR for the solution of the
# (under/over-determined or square) linear system
#
#  Ax = b.
#
# The method seeks to solve the minimum-norm problem
#
#  min ‖x‖  s.t.  x ∈ argmin ‖Ax - b‖,
#
# and is equivalent to applying the conjugate residual method
# to the linear system
#
#  AAᵀy = b.
#
# This method is equivalent to CRMR, and is described in
#
# D. Orban and M. Arioli. Iterative Solution of Symmetric Quasi-Definite Linear Systems,
# Volume 3 of Spotlights. SIAM, Philadelphia, PA, 2017.
#
# D. Orban, The Projected Golub-Kahan Process for Constrained
# Linear Least-Squares Problems. Cahier du GERAD G-2014-15,
# GERAD, Montreal QC, Canada, 2014.
#
# Dominique Orban, <dominique.orban@gerad.ca>
# Montreal, QC, May 2015.

export craigmr, craigmr!


"""
    (x, y, stats) = craigmr(A, b::AbstractVector{FC};
                            M=I, N=I, sqd :: Bool=false, λ :: T=zero(T), atol :: T=√eps(T),
                            rtol::T=√eps(T), itmax::Int=0, verbose::Int=0, history::Bool=false,
                            callback::Function=solver->false)

`T` is an `AbstractFloat` such as `Float32`, `Float64` or `BigFloat`.
`FC` is `T` or `Complex{T}`.

Solve the consistent linear system

    Ax + λ²y = b

using the CRAIGMR method, where λ ≥ 0 is a regularization parameter.
This method is equivalent to applying the Conjugate Residuals method
to the normal equations of the second kind

    (AAᵀ + λ²I) y = b

but is more stable. When λ = 0, this method solves the minimum-norm problem

    min ‖x‖  s.t.  x ∈ argmin ‖Ax - b‖.

If `λ > 0`, CRAIGMR solves the symmetric and quasi-definite system

    [ -F    Aᵀ ] [ x ]   [ 0 ]
    [  A  λ²E  ] [ y ] = [ b ],

where E and F are symmetric and positive definite.
Preconditioners M = E⁻¹ ≻ 0 and N = F⁻¹ ≻ 0 may be provided in the form of linear operators.
If `sqd=true`, `λ` is set to the common value `1`.

The system above represents the optimality conditions of

    min ‖x‖²_F + λ²‖y‖²_E  s.t.  Ax + λ²Ey = b.

For a symmetric and positive definite matrix `K`, the K-norm of a vector `x` is `‖x‖²_K = xᵀKx`.
CRAIGMR is then equivalent to applying MINRES to `(AF⁻¹Aᵀ + λ²E)y = b` with `Fx = Aᵀy`.

If `λ = 0`, CRAIGMR solves the symmetric and indefinite system

    [ -F   Aᵀ ] [ x ]   [ 0 ]
    [  A   0  ] [ y ] = [ b ].

The system above represents the optimality conditions of

    min ‖x‖²_F  s.t.  Ax = b.

In this case, `M` can still be specified and indicates the weighted norm in which residuals are measured.

CRAIGMR produces monotonic residuals ‖r‖₂.
It is formally equivalent to CRMR, though can be slightly more accurate,
and intricate to implement. Both the x- and y-parts of the solution are
returned.

The callback is called as `callback(solver)` and should return `true` if the main loop should terminate,
and `false` otherwise.

#### References

* D. Orban and M. Arioli. [*Iterative Solution of Symmetric Quasi-Definite Linear Systems*](https://doi.org/10.1137/1.9781611974737), Volume 3 of Spotlights. SIAM, Philadelphia, PA, 2017.
* D. Orban, [*The Projected Golub-Kahan Process for Constrained, Linear Least-Squares Problems*](https://dx.doi.org/10.13140/RG.2.2.17443.99360). Cahier du GERAD G-2014-15, 2014.
"""
function craigmr end

function craigmr(A, b :: AbstractVector{FC}; kwargs...) where FC <: FloatOrComplex
  solver = CraigmrSolver(A, b)
  craigmr!(solver, A, b; kwargs...)
  return (solver.x, solver.y, solver.stats)
end

"""
    solver = craigmr!(solver::CraigmrSolver, A, b; kwargs...)

where `kwargs` are keyword arguments of [`craigmr`](@ref).

See [`CraigmrSolver`](@ref) for more details about the `solver`.
"""
function craigmr! end

function craigmr!(solver :: CraigmrSolver{T,FC,S}, A, b :: AbstractVector{FC};
                  M=I, N=I, sqd :: Bool=false, λ :: T=zero(T), atol :: T=√eps(T),
                  rtol :: T=√eps(T), itmax :: Int=0, verbose :: Int=0, history :: Bool=false,
                  callback :: Function = solver -> false) where {T <: AbstractFloat, FC <: FloatOrComplex{T}, S <: DenseVector{FC}}

  m, n = size(A)
  length(b) == m || error("Inconsistent problem size")
  (verbose > 0) && @printf("CRAIGMR: system of %d equations in %d variables\n", m, n)

  # Check sqd and λ parameters
  sqd && (λ ≠ 0) && error("sqd cannot be set to true if λ ≠ 0 !")
  sqd && (λ = one(T))

  # Tests M = Iₘ and N = Iₙ
  MisI = (M === I)
  NisI = (N === I)

  # Check type consistency
  eltype(A) == FC || error("eltype(A) ≠ $FC")
  ktypeof(b) == S || error("ktypeof(b) ≠ $S")

  # Compute the adjoint of A
  Aᵀ = A'

  # Set up workspace.
  allocate_if(!MisI, solver, :u, S, m)
  allocate_if(!NisI, solver, :v, S, n)
  allocate_if(λ > 0, solver, :q, S, n)
  x, Nv, Aᵀu, d, y, Mu = solver.x, solver.Nv, solver.Aᵀu, solver.d, solver.y, solver.Mu
  w, wbar, Av, q, stats = solver.w, solver.wbar, solver.Av, solver.q, solver.stats
  rNorms, ArNorms = stats.residuals, stats.Aresiduals
  reset!(stats)
  u = MisI ? Mu : solver.u
  v = NisI ? Nv : solver.v

  # Compute y such that AAᵀy = b. Then recover x = Aᵀy.
  x .= zero(FC)
  y .= zero(FC)
  Mu .= b
  MisI || mul!(u, M, Mu)
  β = sqrt(@kdotr(m, u, Mu))
  if β == 0
    stats.niter = 0
    stats.solved, stats.inconsistent = true, false
    history && push!(rNorms, β)
    history && push!(ArNorms, zero(T))
    stats.status = "x = 0 is a zero-residual solution"
    return solver
  end

  # Initialize Golub-Kahan process.
  # β₁Mu₁ = b.
  @kscal!(m, one(FC)/β, u)
  MisI || @kscal!(m, one(FC)/β, Mu)
  # α₁Nv₁ = Aᵀu₁.
  mul!(Aᵀu, Aᵀ, u)
  Nv .= Aᵀu
  NisI || mul!(v, N, Nv)
  α = sqrt(@kdotr(n, v, Nv))
  Anorm² = α * α

  iter = 0
  itmax == 0 && (itmax = m + n)

  (verbose > 0) && @printf("%5s  %7s  %7s  %7s  %7s  %8s  %8s  %7s\n", "k", "‖r‖", "‖Aᵀr‖", "β", "α", "cos", "sin", "‖A‖²")
  kdisplay(iter, verbose) && @printf("%5d  %7.1e  %7.1e  %7.1e  %7.1e  %8.1e  %8.1e  %7.1e\n", iter, β, α, β, α, 0, 1, Anorm²)

  # Aᵀb = 0 so x = 0 is a minimum least-squares solution
  if α == 0
    stats.niter = 0
    stats.solved, stats.inconsistent = true, false
    history && push!(rNorms, β)
    history && push!(ArNorms, zero(T))
    stats.status = "x = 0 is a minimum least-squares solution"
    return solver
  end
  @kscal!(n, one(FC)/α, v)
  NisI || @kscal!(n, one(FC)/α, Nv)

  # Regularization.
  λₖ  = λ             # λ₁ = λ
  cpₖ = spₖ = one(T)  # Givens sines and cosines used to zero out λₖ
  cdₖ = sdₖ = one(T)  # Givens sines and cosines used to define λₖ₊₁
  λ > 0 && (q .= v)   # Additional vector needed to update x, by definition q₀ = 0

  if λ > 0
    (cpₖ, spₖ, αhat) = sym_givens(α, λₖ)
    @kscal!(n, spₖ, q)  # q̄₁ = sp₁ * v₁
  else
    αhat = α
  end

  # Initialize other constants.
  ζbar = β
  ρbar = αhat
  θ = zero(T)
  rNorm = ζbar
  history && push!(rNorms, rNorm)
  ArNorm = α
  history && push!(ArNorms, ArNorm)

  ɛ_c = atol + rtol * rNorm  # Stopping tolerance for consistent systems.
  ɛ_i = atol + rtol * ArNorm  # Stopping tolerance for inconsistent systems.

  wbar .= u
  @kscal!(m, one(FC)/αhat, wbar)
  w .= zero(FC)
  d .= zero(FC)

  status = "unknown"
  solved = rNorm ≤ ɛ_c
  inconsistent = (rNorm > 100 * ɛ_c) & (ArNorm ≤ ɛ_i)
  tired  = iter ≥ itmax
  user_requested_exit = false

  while ! (solved || inconsistent || tired || user_requested_exit)
    iter = iter + 1

    # Generate next Golub-Kahan vectors.
    # 1. βₖ₊₁Muₖ₊₁ = Avₖ - αₖMuₖ
    mul!(Av, A, v)
    @kaxpby!(m, one(FC), Av, -α, Mu)
    MisI || mul!(u, M, Mu)
    β = sqrt(@kdotr(m, u, Mu))
    if β ≠ 0
      @kscal!(m, one(FC)/β, u)
      MisI || @kscal!(m, one(FC)/β, Mu)
    end

    Anorm² = Anorm² + β * β  # = ‖B_{k-1}‖²

    if λ > 0
      βhat = cpₖ * β
      λₐᵤₓ = spₖ * β
    else
      βhat = β
    end

    # Continue QR factorization
    #
    # Q [ Lₖ  β₁ e₁ ] = [ Rₖ   zₖ  ] :
    #   [ β    0    ]   [ 0   ζbar ]
    #
    #       k  k+1    k    k+1      k  k+1
    # k   [ c   s ] [ ρbar    ] = [ ρ  θ⁺    ]
    # k+1 [ s  -c ] [ β    α⁺ ]   [    ρbar⁺ ]
    #
    # so that we obtain
    #
    # [ c  s ] [ ζbar ] = [ ζ     ]
    # [ s -c ] [  0   ]   [ ζbar⁺ ]
    (c, s, ρ) = sym_givens(ρbar, βhat)
    ζ = c * ζbar
    ζbar = s * ζbar
    rNorm = abs(ζbar)
    history && push!(rNorms, rNorm)

    @kaxpby!(m, one(FC)/ρ, wbar, -θ/ρ, w)  # w = (wbar - θ * w) / ρ
    @kaxpy!(m, ζ, w, y)                    # y = y + ζ * w

    if λ > 0
      # DₖRₖ = V̅ₖ with v̅ₖ = cpₖvₖ + spₖqₖ₋₁
      if iter == 1
        @kaxpy!(n, one(FC)/ρ, cpₖ * v, d)
      else
        @kaxpby!(n, one(FC)/ρ, cpₖ * v, -θ/ρ, d)
        @kaxpy!(n, one(FC)/ρ, spₖ * q, d)
        @kaxpby!(n, spₖ, v, -cpₖ, q)  # q̄ₖ ← spₖ * vₖ - cpₖ * qₖ₋₁
      end
    else
      # DₖRₖ = Vₖ
      if iter == 1
        @kaxpy!(n, one(FC)/ρ, v, d)
      else
        @kaxpby!(n, one(FC)/ρ, v, -θ/ρ, d)
      end
    end

    # xₖ = Dₖzₖ
    @kaxpy!(n, ζ, d, x)

    # 2. αₖ₊₁Nvₖ₊₁ = Aᵀuₖ₊₁ - βₖ₊₁Nvₖ
    mul!(Aᵀu, Aᵀ, u)
    @kaxpby!(n, one(FC), Aᵀu, -β, Nv)
    NisI || mul!(v, N, Nv)
    α = sqrt(@kdotr(n, v, Nv))
    Anorm² = Anorm² + α * α  # = ‖Lₖ‖
    ArNorm = α * β * abs(ζ/ρ)
    history && push!(ArNorms, ArNorm)

    kdisplay(iter, verbose) && @printf("%5d  %7.1e  %7.1e  %7.1e  %7.1e  %8.1e  %8.1e  %7.1e\n", iter, rNorm, ArNorm, β, α, c, s, Anorm²)

    if λ > 0
      (cdₖ, sdₖ, λₖ₊₁) = sym_givens(λ, λₐᵤₓ)
      @kscal!(n, sdₖ, q)  # qₖ ← sdₖ * q̄ₖ
      (cpₖ, spₖ, αhat) = sym_givens(α, λₖ₊₁)
    else
      αhat = α
    end

    if α ≠ 0
      @kscal!(n, one(FC)/α, v)
      NisI || @kscal!(n, one(FC)/α, Nv)
      @kaxpby!(m, one(T)/αhat, u, -βhat / αhat, wbar)  # wbar = (u - beta * wbar) / alpha
    end
    θ    =  s * αhat
    ρbar = -c * αhat

    user_requested_exit = callback(solver) :: Bool
    solved = rNorm ≤ ɛ_c
    inconsistent = (rNorm > 100 * ɛ_c) & (ArNorm ≤ ɛ_i)
    tired  = iter ≥ itmax
  end
  (verbose > 0) && @printf("\n")
  
  tired               && (status = "maximum number of iterations exceeded")
  solved              && (status = "found approximate minimum-norm solution")
  !tired && !solved   && (status = "found approximate minimum least-squares solution")
  user_requested_exit && (status = "user-requested exit")

  # Update stats
  stats.niter = iter
  stats.solved = solved
  stats.inconsistent = inconsistent
  stats.status = status
  return solver
end
