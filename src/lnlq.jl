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
#  AAᴴy = b with x = Aᴴy and can be reformulated as
#
#  [ -I  Aᴴ ][ x ] = [ 0 ]
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
    (x, y, stats) = lnlq(A, b::AbstractVector{FC};
                         M=I, N=I, ldiv::Bool=false,
                         transfer_to_craig::Bool=true,
                         sqd::Bool=false, λ::T=zero(T),
                         σ::T=zero(T), utolx::T=√eps(T),
                         utoly::T=√eps(T), atol::T=√eps(T),
                         rtol::T=√eps(T), itmax::Int=0,
                         timemax::Float64=Inf, verbose::Int=0, history::Bool=false,
                         callback=solver->false, iostream::IO=kstdout)

`T` is an `AbstractFloat` such as `Float32`, `Float64` or `BigFloat`.
`FC` is `T` or `Complex{T}`.

Find the least-norm solution of the consistent linear system

    Ax + λ²y = b

of size m × n using the LNLQ method, where λ ≥ 0 is a regularization parameter.

For a system in the form Ax = b, LNLQ method is equivalent to applying
SYMMLQ to AAᴴy = b and recovering x = Aᴴy but is more stable.
Note that y are the Lagrange multipliers of the least-norm problem

    minimize ‖x‖  s.t.  Ax = b.

If `λ > 0`, LNLQ solves the symmetric and quasi-definite system

    [ -F    Aᴴ ] [ x ]   [ 0 ]
    [  A  λ²E  ] [ y ] = [ b ],

where E and F are symmetric and positive definite.
Preconditioners M = E⁻¹ ≻ 0 and N = F⁻¹ ≻ 0 may be provided in the form of linear operators.
If `sqd=true`, `λ` is set to the common value `1`.

The system above represents the optimality conditions of

    min ‖x‖²_F + λ²‖y‖²_E  s.t.  Ax + λ²Ey = b.

For a symmetric and positive definite matrix `K`, the K-norm of a vector `x` is `‖x‖²_K = xᴴKx`.
LNLQ is then equivalent to applying SYMMLQ to `(AF⁻¹Aᴴ + λ²E)y = b` with `Fx = Aᴴy`.

If `λ = 0`, LNLQ solves the symmetric and indefinite system

    [ -F   Aᴴ ] [ x ]   [ 0 ]
    [  A   0  ] [ y ] = [ b ].

The system above represents the optimality conditions of

    minimize ‖x‖²_F  s.t.  Ax = b.

In this case, `M` can still be specified and indicates the weighted norm in which residuals are measured.

In this implementation, both the x and y-parts of the solution are returned.

`utolx` and `utoly` are tolerances on the upper bound of the distance to the solution ‖x-x*‖ and ‖y-y*‖, respectively.
The bound is valid if λ>0 or σ>0 where σ should be strictly smaller than the smallest positive singular value.
For instance σ:=(1-1e-7)σₘᵢₙ .

#### Input arguments

* `A`: a linear operator that models a matrix of dimension `m × n`;
* `b`: a vector of length `m`.

#### Keyword arguments

* `M`: linear operator that models a Hermitian positive-definite matrix of size `m` used for centered preconditioning of the augmented system;
* `N`: linear operator that models a Hermitian positive-definite matrix of size `n` used for centered preconditioning of the augmented system;
* `ldiv`: define whether the preconditioners use `ldiv!` or `mul!`;
* `transfer_to_craig`: transfer from the LNLQ point to the CRAIG point, when it exists. The transfer is based on the residual norm;
* `sqd`: if `true`, set `λ=1` for Hermitian quasi-definite systems;
* `λ`: regularization parameter;
* `σ`: strict lower bound on the smallest positive singular value `σₘᵢₙ` such as `σ = (1-10⁻⁷)σₘᵢₙ`;
* `utolx`: tolerance on the upper bound on the distance to the solution `‖x-x*‖`;
* `utoly`: tolerance on the upper bound on the distance to the solution `‖y-y*‖`;
* `atol`: absolute stopping tolerance based on the residual norm;
* `rtol`: relative stopping tolerance based on the residual norm;
* `itmax`: the maximum number of iterations. If `itmax=0`, the default number of iterations is set to `m+n`;
* `timemax`: the time limit in seconds;
* `verbose`: additional details can be displayed if verbose mode is enabled (verbose > 0). Information will be displayed every `verbose` iterations;
* `history`: collect additional statistics on the run such as residual norms, or Aᴴ-residual norms;
* `callback`: function or functor called as `callback(solver)` that returns `true` if the Krylov method should terminate, and `false` otherwise;
* `iostream`: stream to which output is logged.

#### Output arguments

* `x`: a dense vector of length `n`;
* `y`: a dense vector of length `m`;
* `stats`: statistics collected on the run in a [`LNLQStats`](@ref) structure.

#### Reference

* R. Estrin, D. Orban, M.A. Saunders, [*LNLQ: An Iterative Method for Least-Norm Problems with an Error Minimization Property*](https://doi.org/10.1137/18M1194948), SIAM Journal on Matrix Analysis and Applications, 40(3), pp. 1102--1124, 2019.
"""
function lnlq end

"""
    solver = lnlq!(solver::LnlqSolver, A, b; kwargs...)

where `kwargs` are keyword arguments of [`lnlq`](@ref).

See [`LnlqSolver`](@ref) for more details about the `solver`.
"""
function lnlq! end

def_args_lnlq = (:(A                    ),
                 :(b::AbstractVector{FC}))

def_kwargs_lnlq = (:(; M = I                         ),
                   :(; N = I                         ),
                   :(; ldiv::Bool = false            ),
                   :(; transfer_to_craig::Bool = true),
                   :(; sqd::Bool = false             ),
                   :(; λ::T = zero(T)                ),
                   :(; σ::T = zero(T)                ),
                   :(; utolx::T = √eps(T)            ),
                   :(; utoly::T = √eps(T)            ),
                   :(; atol::T = √eps(T)             ),
                   :(; rtol::T = √eps(T)             ),
                   :(; itmax::Int = 0                ),
                   :(; timemax::Float64 = Inf        ),
                   :(; verbose::Int = 0              ),
                   :(; history::Bool = false         ),
                   :(; callback = solver -> false    ),
                   :(; iostream::IO = kstdout        ))

def_kwargs_lnlq = extract_parameters.(def_kwargs_lnlq)

args_lnlq = (:A, :b)
kwargs_lnlq = (:M, :N, :ldiv, :transfer_to_craig, :sqd, :λ, :σ, :utolx, :utoly, :atol, :rtol, :itmax, :timemax, :verbose, :history, :callback, :iostream)

@eval begin
  function lnlq!(solver :: LnlqSolver{T,FC,S}, $(def_args_lnlq...); $(def_kwargs_lnlq...)) where {T <: AbstractFloat, FC <: FloatOrComplex{T}, S <: AbstractVector{FC}}

    # Timer
    start_time = time_ns()
    timemax_ns = 1e9 * timemax

    m, n = size(A)
    (m == solver.m && n == solver.n) || error("(solver.m, solver.n) = ($(solver.m), $(solver.n)) is inconsistent with size(A) = ($m, $n)")
    length(b) == m || error("Inconsistent problem size")
    (verbose > 0) && @printf(iostream, "LNLQ: system of %d equations in %d variables\n", m, n)

    # Check sqd and λ parameters
    sqd && (λ ≠ 0) && error("sqd cannot be set to true if λ ≠ 0 !")
    sqd && (λ = one(T))

    # Tests M = Iₘ and N = Iₙ
    MisI = (M === I)
    NisI = (N === I)

    # Check type consistency
    eltype(A) == FC || @warn "eltype(A) ≠ $FC. This could lead to errors or additional allocations in operator-vector products."
    ktypeof(b) <: S || error("ktypeof(b) is not a subtype of $S")

    # Compute the adjoint of A
    Aᴴ = A'

    # Set up workspace.
    allocate_if(!MisI, solver, :u, S, solver.y)  # The length of u is m
    allocate_if(!NisI, solver, :v, S, solver.x)  # The length of v is n
    allocate_if(λ > 0, solver, :q, S, solver.x)  # The length of q is n
    x, Nv, Aᴴu, y, w̄ = solver.x, solver.Nv, solver.Aᴴu, solver.y, solver.w̄
    Mu, Av, q, stats = solver.Mu, solver.Av, solver.q, solver.stats
    rNorms, xNorms, yNorms = stats.residuals, stats.error_bnd_x, stats.error_bnd_y
    reset!(stats)
    u = MisI ? Mu : solver.u
    v = NisI ? Nv : solver.v

    # Set up parameter σₑₛₜ for the error estimate on x and y
    σₑₛₜ = √(σ^2 + λ^2)
    complex_error_bnd = false

    # Initial solutions (x₀, y₀) and residual norm ‖r₀‖.
    kfill!(x, zero(FC))
    kfill!(y, zero(FC))

    bNorm = knorm(m, b)
    if bNorm == 0
      stats.niter = 0
      stats.solved = true
      stats.error_with_bnd = false
      history && push!(rNorms, bNorm)
      stats.timer = start_time |> ktimer
      stats.status = "x is a zero-residual solution"
      return solver
    end

    history && push!(rNorms, bNorm)
    ε = atol + rtol * bNorm

    iter = 0
    itmax == 0 && (itmax = m + n)

    (verbose > 0) && @printf(iostream, "%5s  %7s  %5s\n", "k", "‖rₖ‖", "timer")
    kdisplay(iter, verbose) && @printf(iostream, "%5d  %7.1e  %.2fs\n", iter, bNorm, start_time |> ktimer)

    # Update iteration index
    iter = iter + 1

    # Initialize generalized Golub-Kahan bidiagonalization.
    # β₁Mu₁ = b.
    kcopy!(m, Mu, b)  # Mu ← b
    MisI || mulorldiv!(u, M, Mu, ldiv)  # u₁ = M⁻¹ * Mu₁
    βₖ = knorm_elliptic(m, u, Mu)       # β₁ = ‖u₁‖_M
    if βₖ ≠ 0
      kscal!(m, one(FC) / βₖ, u)
      MisI || kscal!(m, one(FC) / βₖ, Mu)
    end

    # α₁Nv₁ = Aᴴu₁.
    mul!(Aᴴu, Aᴴ, u)
    kcopy!(n, Nv, Aᴴu)  # Nv ← Aᴴu
    NisI || mulorldiv!(v, N, Nv, ldiv)  # v₁ = N⁻¹ * Nv₁
    αₖ = knorm_elliptic(n, v, Nv)       # α₁ = ‖v₁‖_N
    if αₖ ≠ 0
      kscal!(n, one(FC) / αₖ, v)
      NisI || kscal!(n, one(FC) / αₖ, Nv)
    end

    kcopy!(m, w̄, u)  # Direction w̄₁
    cₖ = zero(T)     # Givens cosines used for the LQ factorization of (Lₖ)ᴴ
    sₖ = zero(FC)    # Givens sines used for the LQ factorization of (Lₖ)ᴴ
    ζₖ₋₁ = zero(FC)  # ζₖ₋₁ and ζbarₖ are the last components of z̅ₖ
    ηₖ = zero(FC)    # Coefficient of M̅ₖ

    # Variable used for the regularization.
    λₖ  = λ              # λ₁ = λ
    cpₖ = spₖ = one(T)   # Givens sines and cosines used to zero out λₖ
    cdₖ = sdₖ = one(FC)  # Givens sines and cosines used to define λₖ₊₁
    λ > 0 && kcopy!(n, q, v)  # Additional vector needed to update x, by definition q₀ = 0

    # Initialize the regularization.
    if λ > 0
      #        k    2k      k   2k           k      2k
      # k   [  αₖ   λₖ ] [ cpₖ  spₖ ] = [  αhatₖ    0   ]
      # k+1 [ βₖ₊₁  0  ] [ spₖ -cpₖ ]   [ βhatₖ₊₁  θₖ₊₁ ]
      (cpₖ, spₖ, αhatₖ) = sym_givens(αₖ, λₖ)

      # q̄₁ = sp₁ * v₁
      kscal!(n, spₖ, q)
    else
      αhatₖ = αₖ
    end

    # Begin the LQ factorization of (Lₖ)ᴴ = M̅ₖQₖ.
    # [ α₁ β₂ 0  •  •  •  0 ]   [ ϵ₁  0   •   •   •   •   0   ]
    # [ 0  α₂ •  •        • ]   [ η₂  ϵ₂  •               •   ]
    # [ •  •  •  •  •     • ]   [ 0   •   •   •           •   ]
    # [ •     •  •  •  •  • ] = [ •   •   •   •   •       •   ] Qₖ
    # [ •        •  •  •  0 ]   [ •       •   •   •   •   •   ]
    # [ •           •  •  βₖ]   [ •           •   •   •   0   ]
    # [ 0  •  •  •  •  0  αₖ]   [ 0   •   •   •   0   ηₖ ϵbarₖ]

    ϵbarₖ = αhatₖ  # ϵbar₁ = αhat₁

    # Hₖ = Bₖ(Lₖ)ᴴ = [   Lₖ(Lₖ)ᴴ   ] ⟹ (Hₖ₋₁)ᴴ = [Lₖ₋₁Mₖ₋₁  0] Qₖ
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
    user_requested_exit = false
    overtimed = false

    if σₑₛₜ > 0
      τtildeₖ = βₖ / σₑₛₜ
      ζtildeₖ = τtildeₖ / σₑₛₜ
      err_x = τtildeₖ
      err_y = ζtildeₖ

      solved_lq = err_x ≤ utolx || err_y ≤ utoly
      history && push!(xNorms, err_x)
      history && push!(yNorms, err_y)

      ρbar = -σₑₛₜ
      csig = -one(T)
    end

    while !(solved_lq || solved_cg || tired || user_requested_exit || overtimed)

      # Update of (xᵃᵘˣ)ₖ = Vₖtₖ
      if λ > 0
        # (xᵃᵘˣ)ₖ ← (xᵃᵘˣ)ₖ₋₁ + τₖ * (cpₖvₖ + spₖqₖ₋₁)
        kaxpy!(n, τₖ * cpₖ, v, x)
        if iter ≥ 2
          kaxpy!(n, τₖ * spₖ, q, x)
          # q̄ₖ ← spₖ * vₖ - cpₖ * qₖ₋₁
          kaxpby!(n, spₖ, v, -cpₖ, q)
        end
      else
        # (xᵃᵘˣ)ₖ ← (xᵃᵘˣ)ₖ₋₁ + τₖ * vₖ
        kaxpy!(n, τₖ, v, x)
      end

      # Continue the generalized Golub-Kahan bidiagonalization.
      # AVₖ    = MUₖ₊₁Bₖ
      # AᴴUₖ₊₁ = NVₖ(Bₖ)ᴴ + αₖ₊₁Nvₖ₊₁(eₖ₊₁)ᴴ = NVₖ₊₁(Lₖ₊₁)ᴴ
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
      kaxpby!(m, one(FC), Av, -αₖ, Mu)
      MisI || mulorldiv!(u, M, Mu, ldiv)  # uₖ₊₁ = M⁻¹ * Muₖ₊₁
      βₖ₊₁ = knorm_elliptic(m, u, Mu)     # βₖ₊₁ = ‖uₖ₊₁‖_M
      if βₖ₊₁ ≠ 0
        kscal!(m, one(FC) / βₖ₊₁, u)
        MisI || kscal!(m, one(FC) / βₖ₊₁, Mu)
      end

      # αₖ₊₁Nvₖ₊₁ = Aᴴuₖ₊₁ - βₖ₊₁Nvₖ
      mul!(Aᴴu, Aᴴ, u)
      kaxpby!(n, one(FC), Aᴴu, -βₖ₊₁, Nv)
      NisI || mulorldiv!(v, N, Nv, ldiv)  # vₖ₊₁ = N⁻¹ * Nvₖ₊₁
      αₖ₊₁ = knorm_elliptic(n, v, Nv)     # αₖ₊₁ = ‖vₖ₊₁‖_N
      if αₖ₊₁ ≠ 0
        kscal!(n, one(FC) / αₖ₊₁, v)
        NisI || kscal!(n, one(FC) / αₖ₊₁, Nv)
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
        kscal!(n, sdₖ, q)

        #       k+1   2k+1      k+1    2k+1        k+1     2k+1
        # k+1 [ αₖ₊₁  λₖ₊₁ ] [ cpₖ₊₁  spₖ₊₁ ] = [ αhatₖ₊₁   0   ]
        # k+2 [ βₖ₊₂   0   ] [ spₖ₊₁ -cpₖ₊₁ ]   [  γₖ₊₂    θₖ₊₂ ]
        (cpₖ₊₁, spₖ₊₁, αhatₖ₊₁) = sym_givens(αₖ₊₁, λₖ₊₁)
      else
        βhatₖ₊₁ = βₖ₊₁
        αhatₖ₊₁ = αₖ₊₁
      end

      if σₑₛₜ > 0 && !complex_error_bnd
        μbar = -csig * αhatₖ
        ρ = √(ρbar^2 + αhatₖ^2)
        csig = ρbar / ρ
        ssig = αhatₖ / ρ
        ρbar = ssig * μbar + csig * σₑₛₜ
        μbar = -csig * βhatₖ₊₁
        θ = βhatₖ₊₁ * csig / ρbar
        ωdisc = σₑₛₜ^2 - σₑₛₜ * βhatₖ₊₁ * θ
        if ωdisc < 0
          complex_error_bnd = true
        else
          ω = √ωdisc
          τtildeₖ = - τₖ * βhatₖ₊₁ / ω
        end

        ρ = √(ρbar^2 + βhatₖ₊₁^2)
        csig = ρbar / ρ
        ssig = βhatₖ₊₁ / ρ
        ρbar = ssig * μbar + csig * σₑₛₜ
      end

      # Continue the LQ factorization of (Lₖ₊₁)ᴴ.
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
      kaxpy!(m, ζₖ * cₖ₊₁, w̄, y)
      kaxpy!(m, ζₖ * sₖ₊₁, u, y)

      # Compute w̄ₖ₊₁
      kaxpby!(m, -cₖ₊₁, u, sₖ₊₁, w̄)

      if σₑₛₜ > 0 && !complex_error_bnd
        if transfer_to_craig
          disc_x = τtildeₖ^2 - τₖ₊₁^2
          disc_x < 0 ? complex_error_bnd = true : err_x = √disc_x
        else
          disc_xL = τtildeₖ^2 - τₖ₊₁^2 + (τₖ₊₁ - ηₖ₊₁ * ζₖ)^2
          disc_xL < 0 ? complex_error_bnd = true : err_x = √disc_xL
        end
        ηtildeₖ = ω * sₖ₊₁
        ϵtildeₖ = -ω * cₖ₊₁
        ζtildeₖ = (τtildeₖ - ηtildeₖ * ζₖ) / ϵtildeₖ
        
        if transfer_to_craig
          disc_y = ζtildeₖ^2 - ζbarₖ₊₁^2
          disc_y < 0 ? complex_error_bnd = true : err_y = √disc_y
        else
          err_y = abs(ζtildeₖ)
        end

        history && push!(xNorms, err_x)
        history && push!(yNorms, err_y)
      end

      # Compute residual norm ‖(rᴸ)ₖ‖ = |αₖ| * √(|ϵbarₖζbarₖ|² + |βₖ₊₁sₖζₖ₋₁|²)
      if iter == 1
        rNorm_lq = bNorm
      else
        rNorm_lq = abs(αhatₖ) * √(abs2(ϵbarₖ * ζbarₖ) + abs2(βhatₖ₊₁ * sₖ * ζₖ₋₁))
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
      user_requested_exit = callback(solver) :: Bool
      tired = iter ≥ itmax
      solved_lq = rNorm_lq ≤ ε
      solved_cg = transfer_to_craig && rNorm_cg ≤ ε
      if σₑₛₜ > 0
        solved_lq = solved_lq || err_x ≤ utolx || err_y ≤ utoly
        solved_cg = transfer_to_craig && (solved_cg || err_x ≤ utolx || err_y ≤ utoly)
      end
      timer = time_ns() - start_time
      overtimed = timer > timemax_ns
      kdisplay(iter, verbose) && @printf(iostream, "%5d  %7.1e  %.2fs\n", iter, rNorm_lq, start_time |> ktimer)

      # Update iteration index.
      iter = iter + 1
    end
    (verbose > 0) && @printf(iostream, "\n")

    if solved_cg
      if λ > 0
        # (xᶜ)ₖ ← (xᵃᵘˣ)ₖ₋₁ + τₖ * (cpₖvₖ + spₖqₖ₋₁)
        kaxpy!(n, τₖ * cpₖ, v, x)
        if iter ≥ 2
          kaxpy!(n, τₖ * spₖ, q, x)
        end
      else
        # (xᶜ)ₖ ← (xᵃᵘˣ)ₖ₋₁ + τₖ * vₖ
        kaxpy!(n, τₖ, v, x)
      end
      # (yᶜ)ₖ ← (yᴸ)ₖ₋₁ + ζbarₖ * w̄ₖ
      kaxpy!(m, ζbarₖ, w̄, y)
    else
      if λ > 0
        # (xᴸ)ₖ ← (xᵃᵘˣ)ₖ₋₁ + ηₖζₖ₋₁ * (cpₖvₖ + spₖqₖ₋₁)
        kaxpy!(n, ηₖ * ζₖ₋₁ * cpₖ, v, x)
        if iter ≥ 2
          kaxpy!(n, ηₖ * ζₖ₋₁ * spₖ, q, x)
        end
      else
        # (xᴸ)ₖ ← (xᵃᵘˣ)ₖ₋₁ + ηₖζₖ₋₁ * vₖ
        kaxpy!(n, ηₖ * ζₖ₋₁, v, x)
      end
    end

    # Termination status
    tired               && (status = "maximum number of iterations exceeded")
    solved_lq           && (status = "solutions (xᴸ, yᴸ) good enough for the tolerances given")
    solved_cg           && (status = "solutions (xᶜ, yᶜ) good enough for the tolerances given")
    user_requested_exit && (status = "user-requested exit")
    overtimed           && (status = "time limit exceeded")

    # Update stats
    stats.niter = iter
    stats.solved = solved_lq || solved_cg
    stats.error_with_bnd = complex_error_bnd
    stats.timer = start_time |> ktimer
    stats.status = status
    return solver
  end
end
