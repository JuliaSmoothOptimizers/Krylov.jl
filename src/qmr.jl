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
    (x, stats) = qmr(A, b::AbstractVector{FC};
                     c::AbstractVector{FC}=b, M=I, N=I, ldiv::Bool=false, atol::T=√eps(T),
                     rtol::T=√eps(T), itmax::Int=0, timemax::Float64=Inf, verbose::Int=0,
                     history::Bool=false, callback=solver->false, iostream::IO=kstdout)

`T` is an `AbstractFloat` such as `Float32`, `Float64` or `BigFloat`.
`FC` is `T` or `Complex{T}`.

    (x, stats) = qmr(A, b, x0::AbstractVector; kwargs...)

QMR can be warm-started from an initial guess `x0` where `kwargs` are the same keyword arguments as above.

Solve the square linear system Ax = b of size n using QMR.

QMR is based on the Lanczos biorthogonalization process and requires two initial vectors `b` and `c`.
The relation `bᴴc ≠ 0` must be satisfied and by default `c = b`.
When `A` is Hermitian and `b = c`, QMR is equivalent to MINRES.
QMR requires support for `adjoint(M)` and `adjoint(N)` if preconditioners are provided.

#### Input arguments

* `A`: a linear operator that models a matrix of dimension `n`;
* `b`: a vector of length `n`.

#### Optional argument

* `x0`: a vector of length `n` that represents an initial guess of the solution `x`.

#### Keyword arguments

* `c`: the second initial vector of length `n` required by the Lanczos biorthogonalization process;
* `M`: linear operator that models a nonsingular matrix of size `n` used for left preconditioning;
* `N`: linear operator that models a nonsingular matrix of size `n` used for right preconditioning;
* `ldiv`: define whether the preconditioners use `ldiv!` or `mul!`;
* `atol`: absolute stopping tolerance based on the residual norm;
* `rtol`: relative stopping tolerance based on the residual norm;
* `itmax`: the maximum number of iterations. If `itmax=0`, the default number of iterations is set to `2n`;
* `timemax`: the time limit in seconds;
* `verbose`: additional details can be displayed if verbose mode is enabled (verbose > 0). Information will be displayed every `verbose` iterations;
* `history`: collect additional statistics on the run such as residual norms, or Aᴴ-residual norms;
* `callback`: function or functor called as `callback(solver)` that returns `true` if the Krylov method should terminate, and `false` otherwise;
* `iostream`: stream to which output is logged.

#### Output arguments

* `x`: a dense vector of length `n`;
* `stats`: statistics collected on the run in a [`SimpleStats`](@ref) structure.

#### References

* R. W. Freund and N. M. Nachtigal, [*QMR : a quasi-minimal residual method for non-Hermitian linear systems*](https://doi.org/10.1007/BF01385726), Numerische mathematik, Vol. 60(1), pp. 315--339, 1991.
* R. W. Freund and N. M. Nachtigal, [*An implementation of the QMR method based on coupled two-term recurrences*](https://doi.org/10.1137/0915022), SIAM Journal on Scientific Computing, Vol. 15(2), pp. 313--337, 1994.
* A. Montoison and D. Orban, [*BiLQ: An Iterative Method for Nonsymmetric Linear Systems with a Quasi-Minimum Error Property*](https://doi.org/10.1137/19M1290991), SIAM Journal on Matrix Analysis and Applications, 41(3), pp. 1145--1166, 2020.
"""
function qmr end

"""
    solver = qmr!(solver::QmrSolver, A, b; kwargs...)
    solver = qmr!(solver::QmrSolver, A, b, x0; kwargs...)

where `kwargs` are keyword arguments of [`qmr`](@ref).

See [`QmrSolver`](@ref) for more details about the `solver`.
"""
function qmr! end

def_args_qmr = (:(A                    ),
                :(b::AbstractVector{FC}))

def_optargs_qmr = (:(x0::AbstractVector),)

def_kwargs_qmr = (:(; c::AbstractVector{FC} = b ),
                  :(; M = I                     ),
                  :(; N = I                     ),
                  :(; ldiv::Bool = false        ),
                  :(; atol::T = √eps(T)         ),
                  :(; rtol::T = √eps(T)         ),
                  :(; itmax::Int = 0            ),
                  :(; timemax::Float64 = Inf    ),
                  :(; verbose::Int = 0          ),
                  :(; history::Bool = false     ),
                  :(; callback = solver -> false),
                  :(; iostream::IO = kstdout    ))

def_kwargs_qmr = extract_parameters.(def_kwargs_qmr)

args_qmr = (:A, :b)
optargs_qmr = (:x0,)
kwargs_qmr = (:c, :M, :N, :ldiv, :atol, :rtol, :itmax, :timemax, :verbose, :history, :callback, :iostream)

@eval begin
  function qmr!(solver :: QmrSolver{T,FC,S}, $(def_args_qmr...); $(def_kwargs_qmr...)) where {T <: AbstractFloat, FC <: FloatOrComplex{T}, S <: AbstractVector{FC}}

    # Timer
    start_time = time_ns()
    timemax_ns = 1e9 * timemax

    m, n = size(A)
    (m == solver.m && n == solver.n) || error("(solver.m, solver.n) = ($(solver.m), $(solver.n)) is inconsistent with size(A) = ($m, $n)")
    m == n || error("System must be square")
    length(b) == m || error("Inconsistent problem size")
    (verbose > 0) && @printf(iostream, "QMR: system of size %d\n", n)

    # Check M = Iₙ and N = Iₙ
    MisI = (M === I)
    NisI = (N === I)

    # Check type consistency
    eltype(A) == FC || @warn "eltype(A) ≠ $FC. This could lead to errors or additional allocations in operator-vector products."
    ktypeof(b) <: S || error("ktypeof(b) is not a subtype of $S")
    ktypeof(c) <: S || error("ktypeof(c) is not a subtype of $S")

    # Compute the adjoint of A, M and N
    Aᴴ = A'
    Mᴴ = M'
    Nᴴ = N'

    # Set up workspace.
    allocate_if(!MisI, solver, :t, S, solver.x)  # The length of t is n
    allocate_if(!NisI, solver, :s, S, solver.x)  # The length of s is n
    uₖ₋₁, uₖ, q, vₖ₋₁, vₖ, p = solver.uₖ₋₁, solver.uₖ, solver.q, solver.vₖ₋₁, solver.vₖ, solver.p
    Δx, x, wₖ₋₂, wₖ₋₁, stats = solver.Δx, solver.x, solver.wₖ₋₂, solver.wₖ₋₁, solver.stats
    warm_start = solver.warm_start
    rNorms = stats.residuals
    reset!(stats)
    r₀ = warm_start ? q : b
    Mᴴuₖ = MisI ? uₖ : solver.t
    t = MisI ? q : solver.t
    Nvₖ = NisI ? vₖ : solver.s
    s = NisI ? p : solver.s

    if warm_start
      mul!(r₀, A, Δx)
      kaxpby!(n, one(FC), b, -one(FC), r₀)
    end
    if !MisI
      mulorldiv!(solver.t, M, r₀, ldiv)
      r₀ = solver.t
    end

    # Initial solution x₀ and residual norm ‖r₀‖.
    kfill!(x, zero(FC))
    rNorm = knorm(n, r₀)  # ‖r₀‖ = ‖b₀ - Ax₀‖

    history && push!(rNorms, rNorm)
    if rNorm == 0
      stats.niter = 0
      stats.solved = true
      stats.inconsistent = false
      stats.timer = start_time |> ktimer
      stats.status = "x is a zero-residual solution"
      warm_start && kaxpy!(n, one(FC), Δx, x)
      solver.warm_start = false
      return solver
    end

    iter = 0
    itmax == 0 && (itmax = 2*n)

    # Initialize the Lanczos biorthogonalization process.
    cᴴb = kdot(n, c, r₀)  # ⟨c,r₀⟩
    if cᴴb == 0
      stats.niter = 0
      stats.solved = false
      stats.inconsistent = false
      stats.timer = start_time |> ktimer
      stats.status = "Breakdown bᴴc = 0"
      warm_start && kaxpy!(n, one(FC), Δx, x)
      solver.warm_start = false
      return solver
    end

    ε = atol + rtol * rNorm
    (verbose > 0) && @printf(iostream, "%5s  %8s  %7s  %5s\n", "k", "αₖ", "‖rₖ‖", "timer")
    kdisplay(iter, verbose) && @printf(iostream, "%5d  %8.1e  %7.1e  %.2fs\n", iter, cᴴb, rNorm, start_time |> ktimer)

    βₖ = √(abs(cᴴb))             # β₁γ₁ = cᴴ(b - Ax₀)
    γₖ = cᴴb / βₖ                # β₁γ₁ = cᴴ(b - Ax₀)
    kfill!(vₖ₋₁, zero(FC))       # v₀ = 0
    kfill!(uₖ₋₁, zero(FC))       # u₀ = 0
    vₖ .= r₀ ./ βₖ               # v₁ = (b - Ax₀) / β₁
    uₖ .= c ./ conj(γₖ)          # u₁ = c / γ̄₁
    cₖ₋₂ = cₖ₋₁ = cₖ = zero(T)   # Givens cosines used for the QR factorization of Tₖ₊₁.ₖ
    sₖ₋₂ = sₖ₋₁ = sₖ = zero(FC)  # Givens sines used for the QR factorization of Tₖ₊₁.ₖ
    kfill!(wₖ₋₂, zero(FC))       # Column k-2 of Wₖ = Vₖ(Rₖ)⁻¹
    kfill!(wₖ₋₁, zero(FC))       # Column k-1 of Wₖ = Vₖ(Rₖ)⁻¹
    ζbarₖ = βₖ                   # ζbarₖ is the last component of z̅ₖ = (Qₖ)ᴴβ₁e₁
    τₖ = kdotr(n, vₖ, vₖ)        # τₖ is used for the residual norm estimate

    # Stopping criterion.
    solved    = rNorm ≤ ε
    breakdown = false
    tired     = iter ≥ itmax
    status    = "unknown"
    user_requested_exit = false
    overtimed = false

    while !(solved || tired || breakdown || user_requested_exit || overtimed)
      # Update iteration index.
      iter = iter + 1

      # Continue the Lanczos biorthogonalization process.
      # MANVₖ    = VₖTₖ    + βₖ₊₁vₖ₊₁(eₖ)ᵀ = Vₖ₊₁Tₖ₊₁.ₖ
      # NᴴAᴴMᴴUₖ = Uₖ(Tₖ)ᴴ + γ̄ₖ₊₁uₖ₊₁(eₖ)ᵀ = Uₖ₊₁(Tₖ.ₖ₊₁)ᴴ

      # Forms vₖ₊₁ : q ← MANvₖ
      NisI || mulorldiv!(Nvₖ, N, vₖ, ldiv)
      mul!(t, A, Nvₖ)
      MisI || mulorldiv!(q, M, t, ldiv)

      # Forms uₖ₊₁ : p ← NᴴAᴴMᴴuₖ
      MisI || mulorldiv!(Mᴴuₖ, Mᴴ, uₖ, ldiv)
      mul!(s, Aᴴ, Mᴴuₖ)
      NisI || mulorldiv!(p, Nᴴ, s, ldiv)

      kaxpy!(n, -γₖ, vₖ₋₁, q)  # q ← q - γₖ * vₖ₋₁
      kaxpy!(n, -βₖ, uₖ₋₁, p)  # p ← p - β̄ₖ * uₖ₋₁

      αₖ = kdot(n, uₖ, q)  # αₖ = ⟨uₖ,q⟩

      kaxpy!(n, -     αₖ , vₖ, q)  # q ← q - αₖ * vₖ
      kaxpy!(n, -conj(αₖ), uₖ, p)  # p ← p - ᾱₖ * uₖ

      pᴴq = kdot(n, p, q)  # pᴴq  = ⟨p,q⟩
      βₖ₊₁ = √(abs(pᴴq))   # βₖ₊₁ = √(|pᴴq|)
      γₖ₊₁ = pᴴq / βₖ₊₁    # γₖ₊₁ = pᴴq / βₖ₊₁

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
      # If k = 1, we don't have any previous reflection.
      # If k = 2, we apply the last reflection.
      # If k ≥ 3, we only apply the two previous reflections.

      # Apply previous Givens reflections Qₖ₋₂.ₖ₋₁
      if iter ≥ 3
        # [cₖ₋₂  sₖ₋₂] [0 ] = [  ϵₖ₋₂ ]
        # [s̄ₖ₋₂ -cₖ₋₂] [γₖ]   [λbarₖ₋₁]
        ϵₖ₋₂    =  sₖ₋₂ * γₖ
        λbarₖ₋₁ = -cₖ₋₂ * γₖ
      end

      # Apply previous Givens reflections Qₖ₋₁.ₖ
      if iter ≥ 2
        iter == 2 && (λbarₖ₋₁ = γₖ)
        # [cₖ₋₁  sₖ₋₁] [λbarₖ₋₁] = [λₖ₋₁ ]
        # [s̄ₖ₋₁ -cₖ₋₁] [   αₖ  ]   [δbarₖ]
        λₖ₋₁  =      cₖ₋₁  * λbarₖ₋₁ + sₖ₋₁ * αₖ
        δbarₖ = conj(sₖ₋₁) * λbarₖ₋₁ - cₖ₋₁ * αₖ

        # Update sₖ₋₂ and cₖ₋₂.
        sₖ₋₂ = sₖ₋₁
        cₖ₋₂ = cₖ₋₁
      end

      # Compute and apply current Givens reflection Qₖ.ₖ₊₁
      iter == 1 && (δbarₖ = αₖ)
      # [cₖ  sₖ] [δbarₖ] = [δₖ]
      # [s̄ₖ -cₖ] [βₖ₊₁ ]   [0 ]
      (cₖ, sₖ, δₖ) = sym_givens(δbarₖ, βₖ₊₁)

      # Update z̅ₖ₊₁ = Qₖ.ₖ₊₁ [ z̄ₖ ]
      #                      [ 0  ]
      #
      # [cₖ  sₖ] [ζbarₖ] = [   ζₖ  ]
      # [s̄ₖ -cₖ] [  0  ]   [ζbarₖ₊₁]
      ζₖ      =      cₖ  * ζbarₖ
      ζbarₖ₊₁ = conj(sₖ) * ζbarₖ

      # Update sₖ₋₁ and cₖ₋₁.
      sₖ₋₁ = sₖ
      cₖ₋₁ = cₖ

      # Compute the direction wₖ, the last column of Wₖ = Vₖ(Rₖ)⁻¹ ⟷ (Rₖ)ᵀ(Wₖ)ᵀ = (Vₖ)ᵀ.
      # w₁ = v₁ / δ₁
      if iter == 1
        wₖ = wₖ₋₁
        kaxpy!(n, one(FC), vₖ, wₖ)
        wₖ .= wₖ ./ δₖ
      end
      # w₂ = (v₂ - λ₁w₁) / δ₂
      if iter == 2
        wₖ = wₖ₋₂
        kaxpy!(n, -λₖ₋₁, wₖ₋₁, wₖ)
        kaxpy!(n, one(FC), vₖ, wₖ)
        wₖ .= wₖ ./ δₖ
      end
      # wₖ = (vₖ - λₖ₋₁wₖ₋₁ - ϵₖ₋₂wₖ₋₂) / δₖ
      if iter ≥ 3
        kscal!(n, -ϵₖ₋₂, wₖ₋₂)
        wₖ = wₖ₋₂
        kaxpy!(n, -λₖ₋₁, wₖ₋₁, wₖ)
        kaxpy!(n, one(FC), vₖ, wₖ)
        wₖ .= wₖ ./ δₖ
      end

      # Compute solution xₖ.
      # xₖ ← xₖ₋₁ + ζₖ * wₖ
      kaxpy!(n, ζₖ, wₖ, x)

      # Compute vₖ₊₁ and uₖ₊₁.
      kcopy!(n, vₖ₋₁, vₖ)  # vₖ₋₁ ← vₖ
      kcopy!(n, uₖ₋₁, uₖ)  # uₖ₋₁ ← uₖ

      if pᴴq ≠ zero(FC)
        vₖ .= q ./ βₖ₊₁        # βₖ₊₁vₖ₊₁ = q
        uₖ .= p ./ conj(γₖ₊₁)  # γ̄ₖ₊₁uₖ₊₁ = p
      end

      # Compute τₖ₊₁ = τₖ + ‖vₖ₊₁‖²
      τₖ₊₁ = τₖ + kdotr(n, vₖ, vₖ)

      # Compute ‖rₖ‖ ≤ |ζbarₖ₊₁|√τₖ₊₁
      rNorm = abs(ζbarₖ₊₁) * √τₖ₊₁
      history && push!(rNorms, rNorm)

      # Update directions for x.
      if iter ≥ 2
        @kswap!(wₖ₋₂, wₖ₋₁)
      end

      # Update ζbarₖ, βₖ, γₖ and τₖ.
      ζbarₖ = ζbarₖ₊₁
      βₖ    = βₖ₊₁
      γₖ    = γₖ₊₁
      τₖ    = τₖ₊₁

      # Stopping conditions that do not depend on user input.
      # This is to guard against tolerances that are unreasonably small.
      resid_decrease_mach = (rNorm + one(T) ≤ one(T))

      # Update stopping criterion.
      user_requested_exit = callback(solver) :: Bool
      resid_decrease_lim = rNorm ≤ ε
      solved = resid_decrease_lim || resid_decrease_mach
      tired = iter ≥ itmax
      breakdown = !solved && (pᴴq == 0)
      timer = time_ns() - start_time
      overtimed = timer > timemax_ns
      kdisplay(iter, verbose) && @printf(iostream, "%5d  %8.1e  %7.1e  %.2fs\n", iter, αₖ, rNorm, start_time |> ktimer)
    end
    (verbose > 0) && @printf(iostream, "\n")

    # Termination status
    tired               && (status = "maximum number of iterations exceeded")
    breakdown           && (status = "Breakdown ⟨uₖ₊₁,vₖ₊₁⟩ = 0")
    solved              && (status = "solution good enough given atol and rtol")
    user_requested_exit && (status = "user-requested exit")
    overtimed           && (status = "time limit exceeded")

    # Update x
    if !NisI
      copyto!(solver.s, x)
      mulorldiv!(x, N, solver.s, ldiv)
    end
    warm_start && kaxpy!(n, one(FC), Δx, x)
    solver.warm_start = false

    # Update stats
    stats.niter = iter
    stats.solved = solved
    stats.inconsistent = false
    stats.timer = start_time |> ktimer
    stats.status = status
    return solver
  end
end
