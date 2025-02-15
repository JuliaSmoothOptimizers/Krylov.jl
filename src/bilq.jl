# An implementation of BiLQ for the solution of unsymmetric
# and square consistent linear system Ax = b.
#
# This method is described in
#
# A. Montoison and D. Orban
# BiLQ: An Iterative Method for Nonsymmetric Linear Systems with a Quasi-Minimum Error Property.
# SIAM Journal on Matrix Analysis and Applications, 41(3), pp. 1145--1166, 2020.
#
# Alexis Montoison, <alexis.montoison@polymtl.ca>
# Montreal, February 2019.

export bilq, bilq!

"""
    (x, stats) = bilq(A, b::AbstractVector{FC};
                      c::AbstractVector{FC}=b, transfer_to_bicg::Bool=true,
                      M=I, N=I, ldiv::Bool=false, atol::T=√eps(T),
                      rtol::T=√eps(T), itmax::Int=0, timemax::Float64=Inf,
                      verbose::Int=0, history::Bool=false,
                      callback=solver->false, iostream::IO=kstdout)

`T` is an `AbstractFloat` such as `Float32`, `Float64` or `BigFloat`.
`FC` is `T` or `Complex{T}`.

    (x, stats) = bilq(A, b, x0::AbstractVector; kwargs...)

BiLQ can be warm-started from an initial guess `x0` where `kwargs` are the same keyword arguments as above.

Solve the square linear system Ax = b of size n using BiLQ.
BiLQ is based on the Lanczos biorthogonalization process and requires two initial vectors `b` and `c`.
The relation `bᴴc ≠ 0` must be satisfied and by default `c = b`.
When `A` is Hermitian and `b = c`, BiLQ is equivalent to SYMMLQ.
BiLQ requires support for `adjoint(M)` and `adjoint(N)` if preconditioners are provided.

#### Input arguments

* `A`: a linear operator that models a matrix of dimension `n`;
* `b`: a vector of length `n`.

#### Optional argument

* `x0`: a vector of length `n` that represents an initial guess of the solution `x`.

#### Keyword arguments

* `c`: the second initial vector of length `n` required by the Lanczos biorthogonalization process;
* `transfer_to_bicg`: transfer from the BiLQ point to the BiCG point, when it exists. The transfer is based on the residual norm;
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

* A. Montoison and D. Orban, [*BiLQ: An Iterative Method for Nonsymmetric Linear Systems with a Quasi-Minimum Error Property*](https://doi.org/10.1137/19M1290991), SIAM Journal on Matrix Analysis and Applications, 41(3), pp. 1145--1166, 2020.
* R. Fletcher, [*Conjugate gradient methods for indefinite systems*](https://doi.org/10.1007/BFb0080116), Numerical Analysis, Springer, pp. 73--89, 1976.
"""
function bilq end

"""
    solver = bilq!(solver::BilqSolver, A, b; kwargs...)
    solver = bilq!(solver::BilqSolver, A, b, x0; kwargs...)

where `kwargs` are keyword arguments of [`bilq`](@ref).

See [`BilqSolver`](@ref) for more details about the `solver`.
"""
function bilq! end

def_args_bilq = (:(A                    ),
                 :(b::AbstractVector{FC}))

def_optargs_bilq = (:(x0::AbstractVector),)

def_kwargs_bilq = (:(; c::AbstractVector{FC} = b    ),
                   :(; transfer_to_bicg::Bool = true),
                   :(; M = I                        ),
                   :(; N = I                        ),
                   :(; ldiv::Bool = false           ),
                   :(; atol::T = √eps(T)            ),
                   :(; rtol::T = √eps(T)            ),
                   :(; itmax::Int = 0               ),
                   :(; timemax::Float64 = Inf       ),
                   :(; verbose::Int = 0             ),
                   :(; history::Bool = false        ),
                   :(; callback = solver -> false   ),
                   :(; iostream::IO = kstdout       ))

def_kwargs_bilq = extract_parameters.(def_kwargs_bilq)

args_bilq = (:A, :b)
optargs_bilq = (:x0,)
kwargs_bilq = (:c, :transfer_to_bicg, :M, :N, :ldiv, :atol, :rtol, :itmax, :timemax, :verbose, :history, :callback, :iostream)

@eval begin
  function bilq!(solver :: BilqSolver{T,FC,S}, $(def_args_bilq...); $(def_kwargs_bilq...)) where {T <: AbstractFloat, FC <: FloatOrComplex{T}, S <: AbstractVector{FC}}

    # Timer
    start_time = time_ns()
    timemax_ns = 1e9 * timemax

    m, n = size(A)
    (m == solver.m && n == solver.n) || error("(solver.m, solver.n) = ($(solver.m), $(solver.n)) is inconsistent with size(A) = ($m, $n)")
    m == n || error("System must be square")
    length(b) == m || error("Inconsistent problem size")
    (verbose > 0) && @printf(iostream, "BILQ: system of size %d\n", n)

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
    uₖ₋₁, uₖ, q, vₖ₋₁, vₖ = solver.uₖ₋₁, solver.uₖ, solver.q, solver.vₖ₋₁, solver.vₖ
    p, Δx, x, d̅, stats = solver.p, solver.Δx, solver.x, solver.d̅, solver.stats
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
    bNorm = knorm(n, r₀)  # ‖r₀‖ = ‖b₀ - Ax₀‖

    history && push!(rNorms, bNorm)
    if bNorm == 0
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

    ε = atol + rtol * bNorm
    (verbose > 0) && @printf(iostream, "%5s  %8s  %7s  %5s\n", "k", "αₖ", "‖rₖ‖", "timer")
    kdisplay(iter, verbose) && @printf(iostream, "%5d  %8.1e  %7.1e  %.2fs\n", iter, cᴴb, bNorm, start_time |> ktimer)

    βₖ = √(abs(cᴴb))            # β₁γ₁ = cᴴ(b - Ax₀)
    γₖ = cᴴb / βₖ               # β₁γ₁ = cᴴ(b - Ax₀)
    kfill!(vₖ₋₁, zero(FC))      # v₀ = 0
    kfill!(uₖ₋₁, zero(FC))      # u₀ = 0
    vₖ .= r₀ ./ βₖ              # v₁ = (b - Ax₀) / β₁
    uₖ .= c ./ conj(γₖ)         # u₁ = c / γ̄₁
    cₖ₋₁ = cₖ = -one(T)         # Givens cosines used for the LQ factorization of Tₖ
    sₖ₋₁ = sₖ = zero(FC)        # Givens sines used for the LQ factorization of Tₖ
    kfill!(d̅, zero(FC))         # Last column of D̅ₖ = Vₖ(Qₖ)ᴴ
    ζₖ₋₁ = ζbarₖ = zero(FC)     # ζₖ₋₁ and ζbarₖ are the last components of z̅ₖ = (L̅ₖ)⁻¹β₁e₁
    ζₖ₋₂ = ηₖ = zero(FC)        # ζₖ₋₂ and ηₖ are used to update ζₖ₋₁ and ζbarₖ
    δbarₖ₋₁ = δbarₖ = zero(FC)  # Coefficients of Lₖ₋₁ and L̅ₖ modified over the course of two iterations
    norm_vₖ = bNorm / βₖ        # ‖vₖ‖ is used for residual norm estimates

    # Stopping criterion.
    solved_lq = bNorm ≤ ε
    solved_cg = false
    breakdown = false
    tired     = iter ≥ itmax
    status    = "unknown"
    user_requested_exit = false
    overtimed = false

    while !(solved_lq || solved_cg || tired || breakdown || user_requested_exit || overtimed)
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

      # Update the LQ factorization of Tₖ = L̅ₖQₖ.
      # [ α₁ γ₂ 0  •  •  •  0 ]   [ δ₁   0    •   •   •    •    0   ]
      # [ β₂ α₂ γ₃ •        • ]   [ λ₁   δ₂   •                 •   ]
      # [ 0  •  •  •  •     • ]   [ ϵ₁   λ₂   δ₃  •             •   ]
      # [ •  •  •  •  •  •  • ] = [ 0    •    •   •   •         •   ] Qₖ
      # [ •     •  •  •  •  0 ]   [ •    •    •   •   •    •    •   ]
      # [ •        •  •  •  γₖ]   [ •         •   •   •    •    0   ]
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
        ϵₖ₋₂  =   sₖ₋₁ * βₖ
        λₖ₋₁  =  -cₖ₋₁ *      cₖ  * βₖ + sₖ * αₖ
        δbarₖ =  -cₖ₋₁ * conj(sₖ) * βₖ - cₖ * αₖ
      end

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
        # (xᴸ)ₖ₋₁ ← (xᴸ)ₖ₋₂ + ζₖ₋₁ * dₖ₋₁
        kaxpy!(n, ζₖ₋₁ * cₖ,  d̅, x)
        kaxpy!(n, ζₖ₋₁ * sₖ, vₖ, x)
      end

      # Compute d̅ₖ.
      if iter == 1
        # d̅₁ = v₁
        kcopy!(n, d̅, vₖ)  # d̅ ← vₖ
      else
        # d̅ₖ = s̄ₖ * d̅ₖ₋₁ - cₖ * vₖ
        kaxpby!(n, -cₖ, vₖ, conj(sₖ), d̅)
      end

      # Compute vₖ₊₁ and uₖ₊₁.
      kcopy!(n, vₖ₋₁, vₖ)  # vₖ₋₁ ← vₖ
      kcopy!(n, uₖ₋₁, uₖ)  # uₖ₋₁ ← uₖ

      if pᴴq ≠ 0
        vₖ .= q ./ βₖ₊₁        # βₖ₊₁vₖ₊₁ = q
        uₖ .= p ./ conj(γₖ₊₁)  # γ̄ₖ₊₁uₖ₊₁ = p
      end

      # Compute ⟨vₖ,vₖ₊₁⟩ and ‖vₖ₊₁‖
      vₖᴴvₖ₊₁ = kdot(n, vₖ₋₁, vₖ)
      norm_vₖ₊₁ = knorm(n, vₖ)

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

      # Compute BiCG residual norm
      # ‖rₖ‖ = |ρₖ| * ‖vₖ₊₁‖
      if transfer_to_bicg && (abs(δbarₖ) > eps(T))
        ζbarₖ = ηₖ / δbarₖ
        ρₖ = βₖ₊₁ * (sₖ * ζₖ₋₁ - cₖ * ζbarₖ)
        rNorm_cg = abs(ρₖ) * norm_vₖ₊₁
      end

      # Update sₖ₋₁, cₖ₋₁, γₖ, βₖ, δbarₖ₋₁ and norm_vₖ.
      sₖ₋₁    = sₖ
      cₖ₋₁    = cₖ
      γₖ      = γₖ₊₁
      βₖ      = βₖ₊₁
      δbarₖ₋₁ = δbarₖ
      norm_vₖ = norm_vₖ₊₁

      # Update stopping criterion.
      user_requested_exit = callback(solver) :: Bool
      solved_lq = rNorm_lq ≤ ε
      solved_cg = transfer_to_bicg && (abs(δbarₖ) > eps(T)) && (rNorm_cg ≤ ε)
      tired = iter ≥ itmax
      breakdown = !solved_lq && !solved_cg && (pᴴq == 0)
      timer = time_ns() - start_time
      overtimed = timer > timemax_ns
      kdisplay(iter, verbose) && @printf(iostream, "%5d  %8.1e  %7.1e  %.2fs\n", iter, αₖ, rNorm_lq, start_time |> ktimer)
    end
    (verbose > 0) && @printf(iostream, "\n")

    # Compute BICG point
    # (xᶜ)ₖ ← (xᴸ)ₖ₋₁ + ζbarₖ * d̅ₖ
    if solved_cg
      kaxpy!(n, ζbarₖ, d̅, x)
    end

    # Termination status
    tired               && (status = "maximum number of iterations exceeded")
    breakdown           && (status = "Breakdown ⟨uₖ₊₁,vₖ₊₁⟩ = 0")
    solved_lq           && (status = "solution xᴸ good enough given atol and rtol")
    solved_cg           && (status = "solution xᶜ good enough given atol and rtol")
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
    stats.solved = solved_lq || solved_cg
    stats.inconsistent = false
    stats.timer = start_time |> ktimer
    stats.status = status
    return solver
  end
end
