# An implementation of BICGSTAB for the solution of unsymmetric and square consistent linear system Ax = b.
#
# This method is described in
#
# H. A. van der Vorst
# Bi-CGSTAB: A fast and smoothly converging variant of Bi-CG for the solution of nonsymmetric linear systems.
# SIAM Journal on Scientific and Statistical Computing, 13(2), pp. 631--644, 1992.
#
# G. L.G. Sleijpen and D. R. Fokkema
# BiCGstab(ℓ) for linear equations involving unsymmetric matrices with complex spectrum.
# Electronic Transactions on Numerical Analysis, 1, pp. 11--32, 1993.
#
# Alexis Montoison, <alexis.montoison@polymtl.ca>
# Montréal, October 2020.

export bicgstab, bicgstab!

"""
    (x, stats) = bicgstab(A, b::AbstractVector{FC};
                          c::AbstractVector{FC}=b, M=I, N=I,
                          ldiv::Bool=false, atol::T=√eps(T),
                          rtol::T=√eps(T), itmax::Int=0,
                          timemax::Float64=Inf, verbose::Int=0, history::Bool=false,
                          callback=solver->false, iostream::IO=kstdout)

`T` is an `AbstractFloat` such as `Float32`, `Float64` or `BigFloat`.
`FC` is `T` or `Complex{T}`.

    (x, stats) = bicgstab(A, b, x0::AbstractVector; kwargs...)

BICGSTAB can be warm-started from an initial guess `x0` where `kwargs` are the same keyword arguments as above.

Solve the square linear system Ax = b of size n using BICGSTAB.
BICGSTAB requires two initial vectors `b` and `c`.
The relation `bᴴc ≠ 0` must be satisfied and by default `c = b`.

The Biconjugate Gradient Stabilized method is a variant of BiCG, like CGS,
but using different updates for the Aᴴ-sequence in order to obtain smoother
convergence than CGS.

If BICGSTAB stagnates, we recommend DQGMRES and BiLQ as alternative methods for unsymmetric square systems.

BICGSTAB stops when `itmax` iterations are reached or when `‖rₖ‖ ≤ atol + ‖b‖ * rtol`.

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

* H. A. van der Vorst, [*Bi-CGSTAB: A fast and smoothly converging variant of Bi-CG for the solution of nonsymmetric linear systems*](https://doi.org/10.1137/0913035), SIAM Journal on Scientific and Statistical Computing, 13(2), pp. 631--644, 1992.
* G. L.G. Sleijpen and D. R. Fokkema, [*BiCGstab(ℓ) for linear equations involving unsymmetric matrices with complex spectrum*](https://etna.math.kent.edu/volumes/1993-2000/vol1/abstract.php?vol=1&pages=11-32), Electronic Transactions on Numerical Analysis, 1, pp. 11--32, 1993.
"""
function bicgstab end

"""
    solver = bicgstab!(solver::BicgstabSolver, A, b; kwargs...)
    solver = bicgstab!(solver::BicgstabSolver, A, b, x0; kwargs...)

where `kwargs` are keyword arguments of [`bicgstab`](@ref).

See [`BicgstabSolver`](@ref) for more details about the `solver`.
"""
function bicgstab! end

def_args_bicgstab = (:(A                    ),
                     :(b::AbstractVector{FC}))

def_optargs_bicgstab = (:(x0::AbstractVector),)

def_kwargs_bicgstab = (:(; c::AbstractVector{FC} = b ),
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

def_kwargs_bicgstab = extract_parameters.(def_kwargs_bicgstab)

args_bicgstab = (:A, :b)
optargs_bicgstab = (:x0,)
kwargs_bicgstab = (:c, :M, :N, :ldiv, :atol, :rtol, :itmax, :timemax, :verbose, :history, :callback, :iostream)

@eval begin
  function bicgstab!(solver :: BicgstabSolver{T,FC,S}, $(def_args_bicgstab...); $(def_kwargs_bicgstab...)) where {T <: AbstractFloat, FC <: FloatOrComplex{T}, S <: AbstractVector{FC}}

    # Timer
    start_time = time_ns()
    timemax_ns = 1e9 * timemax

    m, n = size(A)
    (m == solver.m && n == solver.n) || error("(solver.m, solver.n) = ($(solver.m), $(solver.n)) is inconsistent with size(A) = ($m, $n)")
    m == n || error("System must be square")
    length(b) == m || error("Inconsistent problem size")
    (verbose > 0) && @printf(iostream, "BICGSTAB: system of size %d\n", n)

    # Check M = Iₙ and N = Iₙ
    MisI = (M === I)
    NisI = (N === I)

    # Check type consistency
    eltype(A) == FC || @warn "eltype(A) ≠ $FC. This could lead to errors or additional allocations in operator-vector products."
    ktypeof(b) <: S || error("ktypeof(b) is not a subtype of $S")
    ktypeof(c) <: S || error("ktypeof(c) is not a subtype of $S")

    # Set up workspace.
    allocate_if(!MisI, solver, :t , S, solver.x)  # The length of t is n
    allocate_if(!NisI, solver, :yz, S, solver.x)  # The length of yz is n
    Δx, x, r, p, v, s, qd, stats = solver.Δx, solver.x, solver.r, solver.p, solver.v, solver.s, solver.qd, solver.stats
    warm_start = solver.warm_start
    rNorms = stats.residuals
    reset!(stats)
    q = d = solver.qd
    t = MisI ? d : solver.t
    y = NisI ? p : solver.yz
    z = NisI ? s : solver.yz
    r₀ = MisI ? r : solver.qd

    if warm_start
      mul!(r₀, A, Δx)
      kaxpby!(n, one(FC), b, -one(FC), r₀)
    else
      kcopy!(n, r₀, b)  # r₀ ← b
    end

    kfill!(x, zero(FC))                 # x₀
    kfill!(s, zero(FC))                 # s₀
    kfill!(v, zero(FC))                 # v₀
    MisI || mulorldiv!(r, M, r₀, ldiv)  # r₀
    kcopy!(n, p, r)                     # p₁

    α = one(FC)  # α₀
    ω = one(FC)  # ω₀
    ρ = one(FC)  # ρ₀

    # Compute residual norm ‖r₀‖₂.
    rNorm = knorm(n, r)
    history && push!(rNorms, rNorm)
    if rNorm == 0
      stats.niter = 0
      stats.solved, stats.inconsistent = true, false
      stats.timer = start_time |> ktimer
      stats.status = "x is a zero-residual solution"
      warm_start && kaxpy!(n, one(FC), Δx, x)
      solver.warm_start = false
      return solver
    end

    iter = 0
    itmax == 0 && (itmax = 2*n)

    ε = atol + rtol * rNorm
    (verbose > 0) && @printf(iostream, "%5s  %7s  %8s  %8s  %5s\n", "k", "‖rₖ‖", "|αₖ|", "|ωₖ|", "timer")
    kdisplay(iter, verbose) && @printf(iostream, "%5d  %7.1e  %8.1e  %8.1e  %.2fs\n", iter, rNorm, abs(α), abs(ω), start_time |> ktimer)

    next_ρ = kdot(n, c, r)  # ρ₁ = ⟨r̅₀,r₀⟩
    if next_ρ == 0
      stats.niter = 0
      stats.solved, stats.inconsistent = false, false
      stats.timer = start_time |> ktimer
      stats.status = "Breakdown bᴴc = 0"
      warm_start && kaxpy!(n, one(FC), Δx, x)
      solver.warm_start = false
      return solver
    end

    # Stopping criterion.
    solved = rNorm ≤ ε
    tired = iter ≥ itmax
    breakdown = false
    status = "unknown"
    user_requested_exit = false
    overtimed = false

    while !(solved || tired || breakdown || user_requested_exit || overtimed)
      # Update iteration index and ρ.
      iter = iter + 1
      ρ = next_ρ

      NisI || mulorldiv!(y, N, p, ldiv)  # yₖ = N⁻¹pₖ
      mul!(q, A, y)                      # qₖ = Ayₖ
      mulorldiv!(v, M, q, ldiv)          # vₖ = M⁻¹qₖ
      α = ρ / kdot(n, c, v)              # αₖ = ⟨r̅₀,rₖ₋₁⟩ / ⟨r̅₀,vₖ⟩
      kcopy!(n, s, r)                    # sₖ = rₖ₋₁
      kaxpy!(n, -α, v, s)                # sₖ = sₖ - αₖvₖ
      kaxpy!(n, α, y, x)                 # xₐᵤₓ = xₖ₋₁ + αₖyₖ
      NisI || mulorldiv!(z, N, s, ldiv)  # zₖ = N⁻¹sₖ
      mul!(d, A, z)                      # dₖ = Azₖ
      MisI || mulorldiv!(t, M, d, ldiv)  # tₖ = M⁻¹dₖ
      ω = kdot(n, t, s) / kdot(n, t, t)  # ⟨tₖ,sₖ⟩ / ⟨tₖ,tₖ⟩
      kaxpy!(n, ω, z, x)                 # xₖ = xₐᵤₓ + ωₖzₖ
      kcopy!(n, r, s)                    # rₖ = sₖ
      kaxpy!(n, -ω, t, r)                # rₖ = rₖ - ωₖtₖ
      next_ρ = kdot(n, c, r)             # ρₖ₊₁ = ⟨r̅₀,rₖ⟩
      β = (next_ρ / ρ) * (α / ω)         # βₖ₊₁ = (ρₖ₊₁ / ρₖ) * (αₖ / ωₖ)
      kaxpy!(n, -ω, v, p)                # pₐᵤₓ = pₖ - ωₖvₖ
      kaxpby!(n, one(FC), r, β, p)       # pₖ₊₁ = rₖ₊₁ + βₖ₊₁pₐᵤₓ

      # Compute residual norm ‖rₖ‖₂.
      rNorm = knorm(n, r)
      history && push!(rNorms, rNorm)

      # Stopping conditions that do not depend on user input.
      # This is to guard against tolerances that are unreasonably small.
      resid_decrease_mach = (rNorm + one(T) ≤ one(T))

      # Update stopping criterion.
      user_requested_exit = callback(solver) :: Bool
      resid_decrease_lim = rNorm ≤ ε
      solved = resid_decrease_lim || resid_decrease_mach
      tired = iter ≥ itmax
      breakdown = (α == 0 || isnan(α))
      timer = time_ns() - start_time
      overtimed = timer > timemax_ns
      kdisplay(iter, verbose) && @printf(iostream, "%5d  %7.1e  %8.1e  %8.1e  %.2fs\n", iter, rNorm, abs(α), abs(ω), start_time |> ktimer)
    end
    (verbose > 0) && @printf(iostream, "\n")

    # Termination status
    tired               && (status = "maximum number of iterations exceeded")
    breakdown           && (status = "breakdown αₖ == 0")
    solved              && (status = "solution good enough given atol and rtol")
    user_requested_exit && (status = "user-requested exit")
    overtimed           && (status = "time limit exceeded")

    # Update x
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
