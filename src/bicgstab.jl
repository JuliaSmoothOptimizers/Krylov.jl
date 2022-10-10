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
    (x, stats) = bicgstab(A, b::AbstractVector{FC}; c::AbstractVector{FC}=b,
                          M=I, N=I, atol::T=√eps(T), rtol::T=√eps(T),
                          itmax::Int=0, verbose::Int=0, history::Bool=false,
                          ldiv::Bool=false, callback=solver->false)

`T` is an `AbstractFloat` such as `Float32`, `Float64` or `BigFloat`.
`FC` is `T` or `Complex{T}`.

Solve the square linear system Ax = b of size n using BICGSTAB.
BICGSTAB requires two initial vectors `b` and `c`.
The relation `bᴴc ≠ 0` must be satisfied and by default `c = b`.

The Biconjugate Gradient Stabilized method is a variant of BiCG, like CGS,
but using different updates for the Aᴴ-sequence in order to obtain smoother
convergence than CGS.

If BICGSTAB stagnates, we recommend DQGMRES and BiLQ as alternative methods for unsymmetric square systems.

BICGSTAB stops when `itmax` iterations are reached or when `‖rₖ‖ ≤ atol + ‖b‖ * rtol`.
`atol` is an absolute tolerance and `rtol` is a relative tolerance.

Additional details can be displayed if verbose mode is enabled (verbose > 0).
Information will be displayed every `verbose` iterations.

This implementation allows a left preconditioner `M` and a right preconditioner `N`.

BICGSTAB can be warm-started from an initial guess `x0` with

    (x, stats) = bicgstab(A, b, x0; kwargs...)

where `kwargs` are the same keyword arguments as above.

The callback is called as `callback(solver)` and should return `true` if the main loop should terminate,
and `false` otherwise.

#### Input arguments

* `A`: a linear operator that models a matrix of dimension n;
* `b`: a vector of length n.

#### Output arguments

* `x`: a dense vector of length n;
* `stats`: statistics collected on the run in a [`SimpleStats`](@ref) structure.

#### References

* H. A. van der Vorst, [*Bi-CGSTAB: A fast and smoothly converging variant of Bi-CG for the solution of nonsymmetric linear systems*](https://doi.org/10.1137/0913035), SIAM Journal on Scientific and Statistical Computing, 13(2), pp. 631--644, 1992.
* G. L.G. Sleijpen and D. R. Fokkema, *BiCGstab(ℓ) for linear equations involving unsymmetric matrices with complex spectrum*, Electronic Transactions on Numerical Analysis, 1, pp. 11--32, 1993.
"""
function bicgstab end

function bicgstab(A, b :: AbstractVector{FC}, x0 :: AbstractVector; kwargs...) where FC <: FloatOrComplex
  solver = BicgstabSolver(A, b)
  bicgstab!(solver, A, b, x0; kwargs...)
  return (solver.x, solver.stats)
end

function bicgstab(A, b :: AbstractVector{FC}; kwargs...) where FC <: FloatOrComplex
  solver = BicgstabSolver(A, b)
  bicgstab!(solver, A, b; kwargs...)
  return (solver.x, solver.stats)
end

"""
    solver = bicgstab!(solver::BicgstabSolver, A, b; kwargs...)
    solver = bicgstab!(solver::BicgstabSolver, A, b, x0; kwargs...)

where `kwargs` are keyword arguments of [`bicgstab`](@ref).

See [`BicgstabSolver`](@ref) for more details about the `solver`.
"""
function bicgstab! end

function bicgstab!(solver :: BicgstabSolver{T,FC,S}, A, b :: AbstractVector{FC}, x0 :: AbstractVector; kwargs...) where {T <: AbstractFloat, FC <: FloatOrComplex{T}, S <: DenseVector{FC}}
  warm_start!(solver, x0)
  bicgstab!(solver, A, b; kwargs...)
  return solver
end

function bicgstab!(solver :: BicgstabSolver{T,FC,S}, A, b :: AbstractVector{FC}; c :: AbstractVector{FC}=b,
                   M=I, N=I, atol :: T=√eps(T), rtol :: T=√eps(T),
                   itmax :: Int=0, verbose :: Int=0, history :: Bool=false,
                   ldiv :: Bool=false, callback = solver -> false) where {T <: AbstractFloat, FC <: FloatOrComplex{T}, S <: DenseVector{FC}}

  m, n = size(A)
  m == n || error("System must be square")
  length(b) == m || error("Inconsistent problem size")
  (verbose > 0) && @printf("BICGSTAB: system of size %d\n", n)

  # Check M = Iₙ and N = Iₙ
  MisI = (M === I)
  NisI = (N === I)

  # Check type consistency
  eltype(A) == FC || error("eltype(A) ≠ $FC")
  ktypeof(b) <: S || error("ktypeof(b) is not a subtype of $S")
  ktypeof(c) <: S || error("ktypeof(c) is not a subtype of $S")

  # Set up workspace.
  allocate_if(!MisI, solver, :t , S, n)
  allocate_if(!NisI, solver, :yz, S, n)
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
    @kaxpby!(n, one(FC), b, -one(FC), r₀)
  else
    r₀ .= b
  end

  x .= zero(FC)                       # x₀
  s .= zero(FC)                       # s₀
  v .= zero(FC)                       # v₀
  MisI || mulorldiv!(r, M, r₀, ldiv)  # r₀
  p .= r                              # p₁

  α = one(FC) # α₀
  ω = one(FC) # ω₀
  ρ = one(FC) # ρ₀

  # Compute residual norm ‖r₀‖₂.
  rNorm = @knrm2(n, r)
  history && push!(rNorms, rNorm)
  if rNorm == 0
    stats.niter = 0
    stats.solved, stats.inconsistent = true, false
    stats.status = "x = 0 is a zero-residual solution"
    solver.warm_start = false
    return solver
  end

  iter = 0
  itmax == 0 && (itmax = 2*n)

  ε = atol + rtol * rNorm
  (verbose > 0) && @printf("%5s  %7s  %8s  %8s\n", "k", "‖rₖ‖", "|αₖ|", "|ωₖ|")
  kdisplay(iter, verbose) && @printf("%5d  %7.1e  %8.1e  %8.1e\n", iter, rNorm, abs(α), abs(ω))

  next_ρ = @kdot(n, c, r)  # ρ₁ = ⟨r̅₀,r₀⟩
  if next_ρ == 0
    stats.niter = 0
    stats.solved, stats.inconsistent = false, false
    stats.status = "Breakdown bᴴc = 0"
    solver.warm_start = false
    return solver
  end

  # Stopping criterion.
  solved = rNorm ≤ ε
  tired = iter ≥ itmax
  breakdown = false
  status = "unknown"
  user_requested_exit = false

  while !(solved || tired || breakdown || user_requested_exit)
    # Update iteration index and ρ.
    iter = iter + 1
    ρ = next_ρ

    NisI || mulorldiv!(y, N, p, ldiv)    # yₖ = N⁻¹pₖ
    mul!(q, A, y)                        # qₖ = Ayₖ
    mulorldiv!(v, M, q, ldiv)            # vₖ = M⁻¹qₖ
    α = ρ / @kdot(n, c, v)               # αₖ = ⟨r̅₀,rₖ₋₁⟩ / ⟨r̅₀,vₖ⟩
    @kcopy!(n, r, s)                     # sₖ = rₖ₋₁
    @kaxpy!(n, -α, v, s)                 # sₖ = sₖ - αₖvₖ
    @kaxpy!(n, α, y, x)                  # xₐᵤₓ = xₖ₋₁ + αₖyₖ
    NisI || mulorldiv!(z, N, s, ldiv)    # zₖ = N⁻¹sₖ
    mul!(d, A, z)                        # dₖ = Azₖ
    MisI || mulorldiv!(t, M, d, ldiv)    # tₖ = M⁻¹dₖ
    ω = @kdot(n, t, s) / @kdot(n, t, t)  # ⟨tₖ,sₖ⟩ / ⟨tₖ,tₖ⟩
    @kaxpy!(n, ω, z, x)                  # xₖ = xₐᵤₓ + ωₖzₖ
    @kcopy!(n, s, r)                     # rₖ = sₖ
    @kaxpy!(n, -ω, t, r)                 # rₖ = rₖ - ωₖtₖ
    next_ρ = @kdot(n, c, r)              # ρₖ₊₁ = ⟨r̅₀,rₖ⟩
    β = (next_ρ / ρ) * (α / ω)           # βₖ₊₁ = (ρₖ₊₁ / ρₖ) * (αₖ / ωₖ)
    @kaxpy!(n, -ω, v, p)                 # pₐᵤₓ = pₖ - ωₖvₖ
    @kaxpby!(n, one(FC), r, β, p)        # pₖ₊₁ = rₖ₊₁ + βₖ₊₁pₐᵤₓ

    # Compute residual norm ‖rₖ‖₂.
    rNorm = @knrm2(n, r)
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
    kdisplay(iter, verbose) && @printf("%5d  %7.1e  %8.1e  %8.1e\n", iter, rNorm, abs(α), abs(ω))
  end
  (verbose > 0) && @printf("\n")

  tired               && (status = "maximum number of iterations exceeded")
  breakdown           && (status = "breakdown αₖ == 0")
  solved              && (status = "solution good enough given atol and rtol")
  user_requested_exit && (status = "user-requested exit")

  # Update x
  warm_start && @kaxpy!(n, one(FC), Δx, x)
  solver.warm_start = false

  # Update stats
  stats.niter = iter
  stats.solved = solved
  stats.inconsistent = false
  stats.status = status
  return solver
end
