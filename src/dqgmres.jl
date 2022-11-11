# An implementation of DQGMRES for the solution of the square linear system Ax = b.
#
# This method is described in
#
# Y. Saad and K. Wu, DQGMRES: a quasi minimal residual algorithm based on incomplete orthogonalization.
# Numerical Linear Algebra with Applications, Vol. 3(4), pp. 329--343, 1996.
#
# Alexis Montoison, <alexis.montoison@polymtl.ca>
# Montreal, August 2018.

export dqgmres, dqgmres!

"""
    (x, stats) = dqgmres(A, b::AbstractVector{FC};
                         memory::Int=20, M=I, N=I, atol::T=√eps(T),
                         rtol::T=√eps(T), reorthogonalization::Bool=false,
                         itmax::Int=0, verbose::Int=0, history::Bool=false,
                         ldiv::Bool=false, callback=solver->false, iostream::IO=stdout)

`T` is an `AbstractFloat` such as `Float32`, `Float64` or `BigFloat`.
`FC` is `T` or `Complex{T}`.

    (x, stats) = dqgmres(A, b, x0::AbstractVector; kwargs...)

DQGMRES can be warm-started from an initial guess `x0` where `kwargs` are the same keyword arguments as above.

Solve the consistent linear system Ax = b of size n using DQGMRES.

DQGMRES algorithm is based on the incomplete Arnoldi orthogonalization process
and computes a sequence of approximate solutions with the quasi-minimal residual property.

DQGMRES only orthogonalizes the new vectors of the Krylov basis against the `memory` most recent vectors.
If MINRES is well defined on `Ax = b` and `memory = 2`, DQGMRES is theoretically equivalent to MINRES.
If `k ≤ memory` where `k` is the number of iterations, DQGMRES is theoretically equivalent to GMRES.
Otherwise, DQGMRES interpolates between MINRES and GMRES and is similar to MINRES with partial reorthogonalization.

Partial reorthogonalization is available with the `reorthogonalization` option.

This implementation allows a left preconditioner M and a right preconditioner N.

The callback is called as `callback(solver)` and should return `true` if the main loop should terminate,
and `false` otherwise.

#### Input arguments

* `A`: a linear operator that models a matrix of dimension n;
* `b`: a vector of length n.

#### Optional argument

* `x0`: a vector of length n that represents an initial guess of the solution x.

#### Output arguments

* `x`: a dense vector of length n;
* `stats`: statistics collected on the run in a [`SimpleStats`](@ref) structure.

#### Reference

* Y. Saad and K. Wu, [*DQGMRES: a quasi minimal residual algorithm based on incomplete orthogonalization*](https://doi.org/10.1002/(SICI)1099-1506(199607/08)3:4%3C329::AID-NLA86%3E3.0.CO;2-8), Numerical Linear Algebra with Applications, Vol. 3(4), pp. 329--343, 1996.
"""
function dqgmres end

function dqgmres(A, b :: AbstractVector{FC}, x0 :: AbstractVector; memory :: Int=20, kwargs...) where FC <: FloatOrComplex
  solver = DqgmresSolver(A, b, memory)
  dqgmres!(solver, A, b, x0; kwargs...)
  return (solver.x, solver.stats)
end

function dqgmres(A, b :: AbstractVector{FC}; memory :: Int=20, kwargs...) where FC <: FloatOrComplex
  solver = DqgmresSolver(A, b, memory)
  dqgmres!(solver, A, b; kwargs...)
  return (solver.x, solver.stats)
end

"""
    solver = dqgmres!(solver::DqgmresSolver, A, b; kwargs...)
    solver = dqgmres!(solver::DqgmresSolver, A, b, x0; kwargs...)

where `kwargs` are keyword arguments of [`dqgmres`](@ref).

Note that the `memory` keyword argument is the only exception.
It's required to create a `DqgmresSolver` and can't be changed later.

See [`DqgmresSolver`](@ref) for more details about the `solver`.
"""
function dqgmres! end

function dqgmres!(solver :: DqgmresSolver{T,FC,S}, A, b :: AbstractVector{FC}, x0 :: AbstractVector; kwargs...) where {T <: AbstractFloat, FC <: FloatOrComplex{T}, S <: DenseVector{FC}}
  warm_start!(solver, x0)
  dqgmres!(solver, A, b; kwargs...)
  return solver
end

function dqgmres!(solver :: DqgmresSolver{T,FC,S}, A, b :: AbstractVector{FC};
                  M=I, N=I, atol :: T=√eps(T), rtol :: T=√eps(T),
                  reorthogonalization :: Bool=false, itmax :: Int=0,
                  verbose :: Int=0, history :: Bool=false,
                  ldiv :: Bool=false, callback = solver -> false, iostream :: IO=kstdout) where {T <: AbstractFloat, FC <: FloatOrComplex{T}, S <: DenseVector{FC}}

  m, n = size(A)
  m == n || error("System must be square")
  length(b) == m || error("Inconsistent problem size")
  (verbose > 0) && @printf(iostream, "DQGMRES: system of size %d\n", n)

  # Check M = Iₙ and N = Iₙ
  MisI = (M === I)
  NisI = (N === I)

  # Check type consistency
  eltype(A) == FC || error("eltype(A) ≠ $FC")
  ktypeof(b) <: S || error("ktypeof(b) is not a subtype of $S")

  # Set up workspace.
  allocate_if(!MisI, solver, :w, S, n)
  allocate_if(!NisI, solver, :z, S, n)
  Δx, x, t, P, V = solver.Δx, solver.x, solver.t, solver.P, solver.V
  c, s, H, stats = solver.c, solver.s, solver.H, solver.stats
  warm_start = solver.warm_start
  rNorms = stats.residuals
  reset!(stats)
  w  = MisI ? t : solver.w
  r₀ = MisI ? t : solver.w

  # Initial solution x₀ and residual r₀.
  x .= zero(FC)  # x₀
  if warm_start
    mul!(t, A, Δx)
    @kaxpby!(n, one(FC), b, -one(FC), t)
  else
    t .= b
  end
  MisI || mulorldiv!(r₀, M, t, ldiv)  # M(b - Ax₀)
  rNorm = @knrm2(n, r₀)               # β = ‖r₀‖₂
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
  (verbose > 0) && @printf(iostream, "%5s  %7s\n", "k", "‖rₖ‖")
  kdisplay(iter, verbose) && @printf(iostream, "%5d  %7.1e\n", iter, rNorm)

  # Set up workspace.
  mem = length(V)  # Memory.
  for i = 1 : mem
    V[i] .= zero(FC)  # Orthogonal basis of Kₖ(MAN, Mr₀).
    P[i] .= zero(FC)  # Directions for x : Pₖ = NVₖ(Rₖ)⁻¹.
  end
  c .= zero(T)   # Last mem Givens cosines used for the factorization QₖRₖ = Hₖ.
  s .= zero(FC)  # Last mem Givens sines used for the factorization QₖRₖ = Hₖ.
  H .= zero(FC)  # Last column of the band hessenberg matrix Hₖ.
  # Each column has at most mem + 1 nonzero elements.
  # hᵢ.ₖ is stored as H[k-i+1], i ≤ k. hₖ₊₁.ₖ is not stored in H.
  # k-i+1 represents the indice of the diagonal where hᵢ.ₖ is located.
  # In addition of that, the last column of Rₖ is also stored in H.

  # Initial γ₁ and V₁.
  γₖ = rNorm # γₖ and γₖ₊₁ are the last components of gₖ, right-hand of the least squares problem min ‖ Hₖyₖ - gₖ ‖₂.
  V[1] .= r₀ ./ rNorm

  # The following stopping criterion compensates for the lag in the
  # residual, but usually increases the number of iterations.
  # solved = sqrt(max(1, iter-mem+1)) * |γₖ₊₁| ≤ ε
  solved = rNorm ≤ ε # less accurate, but acceptable.
  tired = iter ≥ itmax
  status = "unknown"
  user_requested_exit = false

  while !(solved || tired || user_requested_exit)

    # Update iteration index.
    iter = iter + 1

    # Set position in circulars stacks.
    pos = mod(iter-1, mem) + 1     # Position corresponding to pₖ and vₖ in circular stacks P and V.
    next_pos = mod(iter, mem) + 1  # Position corresponding to vₖ₊₁ in the circular stack V.

    # Incomplete Arnoldi procedure.
    z = NisI ? V[pos] : solver.z
    NisI || mulorldiv!(z, N, V[pos], ldiv)  # Nvₖ, forms pₖ
    mul!(t, A, z)                           # ANvₖ
    MisI || mulorldiv!(w, M, t, ldiv)       # MANvₖ, forms vₖ₊₁
    for i = max(1, iter-mem+1) : iter
      ipos = mod(i-1, mem) + 1  # Position corresponding to vᵢ in the circular stack V.
      diag = iter - i + 1
      H[diag] = @kdot(n, w, V[ipos])    # hᵢ.ₖ = ⟨MANvₖ, vᵢ⟩
      @kaxpy!(n, -H[diag], V[ipos], w)  # w ← w - hᵢ.ₖvᵢ
    end

    # Partial reorthogonalization of the Krylov basis.
    if reorthogonalization
      for i = max(1, iter-mem+1) : iter
        ipos = mod(i-1, mem) + 1
        diag = iter - i + 1
        Htmp = @kdot(n, w, V[ipos])
        H[diag] += Htmp
        @kaxpy!(n, -Htmp, V[ipos], w)
      end
    end

    # Compute hₖ₊₁.ₖ and vₖ₊₁.
    Haux = @knrm2(n, w)         # hₖ₊₁.ₖ = ‖vₖ₊₁‖₂
    if Haux ≠ 0                 # hₖ₊₁.ₖ = 0 ⇒ "lucky breakdown"
      V[next_pos] .= w ./ Haux  # vₖ₊₁ = w / hₖ₊₁.ₖ
    end
    # rₖ₋ₘₑₘ.ₖ ≠ 0 when k ≥ mem + 1
    # We don't want to use rₖ₋₁₋ₘₑₘ.ₖ₋₁ when we compute rₖ₋ₘₑₘ.ₖ
    if iter ≥ mem + 2
      H[mem+1] = zero(FC)  # rₖ₋ₘₑₘ.ₖ = 0
    end

    # Update the QR factorization of Hₖ.
    # Apply mem previous Givens reflections Ωᵢ.
    for i = max(1,iter-mem) : iter-1
      irot_pos = mod(i-1, mem) + 1  # Position corresponding to cᵢ and sᵢ in circular stacks c and s.
      diag = iter - i
      next_diag = diag + 1
      Htmp         =      c[irot_pos]  * H[next_diag] + s[irot_pos] * H[diag]
      H[diag]      = conj(s[irot_pos]) * H[next_diag] - c[irot_pos] * H[diag]
      H[next_diag] = Htmp
    end

    # Compute and apply current Givens reflection Ωₖ.
    # [cₖ  sₖ] [ hₖ.ₖ ] = [ρₖ]
    # [sₖ -cₖ] [hₖ₊₁.ₖ]   [0 ]
    (c[pos], s[pos], H[1]) = sym_givens(H[1], Haux)
    γₖ₊₁ = conj(s[pos]) * γₖ
    γₖ   =      c[pos]  * γₖ

    # Compute the direction pₖ, the last column of Pₖ = NVₖ(Rₖ)⁻¹.
    for i = max(1,iter-mem) : iter-1
      ipos = mod(i-1, mem) + 1  # Position corresponding to pᵢ in the circular stack P.
      diag = iter - i + 1
      if ipos == pos
        # pₐᵤₓ ← -hₖ₋ₘₑₘ.ₖ * pₖ₋ₘₑₘ
        @kscal!(n, -H[diag], P[pos])
      else
        # pₐᵤₓ ← pₐᵤₓ - hᵢ.ₖ * pᵢ
        @kaxpy!(n, -H[diag], P[ipos], P[pos])
      end
    end
    # pₐᵤₓ ← pₐᵤₓ + Nvₖ
    @kaxpy!(n, one(FC), z, P[pos])
    # pₖ = pₐᵤₓ / hₖ.ₖ
    P[pos] .= P[pos] ./ H[1]

    # Compute solution xₖ.
    # xₖ ← xₖ₋₁ + γₖ * pₖ
    @kaxpy!(n, γₖ, P[pos], x)

    # Update residual norm estimate.
    # ‖ M(b - Axₖ) ‖₂ ≈ |γₖ₊₁|
    rNorm = abs(γₖ₊₁)
    history && push!(rNorms, rNorm)

    # Update γₖ.
    γₖ = γₖ₊₁

    # Stopping conditions that do not depend on user input.
    # This is to guard against tolerances that are unreasonably small.
    resid_decrease_mach = (rNorm + one(T) ≤ one(T))

    # Update stopping criterion.
    user_requested_exit = callback(solver) :: Bool
    resid_decrease_lim = rNorm ≤ ε
    solved = resid_decrease_lim || resid_decrease_mach
    tired = iter ≥ itmax
    kdisplay(iter, verbose) && @printf(iostream, "%5d  %7.1e\n", iter, rNorm)
  end
  (verbose > 0) && @printf(iostream, "\n")
  solved              && (status = "solution good enough given atol and rtol")
  tired               && (status = "maximum number of iterations exceeded")
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
