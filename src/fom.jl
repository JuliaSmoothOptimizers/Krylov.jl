# An implementation of FOM for the solution of the square linear system Ax = b.
#
# This method is described in
#
# Y. Saad, Krylov subspace methods for solving unsymmetric linear systems.
# Mathematics of computation, Vol. 37(155), pp. 105--126, 1981.
#
# Alexis Montoison, <alexis.montoison@polymtl.ca>
# Montreal, December 2018.

export fom, fom!

"""
    (x, stats) = fom(A, b::AbstractVector{FC};
                     memory::Int=20, M=I, N=I, atol::T=√eps(T), rtol::T=√eps(T),
                     reorthogonalization::Bool=false, itmax::Int=0,
                     restart::Bool=false, verbose::Int=0, history::Bool=false,
                     ldiv::Bool=false, callback=solver->false, iostream::IO=kstdout)

`T` is an `AbstractFloat` such as `Float32`, `Float64` or `BigFloat`.
`FC` is `T` or `Complex{T}`.

    (x, stats) = fom(A, b, x0::AbstractVector; kwargs...)

FOM can be warm-started from an initial guess `x0` where `kwargs` are the same keyword arguments as above.

Solve the linear system Ax = b of size n using FOM.

FOM algorithm is based on the Arnoldi process and a Galerkin condition.

This implementation allows a left preconditioner M and a right preconditioner N.
Full reorthogonalization is available with the `reorthogonalization` option.

If `restart = true`, the restarted version FOM(k) is used with `k = memory`.
If `restart = false`, the parameter `memory` should be used as a hint of the number of iterations to limit dynamic memory allocations.
More storage will be allocated only if the number of iterations exceeds `memory`.

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

* Y. Saad, [*Krylov subspace methods for solving unsymmetric linear systems*](https://doi.org/10.1090/S0025-5718-1981-0616364-6), Mathematics of computation, Vol. 37(155), pp. 105--126, 1981.
"""
function fom end

function fom(A, b :: AbstractVector{FC}, x0 :: AbstractVector; memory :: Int=20, kwargs...) where FC <: FloatOrComplex
  solver = FomSolver(A, b, memory)
  fom!(solver, A, b, x0; kwargs...)
  return (solver.x, solver.stats)
end

function fom(A, b :: AbstractVector{FC}; memory :: Int=20, kwargs...) where FC <: FloatOrComplex
  solver = FomSolver(A, b, memory)
  fom!(solver, A, b; kwargs...)
  return (solver.x, solver.stats)
end

"""
    solver = fom!(solver::FomSolver, A, b; kwargs...)
    solver = fom!(solver::FomSolver, A, b, x0; kwargs...)

where `kwargs` are keyword arguments of [`fom`](@ref).

Note that the `memory` keyword argument is the only exception.
It's required to create a `FomSolver` and can't be changed later.

See [`FomSolver`](@ref) for more details about the `solver`.
"""
function fom! end

function fom!(solver :: FomSolver{T,FC,S}, A, b :: AbstractVector{FC}, x0 :: AbstractVector; kwargs...) where {T <: AbstractFloat, FC <: FloatOrComplex{T}, S <: DenseVector{FC}}
  warm_start!(solver, x0)
  fom!(solver, A, b; kwargs...)
  return solver
end

function fom!(solver :: FomSolver{T,FC,S}, A, b :: AbstractVector{FC};
              M=I, N=I, atol :: T=√eps(T), rtol :: T=√eps(T),
              reorthogonalization :: Bool=false, itmax :: Int=0,
              restart :: Bool=false, verbose :: Int=0, history :: Bool=false,
              ldiv :: Bool=false, callback = solver -> false, iostream :: IO=kstdout) where {T <: AbstractFloat, FC <: FloatOrComplex{T}, S <: DenseVector{FC}}

  m, n = size(A)
  m == n || error("System must be square")
  length(b) == m || error("Inconsistent problem size")
  (verbose > 0) && @printf(iostream, "FOM: system of size %d\n", n)

  # Check M = Iₙ and N = Iₙ
  MisI = (M === I)
  NisI = (N === I)

  # Check type consistency
  eltype(A) == FC || error("eltype(A) ≠ $FC")
  ktypeof(b) <: S || error("ktypeof(b) is not a subtype of $S")

  # Set up workspace.
  allocate_if(!MisI  , solver, :q , S, n)
  allocate_if(!NisI  , solver, :p , S, n)
  allocate_if(restart, solver, :Δx, S, n)
  Δx, x, w, V, z = solver.Δx, solver.x, solver.w, solver.V, solver.z
  l, U, stats = solver.l, solver.U, solver.stats
  warm_start = solver.warm_start
  rNorms = stats.residuals
  reset!(stats)
  q  = MisI ? w : solver.q
  r₀ = MisI ? w : solver.q
  xr = restart ? Δx : x

  # Initial solution x₀.
  x .= zero(FC)

  # Initial residual r₀.
  if warm_start
    mul!(w, A, Δx)
    @kaxpby!(n, one(FC), b, -one(FC), w)
    restart && @kaxpy!(n, one(FC), Δx, x)
  else
    w .= b
  end
  MisI || mulorldiv!(r₀, M, w, ldiv)  # r₀ = M(b - Ax₀)
  β = @knrm2(n, r₀)                   # β = ‖r₀‖₂

  rNorm = β
  history && push!(rNorms, β)
  ε = atol + rtol * rNorm

  if β == 0
    stats.niter = 0
    stats.solved, stats.inconsistent = true, false
    stats.status = "x = 0 is a zero-residual solution"
    solver.warm_start = false
    return solver
  end

  mem = length(l)  # Memory
  npass = 0        # Number of pass

  iter = 0        # Cumulative number of iterations
  inner_iter = 0  # Number of iterations in a pass

  itmax == 0 && (itmax = 2*n)
  inner_itmax = itmax

  (verbose > 0) && @printf(iostream, "%5s  %5s  %7s  %7s\n", "pass", "k", "‖rₖ‖", "hₖ₊₁.ₖ")
  kdisplay(iter, verbose) && @printf(iostream, "%5d  %5d  %7.1e  %7s\n", npass, iter, rNorm, "✗ ✗ ✗ ✗")

  # Tolerance for breakdown detection.
  btol = eps(T)^(3/4)

  # Stopping criterion
  breakdown = false
  solved = rNorm ≤ ε
  tired = iter ≥ itmax
  inner_tired = inner_iter ≥ inner_itmax
  status = "unknown"
  user_requested_exit = false

  while !(solved || tired || breakdown || user_requested_exit)

    # Initialize workspace.
    nr = 0  # Number of coefficients stored in Uₖ.
    for i = 1 : mem
      V[i] .= zero(FC)  # Orthogonal basis of Kₖ(MAN, Mr₀).
    end
    l .= zero(FC)  # Lower unit triangular matrix Lₖ.
    U .= zero(FC)  # Upper triangular matrix Uₖ.
    z .= zero(FC)  # Solution of Lₖzₖ = βe₁.

    if restart
      xr .= zero(FC)  # xr === Δx when restart is set to true
      if npass ≥ 1
        mul!(w, A, x)
        @kaxpby!(n, one(FC), b, -one(FC), w)
        MisI || mulorldiv!(r₀, M, w, ldiv)
      end
    end

    # Initial ζ₁ and V₁
    β = @knrm2(n, r₀)
    z[1] = β
    @. V[1] = r₀ / rNorm

    npass = npass + 1
    inner_iter = 0
    inner_tired = false

    while !(solved || inner_tired || breakdown)

      # Update iteration index
      inner_iter = inner_iter + 1

      # Update workspace if more storage is required and restart is set to false
      if !restart && (inner_iter > mem)
        for i = 1 : inner_iter
          push!(U, zero(FC))
        end
        push!(l, zero(FC))
        push!(z, zero(FC))
      end

      # Continue the Arnoldi process.
      p = NisI ? V[inner_iter] : solver.p
      NisI || mulorldiv!(p, N, V[inner_iter], ldiv)  # p ← Nvₖ
      mul!(w, A, p)                                  # w ← ANvₖ
      MisI || mulorldiv!(q, M, w, ldiv)              # q ← MANvₖ
      for i = 1 : inner_iter
        U[nr+i] = @kdot(n, V[i], q)      # hᵢₖ = (vᵢ)ᴴq
        @kaxpy!(n, -U[nr+i], V[i], q)    # q ← q - hᵢₖvᵢ
      end

      # Reorthogonalization of the Krylov basis.
      if reorthogonalization
        for i = 1 : inner_iter
          Htmp = @kdot(n, V[i], q)
          U[nr+i] += Htmp
          @kaxpy!(n, -Htmp, V[i], q)
        end
      end

      # Compute hₖ₊₁.ₖ
      Hbis = @knrm2(n, q)  # hₖ₊₁.ₖ = ‖vₖ₊₁‖₂

      # Update the LU factorization of Hₖ.
      if inner_iter ≥ 2
        for i = 2 : inner_iter
          # uᵢ.ₖ ← hᵢ.ₖ - lᵢ.ᵢ₋₁ * uᵢ₋₁.ₖ
          U[nr+i] = U[nr+i] - l[i-1] * U[nr+i-1]
        end
        # ζₖ = -lₖ.ₖ₋₁ * ζₖ₋₁
        z[inner_iter] = - l[inner_iter-1] * z[inner_iter-1]
      end
      # lₖ₊₁.ₖ = hₖ₊₁.ₖ / uₖ.ₖ
      l[inner_iter] = Hbis / U[nr+inner_iter]

      # Update residual norm estimate.
      # ‖ M(b - Axₖ) ‖₂ = hₖ₊₁.ₖ * |ζₖ / uₖ.ₖ|
      rNorm = Hbis * abs(z[inner_iter] / U[nr+inner_iter])
      history && push!(rNorms, rNorm)

      # Update the number of coefficients in Uₖ
      nr = nr + inner_iter

      # Stopping conditions that do not depend on user input.
      # This is to guard against tolerances that are unreasonably small.
      resid_decrease_mach = (rNorm + one(T) ≤ one(T))

      # Update stopping criterion.
      user_requested_exit = callback(solver) :: Bool
      resid_decrease_lim = rNorm ≤ ε
      breakdown = Hbis ≤ btol
      solved = resid_decrease_lim || resid_decrease_mach
      inner_tired = restart ? inner_iter ≥ min(mem, inner_itmax) : inner_iter ≥ inner_itmax
      kdisplay(iter+inner_iter, verbose) && @printf(iostream, "%5d  %5d  %7.1e  %7.1e\n", npass, iter+inner_iter, rNorm, Hbis)

      # Compute vₖ₊₁.
      if !(solved || inner_tired || breakdown)
        if !restart && (inner_iter ≥ mem)
          push!(V, S(undef, n))
        end
        @. V[inner_iter+1] = q / Hbis  # hₖ₊₁.ₖvₖ₊₁ = q
      end
    end

    # Hₖyₖ = βe₁ ⟺ LₖUₖyₖ = βe₁ ⟺ Uₖyₖ = zₖ.
    # Compute yₖ by solving Uₖyₖ = zₖ with backward substitution.
    y = z  # yᵢ = zᵢ
    for i = inner_iter : -1 : 1
      pos = nr + i - inner_iter      # position of rᵢ.ₖ
      for j = inner_iter : -1 : i+1
        y[i] = y[i] - U[pos] * y[j]  # yᵢ ← yᵢ - uᵢⱼyⱼ
        pos = pos - j + 1            # position of rᵢ.ⱼ₋₁
      end
      y[i] = y[i] / U[pos]  # yᵢ ← yᵢ / rᵢᵢ
    end

    # Form xₖ = NVₖyₖ
    for i = 1 : inner_iter
      @kaxpy!(n, y[i], V[i], xr)
    end
    if !NisI
      solver.p .= xr
      mulorldiv!(xr, N, solver.p, ldiv)
    end
    restart && @kaxpy!(n, one(FC), xr, x)

    # Update inner_itmax, iter and tired variables.
    inner_itmax = inner_itmax - inner_iter
    iter = iter + inner_iter
    tired = iter ≥ itmax
  end
  (verbose > 0) && @printf(iostream, "\n")

  tired               && (status = "maximum number of iterations exceeded")
  breakdown           && (status = "inconsistent linear system")
  solved              && (status = "solution good enough given atol and rtol")
  user_requested_exit && (status = "user-requested exit")

  # Update x
  warm_start && !restart && @kaxpy!(n, one(FC), Δx, x)
  solver.warm_start = false

  # Update stats
  stats.niter = iter
  stats.solved = solved
  stats.inconsistent = !solved && breakdown
  stats.status = status
  return solver
end
