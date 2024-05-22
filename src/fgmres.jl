# An implementation of FGMRES for the solution of the square linear system Ax = b.
#
# This method is described in
#
# Y. Saad, A Flexible Inner-Outer Preconditioned GMRES Algorithms.
# SIAM Journal on Scientific Computing, Vol. 14(2), pp. 461--469, 1993.
#
# Alexis Montoison, <alexis.montoison@polymtl.ca>
# Montreal, September 2022.

export fgmres, fgmres!

"""
    (x, stats) = fgmres(A, b::AbstractVector{FC};
                        memory::Int=20, M=I, N=I, ldiv::Bool=false,
                        restart::Bool=false, reorthogonalization::Bool=false,
                        atol::T=√eps(T), rtol::T=√eps(T), itmax::Int=0,
                        timemax::Float64=Inf, verbose::Int=0, history::Bool=false,
                        callback=solver->false, iostream::IO=kstdout)

`T` is an `AbstractFloat` such as `Float32`, `Float64` or `BigFloat`.
`FC` is `T` or `Complex{T}`.

    (x, stats) = fgmres(A, b, x0::AbstractVector; kwargs...)

FGMRES can be warm-started from an initial guess `x0` where `kwargs` are the same keyword arguments as above.

Solve the linear system Ax = b of size n using FGMRES.

FGMRES computes a sequence of approximate solutions with minimum residual.
FGMRES is a variant of GMRES that allows changes in the right preconditioner at each iteration.

This implementation allows a left preconditioner M and a flexible right preconditioner N.
A situation in which the preconditioner is "not constant" is when a relaxation-type method,
a Chebyshev iteration or another Krylov subspace method is used as a preconditioner. 
Compared to GMRES, there is no additional cost incurred in the arithmetic but the memory requirement almost doubles.
Thus, GMRES is recommended if the right preconditioner N is constant.

#### Input arguments

* `A`: a linear operator that models a matrix of dimension n;
* `b`: a vector of length n.

#### Optional argument

* `x0`: a vector of length n that represents an initial guess of the solution x.

#### Keyword arguments

* `memory`: if `restart = true`, the restarted version FGMRES(k) is used with `k = memory`. If `restart = false`, the parameter `memory` should be used as a hint of the number of iterations to limit dynamic memory allocations. Additional storage will be allocated if the number of iterations exceeds `memory`;
* `M`: linear operator that models a nonsingular matrix of size `n` used for left preconditioning;
* `N`: linear operator that models a nonsingular matrix of size `n` used for right preconditioning;
* `ldiv`: define whether the preconditioners use `ldiv!` or `mul!`;
* `restart`: restart the method after `memory` iterations;
* `reorthogonalization`: reorthogonalize the new vectors of the Krylov basis against all previous vectors;
* `atol`: absolute stopping tolerance based on the residual norm;
* `rtol`: relative stopping tolerance based on the residual norm;
* `itmax`: the maximum number of iterations. If `itmax=0`, the default number of iterations is set to `2n`;
* `timemax`: the time limit in seconds;
* `verbose`: additional details can be displayed if verbose mode is enabled (verbose > 0). Information will be displayed every `verbose` iterations;
* `history`: collect additional statistics on the run such as residual norms, or Aᴴ-residual norms;
* `callback`: function or functor called as `callback(solver)` that returns `true` if the Krylov method should terminate, and `false` otherwise;
* `iostream`: stream to which output is logged.

#### Output arguments

* `x`: a dense vector of length n;
* `stats`: statistics collected on the run in a [`SimpleStats`](@ref) structure.

#### Reference

* Y. Saad, [*A Flexible Inner-Outer Preconditioned GMRES Algorithm*](https://doi.org/10.1137/0914028), SIAM Journal on Scientific Computing, Vol. 14(2), pp. 461--469, 1993.
"""
function fgmres end

"""
    solver = fgmres!(solver::FgmresSolver, A, b; kwargs...)
    solver = fgmres!(solver::FgmresSolver, A, b, x0; kwargs...)

where `kwargs` are keyword arguments of [`fgmres`](@ref).

Note that the `memory` keyword argument is the only exception.
It's required to create a `FgmresSolver` and can't be changed later.

See [`FgmresSolver`](@ref) for more details about the `solver`.
"""
function fgmres! end

def_args_fgmres = (:(A                    ),
                   :(b::AbstractVector{FC}))

def_optargs_fgmres = (:(x0::AbstractVector),)

def_kwargs_fgmres = (:(; M = I                            ),
                     :(; N = I                            ),
                     :(; ldiv::Bool = false               ),
                     :(; restart::Bool = false            ),
                     :(; reorthogonalization::Bool = false),
                     :(; atol::T = √eps(T)                ),
                     :(; rtol::T = √eps(T)                ),
                     :(; itmax::Int = 0                   ),
                     :(; timemax::Float64 = Inf           ),
                     :(; verbose::Int = 0                 ),
                     :(; history::Bool = false            ),
                     :(; callback = solver -> false       ),
                     :(; iostream::IO = kstdout           ))

def_kwargs_fgmres = mapreduce(extract_parameters, vcat, def_kwargs_fgmres)

args_fgmres = (:A, :b)
optargs_fgmres = (:x0,)
kwargs_fgmres = (:M, :N, :ldiv, :restart, :reorthogonalization, :atol, :rtol, :itmax, :timemax, :verbose, :history, :callback, :iostream)

@eval begin
  function fgmres($(def_args_fgmres...), $(def_optargs_fgmres...); memory :: Int=20, $(def_kwargs_fgmres...)) where {T <: AbstractFloat, FC <: FloatOrComplex{T}}
    start_time = time_ns()
    solver = FgmresSolver(A, b, memory)
    warm_start!(solver, $(optargs_fgmres...))
    elapsed_time = ktimer(start_time)
    timemax -= elapsed_time
    fgmres!(solver, $(args_fgmres...); $(kwargs_fgmres...))
    solver.stats.timer += elapsed_time
    return (solver.x, solver.stats)
  end

  function fgmres($(def_args_fgmres...); memory :: Int=20, $(def_kwargs_fgmres...)) where {T <: AbstractFloat, FC <: FloatOrComplex{T}}
    start_time = time_ns()
    solver = FgmresSolver(A, b, memory)
    elapsed_time = ktimer(start_time)
    timemax -= elapsed_time
    fgmres!(solver, $(args_fgmres...); $(kwargs_fgmres...))
    solver.stats.timer += elapsed_time
    return (solver.x, solver.stats)
  end

  function fgmres!(solver :: FgmresSolver{T,FC,S}, $(def_args_fgmres...); $(def_kwargs_fgmres...)) where {T <: AbstractFloat, FC <: FloatOrComplex{T}, S <: AbstractVector{FC}}

    # Timer
    start_time = time_ns()
    timemax_ns = 1e9 * timemax

    m, n = size(A)
    (m == solver.m && n == solver.n) || error("(solver.m, solver.n) = ($(solver.m), $(solver.n)) is inconsistent with size(A) = ($m, $n)")
    m == n || error("System must be square")
    length(b) == m || error("Inconsistent problem size")
    (verbose > 0) && @printf(iostream, "FGMRES: system of size %d\n", n)

    # Check M = Iₙ
    MisI = (M === I)

    # Check type consistency
    eltype(A) == FC || @warn "eltype(A) ≠ $FC. This could lead to errors or additional allocations in operator-vector products."
    ktypeof(b) <: S || error("ktypeof(b) is not a subtype of $S")

    # Set up workspace.
    allocate_if(!MisI  , solver, :q , S, n)
    allocate_if(restart, solver, :Δx, S, n)
    Δx, x, w, V, Z = solver.Δx, solver.x, solver.w, solver.V, solver.Z
    z, c, s, R, stats = solver.z, solver.c, solver.s, solver.R, solver.stats
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
      stats.storage = sizeof(solver)
      stats.timer = ktimer(start_time)
      stats.status = "x = 0 is a zero-residual solution"
      solver.warm_start = false
      return solver
    end

    mem = length(c)  # Memory
    npass = 0        # Number of pass

    iter = 0        # Cumulative number of iterations
    inner_iter = 0  # Number of iterations in a pass

    itmax == 0 && (itmax = 2*n)
    inner_itmax = itmax

    (verbose > 0) && @printf(iostream, "%5s  %5s  %7s  %7s  %5s\n", "pass", "k", "‖rₖ‖", "hₖ₊₁.ₖ", "timer")
    kdisplay(iter, verbose) && @printf(iostream, "%5d  %5d  %7.1e  %7s  %.2fs\n", npass, iter, rNorm, "✗ ✗ ✗ ✗", ktimer(start_time))

    # Tolerance for breakdown detection.
    btol = eps(T)^(3/4)

    # Stopping criterion
    breakdown = false
    inconsistent = false
    solved = rNorm ≤ ε
    tired = iter ≥ itmax
    inner_tired = inner_iter ≥ inner_itmax
    status = "unknown"
    user_requested_exit = false
    overtimed = false

    while !(solved || tired || breakdown || user_requested_exit || overtimed)

      # Initialize workspace.
      nr = 0  # Number of coefficients stored in Rₖ.
      for i = 1 : mem
        V[i] .= zero(FC)  # Orthogonal basis of {Mr₀, MANₖr₀, ..., (MANₖ)ᵏ⁻¹r₀}.
        Z[i] .= zero(FC)  # Zₖ = [N₁v₁, ..., Nₖvₖ]
      end
      s .= zero(FC)  # Givens sines used for the factorization QₖRₖ = Hₖ₊₁.ₖ.
      c .= zero(T)   # Givens cosines used for the factorization QₖRₖ = Hₖ₊₁.ₖ.
      R .= zero(FC)  # Upper triangular matrix Rₖ.
      z .= zero(FC)  # Right-hand of the least squares problem min ‖Hₖ₊₁.ₖyₖ - βe₁‖₂.

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
      V[1] .= r₀ ./ rNorm

      npass = npass + 1
      solver.inner_iter = 0
      inner_tired = false

      while !(solved || inner_tired || breakdown || user_requested_exit || overtimed)

        # Update iteration index
        solver.inner_iter = solver.inner_iter + 1
        inner_iter = solver.inner_iter

        # Update workspace if more storage is required and restart is set to false
        if !restart && (inner_iter > mem)
          for i = 1 : inner_iter
            push!(R, zero(FC))
          end
          push!(s, zero(FC))
          push!(c, zero(T))
          push!(Z, S(undef, n))
        end

        # Continue the process.
        # MAZₖ = Vₖ₊₁Hₖ₊₁.ₖ
        mulorldiv!(Z[inner_iter], N, V[inner_iter], ldiv)  # zₖ ← Nₖvₖ
        mul!(w, A, Z[inner_iter])                          # w  ← Azₖ
        MisI || mulorldiv!(q, M, w, ldiv)                  # q  ← MAzₖ
        for i = 1 : inner_iter
          R[nr+i] = @kdot(n, V[i], q)      # hᵢₖ = (vᵢ)ᴴq
          @kaxpy!(n, -R[nr+i], V[i], q)    # q ← q - hᵢₖvᵢ
        end

        # Reorthogonalization of the basis.
        if reorthogonalization
          for i = 1 : inner_iter
            Htmp = @kdot(n, V[i], q)
            R[nr+i] += Htmp
            @kaxpy!(n, -Htmp, V[i], q)
          end
        end

        # Compute hₖ₊₁.ₖ
        Hbis = @knrm2(n, q)  # hₖ₊₁.ₖ = ‖vₖ₊₁‖₂

        # Update the QR factorization of Hₖ₊₁.ₖ.
        # Apply previous Givens reflections Ωᵢ.
        # [cᵢ  sᵢ] [ r̄ᵢ.ₖ ] = [ rᵢ.ₖ ]
        # [s̄ᵢ -cᵢ] [rᵢ₊₁.ₖ]   [r̄ᵢ₊₁.ₖ]
        for i = 1 : inner_iter-1
          Rtmp      =      c[i]  * R[nr+i] + s[i] * R[nr+i+1]
          R[nr+i+1] = conj(s[i]) * R[nr+i] - c[i] * R[nr+i+1]
          R[nr+i]   = Rtmp
        end

        # Compute and apply current Givens reflection Ωₖ.
        # [cₖ  sₖ] [ r̄ₖ.ₖ ] = [rₖ.ₖ]
        # [s̄ₖ -cₖ] [hₖ₊₁.ₖ]   [ 0  ]
        (c[inner_iter], s[inner_iter], R[nr+inner_iter]) = sym_givens(R[nr+inner_iter], Hbis)

        # Update zₖ = (Qₖ)ᴴβe₁
        ζₖ₊₁          = conj(s[inner_iter]) * z[inner_iter]
        z[inner_iter] =      c[inner_iter]  * z[inner_iter]

        # Update residual norm estimate.
        # ‖ M⁻¹(b - Axₖ) ‖₂ = |ζₖ₊₁|
        rNorm = abs(ζₖ₊₁)
        history && push!(rNorms, rNorm)

        # Update the number of coefficients in Rₖ
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
        timer = time_ns() - start_time
        overtimed = timer > timemax_ns
        kdisplay(iter+inner_iter, verbose) && @printf(iostream, "%5d  %5d  %7.1e  %7.1e  %.2fs\n", npass, iter+inner_iter, rNorm, Hbis, ktimer(start_time))

        # Compute vₖ₊₁
        if !(solved || inner_tired || breakdown || user_requested_exit || overtimed)
          if !restart && (inner_iter ≥ mem)
            push!(V, S(undef, n))
            push!(z, zero(FC))
          end
          V[inner_iter+1] .= q ./ Hbis  # hₖ₊₁.ₖvₖ₊₁ = q
          z[inner_iter+1] = ζₖ₊₁
        end
      end

      # Compute y by solving Ry = z with backward substitution.
      y = z  # yᵢ = ζᵢ
      for i = inner_iter : -1 : 1
        pos = nr + i - inner_iter      # position of rᵢ.ₖ
        for j = inner_iter : -1 : i+1
          y[i] = y[i] - R[pos] * y[j]  # yᵢ ← yᵢ - rᵢⱼyⱼ
          pos = pos - j + 1            # position of rᵢ.ⱼ₋₁
        end
        # Rₖ can be singular if the system is inconsistent
        if abs(R[pos]) ≤ btol
          y[i] = zero(FC)
          inconsistent = true
        else
          y[i] = y[i] / R[pos]  # yᵢ ← yᵢ / rᵢᵢ
        end
      end

      # Form xₖ = N₁v₁y₁ + ... + Nₖvₖyₖ = z₁y₁ + ... + zₖyₖ
      for i = 1 : inner_iter
        @kaxpy!(n, y[i], Z[i], xr)
      end
      restart && @kaxpy!(n, one(FC), xr, x)

      # Update inner_itmax, iter and tired variables.
      inner_itmax = inner_itmax - inner_iter
      iter = iter + inner_iter
      tired = iter ≥ itmax
      timer = time_ns() - start_time
      overtimed = timer > timemax_ns
    end
    (verbose > 0) && @printf(iostream, "\n")

    # Termination status
    tired               && (status = "maximum number of iterations exceeded")
    solved              && (status = "solution good enough given atol and rtol")
    inconsistent        && (status = "found approximate least-squares solution")
    user_requested_exit && (status = "user-requested exit")
    overtimed           && (status = "time limit exceeded")

    # Update x
    warm_start && !restart && @kaxpy!(n, one(FC), Δx, x)
    solver.warm_start = false

    # Update stats
    stats.niter = iter
    stats.solved = solved
    stats.inconsistent = inconsistent
    stats.storage = sizeof(solver)
    stats.timer = ktimer(start_time)
    stats.status = status
    return solver
  end
end
