# An implementation of DIOM for the solution of the square linear system Ax = b.
#
# This method is described in
#
# Y. Saad, Practical use of some krylov subspace methods for solving indefinite and nonsymmetric linear systems.
# SIAM journal on scientific and statistical computing, 5(1), pp. 203--228, 1984.
#
# Alexis Montoison, <alexis.montoison@polymtl.ca>
# Montreal, September 2018.

export diom, diom!

"""
    (x, stats) = diom(A, b::AbstractVector{FC};
                      memory::Int=20, M=I, N=I, ldiv::Bool=false,
                      reorthogonalization::Bool=false, atol::T=√eps(T),
                      rtol::T=√eps(T), itmax::Int=0,
                      timemax::Float64=Inf, verbose::Int=0, history::Bool=false,
                      callback=solver->false, iostream::IO=kstdout)

`T` is an `AbstractFloat` such as `Float32`, `Float64` or `BigFloat`.
`FC` is `T` or `Complex{T}`.

    (x, stats) = diom(A, b, x0::AbstractVector; kwargs...)

DIOM can be warm-started from an initial guess `x0` where `kwargs` are the same keyword arguments as above.

Solve the consistent linear system Ax = b of size n using DIOM.

DIOM only orthogonalizes the new vectors of the Krylov basis against the `memory` most recent vectors.
If CG is well defined on `Ax = b` and `memory = 2`, DIOM is theoretically equivalent to CG.
If `k ≤ memory` where `k` is the number of iterations, DIOM is theoretically equivalent to FOM.
Otherwise, DIOM interpolates between CG and FOM and is similar to CG with partial reorthogonalization.

An advantage of DIOM is that non-Hermitian or Hermitian indefinite or both non-Hermitian
and indefinite systems of linear equations can be handled by this single algorithm.

#### Input arguments

* `A`: a linear operator that models a matrix of dimension n;
* `b`: a vector of length n.

#### Optional argument

* `x0`: a vector of length n that represents an initial guess of the solution x.

#### Keyword arguments

* `memory`: the number of most recent vectors of the Krylov basis against which to orthogonalize a new vector;
* `M`: linear operator that models a nonsingular matrix of size `n` used for left preconditioning;
* `N`: linear operator that models a nonsingular matrix of size `n` used for right preconditioning;
* `ldiv`: define whether the preconditioners use `ldiv!` or `mul!`;
* `reorthogonalization`: reorthogonalize the new vectors of the Krylov basis against the `memory` most recent vectors;
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

* Y. Saad, [*Practical use of some krylov subspace methods for solving indefinite and nonsymmetric linear systems*](https://doi.org/10.1137/0905015), SIAM journal on scientific and statistical computing, 5(1), pp. 203--228, 1984.
"""
function diom end

"""
    solver = diom!(solver::DiomSolver, A, b; kwargs...)
    solver = diom!(solver::DiomSolver, A, b, x0; kwargs...)

where `kwargs` are keyword arguments of [`diom`](@ref).

Note that the `memory` keyword argument is the only exception.
It's required to create a `DiomSolver` and can't be changed later.

See [`DiomSolver`](@ref) for more details about the `solver`.
"""
function diom! end

def_args_diom = (:(A                    ),
                 :(b::AbstractVector{FC}))

def_optargs_diom = (:(x0::AbstractVector),)

def_kwargs_diom = (:(; M = I                            ),
                   :(; N = I                            ),
                   :(; ldiv::Bool = false               ),
                   :(; reorthogonalization::Bool = false),
                   :(; atol::T = √eps(T)                ),
                   :(; rtol::T = √eps(T)                ),
                   :(; itmax::Int = 0                   ),
                   :(; timemax::Float64 = Inf           ),
                   :(; verbose::Int = 0                 ),
                   :(; history::Bool = false            ),
                   :(; callback = solver -> false       ),
                   :(; iostream::IO = kstdout           ))

def_kwargs_diom = extract_parameters.(def_kwargs_diom)

args_diom = (:A, :b)
optargs_diom = (:x0,)
kwargs_diom = (:M, :N, :ldiv, :reorthogonalization, :atol, :rtol, :itmax, :timemax, :verbose, :history, :callback, :iostream)

@eval begin
  function diom!(solver :: DiomSolver{T,FC,S}, $(def_args_diom...); $(def_kwargs_diom...)) where {T <: AbstractFloat, FC <: FloatOrComplex{T}, S <: AbstractVector{FC}}

    # Timer
    start_time = time_ns()
    timemax_ns = 1e9 * timemax

    m, n = size(A)
    (m == solver.m && n == solver.n) || error("(solver.m, solver.n) = ($(solver.m), $(solver.n)) is inconsistent with size(A) = ($m, $n)")
    m == n || error("System must be square")
    length(b) == m || error("Inconsistent problem size")
    (verbose > 0) && @printf(iostream, "DIOM: system of size %d\n", n)

    # Check M = Iₙ and N = Iₙ
    MisI = (M === I)
    NisI = (N === I)

    # Check type consistency
    eltype(A) == FC || @warn "eltype(A) ≠ $FC. This could lead to errors or additional allocations in operator-vector products."
    ktypeof(b) <: S || error("ktypeof(b) is not a subtype of $S")

    # Set up workspace.
    allocate_if(!MisI, solver, :w, S, n)
    allocate_if(!NisI, solver, :z, S, n)
    Δx, x, t, P, V = solver.Δx, solver.x, solver.t, solver.P, solver.V
    L, H, stats = solver.L, solver.H, solver.stats
    warm_start = solver.warm_start
    rNorms = stats.residuals
    reset!(stats)
    w  = MisI ? t : solver.w
    r₀ = MisI ? t : solver.w

    # Initial solution x₀ and residual r₀.
    kfill!(x, zero(FC))  # x₀
    if warm_start
      mul!(t, A, Δx)
      kaxpby!(n, one(FC), b, -one(FC), t)
    else
      kcopy!(n, t, b)  # t ← b
    end
    MisI || mulorldiv!(r₀, M, t, ldiv)  # M(b - Ax₀)
    rNorm = knorm(n, r₀)                # β = ‖r₀‖₂
    history && push!(rNorms, rNorm)
    if rNorm == 0
      stats.niter = 0
      stats.solved, stats.inconsistent = true, false
      stats.storage = sizeof(solver)
      stats.timer = ktimer(start_time)
      stats.status = "x = 0 is a zero-residual solution"
      solver.warm_start = false
      return solver
    end

    iter = 0
    itmax == 0 && (itmax = 2*n)

    ε = atol + rtol * rNorm
    (verbose > 0) && @printf(iostream, "%5s  %7s  %5s\n", "k", "‖rₖ‖", "timer")
    kdisplay(iter, verbose) && @printf(iostream, "%5d  %7.1e  %.2fs\n", iter, rNorm, ktimer(start_time))

    mem = length(V)  # Memory
    for i = 1 : mem
      kfill!(V[i], zero(FC))  # Orthogonal basis of Kₖ(MAN, Mr₀).
    end
    for i = 1 : mem-1
      kfill!(P[i], zero(FC))  # Directions Pₖ = NVₖ(Uₖ)⁻¹.
    end
    kfill!(H, zero(FC))  # Last column of the band hessenberg matrix Hₖ = LₖUₖ.
    # Each column has at most mem + 1 nonzero elements.
    # hᵢ.ₖ is stored as H[k-i+1], i ≤ k. hₖ₊₁.ₖ is not stored in H.
    # k-i+1 represents the indice of the diagonal where hᵢ.ₖ is located.
    # In addition of that, the last column of Uₖ is stored in H.
    kfill!(L, zero(FC))  # Last mem-1 pivots of Lₖ.

    # Initial ξ₁ and V₁.
    ξ = rNorm
    V[1] .= r₀ ./ rNorm

    # Stopping criterion.
    solved = rNorm ≤ ε
    tired = iter ≥ itmax
    status = "unknown"
    user_requested_exit = false
    overtimed = false

    while !(solved || tired || user_requested_exit || overtimed)

      # Update iteration index.
      iter = iter + 1

      # Set position in circulars stacks.
      pos = mod(iter-1, mem) + 1     # Position corresponding to vₖ in the circular stack V.
      next_pos = mod(iter, mem) + 1  # Position corresponding to vₖ₊₁ in the circular stack V.

      # Incomplete Arnoldi procedure.
      z = NisI ? V[pos] : solver.z
      NisI || mulorldiv!(z, N, V[pos], ldiv)  # Nvₖ, forms pₖ
      mul!(t, A, z)                           # ANvₖ
      MisI || mulorldiv!(w, M, t, ldiv)       # MANvₖ, forms vₖ₊₁
      for i = max(1, iter-mem+1) : iter
        ipos = mod(i-1, mem) + 1  # Position corresponding to vᵢ in the circular stack V.
        diag = iter - i + 1
        H[diag] = kdot(n, w, V[ipos])    # hᵢ.ₖ = ⟨MANvₖ, vᵢ⟩
        kaxpy!(n, -H[diag], V[ipos], w)  # w ← w - hᵢ.ₖvᵢ
      end

      # Partial reorthogonalization of the Krylov basis.
      if reorthogonalization
        for i = max(1, iter-mem+1) : iter
          ipos = mod(i-1, mem) + 1
          diag = iter - i + 1
          Htmp = kdot(n, w, V[ipos])
          H[diag] += Htmp
          kaxpy!(n, -Htmp, V[ipos], w)
        end
      end

      # Compute hₖ₊₁.ₖ and vₖ₊₁.
      Haux = knorm(n, w)          # hₖ₊₁.ₖ = ‖vₖ₊₁‖₂
      if Haux ≠ 0                 # hₖ₊₁.ₖ = 0 ⇒ "lucky breakdown"
        V[next_pos] .= w ./ Haux  # vₖ₊₁ = w / hₖ₊₁.ₖ
      end

      # Update the LU factorization of Hₖ.
      # Compute the last column of Uₖ.
      if iter ≥ 2
        # u₁.ₖ ← h₁.ₖ             if iter ≤ mem
        # uₖ₋ₘₑₘ₊₁.ₖ ← hₖ₋ₘₑₘ₊₁.ₖ if iter ≥ mem + 1
        for i = max(2,iter-mem+2) : iter
          lpos = mod(i-1, mem-1) + 1  # Position corresponding to lᵢ.ᵢ₋₁ in the circular stack L.
          diag = iter - i + 1
          next_diag = diag + 1
          # uᵢ.ₖ ← hᵢ.ₖ - lᵢ.ᵢ₋₁ * uᵢ₋₁.ₖ
          H[diag] = H[diag] - L[lpos] * H[next_diag]
          if i == iter
            # Compute ξₖ the last component of zₖ = β(Lₖ)⁻¹e₁.
            # ξₖ = -lₖ.ₖ₋₁ * ξₖ₋₁
            ξ = - L[lpos] * ξ
          end
        end
      end
      # Compute next pivot lₖ₊₁.ₖ = hₖ₊₁.ₖ / uₖ.ₖ
      next_lpos = mod(iter, mem-1) + 1
      L[next_lpos] = Haux / H[1]

      ppos = mod(iter-1, mem-1) + 1 # Position corresponding to pₖ in the circular stack P.

      # Compute the direction pₖ, the last column of Pₖ = NVₖ(Uₖ)⁻¹.
      # u₁.ₖp₁ + ... + uₖ.ₖpₖ = Nvₖ             if k ≤ mem
      # uₖ₋ₘₑₘ₊₁.ₖpₖ₋ₘₑₘ₊₁ + ... + uₖ.ₖpₖ = Nvₖ if k ≥ mem + 1
      for i = max(1,iter-mem+1) : iter-1
        ipos = mod(i-1, mem-1) + 1  # Position corresponding to pᵢ in the circular stack P.
        diag = iter - i + 1
        if ipos == ppos
          # pₖ ← -uₖ₋ₘₑₘ₊₁.ₖ * pₖ₋ₘₑₘ₊₁
          kscal!(n, -H[diag], P[ppos])
        else
          # pₖ ← pₖ - uᵢ.ₖ * pᵢ
          kaxpy!(n, -H[diag], P[ipos], P[ppos])
        end
      end
      # pₐᵤₓ ← pₐᵤₓ + Nvₖ
      kaxpy!(n, one(FC), z, P[ppos])
      # pₖ = pₐᵤₓ / uₖ.ₖ
      P[ppos] .= P[ppos] ./ H[1]

      # Update solution xₖ.
      # xₖ = xₖ₋₁ + ξₖ * pₖ
      kaxpy!(n, ξ, P[ppos], x)

      # Compute residual norm.
      # ‖ M(b - Axₖ) ‖₂ = hₖ₊₁.ₖ * |ξₖ / uₖ.ₖ|
      rNorm = Haux * abs(ξ / H[1])
      history && push!(rNorms, rNorm)

      # Stopping conditions that do not depend on user input.
      # This is to guard against tolerances that are unreasonably small.
      resid_decrease_mach = (rNorm + one(T) ≤ one(T))

      # Update stopping criterion.
      user_requested_exit = callback(solver) :: Bool
      resid_decrease_lim = rNorm ≤ ε
      solved = resid_decrease_lim || resid_decrease_mach
      tired = iter ≥ itmax
      timer = time_ns() - start_time
      overtimed = timer > timemax_ns
      kdisplay(iter, verbose) && @printf(iostream, "%5d  %7.1e  %.2fs\n", iter, rNorm, ktimer(start_time))
    end
    (verbose > 0) && @printf(iostream, "\n")

    # Termination status
    tired               && (status = "maximum number of iterations exceeded")
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
    stats.storage = sizeof(solver)
    stats.timer = ktimer(start_time)
    stats.status = status
    return solver
  end
end
