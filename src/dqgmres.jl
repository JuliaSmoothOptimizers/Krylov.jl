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
                         memory::Int=20, M=I, N=I, ldiv::Bool=false,
                         reorthogonalization::Bool=false, atol::T=√eps(T),
                         rtol::T=√eps(T), itmax::Int=0,
                         timemax::Float64=Inf, verbose::Int=0, history::Bool=false,
                         callback=solver->false, iostream::IO=kstdout)

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

#### Input arguments

* `A`: a linear operator that models a matrix of dimension `n`;
* `b`: a vector of length `n`.

#### Optional argument

* `x0`: a vector of length `n` that represents an initial guess of the solution `x`.

#### Keyword arguments

* `memory`: the number of most recent vectors of the Krylov basis against which to orthogonalize a new vector;
* `M`: linear operator that models a nonsingular matrix of size `n` used for left preconditioning;
* `N`: linear operator that models a nonsingular matrix of size `n` used for right preconditioning;
* `reorthogonalization`: reorthogonalize the new vectors of the Krylov basis against the `memory` most recent vectors;
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

#### Reference

* Y. Saad and K. Wu, [*DQGMRES: a quasi minimal residual algorithm based on incomplete orthogonalization*](https://doi.org/10.1002/(SICI)1099-1506(199607/08)3:4%3C329::AID-NLA86%3E3.0.CO;2-8), Numerical Linear Algebra with Applications, Vol. 3(4), pp. 329--343, 1996.
"""
function dqgmres end

"""
    solver = dqgmres!(solver::DqgmresSolver, A, b; kwargs...)
    solver = dqgmres!(solver::DqgmresSolver, A, b, x0; kwargs...)

where `kwargs` are keyword arguments of [`dqgmres`](@ref).

Note that the `memory` keyword argument is the only exception.
It's required to create a `DqgmresSolver` and can't be changed later.

See [`DqgmresSolver`](@ref) for more details about the `solver`.
"""
function dqgmres! end

def_args_dqgmres = (:(A                    ),
                    :(b::AbstractVector{FC}))

def_optargs_dqgmres = (:(x0::AbstractVector),)

def_kwargs_dqgmres = (:(; M = I                            ),
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

def_kwargs_dqgmres = extract_parameters.(def_kwargs_dqgmres)

args_dqgmres = (:A, :b)
optargs_dqgmres = (:x0,)
kwargs_dqgmres = (:M, :N, :ldiv, :reorthogonalization, :atol, :rtol, :itmax, :timemax, :verbose, :history, :callback, :iostream)

@eval begin
  function dqgmres!(solver :: DqgmresSolver{T,FC,S}, $(def_args_dqgmres...); $(def_kwargs_dqgmres...)) where {T <: AbstractFloat, FC <: FloatOrComplex{T}, S <: AbstractVector{FC}}

    # Timer
    start_time = time_ns()
    timemax_ns = 1e9 * timemax

    m, n = size(A)
    (m == solver.m && n == solver.n) || error("(solver.m, solver.n) = ($(solver.m), $(solver.n)) is inconsistent with size(A) = ($m, $n)")
    m == n || error("System must be square")
    length(b) == m || error("Inconsistent problem size")
    (verbose > 0) && @printf(iostream, "DQGMRES: system of size %d\n", n)

    # Check M = Iₙ and N = Iₙ
    MisI = (M === I)
    NisI = (N === I)

    # Check type consistency
    eltype(A) == FC || @warn "eltype(A) ≠ $FC. This could lead to errors or additional allocations in operator-vector products."
    ktypeof(b) <: S || error("ktypeof(b) is not a subtype of $S")

    # Set up workspace.
    allocate_if(!MisI, solver, :w, S, solver.x)  # The length of w is n
    allocate_if(!NisI, solver, :z, S, solver.x)  # The length of z is n
    Δx, x, t, P, V = solver.Δx, solver.x, solver.t, solver.P, solver.V
    c, s, H, stats = solver.c, solver.s, solver.H, solver.stats
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
      stats.timer = start_time |> ktimer
      stats.status = "x is a zero-residual solution"
      warm_start && kaxpy!(n, one(FC), Δx, x)
      solver.warm_start = false
      return solver
    end

    iter = 0
    itmax == 0 && (itmax = 2*n)

    ε = atol + rtol * rNorm
    (verbose > 0) && @printf(iostream, "%5s  %7s  %5s\n", "k", "‖rₖ‖", "timer")
    kdisplay(iter, verbose) && @printf(iostream, "%5d  %7.1e  %.2fs\n", iter, rNorm, start_time |> ktimer)

    # Set up workspace.
    mem = length(V)  # Memory.
    for i = 1 : mem
      kfill!(V[i], zero(FC))  # Orthogonal basis of Kₖ(MAN, Mr₀).
      kfill!(P[i], zero(FC))  # Directions for x : Pₖ = NVₖ(Rₖ)⁻¹.
    end
    kfill!(c, zero(T))   # Last mem Givens cosines used for the factorization QₖRₖ = Hₖ.
    kfill!(s, zero(FC))  # Last mem Givens sines used for the factorization QₖRₖ = Hₖ.
    kfill!(H, zero(FC))  # Last column of the band hessenberg matrix Hₖ.
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
    overtimed = false

    while !(solved || tired || user_requested_exit || overtimed)

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
          kscal!(n, -H[diag], P[pos])
        else
          # pₐᵤₓ ← pₐᵤₓ - hᵢ.ₖ * pᵢ
          kaxpy!(n, -H[diag], P[ipos], P[pos])
        end
      end
      # pₐᵤₓ ← pₐᵤₓ + Nvₖ
      kaxpy!(n, one(FC), z, P[pos])
      # pₖ = pₐᵤₓ / hₖ.ₖ
      P[pos] .= P[pos] ./ H[1]

      # Compute solution xₖ.
      # xₖ ← xₖ₋₁ + γₖ * pₖ
      kaxpy!(n, γₖ, P[pos], x)

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
      timer = time_ns() - start_time
      overtimed = timer > timemax_ns
      kdisplay(iter, verbose) && @printf(iostream, "%5d  %7.1e  %.2fs\n", iter, rNorm, start_time |> ktimer)
    end
    (verbose > 0) && @printf(iostream, "\n")

    # Termination status
    solved              && (status = "solution good enough given atol and rtol")
    tired               && (status = "maximum number of iterations exceeded")
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
