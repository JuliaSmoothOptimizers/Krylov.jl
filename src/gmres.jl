# An implementation of GMRES for the solution of the square linear system Ax = b.
#
# This method is described in
#
# Y. Saad and M. H. Schultz, GMRES: A Generalized Minimal Residual Algorithm for Solving Nonsymmetric Linear Systems.
# SIAM Journal on Scientific and Statistical Computing, Vol. 7(3), pp. 856--869, 1986.
#
# Alexis Montoison, <alexis.montoison@polymtl.ca>
# Montreal, December 2018.

export gmres, gmres!

"""
    (x, stats) = gmres(A, b::AbstractVector{FC};
                       memory::Int=20, M=I, N=I, ldiv::Bool=false,
                       restart::Bool=false, reorthogonalization::Bool=false,
                       atol::T=√eps(T), rtol::T=√eps(T), itmax::Int=0,
                       timemax::Float64=Inf, verbose::Int=0, history::Bool=false,
                       callback=workspace->false, iostream::IO=kstdout)

`T` is an `AbstractFloat` such as `Float32`, `Float64` or `BigFloat`.
`FC` is `T` or `Complex{T}`.

    (x, stats) = gmres(A, b, x0::AbstractVector; kwargs...)

GMRES can be warm-started from an initial guess `x0` where `kwargs` are the same keyword arguments as above.

Solve the linear system Ax = b of size n using GMRES.

GMRES algorithm is based on the Arnoldi process and computes a sequence of approximate solutions with the minimum residual.

#### Interface

To easily switch between Krylov methods, use the generic interface [`krylov_solve`](@ref) with `method = :gmres`.

For an in-place variant that reuses memory across solves, see [`gmres!`](@ref).

#### Input arguments

* `A`: a linear operator that models a matrix of dimension `n`;
* `b`: a vector of length `n`.

#### Optional argument

* `x0`: a vector of length `n` that represents an initial guess of the solution `x`.

#### Keyword arguments

* `memory`: if `restart = true`, the restarted version GMRES(k) is used with `k = memory`. If `restart = false`, the parameter `memory` should be used as a hint of the number of iterations to limit dynamic memory allocations. Additional storage will be allocated if the number of iterations exceeds `memory`;
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
* `callback`: function or functor called as `callback(workspace)` that returns `true` if the Krylov method should terminate, and `false` otherwise;
* `iostream`: stream to which output is logged.

#### Output arguments

* `x`: a dense vector of length `n`;
* `stats`: statistics collected on the run in a [`SimpleStats`](@ref) structure.

#### Reference

* Y. Saad and M. H. Schultz, [*GMRES: A Generalized Minimal Residual Algorithm for Solving Nonsymmetric Linear Systems*](https://doi.org/10.1137/0907058), SIAM Journal on Scientific and Statistical Computing, Vol. 7(3), pp. 856--869, 1986.
"""
function gmres end

"""
    workspace = gmres!(workspace::GmresWorkspace, A, b; kwargs...)
    workspace = gmres!(workspace::GmresWorkspace, A, b, x0; kwargs...)

In these calls, `kwargs` are keyword arguments of [`gmres`](@ref).
The keyword argument `memory` is the only exception.
It is only supported by [`gmres`](@ref) and is required to create a `GmresWorkspace`.
It cannot be changed later.

See [`GmresWorkspace`](@ref) for instructions on how to create the `workspace`.

For a more generic interface, you can use [`krylov_workspace`](@ref) with `method = :gmres` to allocate the workspace,
and [`krylov_solve!`](@ref) to run the Krylov method in-place.
"""
function gmres! end

def_args_gmres = (:(A                    ),
                  :(b::AbstractVector{FC}))

def_optargs_gmres = (:(x0::AbstractVector),)

def_kwargs_gmres = (:(; M = I                            ),
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
                    :(; callback = workspace -> false    ),
                    :(; iostream::IO = kstdout           ))

def_kwargs_gmres = extract_parameters.(def_kwargs_gmres)

args_gmres = (:A, :b)
optargs_gmres = (:x0,)
kwargs_gmres = (:M, :N, :ldiv, :restart, :reorthogonalization, :atol, :rtol, :itmax, :timemax, :verbose, :history, :callback, :iostream)

@eval begin
  function gmres!(workspace :: GmresWorkspace{T,FC,S}, $(def_args_gmres...); $(def_kwargs_gmres...)) where {T <: AbstractFloat, FC <: FloatOrComplex{T}, S <: AbstractVector{FC}}

    # Timer
    start_time = time_ns()
    timemax_ns = 1e9 * timemax

    m, n = size(A)
    (m == workspace.m && n == workspace.n) || error("(workspace.m, workspace.n) = ($(workspace.m), $(workspace.n)) is inconsistent with size(A) = ($m, $n)")
    m == n || error("System must be square")
    length(b) == m || error("Inconsistent problem size")
    (verbose > 0) && @printf(iostream, "GMRES: system of size %d\n", n)

    # Check M = Iₙ and N = Iₙ
    MisI = (M === I)
    NisI = (N === I)

    # Check type consistency
    eltype(A) == FC || @warn "eltype(A) ≠ $FC. This could lead to errors or additional allocations in operator-vector products."
    ktypeof(b) == S || error("ktypeof(b) must be equal to $S")

    # Set up workspace.
    allocate_if(!MisI  , workspace, :q , S, workspace.x)  # The length of q is n
    allocate_if(!NisI  , workspace, :p , S, workspace.x)  # The length of p is n
    allocate_if(restart, workspace, :Δx, S, workspace.x)  # The length of Δx is n
    Δx, x, w, V, z = workspace.Δx, workspace.x, workspace.w, workspace.V, workspace.z
    c, s, R, stats = workspace.c, workspace.s, workspace.R, workspace.stats
    warm_start = workspace.warm_start
    rNorms = stats.residuals
    reset!(stats)
    q  = MisI ? w : workspace.q
    r₀ = MisI ? w : workspace.q
    xr = restart ? Δx : x

    # Initial solution x₀.
    kfill!(x, zero(FC))

    # Initial residual r₀.
    if warm_start
      mul!(w, A, Δx)
      kaxpby!(n, one(FC), b, -one(FC), w)
      restart && kaxpy!(n, one(FC), Δx, x)
    else
      kcopy!(n, w, b)  # w ← b
    end
    MisI || mulorldiv!(r₀, M, w, ldiv)  # r₀ = M(b - Ax₀)
    β = knorm(n, r₀)                    # β = ‖r₀‖₂

    rNorm = β
    history && push!(rNorms, β)
    ε = atol + rtol * rNorm

    if β == 0
      stats.niter = 0
      stats.solved, stats.inconsistent = true, false
      stats.timer = start_time |> ktimer
      stats.status = "x is a zero-residual solution"
      warm_start && kaxpy!(n, one(FC), Δx, x)
      workspace.warm_start = false
      return workspace
    end

    mem = length(c)  # Memory
    npass = 0        # Number of pass

    iter = 0        # Cumulative number of iterations
    inner_iter = 0  # Number of iterations in a pass

    itmax == 0 && (itmax = 2*n)
    inner_itmax = itmax

    (verbose > 0) && @printf(iostream, "%5s  %5s  %7s  %7s  %5s\n", "pass", "k", "‖rₖ‖", "hₖ₊₁.ₖ", "timer")
    kdisplay(iter, verbose) && @printf(iostream, "%5d  %5d  %7.1e  %7s  %.2fs\n", npass, iter, rNorm, "✗ ✗ ✗ ✗", start_time |> ktimer)

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
        kfill!(V[i], zero(FC))  # Orthogonal basis of Kₖ(MAN, Mr₀).
      end
      kfill!(s, zero(FC))  # Givens sines used for the factorization QₖRₖ = Hₖ₊₁.ₖ.
      kfill!(c, zero(T))   # Givens cosines used for the factorization QₖRₖ = Hₖ₊₁.ₖ.
      kfill!(R, zero(FC))  # Upper triangular matrix Rₖ.
      kfill!(z, zero(FC))  # Right-hand of the least squares problem min ‖Hₖ₊₁.ₖyₖ - βe₁‖₂.

      if restart
        kfill!(xr, zero(FC))  # xr === Δx when restart is set to true
        if npass ≥ 1
          mul!(w, A, x)
          kaxpby!(n, one(FC), b, -one(FC), w)
          MisI || mulorldiv!(r₀, M, w, ldiv)
        end
      end

      # Initial ζ₁ and v₁
      β = knorm(n, r₀)
      z[1] = β
      kdivcopy!(n, V[1], r₀, rNorm)  # v₁ = r₀ / ‖r₀‖

      npass = npass + 1
      workspace.inner_iter = 0
      inner_tired = false

      while !(solved || inner_tired || breakdown || user_requested_exit || overtimed)

        # Update iteration index
        workspace.inner_iter = workspace.inner_iter + 1
        inner_iter = workspace.inner_iter

        # Update workspace if more storage is required and restart is set to false
        if !restart && (inner_iter > mem)
          for i = 1 : inner_iter
            push!(R, zero(FC))
          end
          push!(s, zero(FC))
          push!(c, zero(T))
        end

        # Continue the Arnoldi process.
        p = NisI ? V[inner_iter] : workspace.p
        NisI || mulorldiv!(p, N, V[inner_iter], ldiv)  # p ← Nvₖ
        mul!(w, A, p)                                  # w ← ANvₖ
        MisI || mulorldiv!(q, M, w, ldiv)              # q ← MANvₖ
        for i = 1 : inner_iter
          R[nr+i] = kdot(n, V[i], q)    # hᵢₖ = (vᵢ)ᴴq
          kaxpy!(n, -R[nr+i], V[i], q)  # q ← q - hᵢₖvᵢ
        end

        # Reorthogonalization of the Krylov basis.
        if reorthogonalization
          for i = 1 : inner_iter
            Htmp = kdot(n, V[i], q)
            R[nr+i] += Htmp
            kaxpy!(n, -Htmp, V[i], q)
          end
        end

        # Compute hₖ₊₁.ₖ
        Hbis = knorm(n, q)  # hₖ₊₁.ₖ = ‖vₖ₊₁‖₂

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
        # ‖ M(b - Axₖ) ‖₂ = |ζₖ₊₁|
        rNorm = abs(ζₖ₊₁)
        history && push!(rNorms, rNorm)

        # Update the number of coefficients in Rₖ
        nr = nr + inner_iter

        # Stopping conditions that do not depend on user input.
        # This is to guard against tolerances that are unreasonably small.
        resid_decrease_mach = (rNorm + one(T) ≤ one(T))
        
        # Update stopping criterion.
        user_requested_exit = callback(workspace) :: Bool
        resid_decrease_lim = rNorm ≤ ε
        breakdown = Hbis ≤ btol
        solved = resid_decrease_lim || resid_decrease_mach
        inner_tired = restart ? inner_iter ≥ min(mem, inner_itmax) : inner_iter ≥ inner_itmax
        timer = time_ns() - start_time
        overtimed = timer > timemax_ns
        kdisplay(iter+inner_iter, verbose) && @printf(iostream, "%5d  %5d  %7.1e  %7.1e  %.2fs\n", npass, iter+inner_iter, rNorm, Hbis, start_time |> ktimer)

        # Compute vₖ₊₁.
        if !(solved || inner_tired || breakdown || user_requested_exit || overtimed)
          if !restart && (inner_iter ≥ mem)
            push!(V, similar(workspace.x))
            push!(z, zero(FC))
          end
          kdivcopy!(n, V[inner_iter+1], q, Hbis)  # vₖ₊₁ = q / hₖ₊₁.ₖ
          z[inner_iter+1] = ζₖ₊₁
        end
      end

      # Compute yₖ by solving Rₖyₖ = zₖ with backward substitution.
      y = z  # yᵢ = zᵢ
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

      # Form xₖ = NVₖyₖ
      for i = 1 : inner_iter
        kaxpy!(n, y[i], V[i], xr)
      end
      if !NisI
        kcopy!(n, workspace.p, xr)  # p ← xr
        mulorldiv!(xr, N, workspace.p, ldiv)
      end
      restart && kaxpy!(n, one(FC), xr, x)

      # Update inner_itmax, iter, tired and overtimed variables.
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
    warm_start && !restart && kaxpy!(n, one(FC), Δx, x)
    workspace.warm_start = false

    # Update stats
    stats.niter = iter
    stats.solved = solved
    stats.inconsistent = inconsistent
    stats.timer = start_time |> ktimer
    stats.status = status
    return workspace
  end
end
