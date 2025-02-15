# An implementation of CRMR for the solution of the
# (under/over-determined or square) linear system
#
#  Ax = b.
#
# The method seeks to solve the minimum-norm problem
#
#  min ‖x‖²  s.t. Ax = b,
#
# and is equivalent to applying the conjugate residual method
# to the linear system
#
#  AAᴴy = b.
#
# This method is equivalent to CRAIGMR, described in
#
# D. Orban and M. Arioli. Iterative Solution of Symmetric Quasi-Definite Linear Systems,
# Volume 3 of Spotlights. SIAM, Philadelphia, PA, 2017.
#
# D. Orban, The Projected Golub-Kahan Process for Constrained
# Linear Least-Squares Problems. Cahier du GERAD G-2014-15,
# GERAD, Montreal QC, Canada, 2014.
#
# Dominique Orban, <dominique.orban@gerad.ca>
# Montreal, QC, April 2015.

export crmr, crmr!

"""
    (x, stats) = crmr(A, b::AbstractVector{FC};
                      N=I, ldiv::Bool=false,
                      λ::T=zero(T), atol::T=√eps(T),
                      rtol::T=√eps(T), itmax::Int=0,
                      timemax::Float64=Inf, verbose::Int=0, history::Bool=false,
                      callback=solver->false, iostream::IO=kstdout)

`T` is an `AbstractFloat` such as `Float32`, `Float64` or `BigFloat`.
`FC` is `T` or `Complex{T}`.

Solve the consistent linear system

    Ax + √λs = b

of size m × n using the Conjugate Residual (CR) method, where λ ≥ 0 is a regularization
parameter. This method is equivalent to applying CR to the normal equations
of the second kind

    (AAᴴ + λI) y = b

but is more stable. When λ = 0, this method solves the minimum-norm problem

    min ‖x‖₂  s.t.  x ∈ argmin ‖Ax - b‖₂.

When λ > 0, this method solves the problem

    min ‖(x,s)‖₂  s.t. Ax + √λs = b.

CRMR produces monotonic residuals ‖r‖₂.
It is formally equivalent to CRAIG-MR, though can be slightly less accurate,
but simpler to implement. Only the x-part of the solution is returned.

#### Input arguments

* `A`: a linear operator that models a matrix of dimension `m × n`;
* `b`: a vector of length `m`.

#### Keyword arguments

* `N`: linear operator that models a Hermitian positive-definite matrix of size `n` used for preconditioning;
* `ldiv`: define whether the preconditioner uses `ldiv!` or `mul!`;
* `λ`: regularization parameter;
* `atol`: absolute stopping tolerance based on the residual norm;
* `rtol`: relative stopping tolerance based on the residual norm;
* `itmax`: the maximum number of iterations. If `itmax=0`, the default number of iterations is set to `m+n`;
* `timemax`: the time limit in seconds;
* `verbose`: additional details can be displayed if verbose mode is enabled (verbose > 0). Information will be displayed every `verbose` iterations;
* `history`: collect additional statistics on the run such as residual norms, or Aᴴ-residual norms;
* `callback`: function or functor called as `callback(solver)` that returns `true` if the Krylov method should terminate, and `false` otherwise;
* `iostream`: stream to which output is logged.

#### Output arguments

* `x`: a dense vector of length `n`;
* `stats`: statistics collected on the run in a [`SimpleStats`](@ref) structure.

#### References

* D. Orban and M. Arioli, [*Iterative Solution of Symmetric Quasi-Definite Linear Systems*](https://doi.org/10.1137/1.9781611974737), Volume 3 of Spotlights. SIAM, Philadelphia, PA, 2017.
* D. Orban, [*The Projected Golub-Kahan Process for Constrained Linear Least-Squares Problems*](https://dx.doi.org/10.13140/RG.2.2.17443.99360). Cahier du GERAD G-2014-15, 2014.
"""
function crmr end

"""
    solver = crmr!(solver::CrmrSolver, A, b; kwargs...)

where `kwargs` are keyword arguments of [`crmr`](@ref).

See [`CrmrSolver`](@ref) for more details about the `solver`.
"""
function crmr! end

def_args_crmr = (:(A                    ),
                 :(b::AbstractVector{FC}))

def_kwargs_crmr = (:(; N = I                     ),
                   :(; ldiv::Bool = false        ),
                   :(; λ::T = zero(T)            ),
                   :(; atol::T = √eps(T)         ),
                   :(; rtol::T = √eps(T)         ),
                   :(; itmax::Int = 0            ),
                   :(; timemax::Float64 = Inf    ),
                   :(; verbose::Int = 0          ),
                   :(; history::Bool = false     ),
                   :(; callback = solver -> false),
                   :(; iostream::IO = kstdout    ))

def_kwargs_crmr = extract_parameters.(def_kwargs_crmr)

args_crmr = (:A, :b)
kwargs_crmr = (:N, :ldiv, :λ, :atol, :rtol, :itmax, :timemax, :verbose, :history, :callback, :iostream)

@eval begin
  function crmr!(solver :: CrmrSolver{T,FC,S}, $(def_args_crmr...); $(def_kwargs_crmr...)) where {T <: AbstractFloat, FC <: FloatOrComplex{T}, S <: AbstractVector{FC}}

    # Timer
    start_time = time_ns()
    timemax_ns = 1e9 * timemax

    m, n = size(A)
    (m == solver.m && n == solver.n) || error("(solver.m, solver.n) = ($(solver.m), $(solver.n)) is inconsistent with size(A) = ($m, $n)")
    length(b) == m || error("Inconsistent problem size")
    (verbose > 0) && @printf(iostream, "CRMR: system of %d equations in %d variables\n", m, n)

    # Tests N = Iₙ
    NisI = (N === I)

    # Check type consistency
    eltype(A) == FC || @warn "eltype(A) ≠ $FC. This could lead to errors or additional allocations in operator-vector products."
    ktypeof(b) <: S || error("ktypeof(b) is not a subtype of $S")

    # Compute the adjoint of A
    Aᴴ = A'

    # Set up workspace.
    allocate_if(!NisI, solver, :Nq, S, solver.r)  # The length of Nq is m
    allocate_if(λ > 0, solver, :s , S, solver.r)  # The length of s is m
    x, p, Aᴴr, r = solver.x, solver.p, solver.Aᴴr, solver.r
    q, s, stats = solver.q, solver.s, solver.stats
    rNorms, ArNorms = stats.residuals, stats.Aresiduals
    reset!(stats)
    Nq = NisI ? q : solver.Nq

    kfill!(x, zero(FC))        # initial estimation x = 0
    mulorldiv!(r, N, b, ldiv)  # initial residual r = N * (b - Ax) = N * b
    bNorm = knorm(m, r)        # norm(b - A * x0) if x0 ≠ 0.
    rNorm = bNorm              # + λ * ‖x0‖ if x0 ≠ 0 and λ > 0.
    history && push!(rNorms, rNorm)
    if bNorm == 0
      stats.niter = 0
      stats.solved, stats.inconsistent = true, false
      stats.timer = start_time |> ktimer
      stats.status = "x is a zero-residual solution"
      history && push!(ArNorms, zero(T))
      return solver
    end
    λ > 0 && kcopy!(m, s, r)  # s ← r
    mul!(Aᴴr, Aᴴ, r)          # - λ * x0 if x0 ≠ 0.
    kcopy!(n, p, Aᴴr)         # p ← Aᴴr
    γ = kdotr(n, Aᴴr, Aᴴr)    # Faster than γ = dot(Aᴴr, Aᴴr)
    λ > 0 && (γ += λ * rNorm * rNorm)
    iter = 0
    itmax == 0 && (itmax = m + n)

    ArNorm = sqrt(γ)
    history && push!(ArNorms, ArNorm)
    ɛ_c = atol + rtol * rNorm   # Stopping tolerance for consistent systems.
    ɛ_i = atol + rtol * ArNorm  # Stopping tolerance for inconsistent systems.
    (verbose > 0) && @printf(iostream, "%5s  %8s  %8s  %5s\n", "k", "‖Aᴴr‖", "‖r‖", "timer")
    kdisplay(iter, verbose) && @printf(iostream, "%5d  %8.2e  %8.2e  %.2fs\n", iter, ArNorm, rNorm, start_time |> ktimer)

    status = "unknown"
    solved = rNorm ≤ ɛ_c
    inconsistent = (rNorm > 100 * ɛ_c) && (ArNorm ≤ ɛ_i)
    tired = iter ≥ itmax
    user_requested_exit = false
    overtimed = false

    while ! (solved || inconsistent || tired || user_requested_exit || overtimed)
      mul!(q, A, p)
      λ > 0 && kaxpy!(m, λ, s, q)  # q = q + λ * s
      NisI || mulorldiv!(Nq, N, q, ldiv)
      α = γ / kdotr(m, q, Nq)  # Compute qᴴ * N * q
      kaxpy!(n,  α, p, x)      # Faster than  x =  x + α *  p
      kaxpy!(m, -α, Nq, r)     # Faster than  r =  r - α * Nq
      rNorm = knorm(m, r)      # norm(r)
      mul!(Aᴴr, Aᴴ, r)
      γ_next = kdotr(n, Aᴴr, Aᴴr)  # Faster than γ_next = dot(Aᴴr, Aᴴr)
      λ > 0 && (γ_next += λ * rNorm * rNorm)
      β = γ_next / γ

      kaxpby!(n, one(FC), Aᴴr, β, p)  # Faster than  p = Aᴴr + β * p
      if λ > 0
        kaxpby!(m, one(FC), r, β, s)  # s = r + β * s
      end

      γ = γ_next
      ArNorm = sqrt(γ)
      history && push!(rNorms, rNorm)
      history && push!(ArNorms, ArNorm)
      iter = iter + 1
      kdisplay(iter, verbose) && @printf(iostream, "%5d  %8.2e  %8.2e  %.2fs\n", iter, ArNorm, rNorm, start_time |> ktimer)
      user_requested_exit = callback(solver) :: Bool
      solved = rNorm ≤ ɛ_c
      inconsistent = (rNorm > 100 * ɛ_c) && (ArNorm ≤ ɛ_i)
      tired = iter ≥ itmax
      timer = time_ns() - start_time
      overtimed = timer > timemax_ns
    end
    (verbose > 0) && @printf(iostream, "\n")

    # Termination status
    tired               && (status = "maximum number of iterations exceeded")
    solved              && (status = "solution good enough given atol and rtol")
    inconsistent        && (status = "system probably inconsistent but least squares/norm solution found")
    user_requested_exit && (status = "user-requested exit")
    overtimed           && (status = "time limit exceeded")

    # Update stats
    stats.niter = iter
    stats.solved = solved
    stats.inconsistent = inconsistent
    stats.timer = start_time |> ktimer
    stats.status = status
    return solver
  end
end
