# An implementation of CGNE for the solution of the consistent
# (under/over-determined or square) linear system
#
#  Ax = b.
#
# The method seeks to solve the minimum-norm problem
#
#  min ‖x‖²  s.t. Ax = b,
#
# and is equivalent to applying the conjugate gradient method
# to the linear system
#
#  AAᴴy = b.
#
# This method is also known as Craig's method, CGME, and other
# names, and is described in
#
# J. E. Craig. The N-step iteration procedures.
# Journal of Mathematics and Physics, 34(1), pp. 64--73, 1955.
#
# which is based on Craig's thesis from MIT:
#
# J. E. Craig. Iterations Procedures for Simultaneous Equations.
# Ph.D. Thesis, Department of Electrical Engineering, MIT, 1954.
#
# Dominique Orban, <dominique.orban@gerad.ca>
# Montréal, QC, April 2015.

export cgne, cgne!

"""
    (x, stats) = cgne(A, b::AbstractVector{FC};
                      N=I, ldiv::Bool=false,
                      λ::T=zero(T), atol::T=√eps(T),
                      rtol::T=√eps(T), itmax::Int=0,
                      timemax::Float64=Inf, verbose::Int=0, history::Bool=false,
                      callback=solver->false, iostream::IO=kstdout)

`T` is an `AbstractFloat` such as `Float32`, `Float64` or `BigFloat`.
`FC` is `T` or `Complex{T}`.

Solve the consistent linear system

    Ax + √λs = b

of size m × n using the Conjugate Gradient (CG) method, where λ ≥ 0 is a regularization
parameter. This method is equivalent to applying CG to the normal equations
of the second kind

    (AAᴴ + λI) y = b

but is more stable. When λ = 0, this method solves the minimum-norm problem

  min ‖x‖₂  s.t. Ax = b.

When λ > 0, it solves the problem

    min ‖(x,s)‖₂  s.t. Ax + √λs = b.

CGNE produces monotonic errors ‖x-x*‖₂ but not residuals ‖r‖₂.
It is formally equivalent to CRAIG, though can be slightly less accurate,
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

* J. E. Craig, [*The N-step iteration procedures*](https://doi.org/10.1002/sapm195534164), Journal of Mathematics and Physics, 34(1), pp. 64--73, 1955.
* J. E. Craig, *Iterations Procedures for Simultaneous Equations*, Ph.D. Thesis, Department of Electrical Engineering, MIT, 1954.
"""
function cgne end

"""
    solver = cgne!(solver::CgneSolver, A, b; kwargs...)

where `kwargs` are keyword arguments of [`cgne`](@ref).

See [`CgneSolver`](@ref) for more details about the `solver`.
"""
function cgne! end

def_args_cgne = (:(A                    ),
                 :(b::AbstractVector{FC}))

def_kwargs_cgne = (:(; N = I                     ),
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

def_kwargs_cgne = extract_parameters.(def_kwargs_cgne)

args_cgne = (:A, :b)
kwargs_cgne = (:N, :ldiv, :λ, :atol, :rtol, :itmax, :timemax, :verbose, :history, :callback, :iostream)

@eval begin
  function cgne!(solver :: CgneSolver{T,FC,S}, $(def_args_cgne...); $(def_kwargs_cgne...)) where {T <: AbstractFloat, FC <: FloatOrComplex{T}, S <: AbstractVector{FC}}

    # Timer
    start_time = time_ns()
    timemax_ns = 1e9 * timemax

    m, n = size(A)
    (m == solver.m && n == solver.n) || error("(solver.m, solver.n) = ($(solver.m), $(solver.n)) is inconsistent with size(A) = ($m, $n)")
    length(b) == m || error("Inconsistent problem size")
    (verbose > 0) && @printf(iostream, "CGNE: system of %d equations in %d variables\n", m, n)

    # Tests N = Iₙ
    NisI = (N === I)

    # Check type consistency
    eltype(A) == FC || @warn "eltype(A) ≠ $FC. This could lead to errors or additional allocations in operator-vector products."
    ktypeof(b) <: S || error("ktypeof(b) is not a subtype of $S")

    # Compute the adjoint of A
    Aᴴ = A'

    # Set up workspace.
    allocate_if(!NisI, solver, :z, S, solver.r)  # The length of z is m
    allocate_if(λ > 0, solver, :s, S, solver.r)  # The length of s is m
    x, p, Aᴴz, r, q, s, stats = solver.x, solver.p, solver.Aᴴz, solver.r, solver.q, solver.s, solver.stats
    rNorms = stats.residuals
    reset!(stats)
    z = NisI ? r : solver.z

    kfill!(x, zero(FC))
    kcopy!(m, r, b)  # r ← b
    NisI || mulorldiv!(z, N, r, ldiv)
    rNorm = knorm(m, r)  # Marginally faster than norm(r)
    history && push!(rNorms, rNorm)
    if rNorm == 0
      stats.niter = 0
      stats.solved, stats.inconsistent = true, false
      stats.timer = start_time |> ktimer
      stats.status = "x is a zero-residual solution"
      return solver
    end
    λ > 0 && kcopy!(m, s, r)  # s ← r
    mul!(p, Aᴴ, z)

    # Use ‖p‖ to detect inconsistent system.
    # An inconsistent system will necessarily have AAᴴ singular.
    # Because CGNE is equivalent to CG applied to AAᴴy = b, there will be a
    # conjugate direction u such that uᴴAAᴴu = 0, i.e., Aᴴu = 0. In this
    # implementation, p is a substitute for Aᴴu.
    pNorm = knorm(n, p)

    γ = kdotr(m, r, z)  # Faster than γ = dot(r, z)
    iter = 0
    itmax == 0 && (itmax = m + n)

    ɛ_c = atol + rtol * rNorm  # Stopping tolerance for consistent systems.
    ɛ_i = atol + rtol * pNorm  # Stopping tolerance for inconsistent systems.
    (verbose > 0) && @printf(iostream, "%5s  %8s  %5s\n", "k", "‖r‖", "timer")
    kdisplay(iter, verbose) && @printf(iostream, "%5d  %8.2e  %.2fs\n", iter, rNorm, start_time |> ktimer)

    status = "unknown"
    solved = rNorm ≤ ɛ_c
    inconsistent = (rNorm > 100 * ɛ_c) && (pNorm ≤ ɛ_i)
    tired = iter ≥ itmax
    user_requested_exit = false
    overtimed = false

    while ! (solved || inconsistent || tired || user_requested_exit || overtimed)
      mul!(q, A, p)
      λ > 0 && kaxpy!(m, λ, s, q)
      δ = kdotr(n, p, p)   # Faster than dot(p, p)
      λ > 0 && (δ += λ * kdotr(m, s, s))
      α = γ / δ
      kaxpy!(n,  α, p, x)  # Faster than x = x + α * p
      kaxpy!(m, -α, q, r)  # Faster than r = r - α * q
      NisI || mulorldiv!(z, N, r, ldiv)
      γ_next = kdotr(m, r, z)  # Faster than γ_next = dot(r, z)
      β = γ_next / γ
      mul!(Aᴴz, Aᴴ, z)
      kaxpby!(n, one(FC), Aᴴz, β, p)  # Faster than p = Aᴴz + β * p
      pNorm = knorm(n, p)
      if λ > 0
        kaxpby!(m, one(FC), r, β, s)  # s = r + β * s
      end
      γ = γ_next
      rNorm = sqrt(γ_next)
      history && push!(rNorms, rNorm)
      iter = iter + 1
      kdisplay(iter, verbose) && @printf(iostream, "%5d  %8.2e  %.2fs\n", iter, rNorm, start_time |> ktimer)

      # Stopping conditions that do not depend on user input.
      # This is to guard against tolerances that are unreasonably small.
      resid_decrease_mach = (rNorm + one(T) ≤ one(T))

      user_requested_exit = callback(solver) :: Bool
      resid_decrease_lim = rNorm ≤ ɛ_c
      solved = resid_decrease_lim || resid_decrease_mach
      inconsistent = (rNorm > 100 * ɛ_c) && (pNorm ≤ ɛ_i)
      tired = iter ≥ itmax
      timer = time_ns() - start_time
      overtimed = timer > timemax_ns
    end
    (verbose > 0) && @printf(iostream, "\n")

    # Termination status
    tired               && (status = "maximum number of iterations exceeded")
    inconsistent        && (status = "system probably inconsistent")
    solved              && (status = "solution good enough given atol and rtol")
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
