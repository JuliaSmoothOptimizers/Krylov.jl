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
                      verbose::Int=0, history::Bool=false,
                      callback=solver->false, iostream::IO=kstdout)

`T` is a `Real` such as `Float32`, `Float64` or `BigFloat`.
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

* `A`: a linear operator that models a matrix of dimension m × n;
* `b`: a vector of length m.

#### Keyword arguments

* `N`:
* `ldiv`: define whether the preconditioner uses `ldiv!` or `mul!`;
* `λ`: regularization parameter;
* `atol`: absolute stopping tolerance based on the residual norm;
* `rtol`: relative stopping tolerance based on the residual norm;
* `itmax`: the maximum number of iterations. If `itmax=0`, the default number of iterations is set to `m+n`;
* `verbose`: additional details can be displayed if verbose mode is enabled (verbose > 0). Information will be displayed every `verbose` iterations;
* `history`: collect additional statistics on the run such as residual norms, or Aᴴ-residual norms;
* `callback`: function or functor called as `callback(solver)` that returns `true` if the Krylov method should terminate, and `false` otherwise;
* `iostream`: stream to which output is logged.

#### Output arguments

* `x`: a dense vector of length n;
* `stats`: statistics collected on the run in a [`SimpleStats`](@ref) structure.

#### References

* D. Orban and M. Arioli, [*Iterative Solution of Symmetric Quasi-Definite Linear Systems*](https://doi.org/10.1137/1.9781611974737), Volume 3 of Spotlights. SIAM, Philadelphia, PA, 2017.
* D. Orban, [*The Projected Golub-Kahan Process for Constrained Linear Least-Squares Problems*](https://dx.doi.org/10.13140/RG.2.2.17443.99360). Cahier du GERAD G-2014-15, 2014.
"""
function crmr end

function crmr(A, b :: AbstractVector{FC}; kwargs...) where FC <: RealOrComplex
  solver = CrmrSolver(A, b)
  crmr!(solver, A, b; kwargs...)
  return (solver.x, solver.stats)
end

"""
    solver = crmr!(solver::CrmrSolver, A, b; kwargs...)

where `kwargs` are keyword arguments of [`crmr`](@ref).

See [`CrmrSolver`](@ref) for more details about the `solver`.
"""
function crmr! end

function crmr!(solver :: CrmrSolver{T,FC,S}, A, b :: AbstractVector{FC};
               N=I, ldiv :: Bool=false,
               λ :: T=zero(T), atol :: T=√eps(T),
               rtol :: T=√eps(T), itmax :: Int=0,
               verbose :: Int=0, history :: Bool=false,
               callback = solver -> false, iostream :: IO=kstdout) where {T <: AbstractFloat, FC <: FloatOrComplex{T}, S <: DenseVector{FC}}

  m, n = size(A)
  length(b) == m || error("Inconsistent problem size")
  (verbose > 0) && @printf(iostream, "CRMR: system of %d equations in %d variables\n", m, n)

  # Tests N = Iₙ
  NisI = (N === I)

  # Check type consistency
  eltype(A) == FC || error("eltype(A) ≠ $FC")
  ktypeof(b) <: S || error("ktypeof(b) is not a subtype of $S")

  # Compute the adjoint of A
  Aᴴ = A'

  # Set up workspace.
  allocate_if(!NisI, solver, :Nq, S, m)
  allocate_if(λ > 0, solver, :s , S, m)
  x, p, Aᴴr, r = solver.x, solver.p, solver.Aᴴr, solver.r
  q, s, stats = solver.q, solver.s, solver.stats
  rNorms, ArNorms = stats.residuals, stats.Aresiduals
  reset!(stats)
  Nq = NisI ? q : solver.Nq

  x .= zero(FC)              # initial estimation x = 0
  mulorldiv!(r, N, b, ldiv)  # initial residual r = N * (b - Ax) = N * b
  bNorm = @knrm2(m, r)       # norm(b - A * x0) if x0 ≠ 0.
  rNorm = bNorm              # + λ * ‖x0‖ if x0 ≠ 0 and λ > 0.
  history && push!(rNorms, rNorm)
  if bNorm == 0
    stats.niter = 0
    stats.solved, stats.inconsistent = true, false
    stats.status = "x = 0 is a zero-residual solution"
    history && push!(ArNorms, zero(T))
    return solver
  end
  λ > 0 && (s .= r)
  mul!(Aᴴr, Aᴴ, r)  # - λ * x0 if x0 ≠ 0.
  p .= Aᴴr
  γ = @kdotr(n, Aᴴr, Aᴴr)  # Faster than γ = dot(Aᴴr, Aᴴr)
  λ > 0 && (γ += λ * rNorm * rNorm)
  iter = 0
  itmax == 0 && (itmax = m + n)

  ArNorm = sqrt(γ)
  history && push!(ArNorms, ArNorm)
  ɛ_c = atol + rtol * rNorm  # Stopping tolerance for consistent systems.
  ɛ_i = atol + rtol * ArNorm  # Stopping tolerance for inconsistent systems.
  (verbose > 0) && @printf(iostream, "%5s  %8s  %8s\n", "k", "‖Aᴴr‖", "‖r‖")
  kdisplay(iter, verbose) && @printf(iostream, "%5d  %8.2e  %8.2e\n", iter, ArNorm, rNorm)

  status = "unknown"
  solved = rNorm ≤ ɛ_c
  inconsistent = (rNorm > 100 * ɛ_c) && (ArNorm ≤ ɛ_i)
  tired = iter ≥ itmax
  user_requested_exit = false

  while ! (solved || inconsistent || tired || user_requested_exit)
    mul!(q, A, p)
    λ > 0 && @kaxpy!(m, λ, s, q)  # q = q + λ * s
    NisI || mulorldiv!(Nq, N, q, ldiv)
    α = γ / @kdotr(m, q, Nq)   # Compute qᴴ * N * q
    @kaxpy!(n,  α, p, x)       # Faster than  x =  x + α *  p
    @kaxpy!(m, -α, Nq, r)      # Faster than  r =  r - α * Nq
    rNorm = @knrm2(m, r)       # norm(r)
    mul!(Aᴴr, Aᴴ, r)
    γ_next = @kdotr(n, Aᴴr, Aᴴr)  # Faster than γ_next = dot(Aᴴr, Aᴴr)
    λ > 0 && (γ_next += λ * rNorm * rNorm)
    β = γ_next / γ

    @kaxpby!(n, one(FC), Aᴴr, β, p)  # Faster than  p = Aᴴr + β * p
    if λ > 0
      @kaxpby!(m, one(FC), r, β, s) # s = r + β * s
    end

    γ = γ_next
    ArNorm = sqrt(γ)
    history && push!(rNorms, rNorm)
    history && push!(ArNorms, ArNorm)
    iter = iter + 1
    kdisplay(iter, verbose) && @printf(iostream, "%5d  %8.2e  %8.2e\n", iter, ArNorm, rNorm)
    user_requested_exit = callback(solver) :: Bool
    solved = rNorm ≤ ɛ_c
    inconsistent = (rNorm > 100 * ɛ_c) && (ArNorm ≤ ɛ_i)
    tired = iter ≥ itmax
  end
  (verbose > 0) && @printf(iostream, "\n")

  tired               && (status = "maximum number of iterations exceeded")
  solved              && (status = "solution good enough given atol and rtol")
  inconsistent        && (status = "system probably inconsistent but least squares/norm solution found")
  user_requested_exit && (status = "user-requested exit")

  # Update stats
  stats.niter = iter
  stats.solved = solved
  stats.inconsistent = inconsistent
  stats.status = status
  return solver
end
