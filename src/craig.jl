# An implementation of the Golub-Kahan version of Craig's method
# for the solution of the consistent (under/over-determined or square)
# linear system
#
#  Ax = b.
#
# The method seeks to solve the minimum-norm problem
#
#  min ‖x‖  s.t.  Ax = b,
#
# and is equivalent to applying the conjugate gradient method
# to the linear system
#
#  AAᴴy = b.
#
# This method, sometimes known under the name CRAIG, is the
# Golub-Kahan implementation of CGNE, and is described in
#
# J. E. Craig, The N-step iteration procedures,
# Journal of Mathematics and Physics, 34(1-4), pp. 64--73, 1955.
#
# C. C. Paige and M. A. Saunders, LSQR: An Algorithm for Sparse
# Linear Equations and Sparse Least Squares, ACM Transactions on
# Mathematical Software, 8(1), pp. 43--71, 1982.
#
# M. A. Saunders, Solutions of Sparse Rectangular Systems Using LSQR and CRAIG,
# BIT Numerical Mathematics, 35(4), pp. 588--604, 1995.
#
# Dominique Orban, <dominique.orban@gerad.ca>
# Montréal, QC, April 2015.
#
# This implementation is strongly inspired from Mike Saunders's.

export craig, craig!

"""
    (x, y, stats) = craig(A, b::AbstractVector{FC};
                          M=I, N=I, ldiv::Bool=false,
                          transfer_to_lsqr::Bool=false, sqd::Bool=false,
                          λ::T=zero(T), btol::T=√eps(T),
                          conlim::T=1/√eps(T), atol::T=√eps(T),
                          rtol::T=√eps(T), itmax::Int=0,
                          timemax::Float64=Inf, verbose::Int=0, history::Bool=false,
                          callback=solver->false, iostream::IO=kstdout)

`T` is an `AbstractFloat` such as `Float32`, `Float64` or `BigFloat`.
`FC` is `T` or `Complex{T}`.

Find the least-norm solution of the consistent linear system

    Ax + λ²y = b

of size m × n using the Golub-Kahan implementation of Craig's method, where λ ≥ 0 is a
regularization parameter. This method is equivalent to CGNE but is more
stable.

For a system in the form Ax = b, Craig's method is equivalent to applying
CG to AAᴴy = b and recovering x = Aᴴy. Note that y are the Lagrange
multipliers of the least-norm problem

    minimize ‖x‖  s.t.  Ax = b.

If `λ > 0`, CRAIG solves the symmetric and quasi-definite system

    [ -F     Aᴴ ] [ x ]   [ 0 ]
    [  A   λ²E  ] [ y ] = [ b ],

where E and F are symmetric and positive definite.
Preconditioners M = E⁻¹ ≻ 0 and N = F⁻¹ ≻ 0 may be provided in the form of linear operators.
If `sqd=true`, `λ` is set to the common value `1`.

The system above represents the optimality conditions of

    min ‖x‖²_F + λ²‖y‖²_E  s.t.  Ax + λ²Ey = b.

For a symmetric and positive definite matrix `K`, the K-norm of a vector `x` is `‖x‖²_K = xᴴKx`.
CRAIG is then equivalent to applying CG to `(AF⁻¹Aᴴ + λ²E)y = b` with `Fx = Aᴴy`.

If `λ = 0`, CRAIG solves the symmetric and indefinite system

    [ -F   Aᴴ ] [ x ]   [ 0 ]
    [  A   0  ] [ y ] = [ b ].

The system above represents the optimality conditions of

    minimize ‖x‖²_F  s.t.  Ax = b.

In this case, `M` can still be specified and indicates the weighted norm in which residuals are measured.

In this implementation, both the x and y-parts of the solution are returned.

#### Input arguments

* `A`: a linear operator that models a matrix of dimension `m × n`;
* `b`: a vector of length `m`.

#### Keyword arguments

* `M`: linear operator that models a Hermitian positive-definite matrix of size `m` used for centered preconditioning of the augmented system;
* `N`: linear operator that models a Hermitian positive-definite matrix of size `n` used for centered preconditioning of the augmented system;
* `ldiv`: define whether the preconditioners use `ldiv!` or `mul!`;
* `transfer_to_lsqr`: transfer from the LSLQ point to the LSQR point, when it exists. The transfer is based on the residual norm;
* `sqd`: if `true`, set `λ=1` for Hermitian quasi-definite systems;
* `λ`: regularization parameter;
* `btol`: stopping tolerance used to detect zero-residual problems;
* `conlim`: limit on the estimated condition number of `A` beyond which the solution will be abandoned;
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
* `y`: a dense vector of length `m`;
* `stats`: statistics collected on the run in a [`SimpleStats`](@ref) structure.

#### References

* J. E. Craig, [*The N-step iteration procedures*](https://doi.org/10.1002/sapm195534164), Journal of Mathematics and Physics, 34(1-4), pp. 64--73, 1955.
* C. C. Paige and M. A. Saunders, [*LSQR: An Algorithm for Sparse Linear Equations and Sparse Least Squares*](https://doi.org/10.1145/355984.355989), ACM Transactions on Mathematical Software, 8(1), pp. 43--71, 1982.
* M. A. Saunders, [*Solutions of Sparse Rectangular Systems Using LSQR and CRAIG*](https://doi.org/10.1007/BF01739829), BIT Numerical Mathematics, 35(4), pp. 588--604, 1995.
"""
function craig end

"""
    solver = craig!(solver::CraigSolver, A, b; kwargs...)

where `kwargs` are keyword arguments of [`craig`](@ref).

See [`CraigSolver`](@ref) for more details about the `solver`.
"""
function craig! end

def_args_craig = (:(A                    ),
                  :(b::AbstractVector{FC}))

def_kwargs_craig = (:(; M = I                         ),
                    :(; N = I                         ),
                    :(; ldiv::Bool = false            ),
                    :(; transfer_to_lsqr::Bool = false),
                    :(; sqd::Bool = false             ),
                    :(; λ::T = zero(T)                ),
                    :(; btol::T = √eps(T)             ),
                    :(; conlim::T = 1/√eps(T)         ),
                    :(; atol::T = √eps(T)             ),
                    :(; rtol::T = √eps(T)             ),
                    :(; itmax::Int = 0                ),
                    :(; timemax::Float64 = Inf        ),
                    :(; verbose::Int = 0              ),
                    :(; history::Bool = false         ),
                    :(; callback = solver -> false    ),
                    :(; iostream::IO = kstdout        ))

def_kwargs_craig = extract_parameters.(def_kwargs_craig)

args_craig = (:A, :b)
kwargs_craig = (:M, :N, :ldiv, :transfer_to_lsqr, :sqd, :λ, :btol, :conlim, :atol, :rtol, :itmax, :timemax, :verbose, :history, :callback, :iostream)

@eval begin
  function craig!(solver :: CraigSolver{T,FC,S}, $(def_args_craig...); $(def_kwargs_craig...)) where {T <: AbstractFloat, FC <: FloatOrComplex{T}, S <: AbstractVector{FC}}

    # Timer
    start_time = time_ns()
    timemax_ns = 1e9 * timemax

    m, n = size(A)
    (m == solver.m && n == solver.n) || error("(solver.m, solver.n) = ($(solver.m), $(solver.n)) is inconsistent with size(A) = ($m, $n)")
    length(b) == m || error("Inconsistent problem size")
    (verbose > 0) && @printf(iostream, "CRAIG: system of %d equations in %d variables\n", m, n)

    # Check sqd and λ parameters
    sqd && (λ ≠ 0) && error("sqd cannot be set to true if λ ≠ 0 !")
    sqd && (λ = one(T))

    # Tests M = Iₘ and N = Iₙ
    MisI = (M === I)
    NisI = (N === I)

    # Check type consistency
    eltype(A) == FC || @warn "eltype(A) ≠ $FC. This could lead to errors or additional allocations in operator-vector products."
    ktypeof(b) <: S || error("ktypeof(b) is not a subtype of $S")

    # Compute the adjoint of A
    Aᴴ = A'

    # Set up workspace.
    allocate_if(!MisI, solver, :u , S, solver.y)  # The length of u is m
    allocate_if(!NisI, solver, :v , S, solver.x)  # The length of v is n
    allocate_if(λ > 0, solver, :w2, S, solver.x)  # The length of w2 is n
    x, Nv, Aᴴu, y, w = solver.x, solver.Nv, solver.Aᴴu, solver.y, solver.w
    Mu, Av, w2, stats = solver.Mu, solver.Av, solver.w2, solver.stats
    rNorms = stats.residuals
    reset!(stats)
    u = MisI ? Mu : solver.u
    v = NisI ? Nv : solver.v

    kfill!(x, zero(FC))
    kfill!(y, zero(FC))

    kcopy!(m, Mu, b)  # Mu ← b
    MisI || mulorldiv!(u, M, Mu, ldiv)
    β₁ = knorm_elliptic(m, u, Mu)
    rNorm  = β₁
    history && push!(rNorms, rNorm)
    if β₁ == 0
      stats.niter = 0
      stats.solved, stats.inconsistent = true, false
      stats.timer = start_time |> ktimer
      stats.status = "x is a zero-residual solution"
      return solver
    end
    β₁² = β₁^2
    β = β₁
    θ = β₁      # θ will differ from β when there is regularization (λ > 0).
    ξ = -one(T) # Most recent component of x in Range(V).
    δ = λ
    ρ_prev = one(T)

    # Initialize Golub-Kahan process.
    # β₁Mu₁ = b.
    kscal!(m, one(FC) / β₁, u)
    MisI || kscal!(m, one(FC) / β₁, Mu)

    kfill!(Nv, zero(FC))
    kfill!(w, zero(FC))  # Used to update y.

    λ > 0 && kfill!(w2, zero(FC))

    Anorm² = zero(T) # Estimate of ‖A‖²_F.
    Anorm  = zero(T)
    Dnorm² = zero(T) # Estimate of ‖(AᴴA)⁻¹‖².
    Acond  = zero(T) # Estimate of cond(A).
    xNorm² = zero(T) # Estimate of ‖x‖².
    xNorm  = zero(T)

    iter = 0
    itmax == 0 && (itmax = m + n)

    ɛ_c = atol + rtol * rNorm   # Stopping tolerance for consistent systems.
    ɛ_i = atol                  # Stopping tolerance for inconsistent systems.
    ctol = conlim > 0 ? 1/conlim : zero(T)  # Stopping tolerance for ill-conditioned operators.
    (verbose > 0) && @printf(iostream, "%5s  %8s  %8s  %8s  %8s  %8s  %7s  %5s\n", "k", "‖r‖", "‖x‖", "‖A‖", "κ(A)", "α", "β", "timer")
    kdisplay(iter, verbose) && @printf(iostream, "%5d  %8.2e  %8.2e  %8.2e  %8.2e  %8s  %7s  %.2fs\n", iter, rNorm, xNorm, Anorm, Acond, " ✗ ✗ ✗ ✗", "✗ ✗ ✗ ✗", start_time |> ktimer)

    bkwerr = one(T)  # initial value of the backward error ‖r‖ / √(‖b‖² + ‖A‖² ‖x‖²)

    status = "unknown"

    solved_lim = bkwerr ≤ btol
    solved_mach = one(T) + bkwerr ≤ one(T)
    solved_resid_tol = rNorm ≤ ɛ_c
    solved_resid_lim = rNorm ≤ btol + atol * Anorm * xNorm / β₁
    solved = solved_mach | solved_lim | solved_resid_tol | solved_resid_lim

    ill_cond = ill_cond_mach = ill_cond_lim = false

    inconsistent = false
    tired = iter ≥ itmax
    user_requested_exit = false
    overtimed = false

    while ! (solved || inconsistent || ill_cond || tired || user_requested_exit || overtimed)
      # Generate the next Golub-Kahan vectors
      # 1. αₖ₊₁Nvₖ₊₁ = Aᴴuₖ₊₁ - βₖ₊₁Nvₖ
      mul!(Aᴴu, Aᴴ, u)
      kaxpby!(n, one(FC), Aᴴu, -β, Nv)
      NisI || mulorldiv!(v, N, Nv, ldiv)
      α = knorm_elliptic(n, v, Nv)
      if α == 0
        inconsistent = true
        continue
      end
      kscal!(n, one(FC) / α, v)
      NisI || kscal!(n, one(FC) / α, Nv)

      Anorm² += α * α + λ * λ

      if λ > 0
        # Givens rotation to zero out the δ in position (k, 2k):
        #      k-1  k   2k     k   2k      k-1  k   2k
        # k   [ θ   α   δ ] [ c₁   s₁ ] = [ θ   ρ      ]
        # k+1 [     β     ] [ s₁  -c₁ ]   [     θ+   γ ]
        (c₁, s₁, ρ) = sym_givens(α, δ)
      else
        ρ = α
      end

      ξ = -θ / ρ * ξ

      if λ > 0
        # w1 = c₁ * v + s₁ * w2
        # w2 = s₁ * v - c₁ * w2
        # x  = x + ξ * w1
        kaxpy!(n, ξ * c₁, v, x)
        kaxpy!(n, ξ * s₁, w2, x)
        kaxpby!(n, s₁, v, -c₁, w2)
      else
        kaxpy!(n, ξ, v, x)  # x = x + ξ * v
      end

      # Recur y.
      kaxpby!(m, one(FC), u, -θ/ρ_prev, w)  # w = u - θ/ρ_prev * w
      kaxpy!(m, ξ/ρ, w, y)                  # y = y + ξ/ρ * w

      Dnorm² += knorm(m, w)

      # 2. βₖ₊₁Muₖ₊₁ = Avₖ - αₖMuₖ
      mul!(Av, A, v)
      kaxpby!(m, one(FC), Av, -α, Mu)
      MisI || mulorldiv!(u, M, Mu, ldiv)
      β = knorm_elliptic(m, u, Mu)
      if β ≠ 0
        kscal!(m, one(FC) / β, u)
        MisI || kscal!(m, one(FC) / β, Mu)
      end

      # Finish  updates from the first Givens rotation.
      if λ > 0
        θ =  β * c₁
        γ =  β * s₁
      else
        θ = β
      end

      if λ > 0
        # Givens rotation to zero out the γ in position (k+1, 2k)
        #       2k  2k+1     2k  2k+1      2k  2k+1
        # k+1 [  γ    λ ] [ -c₂   s₂ ] = [  0    δ ]
        # k+2 [  0    0 ] [  s₂   c₂ ]   [  0    0 ]
        c₂, s₂, δ = sym_givens(λ, γ)
        kscal!(n, s₂, w2)
      end

      Anorm² += β * β
      Anorm = sqrt(Anorm²)
      Acond = Anorm * sqrt(Dnorm²)
      xNorm² += ξ * ξ
      xNorm = sqrt(xNorm²)
      rNorm = β * abs(ξ)           # r = - β * ξ * u
      λ > 0 && (rNorm *= abs(c₁))  # r = -c₁ * β * ξ * u when λ > 0.
      history && push!(rNorms, rNorm)
      iter = iter + 1

      bkwerr = rNorm / sqrt(β₁² + Anorm² * xNorm²)

      ρ_prev = ρ   # Only differs from α if λ > 0.

      kdisplay(iter, verbose) && @printf(iostream, "%5d  %8.2e  %8.2e  %8.2e  %8.2e  %8.1e  %7.1e  %.2fs\n", iter, rNorm, xNorm, Anorm, Acond, α, β, start_time |> ktimer)

      solved_lim = bkwerr ≤ btol
      solved_mach = one(T) + bkwerr ≤ one(T)
      solved_resid_tol = rNorm ≤ ɛ_c
      solved_resid_lim = rNorm ≤ btol + atol * Anorm * xNorm / β₁
      solved = solved_mach | solved_lim | solved_resid_tol | solved_resid_lim

      ill_cond_mach = one(T) + one(T) / Acond ≤ one(T)
      ill_cond_lim = 1 / Acond ≤ ctol
      ill_cond = ill_cond_mach | ill_cond_lim

      user_requested_exit = callback(solver) :: Bool
      inconsistent = false
      tired = iter ≥ itmax
      timer = time_ns() - start_time
      overtimed = timer > timemax_ns
    end
    (verbose > 0) && @printf(iostream, "\n")

    # transfer to LSQR point if requested
    if λ > 0 && transfer_to_lsqr
      ξ *= -θ / δ
      kaxpy!(n, ξ, w2, x)
      # TODO: update y
    end

    # Termination status
    tired               && (status = "maximum number of iterations exceeded")
    solved              && (status = "solution good enough for the tolerances given")
    ill_cond_mach       && (status = "condition number seems too large for this machine")
    ill_cond_lim        && (status = "condition number exceeds tolerance")
    inconsistent        && (status = "system may be inconsistent")
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
