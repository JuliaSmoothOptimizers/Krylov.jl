# An implementation of LSQR for the solution of the
# over-determined linear least-squares problem
#
#  minimize ‖Ax - b‖₂
#
# equivalently, of the normal equations
#
#  AᴴAx = Aᴴb.
#
# LSQR is formally equivalent to applying the conjugate gradient method
# to the normal equations but should be more stable. It is also formally
# equivalent to CGLS though LSQR should be expected to be more stable on
# ill-conditioned or poorly scaled problems.
#
# This implementation follows the original implementation by
# Michael Saunders described in
#
# C. C. Paige and M. A. Saunders, LSQR: An Algorithm for Sparse Linear
# Equations and Sparse Least Squares, ACM Transactions on Mathematical
# Software, 8(1), pp. 43--71, 1982.
#
# Dominique Orban, <dominique.orban@gerad.ca>
# Montreal, QC, May 2015.

export lsqr, lsqr!

"""
    (x, stats) = lsqr(A, b::AbstractVector{FC};
                      M=I, N=I, ldiv::Bool=false,
                      window::Int=5, sqd::Bool=false, λ::T=zero(T),
                      radius::T=zero(T), etol::T=√eps(T),
                      axtol::T=√eps(T), btol::T=√eps(T),
                      conlim::T=1/√eps(T), atol::T=zero(T),
                      rtol::T=zero(T), itmax::Int=0,
                      timemax::Float64=Inf, verbose::Int=0, history::Bool=false,
                      callback=solver->false, iostream::IO=kstdout)

`T` is an `AbstractFloat` such as `Float32`, `Float64` or `BigFloat`.
`FC` is `T` or `Complex{T}`.

Solve the regularized linear least-squares problem

    minimize ‖b - Ax‖₂² + λ²‖x‖₂²

of size m × n using the LSQR method, where λ ≥ 0 is a regularization parameter.
LSQR is formally equivalent to applying CG to the normal equations

    (AᴴA + λ²I) x = Aᴴb

(and therefore to CGLS) but is more stable.

LSQR produces monotonic residuals ‖r‖₂ but not optimality residuals ‖Aᴴr‖₂.
It is formally equivalent to CGLS, though can be slightly more accurate.

If `λ > 0`, LSQR solves the symmetric and quasi-definite system

    [ E      A ] [ r ]   [ b ]
    [ Aᴴ  -λ²F ] [ x ] = [ 0 ],

where E and F are symmetric and positive definite.
Preconditioners M = E⁻¹ ≻ 0 and N = F⁻¹ ≻ 0 may be provided in the form of linear operators.
If `sqd=true`, `λ` is set to the common value `1`.

The system above represents the optimality conditions of

    minimize ‖b - Ax‖²_E⁻¹ + λ²‖x‖²_F.

For a symmetric and positive definite matrix `K`, the K-norm of a vector `x` is `‖x‖²_K = xᴴKx`.
LSQR is then equivalent to applying CG to `(AᴴE⁻¹A + λ²F)x = AᴴE⁻¹b` with `r = E⁻¹(b - Ax)`.

If `λ = 0`, we solve the symmetric and indefinite system

    [ E    A ] [ r ]   [ b ]
    [ Aᴴ   0 ] [ x ] = [ 0 ].

The system above represents the optimality conditions of

    minimize ‖b - Ax‖²_E⁻¹.

In this case, `N` can still be specified and indicates the weighted norm in which `x` and `Aᴴr` should be measured.
`r` can be recovered by computing `E⁻¹(b - Ax)`.

#### Input arguments

* `A`: a linear operator that models a matrix of dimension `m × n`;
* `b`: a vector of length `m`.

#### Keyword arguments

* `M`: linear operator that models a Hermitian positive-definite matrix of size `m` used for centered preconditioning of the augmented system;
* `N`: linear operator that models a Hermitian positive-definite matrix of size `n` used for centered preconditioning of the augmented system;
* `ldiv`: define whether the preconditioners use `ldiv!` or `mul!`;
* `window`: number of iterations used to accumulate a lower bound on the error;
* `sqd`: if `true`, set `λ=1` for Hermitian quasi-definite systems;
* `λ`: regularization parameter;
* `radius`: add the trust-region constraint ‖x‖ ≤ `radius` if `radius > 0`. Useful to compute a step in a trust-region method for optimization;
* `etol`: stopping tolerance based on the lower bound on the error;
* `axtol`: tolerance on the backward error;
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
* `stats`: statistics collected on the run in a [`SimpleStats`](@ref) structure.

#### Reference

* C. C. Paige and M. A. Saunders, [*LSQR: An Algorithm for Sparse Linear Equations and Sparse Least Squares*](https://doi.org/10.1145/355984.355989), ACM Transactions on Mathematical Software, 8(1), pp. 43--71, 1982.
"""
function lsqr end

"""
    solver = lsqr!(solver::LsqrSolver, A, b; kwargs...)

where `kwargs` are keyword arguments of [`lsqr`](@ref).

See [`LsqrSolver`](@ref) for more details about the `solver`.
"""
function lsqr! end

def_args_lsqr = (:(A                    ),
                 :(b::AbstractVector{FC}))

def_kwargs_lsqr = (:(; M = I                     ),
                   :(; N = I                     ),
                   :(; ldiv::Bool = false        ),
                   :(; sqd::Bool = false         ),
                   :(; λ::T = zero(T)            ),
                   :(; radius::T = zero(T)       ),
                   :(; etol::T = √eps(T)         ),
                   :(; axtol::T = √eps(T)        ),
                   :(; btol::T = √eps(T)         ),
                   :(; conlim::T = 1/√eps(T)     ),
                   :(; atol::T = zero(T)         ),
                   :(; rtol::T = zero(T)         ),
                   :(; itmax::Int = 0            ),
                   :(; timemax::Float64 = Inf    ),
                   :(; verbose::Int = 0          ),
                   :(; history::Bool = false     ),
                   :(; callback = solver -> false),
                   :(; iostream::IO = kstdout    ))

def_kwargs_lsqr = extract_parameters.(def_kwargs_lsqr)

args_lsqr = (:A, :b)
kwargs_lsqr = (:M, :N, :ldiv, :sqd, :λ, :radius, :etol, :axtol, :btol, :conlim, :atol, :rtol, :itmax, :timemax, :verbose, :history, :callback, :iostream)

@eval begin
  function lsqr!(solver :: LsqrSolver{T,FC,S}, $(def_args_lsqr...); $(def_kwargs_lsqr...)) where {T <: AbstractFloat, FC <: FloatOrComplex{T}, S <: AbstractVector{FC}}

    # Timer
    start_time = time_ns()
    timemax_ns = 1e9 * timemax

    m, n = size(A)
    (m == solver.m && n == solver.n) || error("(solver.m, solver.n) = ($(solver.m), $(solver.n)) is inconsistent with size(A) = ($m, $n)")
    length(b) == m || error("Inconsistent problem size")
    (verbose > 0) && @printf(iostream, "LSQR: system of %d equations in %d variables\n", m, n)

    # Check sqd and λ parameters
    sqd && (λ ≠ 0) && error("sqd cannot be set to true if λ ≠ 0 !")
    sqd && (λ = one(T))

    # Tests M = Iₙ and N = Iₘ
    MisI = (M === I)
    NisI = (N === I)

    # Check type consistency
    eltype(A) == FC || @warn "eltype(A) ≠ $FC. This could lead to errors or additional allocations in operator-vector products."
    ktypeof(b) <: S || error("ktypeof(b) is not a subtype of $S")

    # Compute the adjoint of A
    Aᴴ = A'

    # Set up workspace.
    allocate_if(!MisI, solver, :u, S, solver.Av)  # The length of u is m
    allocate_if(!NisI, solver, :v, S, solver.x)   # The length of v is n
    x, Nv, Aᴴu, w = solver.x, solver.Nv, solver.Aᴴu, solver.w
    Mu, Av, err_vec, stats = solver.Mu, solver.Av, solver.err_vec, solver.stats
    rNorms, ArNorms = stats.residuals, stats.Aresiduals
    reset!(stats)
    u = MisI ? Mu : solver.u
    v = NisI ? Nv : solver.v

    λ² = λ * λ
    ctol = conlim > 0 ? 1/conlim : zero(T)
    kfill!(x, zero(FC))

    # Initialize Golub-Kahan process.
    # β₁ M u₁ = b.
    kcopy!(m, Mu, b)  # Mu ← b
    MisI || mulorldiv!(u, M, Mu, ldiv)
    β₁ = knorm_elliptic(m, u, Mu)
    if β₁ == 0
      stats.niter = 0
      stats.solved, stats.inconsistent = true, false
      stats.timer = start_time |> ktimer
      stats.status = "x is a zero-residual solution"
      history && push!(rNorms, zero(T))
      history && push!(ArNorms, zero(T))
      return solver
    end
    β = β₁

    kscal!(m, one(FC)/β₁, u)
    MisI || kscal!(m, one(FC)/β₁, Mu)
    mul!(Aᴴu, Aᴴ, u)
    kcopy!(n, Nv, Aᴴu)  # Nv ← Aᴴu
    NisI || mulorldiv!(v, N, Nv, ldiv)
    Anorm² = kdotr(n, v, Nv)
    Anorm = sqrt(Anorm²)
    α = Anorm
    Acond  = zero(T)
    xNorm  = zero(T)
    xNorm² = zero(T)
    dNorm² = zero(T)
    c2 = -one(T)
    s2 = zero(T)
    z  = zero(T)

    xENorm² = zero(T)
    err_lbnd = zero(T)
    window = length(err_vec)
    kfill!(err_vec, zero(T))

    iter = 0
    itmax == 0 && (itmax = m + n)

    (verbose > 0) && @printf(iostream, "%5s  %7s  %7s  %7s  %7s  %7s  %7s  %7s  %7s  %5s\n", "k", "α", "β", "‖r‖", "‖Aᴴr‖", "compat", "backwrd", "‖A‖", "κ(A)", "timer")
    kdisplay(iter, verbose) && @printf(iostream, "%5d  %7.1e  %7.1e  %7.1e  %7.1e  %7.1e  %7.1e  %7.1e  %7.1e  %.2fs\n", iter, β₁, α, β₁, α, 0, 1, Anorm, Acond, start_time |> ktimer)

    rNorm = β₁
    r1Norm = rNorm
    r2Norm = rNorm
    res2   = zero(T)
    history && push!(rNorms, r2Norm)
    ArNorm = ArNorm0 = α * β
    history && push!(ArNorms, ArNorm)
    # Aᴴb = 0 so x = 0 is a minimum least-squares solution
    if α == 0
      stats.niter = 0
      stats.solved, stats.inconsistent = true, false
      stats.timer = start_time |> ktimer
      stats.status = "x is a minimum least-squares solution"
      return solver
    end
    kscal!(n, one(FC)/α, v)
    NisI || kscal!(n, one(FC)/α, Nv)
    kcopy!(n, w, v)  # w ← v

    # Initialize other constants.
    ϕbar = β₁
    ρbar = α

    status = "unknown"
    on_boundary = false
    solved_lim = ArNorm / (Anorm * rNorm) ≤ axtol
    solved_mach = one(T) + ArNorm / (Anorm * rNorm) ≤ one(T)
    solved = solved_mach | solved_lim
    tired  = iter ≥ itmax
    ill_cond = ill_cond_mach = ill_cond_lim = false
    zero_resid_lim = rNorm / β₁ ≤ axtol
    zero_resid_mach = one(T) + rNorm / β₁ ≤ one(T)
    zero_resid = zero_resid_mach | zero_resid_lim
    fwd_err = false
    user_requested_exit = false
    overtimed = false

    while ! (solved || tired || ill_cond || user_requested_exit || overtimed)
      iter = iter + 1

      # Generate next Golub-Kahan vectors.
      # 1. βₖ₊₁Muₖ₊₁ = Avₖ - αₖMuₖ
      mul!(Av, A, v)
      kaxpby!(m, one(FC), Av, -α, Mu)
      MisI || mulorldiv!(u, M, Mu, ldiv)
      β = knorm_elliptic(m, u, Mu)
      if β ≠ 0
        kscal!(m, one(FC)/β, u)
        MisI || kscal!(m, one(FC)/β, Mu)
        Anorm² = Anorm² + α * α + β * β  # = ‖B_{k-1}‖²
        λ > 0 && (Anorm² += λ²)

        # 2. αₖ₊₁Nvₖ₊₁ = Aᴴuₖ₊₁ - βₖ₊₁Nvₖ
        mul!(Aᴴu, Aᴴ, u)
        kaxpby!(n, one(FC), Aᴴu, -β, Nv)
        NisI || mulorldiv!(v, N, Nv, ldiv)
        α = knorm_elliptic(n, v, Nv)
        if α ≠ 0
          kscal!(n, one(FC)/α, v)
          NisI || kscal!(n, one(FC)/α, Nv)
        end
      end

      # Continue QR factorization
      # 1. Eliminate the regularization parameter.
      (c1, s1, ρbar1) = sym_givens(ρbar, λ)
      ψ = s1 * ϕbar
      ϕbar = c1 * ϕbar

      # 2. Eliminate β.
      # Q [ Lₖ  β₁ e₁ ] = [ Rₖ   zₖ  ] :
      #   [ β    0    ]   [ 0   ζbar ]
      #
      #       k  k+1    k    k+1      k  k+1
      # k   [ c   s ] [ ρbar    ] = [ ρ  θ⁺    ]
      # k+1 [ s  -c ] [ β    α⁺ ]   [    ρbar⁺ ]
      #
      # so that we obtain
      #
      # [ c  s ] [ ζbar ] = [ ζ     ]
      # [ s -c ] [  0   ]   [ ζbar⁺ ]
      (c, s, ρ) = sym_givens(ρbar1, β)
      ϕ = c * ϕbar
      ϕbar = s * ϕbar

      xENorm² = xENorm² + ϕ * ϕ
      err_vec[mod(iter, window) + 1] = ϕ
      iter ≥ window && (err_lbnd = knorm(window, err_vec))

      τ = s * ϕ
      θ = s * α
      ρbar = -c * α
      dNorm² += kdotr(n, w, w) / ρ^2

      # if a trust-region constraint is give, compute step to the boundary
      # the step ϕ/ρ is not necessarily positive
      σ = ϕ / ρ
      if radius > 0
        t1, t2 = to_boundary(n, x, w, v, radius)
        tmax, tmin = max(t1, t2), min(t1, t2)
        on_boundary = σ > tmax || σ < tmin
        σ = σ > 0 ? min(σ, tmax) : max(σ, tmin)
      end

      kaxpy!(n, σ, w, x)  # x = x + ϕ / ρ * w
      kaxpby!(n, one(FC), v, -θ/ρ, w)  # w = v - θ / ρ * w

      # Use a plane rotation on the right to eliminate the super-diagonal
      # element (θ) of the upper-bidiagonal matrix.
      # Use the result to estimate norm(x).
      δ = s2 * ρ
      γbar = -c2 * ρ
      rhs = ϕ - δ * z
      zbar = rhs / γbar
      xNorm = sqrt(xNorm² + zbar * zbar)
      (c2, s2, γ) = sym_givens(γbar, θ)
      z = rhs / γ
      xNorm² += z * z

      Anorm = sqrt(Anorm²)
      Acond = Anorm * sqrt(dNorm²)
      res1  = ϕbar * ϕbar
      res2 += ψ * ψ
      rNorm = sqrt(res1 + res2)

      ArNorm = α * abs(τ)
      history && push!(ArNorms, ArNorm)

      r1sq = rNorm * rNorm - λ² * xNorm²
      r1Norm = sqrt(abs(r1sq))
      r1sq < 0 && (r1Norm = -r1Norm)
      r2Norm = rNorm
      history && push!(rNorms, r2Norm)

      test1 = rNorm / β₁
      test2 = ArNorm / (Anorm * rNorm)
      test3 = 1 / Acond
      t1    = test1 / (one(T) + Anorm * xNorm / β₁)
      rNormtol = btol + axtol * Anorm * xNorm / β₁

      kdisplay(iter, verbose) && @printf(iostream, "%5d  %7.1e  %7.1e  %7.1e  %7.1e  %7.1e  %7.1e  %7.1e  %7.1e  %.2fs\n", iter, α, β, rNorm, ArNorm, test1, test2, Anorm, Acond, start_time |> ktimer)

      # Stopping conditions that do not depend on user input.
      # This is to guard against tolerances that are unreasonably small.
      ill_cond_mach = (one(T) + test3 ≤ one(T))
      solved_mach = (one(T) + test2 ≤ one(T))
      zero_resid_mach = (one(T) + t1 ≤ one(T))

      # Stopping conditions based on user-provided tolerances.
      user_requested_exit = callback(solver) :: Bool
      tired  = iter ≥ itmax
      ill_cond_lim = (test3 ≤ ctol)
      solved_lim = (test2 ≤ axtol)
      solved_opt = ArNorm ≤ atol + rtol * ArNorm0
      zero_resid_lim = (test1 ≤ rNormtol)
      iter ≥ window && (fwd_err = err_lbnd ≤ etol * sqrt(xENorm²))

      ill_cond = ill_cond_mach || ill_cond_lim
      zero_resid = zero_resid_mach || zero_resid_lim
      solved = solved_mach || solved_lim || solved_opt || zero_resid || fwd_err || on_boundary
      timer = time_ns() - start_time
      overtimed = timer > timemax_ns
    end
    (verbose > 0) && @printf(iostream, "\n")

    # Termination status
    tired               && (status = "maximum number of iterations exceeded")
    ill_cond_mach       && (status = "condition number seems too large for this machine")
    ill_cond_lim        && (status = "condition number exceeds tolerance")
    solved              && (status = "found approximate minimum least-squares solution")
    zero_resid          && (status = "found approximate zero-residual solution")
    fwd_err             && (status = "truncated forward error small enough")
    on_boundary         && (status = "on trust-region boundary")
    user_requested_exit && (status = "user-requested exit")
    overtimed           && (status = "time limit exceeded")

    # Update stats
    stats.niter = iter
    stats.solved = solved
    stats.inconsistent = !zero_resid
    stats.timer = start_time |> ktimer
    stats.status = status
    return solver
  end
end
