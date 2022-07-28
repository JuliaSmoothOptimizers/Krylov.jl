# An implementation of LSQR for the solution of the
# over-determined linear least-squares problem
#
#  minimize ‖Ax - b‖₂
#
# equivalently, of the normal equations
#
#  AᵀAx = Aᵀb.
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

export lsqr_shift, lsqr_shift!


"""
    (x, stats) = lsqr(A, b::AbstractVector{T};
                      M=I, N=I, axtol::T=√eps(T), btol::T=√eps(T),
                      atol::T=zero(T), rtol::T=zero(T),
                      etol::T=√eps(T), window::Int=5,
                      itmax::Int=0, conlim::T=1/√eps(T),
                      radius::T=zero(T), verbose::Int=0, history::Bool=false) where T <: AbstractFloat

Solve the regularized linear least-squares problem

    minimize ‖b - Ax‖₂² + λ²‖x‖₂²

using the LSQR method, where λ ≥ 0 is a regularization parameter.
LSQR is formally equivalent to applying CG to the normal equations

    (AᵀA + λ²I) x = Aᵀb

(and therefore to CGLS) but is more stable.

LSQR produces monotonic residuals ‖r‖₂ but not optimality residuals ‖Aᵀr‖₂.
It is formally equivalent to CGLS, though can be slightly more accurate.

Preconditioners M and N may be provided in the form of linear operators and are
assumed to be symmetric and positive definite. If `sqd` is set to `true`,
we solve the symmetric and quasi-definite system

    [ E    A ] [ r ]   [ b ]
    [ Aᵀ  -F ] [ x ] = [ 0 ],

where E and F are symmetric and positive definite.
The system above represents the optimality conditions of

    minimize ‖b - Ax‖²_E⁻¹ + ‖x‖²_F.

For a symmetric and positive definite matrix `K`, the K-norm of a vector `x` is `‖x‖²_K = xᵀKx`.
LSQR is then equivalent to applying CG to `(AᵀE⁻¹A + F)x = AᵀE⁻¹b` with `r = E⁻¹(b - Ax)`.
Preconditioners M = E⁻¹ ≻ 0 and N = F⁻¹ ≻ 0 may be provided in the form of linear operators.

If `sqd` is set to `false` (the default), we solve the symmetric and
indefinite system

    [ E    A ] [ r ]   [ b ]
    [ Aᵀ   0 ] [ x ] = [ 0 ].

The system above represents the optimality conditions of

    minimize ‖b - Ax‖²_E⁻¹.

In this case, `N` can still be specified and indicates the weighted norm in which `x` and `Aᵀr` should be measured.
`r` can be recovered by computing `E⁻¹(b - Ax)`.

#### Reference

* C. C. Paige and M. A. Saunders, *LSQR: An Algorithm for Sparse Linear Equations and Sparse Least Squares*, ACM Transactions on Mathematical Software, 8(1), pp. 43--71, 1982.
"""
function lsqr_shift(A, b :: AbstractVector{FC}, shifts; window :: Int=5, kwargs...) where {T <: AbstractFloat, FC <: FloatOrComplex{T}}
  solver = LsqrShiftSolver(A, b, length(shifts), window=window)
  lsqr_shift!(solver, A, b, shifts; kwargs...)
  return (solver.x, solver.stats)
end

function lsqr_shift!(solver :: LsqrShiftSolver{T,FC,S}, A, b :: AbstractVector{FC}, shifts;
               M=I, N=I, axtol :: T=√eps(T), btol :: T=√eps(T),
               atol :: T=zero(T), rtol :: T=zero(T),
               etol :: T=√eps(T), itmax :: Int=0, conlim :: T=1/√eps(T),
               radius :: T=zero(T), verbose :: Int=0, history :: Bool=false,
               callback = solver -> false) where {T <: AbstractFloat, FC <: FloatOrComplex{T}, S <: DenseVector{FC}}

  m, n = size(A)
  length(b) == m || error("Inconsistent problem size")
  (verbose > 0) && @printf("LSQR: system of %d equations in %d variables\n", m, n)

  # Tests M == Iₙ and N == Iₘ
  MisI = (M == I)
  NisI = (N == I)

  # Check type consistency
  eltype(A) == FC || error("eltype(A) ≠ $FC")
  ktypeof(b) == S || error("ktypeof(b) ≠ $S")

  # Compute the adjoint of A
  Aᵀ = A'

  # Set up workspace.
  allocate_if(!MisI, solver, :u, S, m)
  allocate_if(!NisI, solver, :v, S, n)
  x, Nv, Aᵀu, w = solver.x, solver.Nv, solver.Aᵀu, solver.w
  Mu, Av, err_vec, stats = solver.Mu, solver.Av, solver.err_vec, solver.stats
  rNorms, ArNorms, Acond = stats.residuals, stats.Aresiduals, stats.Acond
  converged, zero_resid = solver.converged, stats.inconsistent
  Anorm, Anorm² = solver.Anorm, solver.Anorm²
  rNorm, xNorm², dNorm² = solver.rNorm, solver.xNorm², solver.dNorm²
  z, xENorm² = solver.z, solver.xENorm²
  ϕbar, ρbar  = solver.ϕbar, solver.ρbar
  fwd_err, on_boundary = solver.fwd_err, solver.on_boundary
  tired, ill_cond_mach, ill_cond_lim  = solver.tired, solver.ill_cond_mach, solver.ill_cond_lim
  reset!(stats)
  u = MisI ? Mu : solver.u
  v = NisI ? Nv : solver.v

  nshifts = length(shifts)

  ctol = conlim > 0 ? 1/conlim : zero(T)
  for i = 1 : nshifts
    x[i] .= zero(FC)
  end

  # Initialize Golub-Kahan process.
  # β₁ M u₁ = b.
  Mu .= b
  MisI || mul!(u, M, Mu)
  β₁ = sqrt(@kdotr(m, u, Mu))
  if β₁ == 0
    stats.niter = 0
    stats.solved = true
    stats.inconsistent .= false
    stats.status .= "x = 0 is a zero-residual solution"
    for i = 1 : nshifts
      history && push!(rNorms[i], zero(T))
      history && push!(ArNorms[i], zero(T))
    end
    return solver
  end
  β = β₁

  @kscal!(m, one(FC)/β₁, u)
  MisI || @kscal!(m, one(FC)/β₁, Mu)
  mul!(Aᵀu, Aᵀ, u)
  Nv .= Aᵀu
  NisI || mul!(v, N, Nv)
  Anorm² .= @kdotr(n, v, Nv)
  Anorm .= sqrt(Anorm²[1])
  α = Anorm[1]
  Acond .= zero(T)
  rNorm .= zero(T)
  xNorm  = zero(T)
  xNorm² .= zero(T)
  dNorm² .= zero(T)
  c2 = -one(T)
  s2 = zero(T)
  z .= zero(T)

  xENorm² .= zero(T)
  err_lbnd = zero(T)
  window = length(err_vec[1])
  for i = 1 : nshifts
    err_vec[i] .= zero(T)
  end

  iter = 0
  itmax == 0 && (itmax = m + n)

  rNorm .= β₁
  r1Norm = β₁
  r2Norm = β₁
  res2   = zero(T) # Est-ce que c'est pas bizarre ça?
  ArNorm = ArNorm0 = α * β
  for i = 1 : nshifts
    history && push!(rNorms[i], r2Norm)
    history && push!(ArNorms[i], ArNorm)
  end

  #(verbose > 0) && @printf("%5s  %7s  %7s  %7s  %7s  %7s  %7s  %7s  %7s\n", "Aprod", "α", "β", "‖r‖", "‖Aᵀr‖", "compat", "backwrd", "‖A‖", "κ(A)")
  #(verbose > 0) && @printf("%5d  %7.1e  %7.1e  %7.1e  %7.1e  %7.1e  %7.1e  %7.1e  %7.1e\n", 1, β₁, α, β₁, α, 0, 1, Anorm[end], Acond[end])
  # Build format strings for printing.
  if kdisplay(iter, verbose)
    fmt = "%5d" * repeat("  %8.1e", nshifts) * "\n"
    # precompile printf for our particular format
    local_printf(data...) = Core.eval(Main, :(@printf($fmt, $(data)...)))
    local_printf(iter, rNorm...)
  end

  # Aᵀb = 0 so x = 0 is a minimum least-squares solution
  if α == 0
    stats.niter = 0
    stats.solved = true
    stats.inconsistent .= false
    stats.status .= "x = 0 is a minimum least-squares solution"
    return solver
  end
  @kscal!(n, one(FC)/α, v)
  NisI || @kscal!(n, one(FC)/α, Nv)
  for i = 1 : nshifts
    w[i] .= v
  end

  # Initialize other constants.
  ϕbar .= β₁
  ρbar .= α
  # θ = 0.0

  test1 = zero(T)
  t1 = zero(T)
  test2 = zero(T)
  test3 = zero(T)
  rNormtol = zero(T)
  fwd_err .= false
  user_requested_exit = false

  stats.status .= "unknown"
  on_boundary .= false
  solved_lim = (ArNorm / (Anorm[end] * rNorm[end]) ≤ axtol)
  solved_mach = (one(T) + ArNorm / (Anorm[end] * rNorm[end]) ≤ one(T))
  converged .= (solved_mach | solved_lim)
  tired .= (iter ≥ itmax)
  ill_cond_mach .= false
  ill_cond_lim .= false
  zero_resid_lim = (rNorm[end] / β₁ ≤ axtol)
  zero_resid_mach = (one(T) + rNorm[end] / β₁ ≤ one(T))
  # PAS MIS A JOUR ENSUITE. BUG DE LSQR OU MOI?
  zero_resid .= (zero_resid_mach | zero_resid_lim)

  while ! all(converged .| tired .| ill_cond_mach .| ill_cond_lim .| user_requested_exit) # ALLOCATES
    iter = iter + 1

    # Generate next Golub-Kahan vectors.
    # 1. βₖ₊₁Muₖ₊₁ = Avₖ - αₖMuₖ
    mul!(Av, A, v)
    @kaxpby!(m, one(FC), Av, -α, Mu)
    MisI || mul!(u, M, Mu)
    β = sqrt(@kdotr(m, u, Mu))
    if β ≠ 0
      @kscal!(m, one(FC)/β, u)
      MisI || @kscal!(m, one(FC)/β, Mu)
      Anorm² .+= α * α + β * β .+ shifts .* shifts  # = ‖B_{k-1}‖²
      Anorm .= sqrt.(Anorm²)

      # 2. αₖ₊₁Nvₖ₊₁ = Aᵀuₖ₊₁ - βₖ₊₁Nvₖ
      mul!(Aᵀu, Aᵀ, u)
      @kaxpby!(n, one(FC), Aᵀu, -β, Nv)
      NisI || mul!(v, N, Nv)
      α = sqrt(@kdotr(n, v, Nv))
      if α ≠ 0
        @kscal!(n, one(FC)/α, v)
        NisI || @kscal!(n, one(FC)/α, Nv)
      end
    end

    for i in findall(.!(converged .| tired .| ill_cond_mach .| ill_cond_lim)) # ALLOCATES
      λ = shifts[i]

      # Continue QR factorization
      # 1. Eliminate the regularization parameter.
      (c1, s1, ρbar1) = sym_givens(ρbar[i], sqrt(λ))
      ψ = s1 * ϕbar[i]
      ϕbar[i] = c1 * ϕbar[i]

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
      ϕ = c * ϕbar[i]
      ϕbar[i] = s * ϕbar[i]

      xENorm²[i] += ϕ * ϕ
      err_vec[i][mod(iter, window) + 1] = ϕ
      iter ≥ window && (err_lbnd = norm(err_vec[i]))

      τ = s * ϕ
      θ = s * α
      ρbar[i] = -c * α
      dNorm²[i] += @kdotr(n, w[i], w[i]) / ρ^2

      # if a trust-region constraint is give, compute step to the boundary
      # the step ϕ/ρ is not necessarily positive
      σ = ϕ / ρ
      if radius > 0
        t1, t2 = to_boundary(x[i], w[i], radius)
        tmax, tmin = max(t1, t2), min(t1, t2)
        on_boundary[i] = σ > tmax || σ < tmin
        σ = σ > 0 ? min(σ, tmax) : max(σ, tmin)
      end

      @kaxpy!(n, σ, w[i], x[i])  # x = x + ϕ / ρ * w
      @kaxpby!(n, one(FC), v, -θ/ρ, w[i])  # w = v - θ / ρ * w

      # Use a plane rotation on the right to eliminate the super-diagonal
      # element (θ) of the upper-bidiagonal matrix.
      # Use the result to estimate norm(x).
      δ = s2 * ρ
      γbar = -c2 * ρ
      rhs = ϕ - δ * z[i]
      zbar = rhs / γbar
      xNorm = sqrt(xNorm²[i] + zbar * zbar)
      (c2, s2, γ) = sym_givens(γbar, θ)
      z[i] = rhs / γ
      xNorm²[i] += z[i] * z[i]

      Acond[i] = Anorm[i] * sqrt(dNorm²[i])
      res1  = ϕbar[i] * ϕbar[i]
      res2 += ψ * ψ
      rNorm[i] = sqrt(res1 + res2)

      ArNorm = α * abs(τ)
      history && push!(ArNorms[i], ArNorm)

      r1sq = rNorm[i] * rNorm[i] - λ^2 * xNorm²[i]
      r1Norm = sqrt(abs(r1sq))
      r1sq < 0 && (r1Norm = -r1Norm)
      r2Norm = rNorm[i]
      history && push!(rNorms[i], r2Norm)

      test1 = rNorm[i] / β₁
      test2 = ArNorm / (Anorm[i] * rNorm[i])
      test3 = 1 / Acond[i]
      t1    = test1 / (one(T) + Anorm[i] * xNorm / β₁)
      rNormtol = btol + axtol * Anorm[i] * xNorm / β₁

      kdisplay(iter, verbose) && local_printf(iter, rNorm...)

      iter ≥ window && (fwd_err[i] = err_lbnd ≤ etol * sqrt(xENorm²[i]))

      # Stopping conditions that do not depend on user input.
      # This is to guard against tolerances that are unreasonably small.
      ill_cond_mach[i] = one(T) + test3 ≤ one(T)
      solved_mach = one(T) + test2 ≤ one(T)
      zero_resid_mach = one(T) + t1 ≤ one(T)

      # Stopping conditions based on user-provided tolerances.
      user_requested_exit = callback(solver) :: Bool
      tired[i] = iter ≥ itmax
      ill_cond_lim[i] = test3 ≤ ctol
      solved_lim = test2 ≤ axtol
      zero_resid_lim = test1 ≤ rNormtol
      solved_opt = ArNorm ≤ atol + rtol * ArNorm0

      zero_resid[i] = zero_resid_lim | zero_resid_mach

      converged[i] = solved_mach | solved_lim | solved_opt
    end

    #(verbose > 0) && @printf("%5d  %7.1e  %7.1e  %7.1e  %7.1e  %7.1e  %7.1e  %7.1e  %7.1e\n", 1 + 2 * iter, α, β, rNorm[end], ArNorm[end], rNorm[end] / β₁, ArNorm[end] / (Anorm[end] * rNorm[end]), Anorm[end], Acond[end])
    converged .= converged .| zero_resid .| fwd_err .| on_boundary
  end
  (verbose > 0) && @printf("\n")

  user_requested_exit && (stats.status .= "user-requested exit")
  for i = 1 : nshifts
    tired[i]         && (stats.status[i] = "maximum number of iterations exceeded")
    ill_cond_mach[i] && (stats.status[i] = "condition number seems too large for this machine")
    ill_cond_lim[i]  && (stats.status[i] = "condition number exceeds tolerance")
    converged[i]     && (stats.status[i] = "found approximate minimum least-squares solution")
    zero_resid[i]    && (stats.status[i] = "found approximate zero-residual solution")
    fwd_err[i]       && (stats.status[i] = "truncated forward error small enough")
    on_boundary[i]   && (stats.status[i] = "on trust-region boundary")
  end

  # Update stats
  stats.niter = iter
  stats.solved = all(converged)
  stats.inconsistent = .!zero_resid
  return solver
end
