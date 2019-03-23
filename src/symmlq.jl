# An implementation of SYMMLQ for the solution of the
# linear system Ax = b, where A is square and symmetric.
#
# This implementation follows the original implementation by
# Michael Saunders described in
#
# C. C. Paige and M. A. Saunders, Solution of Sparse Indefinite Systems
# of Linear Equations, SIAM Journal on Numerical Analysis, 12(4), 617-629, 1975.
#
# Ron Estrin, <ronestrin756@gmail.com>

export symmlq


"""Solve the shifted linear system

  (A + λ I) x = b

using the SYMMLQ method, where λ is a shift parameter,
and A is square and symmetric.

SYMMLQ produces monotonic errors ‖x*-x‖₂.

A preconditioner M may be provided in the form of a linear operator and is
assumed to be symmetric and positive definite.
"""
function symmlq(A :: AbstractLinearOperator, b :: AbstractVector{T};
                M :: AbstractLinearOperator=opEye(), λ :: Float64=0.0,
                λest :: Float64=0.0, atol :: Float64=1.0e-8, rtol :: Float64=1.0e-8,
                etol :: Float64=1.0e-8, window :: Int=0, itmax :: Int=0,
                conlim :: Float64=1.0e+8, verbose :: Bool=false) where T <: Number

  m, n = size(A)
  m == n || error("System must be square")
  size(b, 1) == m || error("Inconsistent problem size")
  verbose && @printf("SYMMLQ: system of size %d\n", n)

  # Test M == Iₘ
  MisI = isa(M, opEye)

  ϵM = eps(T)
  x_lq = zeros(T, n)
  ctol = conlim > 0.0 ? 1 / conlim : 0.0;

  # Initialize Lanczos process.
  # β₁ M v₁ = b.
  Mvold = copy(b)
  vold = M * Mvold
  β₁ = @kdot(m, vold, Mvold)
  β₁ == 0.0 && return (x_lq, zeros(T, n), SimpleStats(true, true, [0.0], [0.0], "x = 0 is a zero-residual solution"))
  β₁ = sqrt(β₁)
  β = β₁
  @kscal!(m, 1 / β, vold)
  MisI || @kscal!(m, 1 / β, Mvold)

  w̅ = copy(vold)

  Mv = copy(A * vold)
  α = @kdot(m, vold, Mv) + λ
  @kaxpy!(m, -α, Mvold, Mv)  # Mv = Mv - α * Mvold
  v = M * Mv
  β = @kdot(m, v, Mv)
  β < 0.0 && error("Preconditioner is not positive definite")
  β = sqrt(β)
  @kscal!(m, 1 / β, v)
  MisI || @kscal!(m, 1 / β, Mv)

  # Start QR factorization
  γbar = α
  δbar = β
  ϵold = 0.0
  cold = 1.0
  sold = 0.0

  ζold = 0.0
  ζbar = β₁/γbar

  ANorm² = α * α + β * β

  γmax = -Inf
  γmin = Inf
  ANorm = 0.0
  Acond = 0.0
  rNorm = β₁
  rcgNorm = β₁
  xNorm = 0.0
  xcgNorm = abs(ζbar)

  rNorms = [rNorm]
  rcgNorms = [rcgNorm]
  errors = T[]
  errorscg = T[]
  err = Inf
  errcg = Inf

  clist = zeros(window)
  zlist = zeros(window)
  sprod = ones(window)

  if λest ≠ 0
    # Start QR factorization of Tₖ - λest I
    ρbar = α - λest
    σbar = β
    ρ = sqrt(ρbar * ρbar + β * β)
    cwold = -1.0
    cw = ρbar / ρ
    sw = β / ρ

    push!(errors, abs(β₁/λest))
    push!(errorscg, sqrt(errors[1]^2 - ζbar^2))
  end

  verbose && @printf("%5s  %7s  %7s  %8s  %8s  %7s  %7s\n",
                     "Aprod", "‖r‖", "β", "cos", "sin", "‖A‖", "κ(A)")
  verbose && @printf("%5d  %7.1e  %7.1e  %8.1e  %8.1e  %7.1e  %7.1e\n",
                     0, rNorm, β, cold, sold, ANorm, Acond)

  iter = 0
  itmax == 0 && (itmax = 2 * n)

  tol = atol + rtol * β₁
  status = "unknown"
  solved = solved_mach = solved_lim = (rNorm ≤ rtol)
  tired  = iter ≥ itmax
  ill_cond = ill_cond_mach = ill_cond_lim = false
  zero_resid = zero_resid_mach = zero_resid_lim = (rNorm ≤ tol)
  fwd_err = false

  while ! (solved || tired || ill_cond)
    iter = iter + 1

    # Continue QR factorization
    γ = sqrt(γbar^2 + β^2)
    c = γbar/γ
    s = β/γ

    # Update SYMMLQ point
    ζ = ζbar * c
    @kaxpy!(n, c * ζ, w̅, x_lq)
    @kaxpy!(n, s * ζ, v, x_lq)

    # Update w̅
    @kaxpby!(n, -c, v, s, w̅)

    # Generate next Lanczos vector
    oldβ = β
    Mv_next = A * v
    α = @kdot(m, v, Mv_next) + λ
    @kaxpy!(m, -oldβ, Mvold, Mv_next)
    @. Mvold = Mv
    @kaxpy!(m, -α, Mv, Mv_next)
    @. Mv = Mv_next
    v = M * Mv
    β = @kdot(m, v, Mv)
    β < 0.0 && error("Preconditioner is not positive definite")
    β = sqrt(β)
    @kscal!(m, 1 / β, v)
    MisI || @kscal!(m, 1 / β, Mv)

    # Continue A norm estimate
    ANorm² = ANorm² + α * α + oldβ * oldβ + β * β

    if λest ≠ 0
      η = -oldβ * oldβ * cwold / ρbar
      ω = λest + η
      ψ = c * δbar + s * ω
      ωbar = s * δbar - c * ω
    end

    # Continue QR factorization
    δ = δbar * c + α * s
    γbar = δbar * s - α * c
    ϵ = β * s
    δbar = -β * c
    ζbar = -(ϵold * ζold + δ * ζ)/γbar

    rNorm = sqrt(γ * γ * ζ * ζ + ϵold * ϵold * ζold * ζold)
    rcgNorm = abs(rcgNorm*s*cold/c)
    push!(rNorms, rNorm)
    push!(rcgNorms, rcgNorm)

    xNorm = xNorm + ζ * ζ
    xcgNorm = xNorm + ζbar * ζbar

    if window > 0 && λest ≠ 0
      if iter < window && window > 1
         sprod[iter+1:end] = sprod[iter+1:end]*s
      end      

      ix = ((iter-1) % window) + 1
      clist[ix] = c
      zlist[ix] = ζ

      if iter ≥ window
          jx = mod(iter,window)+1
          zetabark = zlist[jx]/clist[jx] 

          theta = abs(@kdot(window, clist, sprod.*zlist))
          theta = zetabark*theta + 
              abs(zetabark*ζbar*sprod[ix]*s) -
              zetabark^2

          errorscg[iter-window+1] = sqrt(abs(errorscg[iter-window+1]^2 - 2*theta))
      end

      ix = ((iter) % window) + 1
      if iter ≥ window && window > 1
         sprod = sprod/sprod[(ix % window) + 1]
         sprod[ix] = sprod[mod(ix-2, window)+1]*s
      end
    end

    if λest ≠ 0
      err = abs((ϵold * ζold + ψ * ζ)/ωbar)
      errcg = sqrt(abs(err * err - ζbar * ζbar))

      push!(errors, err)
      push!(errorscg, errcg)

      ρbar = sw * σbar - cw * (α - λest)
      σbar = -cw * β
      ρ = sqrt(ρbar * ρbar + β * β)

      cwold = cw

      cw = ρbar / ρ
      sw = β / ρ
    end

    # TODO: Use γ or γbar?
    γmax = max(γmax, γ)
    γmin = min(γmin, γ)

    Acond = γmax / γmin
    ANorm = sqrt(ANorm²)
    test1 = rNorm/(ANorm * xNorm)

    verbose && @printf("%5d  %7.1e  %7.1e  %8.1e  %8.1e  %7.1e  %7.1e\n",
                       iter, test1, β, c, s, ANorm, Acond)
    
    # Reset variables
    ϵold = ϵ
    ζold = ζ
    cold = c

    # Stopping conditions that do not depend on user input.
    # This is to guard against tolerances that are unreasonably small.
    ill_cond_mach = (1.0 + 1.0 / Acond ≤ 1.0)
    zero_resid_mach = (1.0 + test1 ≤ 1.0)
    # solved_mach = (ϵx ≥ β₁)

    # Stopping conditions based on user-provided tolerances.
    tired = iter ≥ itmax
    ill_cond_lim = (1.0 / Acond ≤ ctol)
    zero_resid_lim = (test1 ≤ tol)
    fwd_err = (err ≤ etol) | (errcg ≤ etol)

    ill_cond = ill_cond_mach | ill_cond_lim
    solved = solved_mach | solved_lim | zero_resid_mach | zero_resid_lim | fwd_err
  end

  # Compute CG point
  @kaxpby!(m, 1.0, x_lq, ζbar, w̅)
  x_cg = w̅
  
  tired         && (status = "maximum number of iterations exceeded")
  ill_cond_mach && (status = "condition number seems too large for this machine")
  ill_cond_lim  && (status = "condition number exceeds tolerance")
  solved        && (status = "found approximate solution")
  zero_resid    && (status = "found approximate zero-residual solution")

  stats = SymmlqStats(solved, rNorms, rcgNorms, errors, errorscg, ANorm, Acond, status)
  return (x_lq, x_cg, stats)
end
