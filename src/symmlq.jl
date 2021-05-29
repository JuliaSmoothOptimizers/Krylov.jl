# An implementation of SYMMLQ for the solution of the
# linear system Ax = b, where A is square and symmetric.
#
# This implementation follows the original implementation by
# Michael Saunders described in
#
# C. C. Paige and M. A. Saunders, Solution of Sparse Indefinite Systems of Linear Equations,
# SIAM Journal on Numerical Analysis, 12(4), pp. 617--629, 1975.
#
# Ron Estrin, <ronestrin756@gmail.com>

export symmlq, symmlq!


"""
    (x, stats) = symmlq(A, b::AbstractVector{T};
                        M=I, λ::T=zero(T), transfer_to_cg::Bool=true,
                        λest::T=zero(T), atol::T=√eps(T), rtol::T=√eps(T),
                        etol::T=√eps(T), window::Int=0, itmax::Int=0,
                        conlim::T=1/√eps(T), verbose::Int=0, history::Bool=false) where T <: AbstractFloat

Solve the shifted linear system

    (A + λ I) x = b

using the SYMMLQ method, where λ is a shift parameter,
and A is square and symmetric.

SYMMLQ produces monotonic errors ‖x*-x‖₂.

A preconditioner M may be provided in the form of a linear operator and is
assumed to be symmetric and positive definite.

#### Reference

* C. C. Paige and M. A. Saunders, *Solution of Sparse Indefinite Systems of Linear Equations*, SIAM Journal on Numerical Analysis, 12(4), pp. 617--629, 1975.
"""
function symmlq(A, b :: AbstractVector{T}; kwargs...) where T <: AbstractFloat
  solver = SymmlqSolver(A, b)
  symmlq!(solver, A, b; kwargs...)
end

function symmlq!(solver :: SymmlqSolver{T,S}, A, b :: AbstractVector{T};
                 M=I, λ :: T=zero(T), transfer_to_cg :: Bool=true,
                 λest :: T=zero(T), atol :: T=√eps(T), rtol :: T=√eps(T),
                 etol :: T=√eps(T), window :: Int=0, itmax :: Int=0,
                 conlim :: T=1/√eps(T), verbose :: Int=0, history :: Bool=false) where {T <: AbstractFloat, S <: DenseVector{T}}

  m, n = size(A)
  m == n || error("System must be square")
  length(b) == m || error("Inconsistent problem size")
  (verbose > 0) && @printf("SYMMLQ: system of size %d\n", n)

  # Tests M == Iₙ
  MisI = (M == I)

  # Check type consistency
  eltype(A) == T || error("eltype(A) ≠ $T")
  ktypeof(b) == S || error("ktypeof(b) ≠ $S")
  MisI || (eltype(M) == T) || error("eltype(M) ≠ $T")

  # Set up workspace.
  allocate_if(!MisI, solver, :v, S, n)
  x, Mvold, Mv, Mv_next, w̅ = solver.x, solver.Mvold, solver.Mv, solver.Mv_next, solver.w̅
  v = MisI ? Mv : solver.v
  vold = MisI ? Mvold : solver.v

  ϵM = eps(T)
  x .= zero(T)
  ctol = conlim > 0 ? 1 / conlim : zero(T)

  # Initialize Lanczos process.
  # β₁ M v₁ = b.
  Mvold .= b
  MisI || mul!(vold, M, Mvold)
  β₁ = @kdot(m, vold, Mvold)
  β₁ == 0 && return (x, SimpleStats(true, true, [zero(T)], [zero(T)], "x = 0 is a zero-residual solution"))
  β₁ = sqrt(β₁)
  β = β₁
  @kscal!(m, one(T) / β, vold)
  MisI || @kscal!(m, one(T) / β, Mvold)

  w̅ .= vold

  mul!(Mv, A, vold)
  α = @kdot(m, vold, Mv) + λ
  @kaxpy!(m, -α, Mvold, Mv)  # Mv = Mv - α * Mvold
  MisI || mul!(v, M, Mv)
  β = @kdot(m, v, Mv)
  β < 0 && error("Preconditioner is not positive definite")
  β = sqrt(β)
  @kscal!(m, one(T) / β, v)
  MisI || @kscal!(m, one(T) / β, Mv)

  # Start QR factorization
  γbar = α
  δbar = β
  ϵold = zero(T)
  cold = one(T)
  sold = zero(T)

  ηold = zero(T)
  η    = β₁
  ζold = zero(T)

  ANorm² = α * α + β * β

  γmax = T(-Inf)
  γmin = T(Inf)
  ANorm = zero(T)
  Acond = zero(T)

  xNorm = zero(T)
  rNorm = β₁
  rNorms = history ? [rNorm] : T[]

  if γbar ≠ 0
    ζbar = η / γbar
    xcgNorm = abs(ζbar)
    rcgNorm = β₁ * abs(ζbar)
    rcgNorms = history ? Union{T, Missing}[rcgNorm] : Union{T, Missing}[]
  else
    rcgNorms = history ? Union{T, Missing}[missing] : Union{T, Missing}[]
  end

  errors = T[]
  errorscg = Union{T, Missing}[]
  err = T(Inf)
  errcg = T(Inf)

  clist = zeros(T, window)
  zlist = zeros(T, window)
  sprod = ones(T, window)

  if λest ≠ 0
    # Start QR factorization of Tₖ - λest I
    ρbar = α - λest
    σbar = β
    ρ = sqrt(ρbar * ρbar + β * β)
    cwold = -one(T)
    cw = ρbar / ρ
    sw = β / ρ

    history && push!(errors, abs(β₁/λest))
    if γbar ≠ 0
      history && push!(errorscg, sqrt(errors[1]^2 - ζbar^2))
    else
      history && push!(errorscg, missing)
    end
  end

  iter = 0
  itmax == 0 && (itmax = 2 * n)

  (verbose > 0) && @printf("%5s  %7s  %7s  %8s  %8s  %7s  %7s  %7s\n", "Aprod", "‖r‖", "β", "cos", "sin", "‖A‖", "κ(A)", "test1")
  display(iter, verbose) && @printf("%5d  %7.1e  %7.1e  %8.1e  %8.1e  %7.1e  %7.1e\n", 0, rNorm, β, cold, sold, ANorm, Acond)

  tol = atol + rtol * β₁
  status = "unknown"
  solved_lq = solved_mach = solved_lim = (rNorm ≤ tol)
  solved_cg = (γbar ≠ 0) && transfer_to_cg && rcgNorm ≤ tol
  tired  = iter ≥ itmax
  ill_cond = ill_cond_mach = ill_cond_lim = false
  solved = zero_resid = solved_lq || solved_cg
  fwd_err = false

  while ! (solved || tired || ill_cond)
    iter = iter + 1

    # Continue QR factorization
    (c, s, γ) = sym_givens(γbar, β)

    # Update SYMMLQ point
    ηold = η
    ζ = ηold / γ
    @kaxpy!(n, c * ζ, w̅, x)
    @kaxpy!(n, s * ζ, v, x)
    # Update w̅
    @kaxpby!(n, -c, v, s, w̅)

    # Generate next Lanczos vector
    oldβ = β
    mul!(Mv_next, A, v)
    α = @kdot(m, v, Mv_next) + λ
    @kaxpy!(m, -oldβ, Mvold, Mv_next)
    @. Mvold = Mv
    @kaxpy!(m, -α, Mv, Mv_next)
    @. Mv = Mv_next
    MisI || mul!(v, M, Mv)
    β = @kdot(m, v, Mv)
    β < 0 && error("Preconditioner is not positive definite")
    β = sqrt(β)
    @kscal!(m, one(T) / β, v)
    MisI || @kscal!(m, one(T) / β, Mv)

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
    η = -ϵold * ζold - δ * ζ

    rNorm = sqrt(γ * γ * ζ * ζ + ϵold * ϵold * ζold * ζold)
    xNorm = xNorm + ζ * ζ
    history && push!(rNorms, rNorm)

    if γbar ≠ 0
      ζbar = η / γbar
      rcgNorm = β * abs(s * ζ - c * ζbar)
      xcgNorm = xNorm + ζbar * ζbar
      history && push!(rcgNorms, rcgNorm)
    else
      history && push!(rcgNorms, missing)
    end

    if window > 0 && λest ≠ 0
      if iter < window && window > 1
         sprod[iter+1:end] = sprod[iter+1:end] * s
      end

      ix = ((iter-1) % window) + 1
      clist[ix] = c
      zlist[ix] = ζ

      if iter ≥ window
          jx = mod(iter,window) + 1
          zetabark = zlist[jx] / clist[jx]

          if γbar ≠ 0
            theta = abs(@kdot(window, clist, sprod .* zlist))
            theta = zetabark * theta + abs(zetabark * ζbar * sprod[ix] * s) - zetabark^2
            history && (errorscg[iter-window+1] = sqrt(abs(errorscg[iter-window+1]^2 - 2*theta)))
          else
            history && (errorscg[iter-window+1] = missing)
          end
      end

      ix = ((iter) % window) + 1
      if iter ≥ window && window > 1
         sprod = sprod / sprod[(ix % window) + 1]
         sprod[ix] = sprod[mod(ix-2, window)+1] * s
      end
    end

    if λest ≠ 0
      err = abs((ϵold * ζold + ψ * ζ) / ωbar)
      history && push!(errors, err)

      if γbar ≠ 0
        errcg = sqrt(abs(err * err - ζbar * ζbar))
        history && push!(errorscg, errcg)
      else
        history && push!(errorscg, missing)
      end

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

    display(iter, verbose) && @printf("%5d  %7.1e  %7.1e  %8.1e  %8.1e  %7.1e  %7.1e  %7.1e\n", iter, rNorm, β, c, s, ANorm, Acond, test1)

    # Reset variables
    ϵold = ϵ
    ζold = ζ
    cold = c

    # Stopping conditions that do not depend on user input.
    # This is to guard against tolerances that are unreasonably small.
    ill_cond_mach = (one(T) + one(T) / Acond ≤ one(T))
    zero_resid_mach = (one(T) + test1 ≤ one(T))
    # solved_mach = (ϵx ≥ β₁)

    # Stopping conditions based on user-provided tolerances.
    tired = iter ≥ itmax
    ill_cond_lim = (one(T) / Acond ≤ ctol)
    zero_resid_lim = (test1 ≤ tol)
    fwd_err = (err ≤ etol) || ((γbar ≠ 0) && (errcg ≤ etol))
    solved_lq = rNorm ≤ tol
    solved_cg = transfer_to_cg && (γbar ≠ 0) && rcgNorm ≤ tol
    zero_resid = solved_lq || solved_cg
    ill_cond = ill_cond_mach || ill_cond_lim
    solved = solved_mach || zero_resid || zero_resid_mach || zero_resid_lim || fwd_err
  end
  (verbose > 0) && @printf("\n")

  # Compute CG point
  # (xᶜ)ₖ ← (xᴸ)ₖ₋₁ + ζbarₖ * w̅ₖ
  if solved_cg
    @kaxpy!(m, ζbar, w̅, x)
  end

  tired         && (status = "maximum number of iterations exceeded")
  ill_cond_mach && (status = "condition number seems too large for this machine")
  ill_cond_lim  && (status = "condition number exceeds tolerance")
  solved        && (status = "found approximate solution")
  solved_lq     && (status = "solution xᴸ good enough given atol and rtol")
  solved_cg     && (status = "solution xᶜ good enough given atol and rtol")
  stats = SymmlqStats(solved, rNorms, rcgNorms, errors, errorscg, ANorm, Acond, status)
  return (x, stats)
end
