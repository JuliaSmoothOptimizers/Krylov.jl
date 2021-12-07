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
                       M=I, N=I, atol::T=√eps(T), rtol::T=√eps(T),
                       reorthogonalization::Bool=false, itmax::Int=0,
                       restart::Bool=false, memory::Int=20,
                       verbose::Int=0, history::Bool=false)

`T` is an `AbstractFloat` such as `Float32`, `Float64` or `BigFloat`.
`FC` is `T` or `Complex{T}`.

Solve the linear system Ax = b using GMRES method.

GMRES algorithm is based on the Arnoldi process and computes a sequence of approximate solutions with the minimal residual property.

This implementation allows a left preconditioner M and a right preconditioner N.
- Left  preconditioning : M⁻¹Ax = M⁻¹b
- Right preconditioning : AN⁻¹u = b with x = N⁻¹u
- Split preconditioning : M⁻¹AN⁻¹u = M⁻¹b with x = N⁻¹u

#### Reference

* Y. Saad and M. H. Schultz, [*GMRES: A Generalized Minimal Residual Algorithm for Solving Nonsymmetric Linear Systems*](https://doi.org/10.1137/0907058), SIAM Journal on Scientific and Statistical Computing, Vol. 7(3), pp. 856--869, 1986.
"""
function gmres(A, b :: AbstractVector{FC}; memory :: Int=20, kwargs...) where FC <: FloatOrComplex
  solver = GmresSolver(A, b, memory)
  gmres!(solver, A, b; kwargs...)
  return (solver.x, solver.stats)
end

function gmres!(solver :: GmresSolver{T,FC,S}, A, b :: AbstractVector{FC};
                M=I, N=I, atol :: T=√eps(T), rtol :: T=√eps(T),
                reorthogonalization :: Bool=false, itmax :: Int=0,
                restart :: Bool=false, verbose :: Int=0, history :: Bool=false) where {T <: AbstractFloat, FC <: FloatOrComplex{T}, S <: DenseVector{FC}}

  m, n = size(A)
  m == n || error("System must be square")
  length(b) == m || error("Inconsistent problem size")
  (verbose > 0) && @printf("GMRES: system of size %d\n", n)

  # Check M == Iₙ and N == Iₙ
  MisI = (M == I)
  NisI = (N == I)

  # Check type consistency
  eltype(A) == T || error("eltype(A) ≠ $T")
  ktypeof(b) == S || error("ktypeof(b) ≠ $S")
  MisI || (promote_type(eltype(M), T) == T) || error("eltype(M) can't be promoted to $T")
  NisI || (promote_type(eltype(N), T) == T) || error("eltype(N) can't be promoted to $T")

  # Set up workspace.
  allocate_if(!MisI  , solver, :q , S, n)
  allocate_if(!NisI  , solver, :p , S, n)
  allocate_if(restart, solver, :Δx, S, n)
  Δx, x, w, V, z = solver.Δx, solver.x, solver.w, solver.V, solver.z
  c, s, R, stats = solver.c, solver.s, solver.R, solver.stats
  rNorms = stats.residuals
  reset!(stats)
  q  = MisI ? w : solver.q
  r₀ = MisI ? w : solver.q

  # Initial solution x₀ and residual r₀.
  restart && (Δx .= x)
  x .= zero(T)            # x₀
  if restart
    mul!(w, A, Δx)
    @kaxpby!(n, one(T), b, -one(T), w)
  else
    w .= b
  end
  MisI || mul!(r₀, M, w)  # M⁻¹(b - Ax₀)
  β = @knrm2(n, r₀)       # β = ‖r₀‖₂
  rNorm = β
  history && push!(rNorms, β)
  if β == 0
    stats.solved, stats.inconsistent = true, false
    stats.status = "x = 0 is a zero-residual solution"
    return solver
  end

  iter = 0
  itmax == 0 && (itmax = 2*n)

  ε = atol + rtol * rNorm
  (verbose > 0) && @printf("%5s  %7s\n", "k", "‖rₖ‖")
  display(iter, verbose) && @printf("%5d  %7.1e\n", iter, rNorm)

  # Initialize workspace.
  nr = 0           # Number of coefficients stored in Rₖ.
  mem = length(c)  # Memory
  for i = 1 : mem
    V[i] .= zero(T)  # Orthogonal basis of Kₖ(M⁻¹AN⁻¹, M⁻¹b).
  end
  s .= zero(T)  # Givens sines used for the factorization QₖRₖ = Hₖ₊₁.ₖ.
  c .= zero(T)  # Givens cosines used for the factorization QₖRₖ = Hₖ₊₁.ₖ.
  R .= zero(T)  # Upper triangular matrix Rₖ.
  z .= zero(T)  # Right-hand of the least squares problem min ‖Hₖ₊₁.ₖyₖ - βe₁‖₂.

  # Initial ζ₁ and V₁
  z[1] = β
  @. V[1] = r₀ / rNorm

  # Stopping criterion
  solved = rNorm ≤ ε
  tired = iter ≥ itmax
  status = "unknown"

  while !(solved || tired)

    # Update iteration index
    iter = iter + 1

    # Update workspace if more storage is required
    if iter > mem
      for i = 1 : iter
        push!(R, zero(T))
      end
      push!(s, zero(T))
      push!(c, zero(T))
    end

    # Continue the Arnoldi process.
    p = NisI ? V[iter] : solver.p
    NisI || mul!(p, N, V[iter])  # p ← N⁻¹vₖ
    mul!(w, A, p)                # w ← AN⁻¹vₖ
    MisI || mul!(q, M, w)        # q ← M⁻¹AN⁻¹vₖ
    for i = 1 : iter
      R[nr+i] = @kdot(n, V[i], q)    # hᵢₖ = qᵀvᵢ
      @kaxpy!(n, -R[nr+i], V[i], q)  # q ← q - hᵢₖvᵢ
    end

    # Reorthogonalization of the Krylov basis.
    if reorthogonalization
      for i = 1 : iter
        Htmp = @kdot(n, V[i], q)
        R[nr+i] += Htmp
        @kaxpy!(n, -Htmp, V[i], q)
      end
    end

    # Compute hₖ₊₁.ₖ
    Hbis = @knrm2(n, q)  # hₖ₊₁.ₖ = ‖vₖ₊₁‖₂

    # Update the QR factorization of Hₖ₊₁.ₖ.
    # Apply previous Givens reflections Ωᵢ.
    # [cᵢ  sᵢ] [ r̄ᵢ.ₖ ] = [ rᵢ.ₖ ]
    # [sᵢ -cᵢ] [rᵢ₊₁.ₖ]   [r̄ᵢ₊₁.ₖ]
    for i = 1 : iter-1
      Rtmp      = c[i] * R[nr+i] + s[i] * R[nr+i+1]
      R[nr+i+1] = s[i] * R[nr+i] - c[i] * R[nr+i+1]
      R[nr+i]   = Rtmp
    end

    # Compute and apply current Givens reflection Ωₖ.
    # [cₖ  sₖ] [ r̄ₖ.ₖ ] = [rₖ.ₖ]
    # [sₖ -cₖ] [hₖ₊₁.ₖ]   [ 0  ]
    (c[iter], s[iter], R[nr+iter]) = sym_givens(R[nr+iter], Hbis)

    # Update zₖ = (Qₖ)ᵀβe₁
    ζₖ₊₁    = s[iter] * z[iter]
    z[iter] = c[iter] * z[iter]

    # Update residual norm estimate.
    # ‖ M⁻¹(b - Axₖ) ‖₂ = |ζₖ₊₁|
    rNorm = abs(ζₖ₊₁)
    history && push!(rNorms, rNorm)

    # Update the number of coefficients in Rₖ
    nr = nr + iter

    # Update stopping criterion.
    solved = rNorm ≤ ε
    tired = iter ≥ itmax
    display(iter, verbose) && @printf("%5d  %7.1e\n", iter, rNorm)

    # Compute vₖ₊₁
    if !(solved || tired)
      if iter ≥ mem
        push!(V, S(undef, n))
        push!(z, zero(T))
      end
      @. V[iter+1] = q / Hbis  # hₖ₊₁.ₖvₖ₊₁ = q
      z[iter+1] = ζₖ₊₁
    end
  end
  (verbose > 0) && @printf("\n")

  # Compute yₖ by solving Rₖyₖ = zₖ with backward substitution.
  y = z  # yᵢ = zᵢ
  for i = iter : -1 : 1
    pos = nr + i - iter
    for j = iter : -1 : i+1
      y[i] = y[i] - R[pos] * y[j]  # yᵢ ← yᵢ - rᵢⱼyⱼ
      pos = pos - j + 1
    end
    y[i] = y[i] / R[pos]  # yᵢ ← yᵢ / rᵢᵢ
  end

  # Form xₖ = N⁻¹Vₖyₖ
  for i = 1 : iter
    @kaxpy!(n, y[i], V[i], x)
  end
  if !NisI
    solver.p .= x
    mul!(x, N, solver.p)
  end

  status = tired ? "maximum number of iterations exceeded" : "solution good enough given atol and rtol"

  # Update x
  restart && @kaxpy!(n, one(T), Δx, x)

  # Update stats
  stats.solved = solved
  stats.inconsistent = false
  stats.status = status
  return solver
end
