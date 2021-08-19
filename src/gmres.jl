# An implementation of GMRES for the solution of the square linear system Ax = b.
#
# This method is described in
#
# Y. Saad and M. H. Schultz, GMRES: A Generalized Minimal Residual Algorithm for Solving Nonsymmetric Linear Systems.
# SIAM Journal on Scientific and Statistical Computing, Vol. 7(3), pp. 856--869, 1986.
#
# Alexis Montoison, <alexis.montoison@polymtl.ca>
# Montreal, December 2018.

export gmres

"""Solve the consistent linear system Ax = b using GMRES method.

GMRES algorithm is based on the Arnoldi orthogonalization process
and computes a sequence of approximate solutions with the minimal residual property.

When x₀ ≠ 0 (restarted version), GMRES resolves A * Δx = r₀ with r₀ = b - Ax₀ and returns x = x₀ + Δx.

This implementation allows a left preconditioner M and a right preconditioner N.
- Left  preconditioning : M⁻¹Ax = M⁻¹b
- Right preconditioning : AN⁻¹u = b with x = N⁻¹u
- Split preconditioning : M⁻¹AN⁻¹u = M⁻¹b with x = N⁻¹u
"""
function gmres(A :: AbstractLinearOperator, b :: AbstractVector{T};
               x0 :: AbstractVector=zeros(T, size(A,1)),
               M :: AbstractLinearOperator=opEye(),
               N :: AbstractLinearOperator=opEye(),
               atol :: T=√eps(T), rtol :: T=√eps(T), itmax :: Int=0,
               reorthogonalization :: Bool=false, verbose :: Bool=false) where T <: AbstractFloat

  m, n = size(A)
  m == n || error("System must be square")
  length(b) == m || error("Inconsistent problem size")
  verbose && @printf("GMRES: system of size %d\n", n)

  # Initial solution x₀ and residual r₀.
  x = x0 # x₀
  r₀ = A * x0
  @. r₀ = -r₀ + b
  r₀ = M * r₀ # r₀ = M⁻¹(b - Ax₀)

  # Compute β
  rNorm = @knrm2(n, r₀) # β = ‖r₀‖₂
  rNorm == 0 && return x, SimpleStats(true, false, [rNorm], T[], "x = 0 is a zero-residual solution")

  iter = 0
  itmax == 0 && (itmax = 2*n)

  rNorms = [rNorm;]
  ε = atol + rtol * rNorm
  verbose && @printf("%5d  %7.1e\n", iter, rNorm)

  # Set up workspace.
  V  = Vector{T}[zeros(T, n)] # Preconditioned Krylov vectors, orthogonal basis for {r₀, M⁻¹AN⁻¹r₀, (M⁻¹AN⁻¹)²r₀, ..., (M⁻¹AN⁻¹)ᵐ⁻¹r₀}.
  s  = T[]         # Givens sines used for the factorization QₖRₖ = Hₖ.
  c  = T[]         # Givens cosines used for the factorization QₖRₖ = Hₖ.
  H  = Vector{T}[] # Hessenberg matrix Hₖ.
  z  = T[]         # Right-hand of the least squares problem min ‖ Hₖyₖ - zₖ ‖₂.

  # Initial ζ₁ and V₁.
  push!(z, rNorm)
  @. V[1] = r₀ / rNorm

  # Stopping criterion
  solved = rNorm ≤ ε
  tired = iter ≥ itmax
  status = "unknown"

  while !(solved || tired)

    # Update iteration index
    iter = iter + 1

    # Update workspace
    push!(H, zeros(T, iter))
    push!(c, zero(T))
    push!(s, zero(T))
    push!(z, zero(T))

    # Arnoldi procedure
    NV = N * V[iter] # N⁻¹vₖ
    AV = A * NV      # AN⁻¹vₖ
    MV = M * AV      # M⁻¹AN⁻¹vₖ
    for i = 1 : iter
      H[iter][i] = @kdot(n, V[i], MV)
      @kaxpy!(n, -H[iter][i], V[i], MV)
    end

    # Reorthogonalization
    if reorthogonalization
      for i = 1 : iter
        Htmp =  @kdot(n, V[i], MV)
        H[iter][i] += Htmp
        @kaxpy!(n, -Htmp, V[i], MV)
      end
    end

    # Compute hₖ₊₁.ₖ
    Hbis = @knrm2(n, MV) # hₖ₊₁.ₖ = ‖vₖ₊₁‖₂

    # Update the QR factorization of H.
    # Apply previous Givens reflections Ωᵢ.
    for i = 1 : iter-1
      Haux         = c[i] * H[iter][i] + s[i] * H[iter][i+1]
      H[iter][i+1] = s[i] * H[iter][i] - c[i] * H[iter][i+1]
      H[iter][i]   = Haux
    end

    # Compute and apply current Givens reflection Ωₖ.
    # [cₖ  sₖ] [ hₖ.ₖ ] = [ρₖ]
    # [sₖ -cₖ] [hₖ₊₁.ₖ]   [0 ]
    (c[iter], s[iter], H[iter][iter]) = sym_givens(H[iter][iter], Hbis)
    z[iter+1] = s[iter] * z[iter]
    z[iter]   = c[iter] * z[iter]

    # Update residual norm estimate.
    # ‖ M⁻¹(b - Axₖ) ‖₂ = |ζₖ₊₁|
    rNorm = abs(z[iter+1])
    push!(rNorms, rNorm)

    # Update stopping criterion.
    solved = rNorm ≤ ε || Hbis == zero(T) # hₖ₊₁.ₖ = 0 ⇒ "lucky breakdown"
    tired = iter ≥ itmax
    verbose && @printf("%5d  %7.1e\n", iter, rNorm)

    # Compute vₖ₊₁.
    if !(solved || tired)
      push!(V, zeros(T, n))
      @. V[iter+1] = MV / Hbis # Normalization of vₖ₊₁
    end

  end
  verbose && @printf("\n")

  # Compute yₖ by solving Rₖyₖ = (Qₖ)ᵗzₖ with backward substitution
  # We will overwrite zₖ with yₖ
  y = z
  y[iter] = z[iter] / H[iter][iter]
  for i = iter-1 : -1 : 1
    y[i] = (z[i] - sum(H[k][i] * y[k] for k = i+1 : iter)) / H[i][i]
  end

  # Form xₖ = x₀ + Vₖyₖ
  for i = 1 : iter
    Dᵢ = N * V[i] # Needed with right preconditioning
    @kaxpy!(n, y[i], Dᵢ, x)
  end

  status = tired ? "maximum number of iterations exceeded" : "solution good enough given atol and rtol"
  stats = SimpleStats(solved, false, rNorms, T[], status)
  return (x, stats)
end
