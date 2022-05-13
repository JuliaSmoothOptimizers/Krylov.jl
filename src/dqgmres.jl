# An implementation of DQGMRES for the solution of the square linear system Ax = b.
#
# This method is described in
#
# Y. Saad and K. Wu, DQGMRES: a quasi minimal residual algorithm based on incomplete orthogonalization.
# Numerical Linear Algebra with Applications, Vol. 3(4), pp. 329--343, 1996.
#
# Alexis Montoison, <alexis.montoison@polymtl.ca>
# Montreal, August 2018.

export dqgmres, dqgmres!

"""
    (x, stats) = dqgmres(A, b::AbstractVector{FC}; memory::Int=20,
                         M=I, N=I, atol::T=√eps(T), rtol::T=√eps(T),
                         reorthogonalization::Bool=false, itmax::Int=0,
                         verbose::Int=0, history::Bool=false)

`T` is an `AbstractFloat` such as `Float32`, `Float64` or `BigFloat`.
`FC` is `T` or `Complex{T}`.

Solve the consistent linear system Ax = b using DQGMRES method.

DQGMRES algorithm is based on the incomplete Arnoldi orthogonalization process
and computes a sequence of approximate solutions with the quasi-minimal residual property.

DQGMRES only orthogonalizes the new vectors of the Krylov basis against the `memory` most recent vectors.
If MINRES is well defined on `Ax = b` and `memory = 2`, DQGMRES is theoretically equivalent to MINRES.
If `k ≤ memory` where `k` is the number of iterations, DQGMRES is theoretically equivalent to GMRES.
Otherwise, DQGMRES interpolates between MINRES and GMRES and is similar to MINRES with partial reorthogonalization.

Partial reorthogonalization is available with the `reorthogonalization` option.

This implementation allows a left preconditioner M and a right preconditioner N.
- Left  preconditioning : M⁻¹Ax = M⁻¹b
- Right preconditioning : AN⁻¹u = b with x = N⁻¹u
- Split preconditioning : M⁻¹AN⁻¹u = M⁻¹b with x = N⁻¹u

DQGMRES can be warm-started from an initial guess `x0` with the method

    (x, stats) = dqgmres(A, b, x0; kwargs...)

where `kwargs` are the same keyword arguments as above.

#### Reference

* Y. Saad and K. Wu, [*DQGMRES: a quasi minimal residual algorithm based on incomplete orthogonalization*](https://doi.org/10.1002/(SICI)1099-1506(199607/08)3:4%3C329::AID-NLA86%3E3.0.CO;2-8), Numerical Linear Algebra with Applications, Vol. 3(4), pp. 329--343, 1996.
"""
function dqgmres end

function dqgmres(A, b :: AbstractVector{FC}, x0 :: AbstractVector; memory :: Int=20, kwargs...) where FC <: FloatOrComplex
  solver = DqgmresSolver(A, b, memory)
  dqgmres!(solver, A, b, x0; kwargs...)
  return (solver.x, solver.stats)
end

function dqgmres(A, b :: AbstractVector{FC}; memory :: Int=20, kwargs...) where FC <: FloatOrComplex
  solver = DqgmresSolver(A, b, memory)
  dqgmres!(solver, A, b; kwargs...)
  return (solver.x, solver.stats)
end

"""
    solver = dqgmres!(solver::DqgmresSolver, A, b; kwargs...)
    solver = dqgmres!(solver::DqgmresSolver, A, b, x0; kwargs...)

where `kwargs` are keyword arguments of [`dqgmres`](@ref).

Note that the `memory` keyword argument is the only exception.
It's required to create a `DqgmresSolver` and can't be changed later.

See [`DqgmresSolver`](@ref) for more details about the `solver`.
"""
function dqgmres! end

function dqgmres!(solver :: DqgmresSolver{T,FC,S}, A, b :: AbstractVector{FC}, x0 :: AbstractVector; kwargs...) where {T <: AbstractFloat, FC <: FloatOrComplex{T}, S <: DenseVector{FC}}
  warm_start!(solver, x0)
  dqgmres!(solver, A, b; kwargs...)
  return solver
end

function dqgmres!(solver :: DqgmresSolver{T,FC,S}, A, b :: AbstractVector{FC};
                  M=I, N=I, atol :: T=√eps(T), rtol :: T=√eps(T),
                  reorthogonalization :: Bool=false, itmax :: Int=0,
                  verbose :: Int=0, history :: Bool=false) where {T <: AbstractFloat, FC <: FloatOrComplex{T}, S <: DenseVector{FC}}

  m, n = size(A)
  m == n || error("System must be square")
  length(b) == m || error("Inconsistent problem size")
  (verbose > 0) && @printf("DQGMRES: system of size %d\n", n)

  # Check M = Iₙ and N = Iₙ
  MisI = (M === I)
  NisI = (N === I)

  # Check type consistency
  eltype(A) == FC || error("eltype(A) ≠ $FC")
  ktypeof(b) == S || error("ktypeof(b) ≠ $S")

  # Set up workspace.
  allocate_if(!MisI, solver, :w, S, n)
  allocate_if(!NisI, solver, :z, S, n)
  Δx, x, t, P, V = solver.Δx, solver.x, solver.t, solver.P, solver.V
  c, s, H, stats = solver.c, solver.s, solver.H, solver.stats
  warm_start = solver.warm_start
  rNorms = stats.residuals
  reset!(stats)
  w  = MisI ? t : solver.w
  r₀ = MisI ? t : solver.w

  # Initial solution x₀ and residual r₀.
  x .= zero(FC)  # x₀
  if warm_start
    mul!(t, A, Δx)
    @kaxpby!(n, one(FC), b, -one(FC), t)
  else
    t .= b
  end
  MisI || mul!(r₀, M, t)  # M⁻¹(b - Ax₀)
  # Compute β
  rNorm = @knrm2(n, r₀) # β = ‖r₀‖₂
  history && push!(rNorms, rNorm)
  if rNorm == 0
    stats.niter = 0
    stats.solved, stats.inconsistent = true, false
    stats.status = "x = 0 is a zero-residual solution"
    solver.warm_start = false
    return solver
  end

  iter = 0
  itmax == 0 && (itmax = 2*n)

  ε = atol + rtol * rNorm
  (verbose > 0) && @printf("%5s  %7s\n", "k", "‖rₖ‖")
  kdisplay(iter, verbose) && @printf("%5d  %7.1e\n", iter, rNorm)

  # Set up workspace.
  mem = length(c)  # Memory.
  for i = 1 : mem
    V[i] .= zero(FC)  # Orthogonal basis of Kₖ(M⁻¹AN⁻¹, M⁻¹b).
    P[i] .= zero(FC)  # Directions for x : Pₘ = N⁻¹Vₘ(Rₘ)⁻¹.
  end
  c .= zero(T)   # Last mem Givens cosines used for the factorization QₘRₘ = Hₘ.
  s .= zero(FC)  # Last mem Givens sines used for the factorization QₘRₘ = Hₘ.
  H .= zero(FC)  # Last column of the band hessenberg matrix Hₘ.
  # Each column has at most mem + 1 nonzero elements. hᵢ.ₘ is stored as H[m-i+2].
  # m-i+2 represents the indice of the diagonal where hᵢ.ₘ is located.
  # In addition of that, the last column of Rₘ is also stored in H.

  # Initial γ₁ and V₁.
  γₘ = rNorm # γₘ and γₘ₊₁ are the last components of gₘ, right-hand of the least squares problem min ‖ Hₘyₘ - gₘ ‖₂.
  @. V[1] = r₀ / rNorm

  # The following stopping criterion compensates for the lag in the
  # residual, but usually increases the number of iterations.
  # solved = sqrt(max(1, iter-mem+1)) * |γₘ₊₁| ≤ ε
  solved = rNorm ≤ ε # less accurate, but acceptable.
  tired = iter ≥ itmax
  status = "unknown"

  while !(solved || tired)

    # Update iteration index.
    iter = iter + 1

    # Set position in circulars stacks.
    pos = mod(iter-1, mem) + 1 # Position corresponding to pₘ and vₘ in circular stacks P and V.
    next_pos = mod(iter, mem) + 1 # Position corresponding to vₘ₊₁ in the circular stack V.

    # Incomplete Arnoldi procedure.
    z = NisI ? V[pos] : solver.z
    NisI || mul!(z, N, V[pos])  # N⁻¹vₘ, forms pₘ
    mul!(t, A, z)               # AN⁻¹vₘ
    MisI || mul!(w, M, t)       # M⁻¹AN⁻¹vₘ, forms vₘ₊₁
    for i = max(1, iter-mem+1) : iter
      ipos = mod(i-1, mem) + 1 # Position corresponding to vᵢ in the circular stack V.
      diag = iter - i + 2
      H[diag] = @kdot(n, w, V[ipos]) # hᵢ.ₘ = ⟨M⁻¹AN⁻¹vₘ , vᵢ⟩
      @kaxpy!(n, -H[diag], V[ipos], w) # w ← w - hᵢ.ₘ * vᵢ
    end

    # Partial reorthogonalization of the Krylov basis.
    if reorthogonalization
      for i = max(1, iter-mem+1) : iter
        ipos = mod(i-1, mem) + 1
        diag = iter - i + 2
        Htmp = @kdot(n, w, V[ipos])
        H[diag] += Htmp
        @kaxpy!(n, -Htmp, V[ipos], w)
      end
    end

    # Compute hₘ₊₁.ₘ and vₘ₊₁.
    H[1] = @knrm2(n, w) # hₘ₊₁.ₘ = ‖vₘ₊₁‖₂
    if H[1] ≠ 0 # hₘ₊₁.ₘ = 0 ⇒ "lucky breakdown"
      @. V[next_pos] = w / H[1] # vₘ₊₁ = w / hₘ₊₁.ₘ
    end
    # rₘ₋ₘₑₘ.ₘ ≠ 0 when m ≥ mem + 1
    if iter ≥ mem + 2
      H[mem+2] = zero(FC) # hₘ₋ₘₑₘ.ₘ = 0
    end

    # Update the QR factorization of H.
    # Apply mem previous Givens reflections Ωᵢ.
    for i = max(1,iter-mem) : iter-1
      irot_pos = mod(i-1, mem) + 1 # Position corresponding to cᵢ and sᵢ in circular stacks c and s.
      diag = iter - i + 1
      next_diag = diag + 1
      H_aux        =      c[irot_pos]  * H[next_diag] + s[irot_pos] * H[diag]
      H[diag]      = conj(s[irot_pos]) * H[next_diag] - c[irot_pos] * H[diag]
      H[next_diag] = H_aux
    end

    # Compute and apply current Givens reflection Ωₘ.
    # [cₘ  sₘ] [ hₘ.ₘ ] = [ρₘ]
    # [sₘ -cₘ] [hₘ₊₁.ₘ]   [0 ]
    (c[pos], s[pos], H[2]) = sym_givens(H[2], H[1])
    γₘ₊₁ = conj(s[pos]) * γₘ
    γₘ   =      c[pos]  * γₘ

    # Compute the direction pₘ, the last column of Pₘ = N⁻¹Vₘ(Rₘ)⁻¹.
    for i = max(1,iter-mem) : iter-1
      ipos = mod(i-1, mem) + 1 # Position corresponding to pᵢ in the circular stack P.
      diag = iter - i + 2
      if ipos == pos
        # pₐᵤₓ ← -hₘ₋ₘₑₘ.ₘ * pₘ₋ₘₑₘ
        @kscal!(n, -H[diag], P[pos])
      else
        # pₐᵤₓ ← pₐᵤₓ - hᵢ.ₘ * pᵢ
        @kaxpy!(n, -H[diag], P[ipos], P[pos])
      end
    end
    # pₐᵤₓ ← pₐᵤₓ + N⁻¹vₘ
    @kaxpy!(n, one(FC), z, P[pos])
    # pₘ = pₐᵤₓ / hₘ.ₘ
    @. P[pos] = P[pos] / H[2]

    # Compute solution xₘ.
    # xₘ ← xₘ₋₁ + γₘ * pₘ
    @kaxpy!(n, γₘ, P[pos], x)

    # Update residual norm estimate.
    # ‖ M⁻¹(b - Axₘ) ‖₂ ≈ |γₘ₊₁|
    rNorm = abs(γₘ₊₁)
    history && push!(rNorms, rNorm)

    # Update γₘ.
    γₘ = γₘ₊₁

    # Update stopping criterion.
    solved = rNorm ≤ ε
    tired = iter ≥ itmax
    kdisplay(iter, verbose) && @printf("%5d  %7.1e\n", iter, rNorm)
  end
  (verbose > 0) && @printf("\n")
  status = tired ? "maximum number of iterations exceeded" : "solution good enough given atol and rtol"

  # Update x
  warm_start && @kaxpy!(n, one(FC), Δx, x)
  solver.warm_start = false

  # Update stats
  stats.niter = iter
  stats.solved = solved
  stats.inconsistent = false
  stats.status = status
  return solver
end
