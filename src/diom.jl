# An implementation of DIOM for the solution of the square linear system Ax = b.
#
# This method is described in
#
# Y. Saad, Practical use of some krylov subspace methods for solving indefinite and nonsymmetric linear systems.
# SIAM journal on scientific and statistical computing, 5(1), pp. 203--228, 1984.
#
# Alexis Montoison, <alexis.montoison@polymtl.ca>
# Montreal, September 2018.

export diom, diom!

"""
    (x, stats) = diom(A, b::AbstractVector{FC}; memory::Int=20,
                      M=I, N=I, atol::T=√eps(T), rtol::T=√eps(T),
                      reorthogonalization::Bool=false, itmax::Int=0,
                      verbose::Int=0, history::Bool=false,
                      callback::Function=solver->false)

`T` is an `AbstractFloat` such as `Float32`, `Float64` or `BigFloat`.
`FC` is `T` or `Complex{T}`.

Solve the consistent linear system Ax = b using direct incomplete orthogonalization method.

DIOM only orthogonalizes the new vectors of the Krylov basis against the `memory` most recent vectors.
If CG is well defined on `Ax = b` and `memory = 2`, DIOM is theoretically equivalent to CG.
If `k ≤ memory` where `k` is the number of iterations, DIOM is theoretically equivalent to FOM.
Otherwise, DIOM interpolates between CG and FOM and is similar to CG with partial reorthogonalization.

Partial reorthogonalization is available with the `reorthogonalization` option.

An advantage of DIOM is that nonsymmetric or symmetric indefinite or both nonsymmetric
and indefinite systems of linear equations can be handled by this single algorithm.

This implementation allows a left preconditioner M and a right preconditioner N.
- Left  preconditioning : M⁻¹Ax = M⁻¹b
- Right preconditioning : AN⁻¹u = b with x = N⁻¹u
- Split preconditioning : M⁻¹AN⁻¹u = M⁻¹b with x = N⁻¹u

DIOM can be warm-started from an initial guess `x0` with the method

    (x, stats) = diom(A, b, x0; kwargs...)

where `kwargs` are the same keyword arguments as above.

The callback is called as `callback(solver)` and should return `true` if the main loop should terminate,
and `false` otherwise.

#### Reference

* Y. Saad, [*Practical use of some krylov subspace methods for solving indefinite and nonsymmetric linear systems*](https://doi.org/10.1137/0905015), SIAM journal on scientific and statistical computing, 5(1), pp. 203--228, 1984.
"""
function diom end

function diom(A, b :: AbstractVector{FC}, x0 :: AbstractVector; memory :: Int=20, kwargs...) where FC <: FloatOrComplex
  solver = DiomSolver(A, b, memory)
  diom!(solver, A, b, x0; kwargs...)
  return (solver.x, solver.stats)
end

function diom(A, b :: AbstractVector{FC}; memory :: Int=20, kwargs...) where FC <: FloatOrComplex
  solver = DiomSolver(A, b, memory)
  diom!(solver, A, b; kwargs...)
  return (solver.x, solver.stats)
end

"""
    solver = diom!(solver::DiomSolver, A, b; kwargs...)
    solver = diom!(solver::DiomSolver, A, b, x0; kwargs...)

where `kwargs` are keyword arguments of [`diom`](@ref).

Note that the `memory` keyword argument is the only exception.
It's required to create a `DiomSolver` and can't be changed later.

See [`DiomSolver`](@ref) for more details about the `solver`.
"""
function diom! end

function diom!(solver :: DiomSolver{T,FC,S}, A, b :: AbstractVector{FC}, x0 :: AbstractVector; kwargs...) where {T <: AbstractFloat, FC <: FloatOrComplex{T}, S <: DenseVector{FC}}
  warm_start!(solver, x0)
  diom!(solver, A, b; kwargs...)
  return solver
end

function diom!(solver :: DiomSolver{T,FC,S}, A, b :: AbstractVector{FC};
               M=I, N=I, atol :: T=√eps(T), rtol :: T=√eps(T),
               reorthogonalization :: Bool=false, itmax :: Int=0,
               verbose :: Int=0, history :: Bool=false,
               callback :: Function = solver -> false) where {T <: AbstractFloat, FC <: FloatOrComplex{T}, S <: DenseVector{FC}}

  m, n = size(A)
  m == n || error("System must be square")
  length(b) == m || error("Inconsistent problem size")
  (verbose > 0) && @printf("DIOM: system of size %d\n", n)

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
  L, H, stats = solver.L, solver.H, solver.stats
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
  # Compute β.
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

  mem = length(L)  # Memory
  for i = 1 : mem
    V[i] .= zero(FC)  # Orthogonal basis of Kₖ(M⁻¹AN⁻¹, M⁻¹b).
    P[i] .= zero(FC)  # Directions for x : Pₘ = N⁻¹Vₘ(Uₘ)⁻¹.
  end
  H .= zero(FC)  # Last column of the band hessenberg matrix Hₘ = LₘUₘ.
  # Each column has at most mem + 1 nonzero elements. hᵢ.ₘ is stored as H[m-i+2].
  # m-i+2 represents the indice of the diagonal where hᵢ.ₘ is located.
  # In addition of that, the last column of Uₘ is stored in H.
  L .= zero(FC)  # Last mem pivots of Lₘ.

  # Initial ξ₁ and V₁.
  ξ = rNorm
  @. V[1] = r₀ / rNorm

  # Stopping criterion.
  solved = rNorm ≤ ε
  tired = iter ≥ itmax
  status = "unknown"
  user_requested_exit = false

  while !(solved || tired || user_requested_exit)

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
    # It's possible that uₘ₋ₘₑₘ.ₘ ≠ 0 when m ≥ mem + 1
    if iter ≥ mem + 2
      H[mem+2] = zero(FC) # hₘ₋ₘₑₘ.ₘ = 0
    end

    # Update the LU factorization with partial pivoting of H.
    # Compute the last column of Uₘ.
    if iter ≥ 2
      for i = max(2,iter-mem+1) : iter
        lpos = mod(i-1, mem) + 1 # Position corresponding to lᵢ.ᵢ₋₁ in the circular stack L.
        diag = iter - i + 2
        next_diag = diag + 1
        # uᵢ.ₘ ← hᵢ.ₘ - lᵢ.ᵢ₋₁ * uᵢ₋₁.ₘ
        H[diag] = H[diag] - L[lpos] * H[next_diag]
      end
      # Compute ξₘ the last component of zₘ = β(Lₘ)⁻¹e₁.
      # ξₘ = -lₘ.ₘ₋₁ * ξₘ₋₁
      ξ = - L[pos] * ξ
    end
    # Compute next pivot lₘ₊₁.ₘ = hₘ₊₁.ₘ / uₘ.ₘ
    L[next_pos] = H[1] / H[2]

    # Compute the direction pₘ, the last column of Pₘ = N⁻¹Vₘ(Uₘ)⁻¹.
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
    # pₘ = pₐᵤₓ / uₘ.ₘ
    @. P[pos] = P[pos] / H[2]

    # Update solution xₘ.
    # xₘ = xₘ₋₁ + ξₘ * pₘ
    @kaxpy!(n, ξ, P[pos], x)

    # Compute residual norm.
    # ‖ M⁻¹(b - Axₘ) ‖₂ = hₘ₊₁.ₘ * |ξₘ / uₘ.ₘ|
    rNorm = real(H[1]) * abs(ξ / H[2])
    history && push!(rNorms, rNorm)

    # Update stopping criterion.
    user_requested_exit = callback(solver) :: Bool
    solved = rNorm ≤ ε
    tired = iter ≥ itmax
    kdisplay(iter, verbose) && @printf("%5d  %7.1e\n", iter, rNorm)
  end
  (verbose > 0) && @printf("\n")
  tired               && (status = "maximum number of iterations exceeded")
  solved              && (status = "solution good enough given atol and rtol")
  user_requested_exit && (status = "user-requested exit")

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
