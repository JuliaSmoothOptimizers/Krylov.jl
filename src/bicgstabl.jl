# An implementation of BiCGstab(ℓ) for the solution of unsymmetric and square consistent linear system Ax = b.
#
# This method is described in
#
# G. L. G. Sleijpen and D. R. Fokkema
# BiCGstab(ℓ) for linear equations involving unsymmetric matrices with complex spectrum.
# Electronic Transactions on Numerical Analysis, 1, pp. 11--32, 1993.
#
# G. L. G. Sleijpen, H. A. van der Vorst and D. R. Fokkema
# BiCGstab(ℓ) and other hybrid Bi-CG methods.
# Numerical Algorithms, 7, pp. 75--109, 1994.
#
# Suguru Kurita, <skurita0208@gmail.com>
# Tokyo, October 2026.

export bicgstabl, bicgstabl!

"""
    (x, stats) = bicgstabl(A, b::AbstractVector{FC};
                           ℓ::Int=2, c::AbstractVector{FC}=b, M=I, N=I,
                           ldiv::Bool=false, atol::T=√eps(T),
                           rtol::T=√eps(T), itmax::Int=0,
                           timemax::Float64=Inf, verbose::Int=0, history::Bool=false,
                           callback=workspace->false, iostream::IO=kstdout)

`T` is an `AbstractFloat` such as `Float32`, `Float64` or `BigFloat`.
`FC` is `T` or `Complex{T}`.

    (x, stats) = bicgstabl(A, b, x0::AbstractVector; kwargs...)

BiCGstab(ℓ) can be warm-started from an initial guess `x0` where `kwargs` are the same keyword arguments as above.

Solve the square linear system Ax = b of size n using BiCGstab(ℓ).
BiCGstab(ℓ) requires two initial vectors `b` and `c`.
The relation `bᴴc ≠ 0` must be satisfied and by default `c = b`.

Each iteration performs ℓ steps of BiCG followed by a minimal residual step of degree ℓ,
whereas BiCGSTAB uses a minimal residual step of degree one after each BiCG step.
For ℓ = 1, BiCGstab(ℓ) is mathematically equivalent to BiCGSTAB.
Larger values of ℓ can improve robustness, in particular when `A` has eigenvalues with large imaginary parts,
for which BiCGSTAB tends to stagnate.
Each iteration requires 2ℓ operator-vector products with `A`.

BiCGstab(ℓ) stops when `itmax` iterations are reached or when `‖rₖ‖ ≤ atol + ‖r₀‖ * rtol`.

#### Interface

To easily switch between Krylov methods, use the generic interface [`krylov_solve`](@ref) with `method = :bicgstabl`.

For an in-place variant that reuses memory across solves, see [`bicgstabl!`](@ref).

#### Input arguments

* `A`: a linear operator that models a matrix of dimension `n`;
* `b`: a vector of length `n`.

#### Optional argument

* `x0`: a vector of length `n` that represents an initial guess of the solution `x`.

#### Keyword arguments

* `ℓ`: the number of BiCG steps performed in each iteration and the degree of the minimal residual polynomial. `ℓ` must be at least 1 and is capped at `n`. Typical choices are `ℓ = 2, 4, 6, 8`;
* `c`: the second initial vector of length `n` required by the Lanczos biorthogonalization process;
* `M`: linear operator that models a nonsingular matrix of size `n` used for left preconditioning;
* `N`: linear operator that models a nonsingular matrix of size `n` used for right preconditioning;
* `ldiv`: define whether the preconditioners use `ldiv!` or `mul!`;
* `atol`: absolute stopping tolerance based on the residual norm;
* `rtol`: relative stopping tolerance based on the residual norm;
* `itmax`: the maximum number of iterations. If `itmax=0`, the default number of iterations is set to `2n`;
* `timemax`: the time limit in seconds;
* `verbose`: additional details can be displayed if verbose mode is enabled (verbose > 0). Information will be displayed every `verbose` iterations;
* `history`: collect additional statistics on the run such as residual norms, or Aᴴ-residual norms;
* `callback`: function or functor called as `callback(workspace)` that returns `true` if the Krylov method should terminate, and `false` otherwise;
* `iostream`: stream to which output is logged.

#### Output arguments

* `x`: a dense vector of length `n`;
* `stats`: statistics collected on the run in a [`SimpleStats`](@ref) structure.

#### References

* G. L. G. Sleijpen and D. R. Fokkema, [*BiCGstab(ℓ) for linear equations involving unsymmetric matrices with complex spectrum*](https://www.emis.de/journals/ETNA/volumes/1993-2000/vol1/abstract_vol1_pp11-32.html), Electronic Transactions on Numerical Analysis, 1, pp. 11--32, 1993.
* G. L. G. Sleijpen, H. A. van der Vorst and D. R. Fokkema, [*BiCGstab(ℓ) and other hybrid Bi-CG methods*](https://doi.org/10.1007/BF02141261), Numerical Algorithms, 7, pp. 75--109, 1994.
"""
function bicgstabl end

"""
    workspace = bicgstabl!(workspace::BicgstablWorkspace, A, b; kwargs...)
    workspace = bicgstabl!(workspace::BicgstablWorkspace, A, b, x0; kwargs...)

In these calls, `kwargs` are keyword arguments of [`bicgstabl`](@ref).
The keyword argument `ℓ` is the only exception.
It is only supported by [`bicgstabl`](@ref) and is required to create a `BicgstablWorkspace`.
It cannot be changed later.

See [`BicgstablWorkspace`](@ref) for instructions on how to create the `workspace`.

For a more generic interface, you can use [`krylov_workspace`](@ref) with `method = :bicgstabl` to allocate the workspace,
and [`krylov_solve!`](@ref) to run the Krylov method in-place.
"""
function bicgstabl! end

def_args_bicgstabl = (:(A                    ),
                      :(b::AbstractVector{FC}))

def_optargs_bicgstabl = (:(x0::AbstractVector),)

def_kwargs_bicgstabl = (:(; c::AbstractVector{FC} = b    ),
                        :(; M = I                        ),
                        :(; N = I                        ),
                        :(; ldiv::Bool = false           ),
                        :(; atol::T = √eps(T)            ),
                        :(; rtol::T = √eps(T)            ),
                        :(; itmax::Int = 0               ),
                        :(; timemax::Float64 = Inf       ),
                        :(; verbose::Int = 0             ),
                        :(; history::Bool = false        ),
                        :(; callback = workspace -> false),
                        :(; iostream::IO = kstdout       ))

def_kwargs_workspace_bicgstabl = (:(; ℓ::Int = 2),)

def_kwargs_bicgstabl = extract_parameters.(def_kwargs_bicgstabl)
def_kwargs_workspace_bicgstabl = extract_parameters.(def_kwargs_workspace_bicgstabl)

args_bicgstabl = (:A, :b)
optargs_bicgstabl = (:x0,)
kwargs_bicgstabl = (:c, :M, :N, :ldiv, :atol, :rtol, :itmax, :timemax, :verbose, :history, :callback, :iostream)
kwargs_workspace_bicgstabl = (:ℓ,)

@eval begin
  function bicgstabl!(workspace :: BicgstablWorkspace{T,FC,S}, $(def_args_bicgstabl...); $(def_kwargs_bicgstabl...)) where {T <: AbstractFloat, FC <: FloatOrComplex{T}, S <: AbstractVector{FC}}
    # Timer
    start_time = time_ns()
    timemax_ns = 1e9 * timemax

    m, n = size(A)
    ℓ = workspace.ℓ
    (m == workspace.m && n == workspace.n) || error("(workspace.m, workspace.n) = ($(workspace.m), $(workspace.n)) is inconsistent with size(A) = ($m, $n)")
    m == n || error("System must be square")
    length(b) == m || error("Inconsistent problem size")
    (verbose > 0) && @printf(iostream, "BiCGstab(ℓ): system of size %d\n", n)
    ℓ ≥ 1 || error("ℓ must be at least 1")

    # Check M = Iₙ and N = Iₙ
    MisI = (M === I)
    NisI = (N === I)

    # Check type consistency.
    eltype(A) == FC || @warn "eltype(A) ≠ $FC. This could lead to errors or additional allocations in operator-vector products."
    ktypeof(b) == S || error("ktypeof(b) must be equal to $S")
    ktypeof(c) == S || error("ktypeof(c) must be equal to $S")

    # Set up workspace.
    allocate_if(!MisI, workspace, :q , S, workspace.x)  # The length of q is n
    allocate_if(!NisI, workspace, :yz, S, workspace.x)  # The length of yz is n
    Δx, x, q, yz, R, U = workspace.Δx, workspace.x, workspace.q, workspace.yz, workspace.R, workspace.U
    τ, σ, γ, γ′, γ″ = workspace.τ, workspace.σ, workspace.γ, workspace.γ′, workspace.γ″
    warm_start = workspace.warm_start

    stats = workspace.stats
    rNorms = stats.residuals
    reset!(stats)

    r₀ = MisI ? R[1] : q
    if warm_start
      kmul!(r₀, A, Δx)                      # r₀ = A * Δx
      kaxpby!(n, one(FC), b, -one(FC), r₀)  # r₀ = b - r₀
    else
      kcopy!(n, r₀, b)  # r₀ = b
    end
    MisI || mulorldiv!(R[1], M, r₀, ldiv)  # r₀ = M⁻¹r₀

    kfill!(x, zero(FC))
    kfill!(U[1], zero(FC))

    α = zero(FC)
    ω = one(FC)
    ρ₀ = one(FC)

    # Compute residual norm ‖r₀‖₂.
    rNorm = knorm(n, R[1])
    history && push!(rNorms, rNorm)
    if rNorm == 0
      stats.niter = 0
      stats.solved, stats.inconsistent = true, false
      stats.timer = start_time |> ktimer
      stats.status = "x is a zero-residual solution"
      warm_start && kaxpy!(n, one(FC), Δx, x)
      workspace.warm_start = false
      return workspace
    end

    iter = 0
    itmax == 0 && (itmax = 2 * n)

    ε = atol + rtol * rNorm
    (verbose > 0) && @printf(iostream, "%5s  %7s  %8s  %8s  %5s\n", "k", "‖rₖ‖", "|αₖ|", "|ωₖ|", "timer")
    kdisplay(iter, verbose) && @printf(iostream, "%5d  %7.1e  %8.1e  %8.1e  %.2fs\n", iter, rNorm, abs(α), abs(ω), start_time |> ktimer)

    if kdot(n, c, R[1]) == 0  # ⟨c, r₀⟩
      stats.niter = 0
      stats.solved, stats.inconsistent = false, false
      stats.timer = start_time |> ktimer
      stats.status = "Breakdown bᴴc = 0"
      warm_start && kaxpy!(n, one(FC), Δx, x)
      workspace.warm_start = false
      return workspace
    end

    # Stopping criterion.
    solved = rNorm ≤ ε
    tired = iter ≥ itmax
    breakdown = false
    breakdown_status = "unknown"
    status = "unknown"
    user_requested_exit = false
    overtimed = false

    while !(solved || tired || breakdown || user_requested_exit || overtimed)
      iter = iter + 1
      ρ₀ = -ω * ρ₀

      # Bi-CG part.
      for j = 1 : ℓ
        ρ₁ = kdot(n, c, R[j])  # ρ₁ = (c, rⱼ)
        if ρ₁ == 0
          breakdown = true
          breakdown_status = "breakdown ρ = 0"
          break
        end
        β = α * ρ₁ / ρ₀
        ρ₀ = ρ₁

        for i = 1 : j
          kaxpby!(n, one(FC), R[i], -β, U[i])  # uᵢ = rᵢ - β uᵢ
        end

        # uⱼ₊₁ = M⁻¹AN⁻¹uⱼ
        y = NisI ? U[j] : yz
        NisI || mulorldiv!(y, N, U[j], ldiv)
        if MisI
          kmul!(U[j+1], A, y)
        else
          kmul!(q, A, y)
          mulorldiv!(U[j+1], M, q, ldiv)
        end

        α = ρ₁ / kdot(n, c, U[j+1])  # α = ρ₁ / (c, uⱼ₊₁)
        if !isfinite(α)
          breakdown = true
          breakdown_status = "breakdown (c, uⱼ₊₁) = 0"
          break
        end

        for i = 1 : j
          kaxpy!(n, -α, U[i+1], R[i])  # rᵢ = rᵢ - α uᵢ₊₁
        end

        # rⱼ₊₁ = M⁻¹AN⁻¹rⱼ
        z = NisI ? R[j] : yz
        NisI || mulorldiv!(z, N, R[j], ldiv)
        if MisI
          kmul!(R[j+1], A, z)
        else
          kmul!(q, A, z)
          mulorldiv!(R[j+1], M, q, ldiv)
        end
        kaxpy!(n, α, U[1], x)   # x = x + α * u₁
      end
      if breakdown
        rNorm = knorm(n, R[1])
        history && push!(rNorms, rNorm)
        solved = rNorm ≤ ε
        break
      end

      # MR part (mod. Gram-Schmidt).
      pos = 0  # τᵢⱼ's position in the flattened array
      for j = 1 : ℓ
        for i = 1 : j-1
          pos += 1
          τ[pos] = kdot(n, R[i+1], R[j+1]) / σ[i]
          kaxpy!(n, -τ[pos], R[i+1], R[j+1])  # rⱼ = rⱼ - τᵢⱼ rᵢ
        end
        σ[j] = kdotr(n, R[j+1], R[j+1])
        if σ[j] == 0
          breakdown = true
          breakdown_status = "breakdown σⱼ = 0"
          break
        end
        γ′[j] = kdot(n, R[j+1], R[1]) / σ[j]
      end
      if breakdown
        rNorm = knorm(n, R[1])
        history && push!(rNorms, rNorm)
        solved = rNorm ≤ ε
        break
      end

      γ[ℓ] = γ′[ℓ]
      ω = γ[ℓ]

      # γ = T⁻¹ γ′
      for j = ℓ-1 : -1 : 1
        γ[j] = γ′[j]
        for i = j+1 : ℓ
          γ[j] = γ[j] - τ[div((i-1)*(i-2), 2) + j] * γ[i]  # γⱼ -= τⱼᵢ γᵢ
        end
      end

      # γ″ = T S γ
      for j = 1 : ℓ-1
        γ″[j] = γ[j+1]
        for i = j+1 : ℓ-1
          γ″[j] = γ″[j] + τ[div((i-1)*(i-2), 2) + j] * γ[i+1]  # γ″ⱼ += τⱼᵢ γᵢ₊₁
        end
      end

      kaxpy!(n, γ[1], R[1], x)
      kaxpy!(n, -γ′[ℓ], R[ℓ+1], R[1])
      kaxpy!(n, -γ[ℓ], U[ℓ+1], U[1])

      for j = 1 : ℓ-1
        kaxpy!(n, -γ[j], U[j+1], U[1])
        kaxpy!(n, γ″[j], R[j+1], x)
        kaxpy!(n, -γ′[j], R[j+1], R[1])
      end

      # Compute residual norm ‖rₖ‖.
      rNorm = knorm(n, R[1])
      history && push!(rNorms, rNorm)

      # Update stopping criterion.
      user_requested_exit = callback(workspace) :: Bool
      solved = rNorm ≤ ε
      tired = iter ≥ itmax
      if ω == 0
        breakdown = true
        breakdown_status = "breakdown ω = 0"
      end
      timer = time_ns() - start_time
      overtimed = timer > timemax_ns
      kdisplay(iter, verbose) && @printf(iostream, "%5d  %7.1e  %8.1e  %8.1e  %.2fs\n", iter, rNorm, abs(α), abs(ω), start_time |> ktimer)
    end
    (verbose > 0) && @printf(iostream, "\n")

    # Termination status
    tired               && (status = "maximum number of iterations exceeded")
    breakdown           && (status = breakdown_status)
    solved              && (status = "solution good enough given atol and rtol")
    user_requested_exit && (status = "user-requested exit")
    overtimed           && (status = "time limit exceeded")

    # x = N⁻¹x̂
    if !NisI
      mulorldiv!(yz, N, x, ldiv)
      kcopy!(n, x, yz)
    end

    warm_start && kaxpy!(n, one(FC), Δx, x)
    workspace.warm_start = false

    # Update stats
    stats.niter = iter
    stats.solved = solved
    stats.inconsistent = false
    stats.timer = start_time |> ktimer
    stats.status = status

    return workspace
  end
end
