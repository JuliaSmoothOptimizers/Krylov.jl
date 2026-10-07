# An implementation of MINARES.
#
# This method is described in
#
# A. Montoison, D. Orban and M. A. Saunders
# MinAres: An Iterative Solver for Symmetric Linear Systems
# SIAM Journal on Matrix Analysis and Applications, 46(1), pp. 509--529, 2025.
#
# `complex_symmetric = true` runs the same recurrence on the conjugate
# Bunse-Gerstner-Stover (BGS) process instead of the Hermitian Lanczos
# process, for complex symmetric A (transpose(A) = A); see
#
# S.-C. T. Choi, A. Montoison, D. Orban and M. A. Saunders, CS-MinAres:
# Normal-residual minimization for complex symmetric linear systems.
# Manuscript, 2026.
#
# Alexis Montoison, <alexis.montoison@polymtl.ca>
# Sou-Cheng T. Choi, <schoi32@illinoistech.edu>
# Palo Alto, March 2022 -- Chicago, October 2026.

export minares, minares!

"""
    (x, stats) = minares(A, b::AbstractVector{FC};
                         M=I, ldiv::Bool=false,
                         λ::T = zero(T), atol::T=√eps(T),
                         rtol::T=√eps(T), Artol::T = √eps(T),
                         itmax::Int=0, timemax::Float64=Inf,
                         verbose::Int=0, history::Bool=false,
                         callback=workspace->false, iostream::IO=kstdout,
                         complex_symmetric::Bool=false)

`T` is an `AbstractFloat` such as `Float32`, `Float64` or `BigFloat`.
`FC` is `T` or `Complex{T}`.

    (x, stats) = minares(A, b, x0::AbstractVector; kwargs...)

MINARES can be warm-started from an initial guess `x0` where `kwargs` are the same keyword arguments as above.

MINARES solves the Hermitian linear system Ax = b of size n.
MINARES minimizes ‖Arₖ‖₂ when M = Iₙ and ‖AMrₖ‖_M otherwise.
The estimates computed every iteration are ‖Mrₖ‖₂ and ‖AMrₖ‖_M.

If `complex_symmetric = true`, `A` is complex symmetric (`transpose(A) == A`)
instead of Hermitian, and MINARES runs on the conjugate Bunse-Gerstner-Stover
(BGS) process instead of the Hermitian Lanczos process. It then minimizes
‖Aᴴrₖ‖₂ over the conjugate Saunders trial space and estimates ‖rₖ‖₂ and
‖Aᴴrₖ‖₂; this is CS-MinAres. `complex_symmetric = true` requires
`FC <: Complex` and `M = I`. A real symmetric matrix is Hermitian, so use
`complex_symmetric = false` (the default) for it.

#### Interface

To easily switch between Krylov methods, use the generic interface [`krylov_solve`](@ref) with `method = :minares`.

For an in-place variant that reuses memory across solves, see [`minares!`](@ref).

#### Input arguments

* `A`: a linear operator that models a Hermitian (possibly singular) matrix of dimension `n`, or a complex symmetric matrix if `complex_symmetric = true`;
* `b`: a vector of length `n`.

#### Optional argument

* `x0`: a vector of length `n` that represents an initial guess of the solution `x`.

#### Keyword arguments

* `M`: linear operator that models a Hermitian positive-definite matrix of size `n` used for centered preconditioning; not supported if `complex_symmetric = true`;
* `ldiv`: define whether the preconditioner uses `ldiv!` or `mul!`;
* `λ`: regularization parameter;
* `atol`: absolute stopping tolerance based on the residual norm;
* `rtol`: relative stopping tolerance based on the residual norm;
* `Artol`: relative stopping tolerance based on the Aᴴ-residual norm;
* `itmax`: the maximum number of iterations. If `itmax=0`, the default number of iterations is set to `2n`;
* `timemax`: the time limit in seconds;
* `verbose`: additional details can be displayed if verbose mode is enabled (verbose > 0). Information will be displayed every `verbose` iterations;
* `history`: collect additional statistics on the run such as residual norms, or Aᴴ-residual norms;
* `callback`: function or functor called as `callback(workspace)` that returns `true` if the Krylov method should terminate, and `false` otherwise;
* `iostream`: stream to which output is logged;
* `complex_symmetric`: run MINARES on the conjugate BGS process for a complex symmetric `A` instead of the Hermitian Lanczos process.

#### Output arguments

* `x`: a dense vector of length `n`;
* `stats`: statistics collected on the run in a [`SimpleStats`](@ref) structure.

#### References

* A. Montoison, D. Orban and M. A. Saunders, [*MinAres: An Iterative Solver for Symmetric Linear Systems*](https://doi.org/10.1137/23M1605454), SIAM Journal on Matrix Analysis and Applications, 46(1), pp. 509--529, 2025.
* S.-C. T. Choi, [*Minimal residual methods for complex symmetric, skew symmetric and skew Hermitian systems*](https://arxiv.org/abs/1304.6782), Technical Report ANL/MCS-P3028-0812, Computation Institute, University of Chicago, 2013, for `complex_symmetric = true`.
* S.-C. T. Choi, A. Montoison, D. Orban and M. A. Saunders, *CS-MinAres: Normal-Residual Minimization for Complex Symmetric Linear Systems*, manuscript, 2026, for `complex_symmetric = true`.
"""
function minares end

"""
    workspace = minares!(workspace::MinaresWorkspace, A, b; kwargs...)
    workspace = minares!(workspace::MinaresWorkspace, A, b, x0; kwargs...)

In these calls, `kwargs` are keyword arguments of [`minares`](@ref).

See [`MinaresWorkspace`](@ref) for instructions on how to create the `workspace`.

For a more generic interface, you can use [`krylov_workspace`](@ref) with `method = :minares` to allocate the workspace,
and [`krylov_solve!`](@ref) to run the Krylov method in-place.
"""
function minares! end

def_args_minares = (:(A                    ),
                    :(b::AbstractVector{FC}))

def_optargs_minares = (:(x0::AbstractVector),)

def_kwargs_minares = (:(; M = I                        ),
                      :(; ldiv::Bool = false           ),
                      :(; λ::T = zero(T)               ),
                      :(; atol::T = √eps(T)            ),
                      :(; rtol::T = √eps(T)            ),
                      :(; Artol::T = √eps(T)           ),
                      :(; itmax::Int = 0               ),
                      :(; timemax::Float64 = Inf       ),
                      :(; verbose::Int = 0             ),
                      :(; history::Bool = false        ),
                      :(; callback = workspace -> false),
                      :(; iostream::IO = kstdout       ),
                      :(; complex_symmetric::Bool = false))

def_kwargs_minares = extract_parameters.(def_kwargs_minares)

args_minares = (:A, :b)
optargs_minares = (:x0,)
kwargs_minares = (:M, :ldiv, :λ, :atol, :rtol, :Artol, :itmax, :timemax, :verbose, :history, :callback, :iostream, :complex_symmetric)

@eval begin
  function minares!(workspace :: MinaresWorkspace{T,FC,S}, $(def_args_minares...); $(def_kwargs_minares...)) where {T <: AbstractFloat, FC <: FloatOrComplex{T}, S <: AbstractVector{FC}}
    if complex_symmetric
      FC <: Complex || error("complex_symmetric = true requires a complex element type")
      M === I || throw(ArgumentError("complex_symmetric = true does not support preconditioning"))
      return _minares!(ComplexSymmetricStructure(), workspace, A, b, M, λ, atol, rtol, Artol,
                       itmax, timemax, verbose, history, callback, iostream)
    end
    return _minares!(HermitianStructure(), workspace, A, b, M, λ, atol, rtol, Artol,
                     itmax, timemax, verbose, history, callback, iostream)
  end
end

# Shared kernel, as in _minres_qlp!: the structures differ only at the
# structure hooks and at conj() calls, which are the identity on the real
# scalars of the Hermitian structure.
function _minares!(structure :: LanczosStructure, workspace :: MinaresWorkspace{T,FC,S}, A, b :: AbstractVector{FC}, M,
                   λ :: T, atol :: T, rtol :: T, Artol :: T, itmax :: Int, timemax :: Float64,
                   verbose :: Int, history :: Bool, callback, iostream :: IO) where {T <: AbstractFloat, FC <: FloatOrComplex{T}, S <: AbstractVector{FC}}

    Tr = lanczos_scalar(structure, T, FC)

    # Timer
    start_time = time_ns()
    timemax_ns = 1e9 * timemax

    n, m = size(A)
    (m == workspace.m && n == workspace.n) || error("(workspace.m, workspace.n) = ($(workspace.m), $(workspace.n)) is inconsistent with size(A) = ($m, $n)")
    m == n || error("System must be square")
    length(b) == m || error("Inconsistent problem size")
    (verbose > 0) && @printf(iostream, "MINARES: system of size %d\n", n)

    # Tests M = Iₙ
    MisI = (M === I)
    !MisI && error("Preconditioners are not yet supported")

    # Check type consistency
    eltype(A) == FC || error("eltype(A) ≠ $FC")
    ktypeof(b) == S || error("ktypeof(b) must be equal to $S")

    # Set up workspace.
    Δx, vₖ, vₖ₊₁, x, q, stats = workspace.Δx, workspace.vₖ, workspace.vₖ₊₁, workspace.x, workspace.q, workspace.stats
    wₖ₋₂, wₖ₋₁ = workspace.wₖ₋₂, workspace.wₖ₋₁
    dₖ₋₂, dₖ₋₁ = workspace.dₖ₋₂, workspace.dₖ₋₁
    warm_start = workspace.warm_start
    rNorms, ArNorms = stats.residuals, stats.Aresiduals
    reset!(stats)

    iter = 0
    itmax == 0 && (itmax = 2*n)

    kfill!(x, zero(FC))  # x₀

    # Initialize the Lanczos process.
    # β₁v₁ = r₀
    if warm_start
      kmul!(vₖ, A, Δx)  # r₀ = b - Ax₀
      (λ ≠ 0) && kaxpy!(n, λ, Δx, vₖ)
      kaxpby!(n, one(FC), b, -one(FC), vₖ)
    else
      kcopy!(n, vₖ, b)  # r₀ = b
    end
    βₖ = knorm(n, vₖ)  # β₁ = ‖v₁‖
    if βₖ ≠ 0
      kdiv!(n, vₖ, βₖ)
    end
    β₁ = βₖ

    # β₂v₂ = (A + λI)v₁ - α₁v₁, with v̄₁ in place of v₁ on the right for the complex-symmetric structure
    lanczos_mul!(structure, n, vₖ₊₁, A, vₖ, λ)
    αₖ = lanczos_dot(structure, n, vₖ, vₖ₊₁)   # α₁ = (vₖ)ᴴ(A + λI)vₖ, or (vₖ)ᴴ(A + λI)v̄ₖ
    kaxpy!(n, -αₖ, vₖ, vₖ₊₁)
    βₖ₊₁ = knorm(n, vₖ₊₁)    # β₂ = ‖v₂‖
    if βₖ₊₁ ≠ 0
      kdiv!(n, vₖ₊₁, βₖ₊₁)
    end

    ξₖ₋₁ = zero(Tr)
    τₖ₋₂ = τₖ₋₁ = τₖ = zero(Tr)
    θbarₖ₋₂ = zero(Tr)
    ψbisₖ₋₂ = ψbarₖ₋₁ = zero(Tr)
    πₖ₋₂ = πₖ₋₁ = πₖ = zero(Tr)
    χbarₖ = zero(Tr)
    ζbisₖ = ζbarₖ₊₁ = γbarₖ = zero(Tr)
    λbarₖ = γₖ₋₁ = zero(Tr)
    c̃₂ₖ₋₄ = zero(T); s̃₂ₖ₋₄ = zero(Tr)
    c̃₂ₖ₋₃ = zero(T); s̃₂ₖ₋₃ = zero(Tr)
    c̃₂ₖ₋₂ = zero(T); s̃₂ₖ₋₂ = zero(Tr)
    c̃₂ₖ₋₁ = zero(T); s̃₂ₖ₋₁ = zero(Tr)
    c̃₂ₖ   = zero(T); s̃₂ₖ   = zero(Tr)
    kfill!(wₖ₋₂, zero(FC))  # Column k-2 of Wₖ = Vₖ(Rₖ)⁻¹
    kfill!(wₖ₋₁, zero(FC))  # Column k-1 of Wₖ = Vₖ(Rₖ)⁻¹
    kfill!(dₖ₋₂, zero(FC))  # Column k-2 of Dₖ = Wₖ(Uₖ)⁻¹
    kfill!(dₖ₋₁, zero(FC))  # Column k-1 of Dₖ = Wₖ(Uₖ)⁻¹
    β₁α₁ = βₖ * αₖ           # Variable used to update zₖ
    β₁β₂ = βₖ * βₖ₊₁ * one(Tr)  # Variable used to update zₖ
    ϵₖ₋₂ = ϵₖ₋₁ = zero(Tr)
    ℓ = itmax + 2

    rNorm = β₁
    ε = atol + rtol * rNorm
    history && push!(rNorms, rNorm)

    ArNorm = sqrt(abs2(β₁α₁) + abs2(β₁β₂))
    κ = atol + Artol * ArNorm
    history && push!(ArNorms, ArNorm)

    if rNorm == 0
      stats.niter = 0
      stats.solved, stats.inconsistent = true, false
      stats.timer = start_time |> ktimer
      stats.status = "x is a zero-residual solution"
      warm_start && kaxpy!(n, one(FC), Δx, x)
      workspace.warm_start = false
      return workspace
    end

    (verbose > 0) && @printf(iostream, "%5s  %7s  %7s  %7s  %8s  %5s\n", "k", "‖rₖ‖", "‖Arₖ‖", "βₖ₊₁", "ζₖ", "timer")
    kdisplay(iter, verbose) && @printf(iostream, "%5d  %7.1e  %7.1e  %7.1e  %8s  %.2fs\n", iter, rNorm, ArNorm, β₁, " ✗ ✗ ✗ ✗", start_time |> ktimer)

    # Tolerance for breakdown detection.
    btol = eps(T)^(3//4)

    # Stopping criterion.
    solved = (rNorm ≤ ε) || (ArNorm ≤ κ)
    breakdown = false
    tired = iter ≥ itmax
    user_requested_exit = false
    status = "unknown"
    overtimed = false

    while !(solved || tired || breakdown || user_requested_exit || overtimed)
      # Update iteration index.
      iter = iter + 1

      # Update the QR factorization Tₖ₊₁.ₖ = Qₖ [ Rₖ ].
      #                                         [ Oᵀ ]
      #
      # [ α₁ β₂ 0  •  •  •   0  ]      [ λ₁ γ₁ ϵ₁ 0  •  •  0  ]
      # [ β₂ α₂ β₃ •         •  ]      [ 0  λ₂ γ₂ •  •     •  ]
      # [ 0  •  •  •  •      •  ]      [ •  •  λ₃ •  •  •  •  ]
      # [ •  •  •  •  •  •   •  ] = Qₖ [ •     •  •  •  •  0  ]
      # [ •     •  •  •  •   0  ]      [ •        •  •  • ϵₖ₋₂]
      # [ •        •  •  •   βₖ ]      [ •           •  • γₖ₋₁]
      # [ •           •  βₖ  αₖ ]      [ 0  •  •  •  •  0  λₖ ]
      # [ 0  •  •  •  •  0  βₖ₊₁]      [ 0  •  •  •  •  •  0  ]

      (iter == 1) && (λbarₖ = αₖ)
      (iter == 1) && (γbarₖ = βₖ₊₁)

      # Compute the Givens reflection Qₖ.ₖ₊₁
      # [ cₖ  sₖ ] [ λbarₖ γbarₖ   0  ] = [ λₖ    γₖ      ϵₖ   ]
      # [ sₖ -cₖ ] [ βₖ₊₁  αₖ₊₁  βₖ₊₂ ]   [ 0  λbarₖ₊₁ γbarₖ₊₁ ]
      (cₖ, sₖ, λₖ) = sym_givens(λbarₖ, βₖ₊₁)

      # Compute the direction wₖ, the last column of Wₖ = Vₖ(Rₖ)⁻¹ ⟷ (Rₖ)ᵀ(Wₖ)ᵀ = (Vₖ)ᵀ.
      # The complex-symmetric structure uses conj(Vₖ) in place of Vₖ.
      trial_conj!(structure, n, vₖ)
      # w₁ = v₁ / λ₁
      if iter == 1
        wₖ = wₖ₋₁
        kdivcopy!(n, wₖ, vₖ, λₖ)
      end
      # w₂ = (v₂ - γ₁w₁) / λ₂
      if iter == 2
        wₖ = wₖ₋₂
        kaxpy!(n, -γₖ₋₁, wₖ₋₁, wₖ)
        kaxpy!(n, one(T), vₖ, wₖ)
        kdiv!(n, wₖ, λₖ)
      end
      # wₖ = (vₖ - γₖ₋₁wₖ₋₁ - ϵₖ₋₂wₖ₋₂) / λₖ
      if iter ≥ 3
        kscal!(n, -ϵₖ₋₂, wₖ₋₂)
        wₖ = wₖ₋₂
        kaxpy!(n, -γₖ₋₁, wₖ₋₁, wₖ)
        kaxpy!(n, one(T), vₖ, wₖ)
        kdiv!(n, wₖ, λₖ)
      end
      trial_conj!(structure, n, vₖ)

      # Continue the Lanczos process.
      # M(A + λI)Vₖ₊₁ = Vₖ₊₂Tₖ₊₂.ₖ₊₁
      # βₖ₊₂vₖ₊₂ = M(A + λI)vₖ₊₁ - αₖ₊₁vₖ₊₁ - βₖ₊₁vₖ
      if iter ≤ ℓ-1
        trial_conj!(structure, n, vₖ₊₁)  # v̄ₖ₊₁ for the complex-symmetric structure
        kmul!(q, A, vₖ₊₁)  # q ← Avₖ
        kaxpby!(n, one(T), q, -βₖ₊₁, vₖ)  # Forms vₖ₊₂ : vₖ ← Avₖ₊₁ - βₖ₊₁vₖ
        if λ ≠ 0
          kaxpy!(n, λ, vₖ₊₁, vₖ)          # vₖ ← vₖ + λvₖ₊₁
        end
        trial_conj!(structure, n, vₖ₊₁)
        αₖ₊₁ = lanczos_dot(structure, n, vₖ₊₁, vₖ)  # αₖ₊₁ = (vₖ₊₁)ᴴvₖ: the basis vector goes first
        kaxpy!(n, -αₖ₊₁, vₖ₊₁, vₖ)        # vₖ ← vₖ - αₖ₊₁vₖ₊₁
        βₖ₊₂ = knorm(n, vₖ)               # βₖ₊₂ = ‖vₖ₊₂‖
      
        # Detection of early termination
        if βₖ₊₂ ≤ btol
          ℓ = iter + 1
        else
          kdiv!(n, vₖ, βₖ₊₂)
        end
      end

      # Apply the Givens reflection Qₖ.ₖ₊₁
      if iter ≤ ℓ-2
        ϵₖ      =  sₖ * βₖ₊₂
        γbarₖ₊₁ = -cₖ * βₖ₊₂
      end
      if iter ≤ ℓ-1
        γₖ      = cₖ * γbarₖ + sₖ * αₖ₊₁
        λbarₖ₊₁ = conj(sₖ) * γbarₖ - cₖ * αₖ₊₁
      end

      # Update the QR factorization Nₖ = Q̃ₖ [ Uₖ ].
      #                                     [ Oᵀ ]
      # For the complex-symmetric structure, Nₖ = conj(Tₖ₊₂.ₖ₊₁)Qₖ[I; 0] has entries conj(λⱼ), conj(γⱼ), conj(ϵⱼ).
      #
      # [ λ₁  0   •   •   •    •   0  ]      [ μ₁  ϕ₁  ρ₁  0   •    •   0    ]
      # [ γ₁  λ₂  •                •  ]      [ 0   μ₂  ϕ₂  •   •        •    ]
      # [ ϵ₁  γ₂  λ₃  •            •  ]      [ •   •   μ₃  •   •    •   •    ]
      # [ 0   •   •   •   •        •  ]      [ •       •   •   •    •   0    ]
      # [ •   •   •   •   •    •   •  ] = Q̃ₖ [ •           •  μₖ₋₂ ϕₖ₋₂ ρₖ₋₂ ]
      # [ •       •   •   •    •   0  ]      [ •               •   μₖ₋₁ ϕₖ₋₁ ]
      # [ •           •  ϵₖ₋₂ γₖ₋₁ λₖ ]      [ •                    •   μₖ   ]
      # [ •               •   ϵₖ₋₁ γₖ ]      [ 0   •   •   •   •    •   0    ]
      # [ 0  •    •   •   •    0   ϵₖ ]      [ 0   •   •   •   •    •   0    ]

      # If k = 1, we don't have any previous reflection.
      # If k = 2, we apply the reflections Q̃ₖ₊₁.ₖ₋₁ and Q̃ₖ.ₖ₋₁.
      # If k ≥ 3, we apply the reflections Q̃ₖ.ₖ₋₁, Q̃ₖ₊₁.ₖ₋₁ and Q̃ₖ.ₖ₋₂.

      if iter ≥ 3
        # Apply previous reflection Q̃ₖ.ₖ₋₂
        # [ c̃₂ₖ₋₄      s̃₂ₖ₋₄ ] [   0   ]   [ ρₖ₋₂  ] 
        # [        1         ] [   0   ] = [   0   ]
        # [ s̃₂ₖ₋₄     -c̃₂ₖ₋₄ ] [   λₖ  ]   [ λhatₖ ]
        ρₖ₋₂  =  s̃₂ₖ₋₄ * conj(λₖ)
        λhatₖ = -c̃₂ₖ₋₄ * conj(λₖ)
      end
   
      iter == 2 && (λhatₖ = conj(λₖ))
      if iter ≥ 2
        # Apply previous reflection Q̃ₖ.ₖ₋₁
        # [ c̃₂ₖ₋₃   s̃₂ₖ₋₃    ] [   0   ]   [ ϕbarₖ₋₁ ]
        # [ s̃₂ₖ₋₃  -c̃₂ₖ₋₃    ] [ λhatₖ ] = [  μbarₖ  ]
        # [                1 ] [   γₖ  ]   [    γₖ   ]
        ϕbarₖ₋₁ =  s̃₂ₖ₋₃ * λhatₖ
        μbarₖ   = -c̃₂ₖ₋₃ * λhatₖ

        if iter ≤ ℓ-1
          # Apply previous reflection Q̃ₖ₊₁.ₖ₋₁
          # [ c̃₂ₖ₋₂      s̃₂ₖ₋₂ ] [ ϕbarₖ₋₁ ]   [  ϕₖ₋₁  ]
          # [        1         ] [  μbarₖ  ] = [  μbarₖ ]
          # [ s̃₂ₖ₋₂     -c̃₂ₖ₋₂ ] [    γₖ   ]   [  γhatₖ ]
          ϕₖ₋₁  = c̃₂ₖ₋₂ * ϕbarₖ₋₁ + s̃₂ₖ₋₂ * conj(γₖ)
          γhatₖ = conj(s̃₂ₖ₋₂) * ϕbarₖ₋₁ - c̃₂ₖ₋₂ * conj(γₖ)
        else
          ϕₖ₋₁ = ϕbarₖ₋₁
        end
      end

      iter == 1 && (μbarₖ = conj(λₖ))
      iter == 1 && (γhatₖ = conj(γₖ))
      if iter ≤ ℓ-1
        # Compute and apply current Givens reflection Q̃ₖ₊₁.ₖ
        # [ c̃₂ₖ₋₁   s̃₂ₖ₋₁    ] [ μbarₖ ] = [ μbisₖ ]
        # [ s̃₂ₖ₋₁  -c̃₂ₖ₋₁    ] [ γhatₖ ]   [   0   ]
        # [                1 ] [  ϵₖ   ]   [   ϵₖ  ]
        (c̃₂ₖ₋₁, s̃₂ₖ₋₁, μbisₖ) = sym_givens(μbarₖ, γhatₖ)
      else
        μbisₖ = μbarₖ
      end

      if iter ≤ ℓ-2
        # Compute and apply current Givens reflection Q̃ₖ₊₂.ₖ
        # [ c̃₂ₖ      s̃₂ₖ ] [ μbisₖ ] = [ μₖ ] 
        # [      1       ] [   0   ]   [ 0  ]
        # [ s̃₂ₖ     -c̃₂ₖ ] [   ϵₖ  ]   [ 0  ]
        (c̃₂ₖ, s̃₂ₖ, μₖ) = sym_givens(μbisₖ, conj(ϵₖ))
      else
        μₖ = μbisₖ
      end

      # Update zₖ = (Q̃ₖ)ᴴ(β₁ᾱ₁e₁ + β₁β₂e₂)
      iter == 1 && (ζbisₖ   = conj(β₁α₁))
      iter == 1 && (ζbarₖ₊₁ = β₁β₂)

      if iter ≤ ℓ-1
        # [ c̃₂ₖ₋₁   s̃₂ₖ₋₁    ] [  ζbisₖ  ] = [ ζringₖ  ]
        # [ s̃₂ₖ₋₁  -c̃₂ₖ₋₁    ] [ ζbarₖ₊₁ ]   [ ζbisₖ₊₁ ]
        # [                1 ] [    0    ]   [    0    ]
        ζringₖ  = c̃₂ₖ₋₁ * ζbisₖ + s̃₂ₖ₋₁ * ζbarₖ₊₁
        ζbisₖ₊₁ = conj(s̃₂ₖ₋₁) * ζbisₖ - c̃₂ₖ₋₁ * ζbarₖ₊₁
      else
        ζringₖ = ζbisₖ
      end

      if iter ≤ ℓ-2
        # [ c̃₂ₖ      s̃₂ₖ ] [ ζringₖ  ] = [   ζₖ    ]
        # [      1       ] [ ζbisₖ₊₁ ]   [ ζbisₖ₊₁ ]
        # [ s̃₂ₖ     -c̃₂ₖ ] [    0    ]   [ ζbarₖ₊₂ ]
        ζₖ      = c̃₂ₖ * ζringₖ
        ζbarₖ₊₂ = conj(s̃₂ₖ) * ζringₖ
      else
        ζₖ = ζringₖ
      end

      # Compute the direction dₖ, the last column of Dₖ = Wₖ(Uₖ)⁻¹ ⟷ (Uₖ)ᵀ(Dₖ)ᵀ = (Wₖ)ᵀ.
      # d₁ = w₁ / μ₁
      if iter == 1
        dₖ = dₖ₋₁
        kdivcopy!(n, dₖ, wₖ, μₖ)
      end
      # d₂ = (w₂ - ϕ₁d₁) / μ₂
      if iter == 2
        dₖ = dₖ₋₂
        kaxpy!(n, -ϕₖ₋₁, dₖ₋₁, dₖ)
        kaxpy!(n, one(T), wₖ, dₖ)
        kdiv!(n, dₖ, μₖ)
      end
      # dₖ = (wₖ - ϕₖ₋₁dₖ₋₁ - ρₖ₋₂dₖ₋₂) / μₖ
      if iter ≥ 3
        kscal!(n, -ρₖ₋₂, dₖ₋₂)
        dₖ = dₖ₋₂
        kaxpy!(n, -ϕₖ₋₁, dₖ₋₁, dₖ)
        kaxpy!(n, one(T), wₖ, dₖ)
        kdiv!(n, dₖ, μₖ)
      end

      # x = Vₖyₖ = Dₖzₖ = x₋₁ + ζₖdₖ
      kaxpy!(n, ζₖ, dₖ, x)

      # Update ‖Arₖ‖ estimate
      (iter ≤ ℓ-2)  && (ArNorm = sqrt(abs2(ζbisₖ₊₁) + abs2(ζbarₖ₊₂)))
      (iter == ℓ-1) && (ArNorm = abs(ζbisₖ₊₁))
      (iter == ℓ)   && (ArNorm = zero(T))
      history && push!(ArNorms, ArNorm)

      # Update the LQ factorization Uₖ = L̂ₖP̂ₖ.
      # A row [a b] is reduced by (c, s, ρ) = sym_givens(conj(a), conj(b)): [a b][c s; conj(s) -c] = [conj(ρ) 0].
      #
      # [ μ₁  ϕ₁  ρ₁  0   •    •   0    ]   [ ψ₁   0    •    •     •      •      0  ]
      # [ 0   μ₂  ϕ₂  •   •        •    ]   [ θ₁   ψ₂   •                        •  ] 
      # [ •   •   μ₃  •   •    •   •    ]   [ ω₁   θ₂   ψ₃   •                   •  ]
      # [ •       •   •   •    •   0    ] = [ 0    •    •    •     •             •  ] Pₖ
      # [ •           •  μₖ₋₂ ϕₖ₋₂ ρₖ₋₂ ]   [ •    •    •    •   ψₖ₋₂     •      •  ]
      # [ •                   μₖ₋₁ ϕₖ₋₁ ]   [ •         •    •   θₖ₋₂  ψbisₖ₋₁   0  ]
      # [ •                        μₖ   ]   [ 0    •    •    0   ωₖ₋₂  θbarₖ₋₁ ψbarₖ]

      if iter == 1
        ψbarₖ = μₖ
      elseif iter == 2
        # [ ψbar₁  ϕ₁ ] [ ĉ₁   ŝ₁ ] = [ ψbis₁    0   ]
        # [   0    μ₂ ] [ ŝ₁  -ĉ₁ ]   [ θbar₁  ψbar₂ ]
        (ĉ₂ₖ₋₃, ŝ₂ₖ₋₃, ψbisₖ₋₁) = sym_givens(conj(ψbarₖ₋₁), conj(ϕₖ₋₁))
        ψbisₖ₋₁ = conj(ψbisₖ₋₁)
        θbarₖ₋₁ =  conj(ŝ₂ₖ₋₃) * μₖ
        ψbarₖ   = -ĉ₂ₖ₋₃ * μₖ
      else
        # [ ψbisₖ₋₂   0     ρₖ₋₂ ] [ ĉ₂ₖ₋₄      ŝ₂ₖ₋₄ ]   [ ψₖ₋₂     0     0  ]
        # [ θbarₖ₋₂ ψbarₖ₋₁ ϕₖ₋₁ ] [        1         ] = [ θₖ₋₂  ψbarₖ₋₁  δₖ ]
        # [   0       0      μₖ  ] [ ŝ₂ₖ₋₄     -ĉ₂ₖ₋₄ ]   [ ωₖ₋₂     0     ηₖ ]
        (ĉ₂ₖ₋₄, ŝ₂ₖ₋₄, ψₖ₋₂) = sym_givens(conj(ψbisₖ₋₂), conj(ρₖ₋₂))
        ψₖ₋₂ = conj(ψₖ₋₂)
        θₖ₋₂ =  ĉ₂ₖ₋₄ * θbarₖ₋₂ + conj(ŝ₂ₖ₋₄) * ϕₖ₋₁
        δₖ   =  ŝ₂ₖ₋₄ * θbarₖ₋₂ - ĉ₂ₖ₋₄ * ϕₖ₋₁
        ωₖ₋₂ =  conj(ŝ₂ₖ₋₄) * μₖ
        ηₖ   = -ĉ₂ₖ₋₄ * μₖ

        # [ ψₖ₋₂     0     0  ] [ 1                ]   [ ψₖ₋₂    0        0   ]
        # [ θₖ₋₂  ψbarₖ₋₁  δₖ ] [    ĉ₂ₖ₋₃   ŝ₂ₖ₋₃ ] = [ θₖ₋₂  ψbisₖ₋₁    0   ]
        # [ ωₖ₋₂     0     ηₖ ] [    ŝ₂ₖ₋₃  -ĉ₂ₖ₋₃ ]   [ ωₖ₋₂  θbarₖ₋₁  ψbarₖ ]
        (ĉ₂ₖ₋₃, ŝ₂ₖ₋₃, ψbisₖ₋₁) = sym_givens(conj(ψbarₖ₋₁), conj(δₖ))
        ψbisₖ₋₁ = conj(ψbisₖ₋₁)
        θbarₖ₋₁ =  conj(ŝ₂ₖ₋₃) * ηₖ
        ψbarₖ   = -ĉ₂ₖ₋₃ * ηₖ
      end

      # Solve L̂ₖtₖ = zₖ
      # [ ψ₁   0    •    •     •      •      0  ] [τ₁]   [ζ₁]
      # [ θ₁   ψ₂   •                        •  ] [τ₂]   [ζ₂]
      # [ ω₁   θ₂   ψ₃   •                   •  ] [τ₃]   [ζ₃]
      # [ 0    •    •    •     •             •  ] [••] = [••]
      # [ •    •    •    •   ψₖ₋₂     •      •  ] [••]   [••]
      # [ •         •    •   θₖ₋₂  ψbisₖ₋₁   0  ] [••]   [••]
      # [ 0    •    •    0   ωₖ₋₂  θbarₖ₋₁ ψbarₖ] [τₖ]   [ζₖ]
      if iter == 1
        τₖ = ζₖ / ψbarₖ
      elseif iter == 2
        τₖ₋₁ = τₖ
        τₖ₋₁ = τₖ₋₁ * ψbarₖ₋₁ / ψbisₖ₋₁
        ξₖ   = ζₖ
        τₖ   = (ξₖ - θbarₖ₋₁ * τₖ₋₁) / ψbarₖ
      else
        τₖ₋₂ = τₖ₋₁
        τₖ₋₂ = τₖ₋₂ * ψbisₖ₋₂ / ψₖ₋₂
        τₖ₋₁ = (ξₖ₋₁ - θₖ₋₂ * τₖ₋₂) / ψbisₖ₋₁
        ξₖ   = ζₖ - ωₖ₋₂ * τₖ₋₂
        τₖ   = (ξₖ - θbarₖ₋₁ * τₖ₋₁) / ψbarₖ
      end

      # The components of (Qₖ)ᴴβ₁e₁ are (χ₁, ..., χₖ, χbarₖ₊₁)
      (iter == 1) && (χbarₖ = β₁)

      # [ cₖ  sₖ ] [ χbarₖ ] = [    χₖ   ]
      # [ sₖ -cₖ ] [   0   ]   [ χbarₖ₊₁ ]
      χₖ      = cₖ * χbarₖ
      χbarₖ₊₁ = conj(sₖ) * χbarₖ

      # Update pₖ₊₁ = [ P̂ₖ  0 ](Qₖ)ᴴβ₁e₁
      #               [ 0   1 ]
      if iter == 1
        πₖ = χₖ
      elseif iter == 2
        # [ ĉ₁   ŝ₁ ] [ π₁ ] = [ π₁ ]
        # [ ŝ₁  -ĉ₁ ] [ χ₂ ]   [ π₂ ]
        πaux₋₁ = πₖ₋₁
        πₖ₋₁ = ĉ₂ₖ₋₃ * πaux₋₁ + ŝ₂ₖ₋₃ * χₖ
        πₖ   = conj(ŝ₂ₖ₋₃) * πaux₋₁ - ĉ₂ₖ₋₃ * χₖ
      else
        # [ ĉ₂ₖ₋₄      ŝ₂ₖ₋₄ ] [ πₖ₋₂ ]   [ πₖ₋₂ ]
        # [        1         ] [ πₖ₋₁ ] = [ πₖ₋₁ ]
        # [ ŝ₂ₖ₋₄     -ĉ₂ₖ₋₄ ] [  χₖ  ]   [  πₖ  ]
        πaux₋₂ = πₖ₋₂
        πₖ₋₂ = ĉ₂ₖ₋₄ * πaux₋₂ + ŝ₂ₖ₋₄ * χₖ
        πₖ   = conj(ŝ₂ₖ₋₄) * πaux₋₂ - ĉ₂ₖ₋₄ * χₖ

        # [ 1                ] [ πₖ₋₂ ]   [ πₖ₋₂ ]
        # [    ĉ₂ₖ₋₃   ŝ₂ₖ₋₃ ] [ πₖ₋₁ ] = [ πₖ₋₁ ]
        # [    ŝ₂ₖ₋₃  -ĉ₂ₖ₋₃ ] [  πₖ  ]   [  πₖ  ]
        πaux₋₁ = πₖ₋₁
        πₖ₋₁ = ĉ₂ₖ₋₃ * πaux₋₁ + ŝ₂ₖ₋₃ * πₖ
        πₖ   = conj(ŝ₂ₖ₋₃) * πaux₋₁ - ĉ₂ₖ₋₃ * πₖ
      end
      πₖ₊₁ = χbarₖ₊₁

      # Update ‖rₖ‖ estimate
      # ‖ rₖ ‖ = √((πₖ₋₁ - τₖ₋₁)² + (πₖ - τₖ)² + (πₖ₊₁)²)
      if iter == 1
        rNorm = sqrt(abs2(πₖ - τₖ) + abs2(πₖ₊₁))
      else
        rNorm = sqrt(abs2(πₖ₋₁ - τₖ₋₁) + abs2(πₖ - τₖ) + abs2(πₖ₊₁))
      end
      history && push!(rNorms, rNorm)

      # Update stopping criterion.
      breakdown = βₖ₊₁ ≤ btol
      solved = (rNorm ≤ ε) || (ArNorm ≤ κ)
      tired = iter ≥ itmax
      timer = time_ns() - start_time
      overtimed = timer > timemax_ns
      user_requested_exit = callback(workspace) :: Bool

      # Update variables
      @kswap!(vₖ, vₖ₊₁)
      if iter ≥ 2
        @kswap!(wₖ₋₂, wₖ₋₁)
        @kswap!(dₖ₋₂, dₖ₋₁)
        ϵₖ₋₂ = ϵₖ₋₁
        c̃₂ₖ₋₄ = c̃₂ₖ₋₂
        s̃₂ₖ₋₄ = s̃₂ₖ₋₂
        ξₖ₋₁ = ξₖ
        ψbisₖ₋₂ = ψbisₖ₋₁
        θbarₖ₋₂ = θbarₖ₋₁
        πₖ₋₂ = πₖ₋₁
      end
      c̃₂ₖ₋₃ = c̃₂ₖ₋₁
      s̃₂ₖ₋₃ = s̃₂ₖ₋₁
      c̃₂ₖ₋₂ = c̃₂ₖ
      s̃₂ₖ₋₂ = s̃₂ₖ
      βₖ = βₖ₊₁
      χbarₖ = χbarₖ₊₁
      ψbarₖ₋₁ = ψbarₖ
      πₖ₋₁ = πₖ
      if iter ≤ ℓ-1
        αₖ = αₖ₊₁
        βₖ₊₁ = βₖ₊₂
        γₖ₋₁ = γₖ
        λbarₖ = λbarₖ₊₁
        ζbisₖ = ζbisₖ₊₁
      end
      if iter ≤ ℓ-2
        ϵₖ₋₁ = ϵₖ
        γbarₖ = γbarₖ₊₁
        ζbarₖ₊₁ = ζbarₖ₊₂
      end

      kdisplay(iter, verbose) && @printf(iostream, "%5d  %7.1e  %7.1e  %7.1e  %8.1e  %.2fs\n", iter, rNorm, ArNorm, βₖ, ζₖ isa Complex ? abs(ζₖ) : ζₖ, start_time |> ktimer)
    end
    (verbose > 0) && @printf(iostream, "\n")

    # Termination status
    solved              && (status = "solution good enough given atol, rtol and Artol")
    tired               && (status = "maximum number of iterations exceeded")
    user_requested_exit && (status = "user-requested exit")
    overtimed           && (status = "time limit exceeded")

    # Update x
    warm_start && kaxpy!(n, one(FC), Δx, x)
    workspace.warm_start = false

    # Update stats
    stats.niter = iter
    stats.solved = solved
    # stats.inconsistent = inconsistent
    stats.timer = start_time |> ktimer
    stats.status = status
    return workspace
end
