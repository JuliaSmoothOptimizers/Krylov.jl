# An implementation of USYMLQR for the solution of symmetric saddle-point systems.
#
# This method is described in
#
# M. A. Saunders, H. D. Simon, and E. L. Yip
# Two Conjugate-Gradient-Type Methods for Unsymmetric Linear Equations.
# SIAM Journal on Numerical Analysis, 25(4), pp. 927--940, 1988.
#
# A. Buttari, D. Orban, D. Ruiz and D. Titley-Peloquin
# A tridiagonalization method for symmetric saddle-point systems.
# SIAM Journal on Scientific Computing, 41(5), pp. 409--432, 2019
#
# Dominique Orban, <dominique.orban@gerad.ca>
# Alexis Montoison, <alexis.montoison@polymtl.ca> -- <amontoison@anl.gov>
# Montréal, November 2021 -- Chicago, January 2026.

export usymlqr, usymlqr!

"""
   (x, y, stats) = usymlqr(A, b::AbstractVector{FC}, c::AbstractVector{FC};
                           M=I, N=I, ls::Bool=true, ln::Bool=true,
                           ldiv::Bool=false, atol::T=√eps(T), rtol::T=√eps(T),
                           itmax::Int=0, timemax::Float64=Inf,
                           verbose::Int=0, history::Bool=false,
                           callback=workspace->false, iostream::IO=kstdout)

`T` is an `AbstractFloat` such as `Float32`, `Float64` or `BigFloat`.
`FC` is `T` or `Complex{T}`.

    (x, y, stats) = usymlqr(A, b, c, x0::AbstractVector, y0::AbstractVector; kwargs...)

USYMLQR can be warm-started from initial guesses `x0` and `y0` where `kwargs` are the same keyword arguments as above.

Solve the symmetric saddle-point system

    [ E   A ] [ x ] = [ b ]
    [ Aᴴ    ] [ y ]   [ c ]

where E = M⁻¹ ≻ 0 by way of the Saunders-Simon-Yip tridiagonalization using USYMLQ and USYMQR methods.
The method solves the least-squares problem when `ls = true`

    [ E   A ] [ r ] = [ b ]
    [ Aᴴ    ] [ s ]   [ 0 ]

and the least-norm problem when `ln = true`

    [ E   A ] [ w ] = [ 0 ]
    [ Aᴴ    ] [ z ]   [ c ]

and simply adds the solutions.

    [ M   O ]
    [ 0   N ]

indicates the weighted norm in which residuals are measured.
It's the Euclidean norm when `M` and `N` are identity operators.

#### Interface

To easily switch between Krylov methods, use the generic interface [`krylov_solve`](@ref) with `method = :usymlqr`.

For an in-place variant that reuses memory across solves, see [`usymlqr!`](@ref).

#### Input arguments

* `A`: a linear operator that models a matrix of dimension `m × n`;
* `b`: a vector of length `m`;
* `c`: a vector of length `n`.

#### Optional arguments

* `x0`: a vector of length `m` that represents an initial guess of the solution `x`;
* `y0`: a vector of length `n` that represents an initial guess of the solution `y`.

#### Keyword arguments

* `M`: linear operator that models a Hermitian positive-definite matrix of size `m` used for centered preconditioning of the partitioned system;
* `N`: linear operator that models a Hermitian positive-definite matrix of size `n` used for centered preconditioning of the partitioned system;
* `ls`: define whether the least-squares problem is solved;
* `ln`: define whether the least-norm problem is solved;
* `ldiv`: define whether the preconditioners use `ldiv!` or `mul!`;
* `atol`: absolute stopping tolerance based on the residual norm;
* `rtol`: relative stopping tolerance based on the residual norm;
* `itmax`: the maximum number of iterations. If `itmax=0`, the default number of iterations is set to `m+n`;
* `timemax`: the time limit in seconds;
* `verbose`: additional details can be kdisplayed if verbose mode is enabled (verbose > 0). Information will be kdisplayed every `verbose` iterations;
* `history`: collect additional statistics on the run such as residual norms, or Aᴴ-residual norms;
* `callback`: function or functor called as `callback(workspace)` that returns `true` if the Krylov method should terminate, and `false` otherwise;
* `iostream`: stream to which output is logged.

#### Output arguments

* `x`: a dense vector of length `m`;
* `y`: a dense vector of length `n`;
* `stats`: statistics collected on the run in a [`SimpleStats`](@ref) structure.

#### References

* M. A. Saunders, H. D. Simon, and E. L. Yip, [*Two Conjugate-Gradient-Type Methods for Unsymmetric Linear Equations*](https://doi.org/10.1137/0725052), SIAM Journal on Numerical Analysis, 25(4), pp. 927--940, 1988.
* A. Buttari, D. Orban, D. Ruiz and D. Titley-Peloquin, [*A tridiagonalization method for symmetric saddle-point and quasi-definite systems*](https://doi.org/10.1137/18M1194900), SIAM Journal on Scientific Computing, 41(5), pp. 409--432, 2019.
"""
function usymlqr end

"""
    workspace = usymlqr!(workspace::UsymlqrWorkspace, A, b, c; kwargs...)
    workspace = usymlqr!(workspace::UsymlqrWorkspace, A, b, c, x0, y0; kwargs...)

In these calls, `kwargs` are keyword arguments of [`usymlqr`](@ref).

See [`UsymlqrWorkspace`](@ref) for instructions on how to create the `workspace`.

For a more generic interface, you can use [`krylov_workspace`](@ref) `method = :usymlqr` to allocate the workspace,
and [`krylov_solve!`](@ref) to run the Krylov method in-place.
"""
function usymlqr! end

def_args_usymlqr = (:(A                    ),
                    :(b::AbstractVector{FC}),
                    :(c::AbstractVector{FC}))

def_optargs_usymlqr = (:(x0::AbstractVector),
                       :(y0::AbstractVector))

def_kwargs_usymlqr = (:(; M = I                        ),
                      :(; N = I                        ),
                      :(; ls::Bool = true              ),
                      :(; ln::Bool = true              ),
                      :(; ldiv::Bool = false           ),
                      :(; atol::T = √eps(T)            ),
                      :(; rtol::T = √eps(T)            ),
                      :(; itmax::Int = 0               ),
                      :(; timemax::Float64 = Inf       ),
                      :(; verbose::Int = 0             ),
                      :(; history::Bool = false        ),
                      :(; callback = workspace -> false),
                      :(; iostream::IO = kstdout       ))

def_kwargs_usymlqr = extract_parameters.(def_kwargs_usymlqr)

args_usymlqr = (:A, :b, :c)
optargs_usymlqr = (:x0, :y0)
kwargs_usymlqr = (:M, :N, :ls, :ln, :ldiv, :atol, :rtol, :itmax, :timemax, :verbose, :history, :callback, :iostream)

@eval begin
  function usymlqr!(workspace :: UsymlqrWorkspace{T,FC,Sm,Sn}, $(def_args_usymlqr...); $(def_kwargs_usymlqr...)) where {T <: AbstractFloat, FC <: FloatOrComplex{T}, Sm <: AbstractVector{FC}, Sn <: AbstractVector{FC}}
    # Timer
    start_time = time_ns()
    timemax_ns = 1e9 * timemax

    m, n = size(A)
    (m == workspace.m && n == workspace.n) || error("(workspace.m, workspace.n) = ($(workspace.m), $(workspace.n)) is inconsistent with size(A) = ($m, $n)")
    length(b) == m || error("Inconsistent problem size")
    length(c) == n || error("Inconsistent problem size")
    (ls || ln) || error("The keyword arguments `ls` and `ln` can't be both `false`.")
    (verbose > 0) && @printf(iostream, "USYMLQR: system of %d equations in %d variables\n", m+n, m+n)

    # Check M = Iₘ and N = Iₙ
    MisI = (M === I)
    NisI = (N === I)

    # Check type consistency
    eltype(A) == FC || @warn "eltype(A) ≠ $FC. This could lead to errors or additional allocations in operator-vector products."
    ktypeof(b) == Sm || error("ktypeof(b) is not a subtype of $Sm")
    ktypeof(c) == Sn || error("ktypeof(c) is not a subtype of $Sn")

    # Compute the adjoint of A
    Aᴴ = A'

    # Set up workspace.
    allocate_if(!MisI, workspace, :vₖ, Sm, workspace.x)  # The length of vₖ is m
    allocate_if(!NisI, workspace, :uₖ, Sn, workspace.y)  # The length of uₖ is n
    Δx, Δy = workspace.Δx, workspace.Δy
    M⁻¹vₖ₋₁, M⁻¹vₖ, q, xₖ, rₖ = workspace.M⁻¹vₖ₋₁, workspace.M⁻¹vₖ, workspace.q, workspace.x, workspace.r
    N⁻¹uₖ₋₁, N⁻¹uₖ, p, yₖ, zₖ = workspace.N⁻¹uₖ₋₁, workspace.N⁻¹uₖ, workspace.p, workspace.y, workspace.z
    d̅, wₖ₋₂, wₖ₋₁ = workspace.d̅, workspace.wₖ₋₂, workspace.wₖ₋₁
    vₖ = MisI ? M⁻¹vₖ : workspace.vₖ
    uₖ = NisI ? N⁻¹uₖ : workspace.uₖ
    warm_start = workspace.warm_start
    b₀ = warm_start ? q : b
    c₀ = warm_start ? p : c

    stats = workspace.stats
    rNorms = stats.residuals
    reset!(stats)

    iter = 0
    itmax == 0 && (itmax = n+m)

    # Initialize preconditioned orthogonal tridiagonalization process.
    kfill!(M⁻¹vₖ₋₁, zero(FC))  # v₀ = 0
    kfill!(N⁻¹uₖ₋₁, zero(FC))  # u₀ = 0

    # [ I   A ] [ xₖ ] = [ b -   Δx - AΔy ] = [ b₀ ]
    # [ Aᴴ    ] [ yₖ ]   [ c - AᴴΔx       ]   [ c₀ ]
    if warm_start
      mul!(b₀, A, Δy)
      kaxpy!(m, one(T), Δx, b₀)
      kaxpby!(m, one(T), b, -one(T), b₀)
      mul!(c₀, Aᴴ, Δx)
      kaxpby!(n, one(T), c, -one(T), c₀)
    end

    # Initial solutions r₀ , x₀, y₀ and z₀.
    ls ? kcopy!(m, rₖ, b₀) : kfill!(rₖ, zero(FC))
    kfill!(xₖ, zero(FC))
    kfill!(yₖ, zero(FC))
    kfill!(zₖ, zero(FC))

    # β₁Ev₁ = b ↔ β₁v₁ = Mb
    kcopy!(m, M⁻¹vₖ, b₀)
    MisI || mulorldiv!(vₖ, M, M⁻¹vₖ, ldiv)
    βₖ = knorm_elliptic(m, vₖ, M⁻¹vₖ)  # β₁ = ‖v₁‖_E
    if βₖ ≠ 0
      kdiv!(m, M⁻¹vₖ, βₖ)
      MisI || kdiv!(m, vₖ, βₖ)
    else
      # v₁ = 0 such that v₁ ⊥ Span{v₁, ..., vₖ}
      kfill!(M⁻¹vₖ, zero(FC))
      MisI || kfill!(vₖ, zero(FC))
    end

    # γ₁Fu₁ = c ↔ γ₁u₁ = Nc
    kcopy!(n, N⁻¹uₖ, c₀)
    NisI || mulorldiv!(uₖ, N, N⁻¹uₖ, ldiv)
    γₖ = knorm_elliptic(n, uₖ, N⁻¹uₖ)  # γ₁ = ‖u₁‖_F
    if γₖ ≠ 0
      kdiv!(n, N⁻¹uₖ, γₖ)
      NisI || kdiv!(n, uₖ, γₖ)
    else
      # u₁ = 0 such that u₁ ⊥ Span{u₁, ..., uₖ}
      kfill!(N⁻¹uₖ, zero(FC))
      NisI || kfill!(uₖ, zero(FC))
    end

    cₖ₋₂ = cₖ₋₁ = cₖ = -one(T)      # Givens cosines used for the QR factorization of Tₖ₊₁.ₖ
    sₖ₋₂ = sₖ₋₁ = sₖ = zero(FC)     # Givens sines used for the QR factorization of Tₖ₊₁.ₖ
    kfill!(wₖ₋₂, zero(FC))          # Column k-2 of Wₖ = Uₖ(Rₖ)⁻¹
    kfill!(wₖ₋₁, zero(FC))          # Column k-1 of Wₖ = Uₖ(Rₖ)⁻¹
    kfill!(d̅, zero(FC))             # Last column of D̅ₖ = Vₖ(Qₖ₋₁)ᴴ
    ϕbarₖ = βₖ                      # ϕbarₖ is the last component of f̄ₖ = (Qₖ)ᴴβ₁e₁
    ζₖ₋₂ = ζₖ₋₁ = ζbarₖ = zero(FC)  # ζₖ₋₂, ζₖ₋₁ and ζbarₖ are the last components of h̄ₖ = (R̅ₖ)⁻ᴴγ₁e₁
    ηₖ₋₁ = zero(FC)                 # ηₖ₋₁ is used to update ζₖ₋₁ and ζbarₖ
    δₖ₋₁ = δbarₖ = zero(FC)         # Coefficients of Rₖ₋₁ and Rₖ modified over the course of two iterations

    # Stopping criterion.
    κ = zero(T)
    AᴴrNorm = Inf
    rNorm_ls = bNorm = βₖ
    rNorm_ln = cNorm = γₖ
    ε_ls = atol + rtol * rNorm_ls
    ε_ln = atol + rtol * rNorm_ln
    solved_ls = !ls || rNorm_ls ≤ ε_ls
    solved_ln = !ln || rNorm_ln ≤ ε_ln
    solved = solved_ls && solved_ln
    inconsistent = false
    tired = iter ≥ itmax
    status = "unknown"
    ill_cond = false
    user_requested_exit = false
    overtimed = false

    (verbose > 0) && @printf(iostream, "%5s  %7s  %7s  %7s  %7s  %5s\n", "k", "βₖ₊₁", "γₖ₊₁", "‖rₖ‖_LS", "‖rₖ‖_LN", "timer")
    !ls &&  ln && kdisplay(iter, verbose) && @printf(iostream, "%5d  %7.1e  %7.1e  %7s  %7.1e  %.2fs\n", iter, βₖ, γₖ, "✗ ✗ ✗ ✗", rNorm_ln, start_time |> ktimer)
     ls && !ln && kdisplay(iter, verbose) && @printf(iostream, "%5d  %7.1e  %7.1e  %7.1e  %7s  %.2fs\n", iter, βₖ, γₖ, rNorm_ls, "✗ ✗ ✗ ✗", start_time |> ktimer)
     ls &&  ln && kdisplay(iter, verbose) && @printf(iostream, "%5d  %7.1e  %7.1e  %7.1e  %7.1e  %.2fs\n", iter, βₖ, γₖ, rNorm_ls, rNorm_ln, start_time |> ktimer)

    while !(solved || tired || ill_cond || inconsistent || user_requested_exit || overtimed)
      # Update iteration index.
      iter = iter + 1

      # Continue the orthogonal tridiagonalization process.
      # AUₖ  = EVₖTₖ    + βₖ₊₁Evₖ₊₁(eₖ)ᵀ = EVₖ₊₁Tₖ₊₁.ₖ
      # AᴴVₖ = FUₖ(Tₖ)ᴴ + γₖ₊₁Fuₖ₊₁(eₖ)ᵀ = FUₖ₊₁(Tₖ.ₖ₊₁)ᴴ

      mul!(q, A , uₖ)  # Forms Evₖ₊₁ : q ← Auₖ
      mul!(p, Aᴴ, vₖ)  # Forms Fuₖ₊₁ : p ← Aᴴvₖ

      if iter ≥ 2
        kaxpy!(m, -γₖ, M⁻¹vₖ₋₁, q) # q ← q - γₖ * M⁻¹vₖ₋₁
        kaxpy!(n, -βₖ, N⁻¹uₖ₋₁, p) # p ← p - βₖ * N⁻¹uₖ₋₁
      end

      αₖ = kdot(m, vₖ, q)  # αₖ = ⟨uₖ,q⟩

      kaxpy!(m, -     αₖ , M⁻¹vₖ, q)   # q ← q - αₖ * M⁻¹vₖ
      kaxpy!(n, -conj(αₖ), N⁻¹uₖ, p)   # p ← p - ᾱₖ * N⁻¹uₖ

      # Update M⁻¹vₖ₋₁ and N⁻¹uₖ₋₁
      kcopy!(m, M⁻¹vₖ₋₁, M⁻¹vₖ)
      kcopy!(n, N⁻¹uₖ₋₁, N⁻¹uₖ)

      # Compute M⁻¹vₖ and N⁻¹uₖ
      MisI || mulorldiv!(M⁻¹vₖ, M, q, ldiv)  # βₖ₊₁vₖ₊₁ = MAuₖ  - γₖvₖ₋₁ - αₖvₖ
      NisI || mulorldiv!(N⁻¹uₖ, N, p, ldiv)  # γₖ₊₁uₖ₊₁ = NAᴴvₖ - βₖuₖ₋₁ - ᾱₖuₖ

      βₖ₊₁ = MisI ? knorm(m, q) : knorm_elliptic(m, M⁻¹vₖ, q)  # βₖ₊₁ = ‖vₖ₊₁‖_E
      γₖ₊₁ = NisI ? knorm(n, p) : knorm_elliptic(n, N⁻¹uₖ, p)  # γₖ₊₁ = ‖uₖ₊₁‖_F

      # Update the QR factorization of Tₖ₊₁.ₖ = Qₖ [ Rₖ ].
      #                                            [ Oᵀ ]
      #
      # [ α₁ γ₂ 0  •  •  •   0  ]      [ δ₁ λ₁ ϵ₁ 0  •  •  0  ]
      # [ β₂ α₂ γ₃ •         •  ]      [ 0  δ₂ λ₂ •  •     •  ]
      # [ 0  •  •  •  •      •  ]      [ •  •  δ₃ •  •  •  •  ]
      # [ •  •  •  •  •  •   •  ] = Qₖ [ •     •  •  •  •  0  ]
      # [ •     •  •  •  •   0  ]      [ •        •  •  • ϵₖ₋₂]
      # [ •        •  •  •   γₖ ]      [ •           •  • λₖ₋₁]
      # [ •           •  βₖ  αₖ ]      [ 0  •  •  •  •  0  δₖ ]
      # [ 0  •  •  •  •  0  βₖ₊₁]      [ 0  •  •  •  •  •  0  ]
      #
      # If k = 1, we don't have any previous reflexion.
      # If k = 2, we apply the last reflexion.
      # If k ≥ 3, we only apply the two previous reflexions.

      # Apply previous Givens reflections Qₖ₋₂.ₖ₋₁
      if iter ≥ 3
        # [cₖ₋₂  sₖ₋₂] [0 ] = [  ϵₖ₋₂ ]
        # [s̄ₖ₋₂ -cₖ₋₂] [γₖ]   [λbarₖ₋₁]
        ϵₖ₋₂    =  sₖ₋₂ * γₖ
        λbarₖ₋₁ = -cₖ₋₂ * γₖ
      end

      # Apply previous Givens reflections Qₖ₋₁.ₖ
      if iter ≥ 2
        iter == 2 && (λbarₖ₋₁ = γₖ)
        # [cₖ₋₁  sₖ₋₁] [λbarₖ₋₁] = [λₖ₋₁ ]
        # [s̄ₖ₋₁ -cₖ₋₁] [   αₖ  ]   [δbarₖ]
        λₖ₋₁  =      cₖ₋₁  * λbarₖ₋₁ + sₖ₋₁ * αₖ
        δbarₖ = conj(sₖ₋₁) * λbarₖ₋₁ - cₖ₋₁ * αₖ
      end

      # Compute and apply current Givens reflection Qₖ.ₖ₊₁
      iter == 1 && (δbarₖ = αₖ)
      # [cₖ  sₖ] [δbarₖ] = [δₖ]
      # [s̄ₖ -cₖ] [βₖ₊₁ ]   [0 ]
      (cₖ, sₖ, δₖ) = sym_givens(δbarₖ, βₖ₊₁)

      # Compute the direction wₖ, the last column of Wₖ = Uₖ(Rₖ)⁻¹ ⟷ (Rₖ)ᵀ(Wₖ)ᵀ = (Uₖ)ᵀ.
      # w₁ = u₁ / δ₁
      if iter == 1
        wₖ = wₖ₋₁
        kdivcopy!(n, wₖ, uₖ, δₖ)
      end
      # w₂ = (u₂ - λ₁w₁) / δ₂
      if iter == 2
        wₖ = wₖ₋₂
        kaxpy!(n, -λₖ₋₁, wₖ₋₁, wₖ)
        kaxpy!(n, one(FC), uₖ, wₖ)
        kdiv!(n, wₖ, δₖ)
      end
      # wₖ = (uₖ - λₖ₋₁wₖ₋₁ - ϵₖ₋₂wₖ₋₂) / δₖ
      if iter ≥ 3
        kscal!(n, -ϵₖ₋₂, wₖ₋₂)
        wₖ = wₖ₋₂
        kaxpy!(n, -λₖ₋₁, wₖ₋₁, wₖ)
        kaxpy!(n, one(FC), uₖ, wₖ)
        kdiv!(n, wₖ, δₖ)
      end

      if ls && !solved_ls
        # Compute f̄ₖ₊₁ = [   fₖ  ] = (Qₖ)ᴴβ₁e₁
        #                [ϕbarₖ₊₁]
        #
        # [cₖ  sₖ] [ϕbarₖ] = [   ϕₖ  ]
        # [s̄ₖ -cₖ] [  0  ]   [ϕbarₖ₊₁]
        ϕₖ      =      cₖ  * ϕbarₖ
        ϕbarₖ₊₁ = conj(sₖ) * ϕbarₖ

        # println("iter = $iter")
        # println("Exact ‖Aᴴrₖ₋₁‖ = ", norm(A' * (b - A * yₖ)))
        # println("Exact ‖Aᴴrₖ₋₁‖ = ", norm(A' * rₖ))

        # Update the solution yₖ = Wₖfₖ.
        # yₖ ← yₖ₋₁ + ϕₖ * wₖ
        kaxpy!(n, ϕₖ, wₖ, yₖ)

        # Update the residual rₖ.
        # rₖ ← |sₖ|² * rₖ₋₁ - cₖ * ϕbarₖ₊₁ * vₖ₊₁
        # Note: vₖ₊₁ = q / βₖ₊₁
        kaxpby!(m, -cₖ * ϕbarₖ₊₁ / βₖ₊₁, q, abs2(sₖ), rₖ)

        # Compute ‖rₖ‖ = ‖ b - Ayₖ‖ = |ϕbarₖ₊₁|.
        rNorm_ls = abs(ϕbarₖ₊₁)
        history && push!(rNorms, rNorm_ls)

        # Compute ‖Aᴴrₖ₋₁‖ = |ϕbarₖ| * √(|δbarₖ|² + |λbarₖ|²).
        AᴴrNorm = abs(ϕbarₖ) * √(abs2(δbarₖ) + abs2(cₖ₋₁ * γₖ₊₁))
        history && push!(AᴴrNorms, AᴴrNorm)

        # LS -- Update ϕbarₖ₊₁
        ϕbarₖ = ϕbarₖ₊₁

        # println("Estimate ‖Aᴴrₖ₋₁‖ = ", AᴴrNorm)
        # println("Exact ‖rₖ‖ = ", norm(b - A * yₖ))
        # println("Exact ‖rₖ‖ = ", norm(rₖ))
        # println("Estimate ‖rₖ‖ = ", norm(rNorm))

        solved_ls = rNorm_ls ≤ ε_ls
        iter == 1 && (κ = atol + rtol * AᴴrNorm)
        inconsistent = !solved_ls && AᴴrNorm ≤ κ
      end

      if ln && !solved_ln
        # Update the solution of (R̅ₖ)ᴴh̄ₖ = γ₁e₁.
        #
        # [ δ₁  0   •   •   •   •   0  ] [ ζ₁ ]   [ γ₁ ]
        # [ λ₁  δ₂  •               •  ] [ ζ₂ ]   [ 0  ]
        # [ ϵ₁  λ₂  δ₃  •           •  ] [ ζ₃ ]   [ •  ]
        # [ 0   •   •   •   •       •  ] [ •  ] = [ •  ]
        # [ •   •   •   •   •   •   •  ] [ •  ]   [ •  ]
        # [ •       •   •   •   •   0  ] [ •  ]   [ •  ]
        # [ 0   •   •   0 ϵₖ₋₂ λₖ₋₁ δ̄ₖ ] [ ζ̄ₖ ]   [ 0  ]

        # Compute ζₖ₋₁ and ζbarₖ, the last components of the h̄ₖ.
        # [δbar₁] [ζbar₁] = [γ₁]
        if iter == 1
          ηₖ = γₖ
        end
        # [δ₁    0  ] [  ζ₁ ] = [γ₁]
        # [λ₁  δbar₂] [ζbar₂]   [0 ]
        if iter == 2
          ζₖ₋₁ = ηₖ₋₁ / conj(δₖ₋₁)
          ηₖ   = -conj(λₖ₋₁) * ζₖ₋₁
        end
        # [λₖ₋₂  δₖ₋₁    0  ] [ζₖ₋₂ ] = [0]
        # [ϵₖ₋₂  λₖ₋₁  δbarₖ] [ζₖ₋₁ ]   [0]
        #                     [ζbarₖ]
        if iter ≥ 3
          ζₖ₋₂ = ζₖ₋₁
          ζₖ₋₁ = ηₖ₋₁ / conj(δₖ₋₁)
          ηₖ   = -conj(ϵₖ₋₂) * ζₖ₋₂ - conj(λₖ₋₁) * ζₖ₋₁
        end

        # Relations for the directions dₖ₋₁ and d̅ₖ, the last two columns of D̅ₖ = Vₖ(Qₖ₋₁)ᴴ.
        # Note: D̄ₖ represents the matrix P̄ₖ in the paper of USYMLQR.
        # [d̅ₖ₋₁ vₖ] [cₖ₋₁  sₖ₋₁] = [dₖ₋₁ d̅ₖ] ⟷ dₖ₋₁ = cₖ₋₁ * d̅ₖ₋₁ + s̄ₖ₋₁ * vₖ
        #           [s̄ₖ₋₁ -cₖ₋₁]             ⟷ d̅ₖ   = sₖ₋₁ * d̅ₖ₋₁ - cₖ₋₁ * vₖ
        if iter == 1
          # d̅₁ = v₁
          kcopy!(m, d̅, vₖ)  # d̅ ← vₖ
        else
          # Compute solution xₖ.
          # (xᴸ)ₖ ← (xᴸ)ₖ₋₁ + ζₖ₋₁ * dₖ₋₁
          kaxpy!(m, ζₖ₋₁ *      cₖ₋₁,  d̅ , xₖ)
          kaxpy!(m, ζₖ₋₁ * conj(sₖ₋₁), vₖ, xₖ)

          # Compute solution zₖ.
          # (zᴸ)ₖ ← (zᴸ)ₖ₋₁ - ζₖ₋₁ * wₖ₋₁
          kaxpy!(n, -ζₖ₋₁, wₖ₋₁, zₖ)

          # Compute the direction d̅ₖ.
          # d̅ₖ = sₖ₋₁ * d̅ₖ₋₁ - cₖ₋₁ * vₖ
          kaxpby!(m, -cₖ₋₁, vₖ, sₖ₋₁, d̅)
        end

        # Compute USYMLQ residual norm
        # ‖rₖ‖ = √(|μₖ|² + |ωₖ|²)
        if iter == 1
          rNorm_ln = cNorm
        else
          μₖ = γₖ * (conj(sₖ₋₂) * ζₖ₋₂ - cₖ₋₂ * cₖ₋₁ * ζₖ₋₁) + conj(αₖ * sₖ₋₁) * ζₖ₋₁
          ωₖ = γₖ₊₁ * conj(sₖ₋₁) * ζₖ₋₁
          rNorm_ln = sqrt(abs2(μₖ) + abs2(ωₖ))
        end
        history && push!(rNorms, rNorm_ln)

        # LN -- Update ηₖ₋₁
        ηₖ₋₁ = ηₖ

        # println("iter = $iter")

        # println("Exact ‖rₖ‖_LQ = ", norm(c - A'     * xₖ))
        # println("Exact ‖rₖ‖_LQ = ", norm(c + A' * A * zₖ))
        # println("Estimate ‖rₖ‖_LQ = ", norm(rNorm_lq))

        # println("Exact ‖rₖ‖_CG = ", norm(c - A'     * (xₖ + ζbarₖ * d̅)))
        # println("Exact ‖rₖ‖_CG = ", norm(c + A' * A * (zₖ - (ζbarₖ * δₖ / δbarₖ) * wₖ)))
        # println("Estimate ‖rₖ‖_CG = ", norm(rNorm_cg))

        # println("Exact ‖xₖ + Azₖ‖_LQ = ", norm(xₖ + A * zₖ))
        # println("Exact ‖xₖ + Azₖ‖_CG = ", norm(xₖ + ζbarₖ * d̅ + A * (zₖ - (ζbarₖ * δₖ / δbarₖ) * wₖ)))

        solved_ln = rNorm_ln ≤ ε_ln
      end

      # Compute uₖ₊₁ and vₖ₊₁.
      if βₖ₊₁ ≠ zero(T)
        MisI || kdiv!(m, M⁻¹vₖ, βₖ₊₁)
        kdivcopy!(m, vₖ, q, βₖ₊₁)  # vₖ₊₁ = q / βₖ₊₁
      else
        # If βₖ₊₁ == 0 then vₖ₊₁ = 0 and Auₖ ∈ Span{v₁, ..., vₖ}
        # We can keep vₖ₊₁ = 0 such that vₖ₊₁ ⊥ Span{v₁, ..., vₖ}
        MisI || kfill!(M⁻¹vₖ, zero(FC))
        kfill!(vₖ, zero(FC))
      end

      if γₖ₊₁ ≠ zero(T)
        NisI || kdiv!(n, N⁻¹uₖ, γₖ₊₁)
        kdivcopy!(n, uₖ, p, γₖ₊₁)  # uₖ₊₁ = p / γₖ₊₁
      else
        # If γₖ₊₁ == 0 then uₖ₊₁ = 0 and Aᴴvₖ ∈ Span{u₁, ..., uₖ}
        # We can keep uₖ₊₁ = 0 such that uₖ₊₁ ⊥ Span{u₁, ..., uₖ}
        NisI || kfill!(N⁻¹uₖ, zero(FC))
        kfill!(uₖ, zero(FC))
      end

      # Swap the pointers for wₖ₋₂ and wₖ₋₁
      if iter ≥ 2
        @kswap!(wₖ₋₂, wₖ₋₁)
      end

      # Update sₖ₋₂, cₖ₋₂, sₖ₋₁, cₖ₋₁, γₖ, βₖ and δₖ₋₁
      if iter ≥ 2
        sₖ₋₂ = sₖ₋₁
        cₖ₋₂ = cₖ₋₁
      end
      sₖ₋₁ = sₖ
      cₖ₋₁ = cₖ
      δₖ₋₁ = δₖ
      γₖ   = γₖ₊₁
      βₖ   = βₖ₊₁

      # Update stopping criterion.
      user_requested_exit = callback(workspace) :: Bool
      solved = solved_ls && solved_ln
      tired = iter ≥ itmax
      timer = time_ns() - start_time
      overtimed = timer > timemax_ns
      !ls &&  ln && kdisplay(iter, verbose) && @printf(iostream, "%5d  %7.1e  %7.1e  %7s  %7.1e  %.2fs\n", iter, βₖ₊₁, γₖ₊₁, "✗ ✗ ✗ ✗", rNorm_ln, start_time |> ktimer)
       ls && !ln && kdisplay(iter, verbose) && @printf(iostream, "%5d  %7.1e  %7.1e  %7.1e  %7s  %.2fs\n", iter, βₖ₊₁, γₖ₊₁, rNorm_ls, "✗ ✗ ✗ ✗", start_time |> ktimer)
       ls &&  ln && kdisplay(iter, verbose) && @printf(iostream, "%5d  %7.1e  %7.1e  %7.1e  %7.1e  %.2fs\n", iter, βₖ₊₁, γₖ₊₁, rNorm_ls, rNorm_ln, start_time |> ktimer)
    end
    (verbose > 0) && @printf(iostream, "\n")

    # Termination status
    tired               && (status = "maximum number of iterations exceeded")
    solved              && (status = "solution good enough given atol and rtol")
    user_requested_exit && (status = "user-requested exit")
    overtimed           && (status = "time limit exceeded")

    # Compute the solution of the saddle point system
    # xₖ ← xₖ + rₖ
    # yₖ ← yₖ + zₖ
    kaxpy!(m, one(FC), rₖ, xₖ)
    kaxpy!(n, one(FC), zₖ, yₖ)

    # Update xₖ and yₖ
    warm_start && kaxpy!(m, one(FC), Δx, xₖ)
    warm_start && kaxpy!(n, one(FC), Δy, yₖ)
    workspace.warm_start = false

    # Update stats
    stats.niter = iter
    stats.solved = solved
    stats.inconsistent = false
    stats.timer = ktimer(start_time)
    stats.status = status
    return workspace
  end
end
