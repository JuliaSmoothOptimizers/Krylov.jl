# An implementation of CS-MINRES-QLP.
#
# This method is described in
#
# S.-C. T. Choi, Iterative methods for singular linear equations and least-squares problems.
# Ph.D. thesis, ICME, Stanford University, 2006.
#
# S.-C. T. Choi, C. C. Paige and M. A. Saunders, MINRES-QLP: A Krylov subspace method for indefinite or singular symmetric systems.
# SIAM Journal on Scientific Computing, Vol. 33(4), pp. 1810--1836, 2011.
#
# S.-C. T. Choi, csminresqlp.m, MATLAB Central File Exchange 61151, version 1.1.0.0, 2017
# (the always-QLP recurrence generalized here to the conjugated Saunders process), and
# IsOpSym6.m from the same submission (the complex symmetric Givens rotation it uses is
# the same [c s; conj(s) -c], c real, convention implemented generically by `sym_givens`
# for `Complex{T}` arguments in krylov_utils.jl).
#
# Sou-Cheng T. Choi
# Alexis Montoison, <alexis.montoison@polymtl.ca>

export csminresqlp, csminresqlp!

"""
    (x, stats) = csminresqlp(A, b::AbstractVector{FC};
                             λ::FC=zero(FC), atol::T=√eps(T),
                             rtol::T=√eps(T), itmax::Int=0,
                             reorthogonalize::Bool=false,
                             timemax::Float64=Inf, verbose::Int=0, history::Bool=false,
                             callback=workspace->false, iostream::IO=kstdout)

`T` is an `AbstractFloat` such as `Float32` or `Float64`. `FC` is `Complex{T}`.
CS-MINRES-QLP is for complex symmetric systems; a real symmetric matrix is
also Hermitian, so use [`minres_qlp`](@ref) for it instead.

    (x, stats) = csminresqlp(A, b, x0::AbstractVector; kwargs...)

CS-MINRES-QLP can be warm-started from an initial guess `x0` where `kwargs` are the same keyword arguments as above.

CS-MINRES-QLP solves the complex symmetric linear system (A + λI)x = b of size
`n`, where `A` satisfies `transpose(A) == A` (`A` may be singular and the
system may be inconsistent), using the conjugated Saunders process in place
of the ordinary Hermitian Lanczos process that [`minres_qlp`](@ref) uses.
It is a short-recurrence, fixed-storage generalization of the always-QLP
MINRES-QLP recurrence: the QR factorization of the Saunders tridiagonal is
followed by a second (QLP) factorization that selects the minimum-norm
solution among residual minimizers, exactly as in the Hermitian case.

Without `reorthogonalize=true`, the returned solution is not guaranteed to be
the minimum-norm one on severely rank-deficient, inconsistent systems: the
short-recurrence always-QLP generalization can lose that guarantee to
roundoff, unlike ordinary Hermitian MINRES-QLP (which keeps it without
reorthogonalization). `reorthogonalize=true` restores the guarantee by storing
the full generated basis (`O(n)` extra vectors, `O(n²)` work) and caps `itmax`
at `n`.

#### Interface

To easily switch between Krylov methods, use the generic interface [`krylov_solve`](@ref) with `method = :csminresqlp`.

For an in-place variant that reuses memory across solves, see [`csminresqlp!`](@ref).

#### Input arguments

* `A`: a linear operator that models a complex symmetric (possibly
  singular) matrix of dimension `n`, satisfying `transpose(A) == A`;
* `b`: a vector of length `n`.

#### Optional argument

* `x0`: a vector of length `n` that represents an initial guess of the solution `x`.

#### Keyword arguments

* `λ`: shift parameter; CS-MINRES-QLP then solves `(A + λI)x = b`, which remains
  complex symmetric for any `λ`;
* `atol`: absolute stopping tolerance based on the residual norm;
* `rtol`: relative stopping tolerance based on the residual norm;
* `itmax`: the maximum number of iterations. If `itmax=0`, the default number of iterations is set to `2n`;
* `reorthogonalize`: if `true`, reorthogonalize the generated basis against all
  previous basis vectors to guarantee the minimum-norm solution on
  rank-deficient, inconsistent systems; see above;
* `timemax`: the time limit in seconds;
* `verbose`: additional details can be displayed if verbose mode is enabled (verbose > 0). Information will be displayed every `verbose` iterations;
* `history`: collect additional statistics on the run such as residual norms;
* `callback`: function or functor called as `callback(workspace)` that returns `true` if the Krylov method should terminate, and `false` otherwise;
* `iostream`: stream to which output is logged.

#### Output arguments

* `x`: a dense vector of length `n`;
* `stats`: statistics collected on the run in a [`SimpleStats`](@ref) structure.

#### References

* S.-C. T. Choi, *Iterative methods for singular linear equations and least-squares problems*, Ph.D. thesis, ICME, Stanford University, 2006.
* S.-C. T. Choi, C. C. Paige and M. A. Saunders, [*MINRES-QLP: A Krylov subspace method for indefinite or singular symmetric systems*](https://doi.org/10.1137/100787921), SIAM Journal on Scientific Computing, Vol. 33(4), pp. 1810--1836, 2011 (reference MATLAB implementation at the [SOL MINRES-QLP page](https://web.stanford.edu/group/SOL/software/minresqlp/minresqlp-matlab/)).
* S.-C. T. Choi, [*CS-MINRES-QLP*](https://www.mathworks.com/matlabcentral/fileexchange/61151-cs-minres-qlp), MATLAB Central File Exchange 61151, version 1.1.0.0, 2017 (`csminresqlp.m` and `IsOpSym6.m`).
"""
function csminresqlp end

"""
    workspace = csminresqlp!(workspace::CsMinresQlpWorkspace, A, b; kwargs...)
    workspace = csminresqlp!(workspace::CsMinresQlpWorkspace, A, b, x0; kwargs...)

In these calls, `kwargs` are keyword arguments of [`csminresqlp`](@ref).

See [`CsMinresQlpWorkspace`](@ref) for instructions on how to create the `workspace`.

For a more generic interface, you can use [`krylov_workspace`](@ref) with `method = :csminresqlp` to allocate the workspace,
and [`krylov_solve!`](@ref) to run the Krylov method in-place.
"""
function csminresqlp! end

def_args_csminresqlp = (:(A                    ),
                        :(b::AbstractVector{FC}))

def_optargs_csminresqlp = (:(x0::AbstractVector),)

def_kwargs_csminresqlp = (:(; λ::FC = zero(FC)              ),
                          :(; atol::T = √eps(T)            ),
                          :(; rtol::T = √eps(T)            ),
                          :(; itmax::Int = 0               ),
                          :(; reorthogonalize::Bool = false),
                          :(; timemax::Float64 = Inf       ),
                          :(; verbose::Int = 0             ),
                          :(; history::Bool = false        ),
                          :(; callback = workspace -> false),
                          :(; iostream::IO = kstdout       ))

def_kwargs_csminresqlp = extract_parameters.(def_kwargs_csminresqlp)

args_csminresqlp = (:A, :b)
optargs_csminresqlp = (:x0,)
kwargs_csminresqlp = (:λ, :atol, :rtol, :itmax, :reorthogonalize, :timemax, :verbose, :history, :callback, :iostream)

@eval begin
  function csminresqlp!(workspace :: CsMinresQlpWorkspace{T,FC,S}, $(def_args_csminresqlp...); $(def_kwargs_csminresqlp...)) where {T <: AbstractFloat, FC <: Complex{T}, S <: AbstractVector{FC}}

    # Timer
    start_time = time_ns()
    timemax_ns = 1e9 * timemax

    m, n = size(A)
    (m == workspace.m && n == workspace.n) || error("(workspace.m, workspace.n) = ($(workspace.m), $(workspace.n)) is inconsistent with size(A) = ($m, $n)")
    m == n || error("System must be square")
    length(b) == m || error("Inconsistent problem size")
    (verbose > 0) && @printf(iostream, "CS-MINRES-QLP: system of size %d\n", n)

    # Check type consistency
    eltype(A) == FC || error("eltype(A) ≠ $FC")
    ktypeof(b) == S || error("ktypeof(b) must be equal to $S")

    # Set up workspace.
    if reorthogonalize && length(workspace.V) < n
      workspace.V = S[similar(workspace.x) for _ in 1:n]
    end
    Δx, x, p = workspace.Δx, workspace.x, workspace.p
    vₖ, vₖ₋₁, vbar = workspace.vₖ, workspace.vₖ₋₁, workspace.vbar
    wₖ, wₖ₋₁, wₖ₋₂ = workspace.wₖ, workspace.wₖ₋₁, workspace.wₖ₋₂
    warm_start = workspace.warm_start
    stats = workspace.stats
    rNorms = stats.residuals
    reset!(stats)

    itmax == 0 && (itmax = 2 * n)
    reorthogonalize && (itmax = min(itmax, n))

    kfill!(x, zero(FC))  # x₀

    # r₀ = b - (A + λI)x₀
    if warm_start
      kmul!(p, A, Δx)
      (λ ≠ 0) && kaxpy!(n, λ, Δx, p)
      kaxpby!(n, one(FC), b, -one(FC), p)
      kcopy!(n, vₖ, p)
    else
      kcopy!(n, vₖ, b)
    end
    βₖ = knorm(n, vₖ)
    rNorm = βₖ
    ε = atol + rtol * rNorm
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
    kdiv!(n, vₖ, βₖ)

    (verbose > 0) && @printf(iostream, "%5s  %7s  %.2fs\n", "k", "‖rₖ‖", 0.0)
    kdisplay(0, verbose) && @printf(iostream, "%5d  %7.1e  %.2fs\n", 0, rNorm, start_time |> ktimer)

    # Short-recurrence CS-MINRES-QLP generalizing ordinary Hermitian MINRES-QLP to
    # complex symmetric A via the conjugated Saunders process. The QR and QLP
    # recurrences below are ported from the always-QLP formulation validated
    # against the Moore-Penrose solution (S.-C. T. Choi's csminresqlp.m), not
    # re-derived from minres_qlp's own two-rotation-history formulas: those
    # exploit that the Hermitian-Lanczos tridiagonal is real, which does not
    # hold here (the Saunders-process diagonal αₖ is genuinely complex).
    kfill!(vₖ₋₁, zero(FC))
    kfill!(wₖ₋₁, zero(FC))
    kfill!(wₖ, zero(FC))
    kfill!(wₖ₋₂, zero(FC))
    csk, snk = -one(T), zero(FC)
    cr1, sr1, cr2, sr2 = one(T), zero(FC), -one(T), zero(FC)
    tauk = taukm1 = taukm2 = zero(FC)
    ϕₖ = FC(βₖ)
    dltan = eplnn = gama = gamal = gamal2 = zero(FC)
    eta = etal = etal2 = vepln = veplnl = veplnl2 = zero(FC)
    uk3 = uk2 = uk = u = zero(FC)
    operator_scale = zero(T)
    ranktol = n * eps(T)
    breakdown_tol = 100 * eps(T)
    closed = false
    iter = 0
    status = "maximum number of iterations exceeded"
    solved = inconsistent = tired = user_requested_exit = overtimed = false

    while !(solved || inconsistent || closed || tired || user_requested_exit || overtimed)
      iter = iter + 1
      k = iter

      # Saunders basis extension: p = (A + λI)*conj(vₖ) - βₖ*vₖ₋₁, αₖ = ⟨vₖ,p⟩.
      vbar .= conj.(vₖ)
      kmul!(p, A, vbar)
      (λ ≠ 0) && kaxpy!(n, λ, vbar, p)
      scale = knorm(n, p)
      (k > 1) && kaxpy!(n, -βₖ, vₖ₋₁, p)
      alfa = kdot(n, vₖ, p)
      kaxpy!(n, -alfa, vₖ, p)
      if reorthogonalize
        kcopy!(n, workspace.V[k], vₖ)
        for pass in 1:2, j in 1:k
          kaxpy!(n, -kdot(n, workspace.V[j], p), workspace.V[j], p)
        end
      end
      βₖ₊₁ = knorm(n, p)
      operator_scale = max(operator_scale, scale, abs(alfa), k > 1 ? βₖ : zero(T))
      # Absolute test (tiny relative to the operator's own scale) plus a
      # relative-drop test (βₖ₊₁ many orders of magnitude below its immediate
      # predecessor βₖ): a genuine closure drops β by ~13 orders of magnitude
      # in one step, which the relative test catches robustly even when the
      # absolute test is a near-miss against accumulated rounding noise.
      closed = βₖ₊₁ ≤ breakdown_tol * operator_scale || βₖ₊₁ ≤ eps(T)^(T(1) / 3) * βₖ
      closed && (βₖ₊₁ = zero(T))

      # Advance the basis now: vₖ ← vₖ₊₁, freeing p and vₖ₋₁ as scratch for
      # the rest of this iteration (QR/QLP stage below only needs scalars and z).
      kcopy!(n, vₖ₋₁, vₖ)
      if !closed && βₖ₊₁ ≠ 0
        kdiv!(n, p, βₖ₊₁)
      end
      kcopy!(n, vₖ, p)
      βₖ = βₖ₊₁

      # Left reflections (QR of the Saunders tridiagonal Tbar).
      dbar, epln = dltan, eplnn
      dlta = csk * dbar + snk * alfa
      gbar = conj(snk) * dbar - csk * alfa
      eplnn, dltan = snk * βₖ₊₁, -csk * βₖ₊₁
      gamal2, gamal = gamal, gama
      csk, snk, gama = sym_givens(gbar, Complex{T}(βₖ₊₁))
      taukm2, taukm1, tauk = taukm1, tauk, csk * ϕₖ
      ϕₖ = conj(snk) * ϕₖ

      # Right reflections turning the upper factor into the lower QLP factor.
      if k > 2
        veplnl2, etal2, etal = veplnl, etal, eta
        dlta, veplnl = sr2 * vepln - cr2 * dlta, cr2 * vepln + conj(sr2) * dlta
        eta, gama = conj(sr2) * gama, -cr2 * gama
      end
      if k > 1
        cr1, sr1, reflected = sym_givens(conj(gamal), conj(dlta))
        gamal = conj(reflected)
        vepln, gama = conj(sr1) * gama, -cr1 * gama
      end

      # Delayed lower-triangular solve; a near-zero pivot (rank deficiency)
      # drops that direction's coefficient to zero instead of dividing by it.
      uk4, uk3 = uk3, uk2
      (k > 2) && (uk2 = abs(gamal2) ≤ ranktol * operator_scale ? zero(FC) : (taukm2 - etal2 * uk4 - veplnl2 * uk3) / gamal2)
      (k > 1) && (uk  = abs(gamal)  ≤ ranktol * operator_scale ? zero(FC) : (taukm1 - etal * uk3 - veplnl * uk2) / gamal)
      pivot_small = abs(gama) ≤ ranktol * operator_scale
      u = pivot_small ? zero(FC) : (tauk - eta * uk2 - vepln * uk) / gama

      # Always-QLP direction update: wₖ₋₂, wₖ₋₁, wₖ are the last three columns of Vₖ(Pₖ)ᴴ,
      # z = vbar = conj(vₖ) at the start of this iteration (before the basis advance above).
      if k == 1
        kaxpby!(n, conj(sr1), vbar, zero(FC), wₖ₋₁)
        kaxpby!(n, cr1, vbar, zero(FC), wₖ)
      elseif k == 2
        kcopy!(n, p, wₖ)                     # p ← old wₖ (p is free scratch after the basis advance)
        kcopy!(n, wₖ₋₁, vbar)
        kaxpby!(n, cr1, p, conj(sr1), wₖ₋₁)          # wₖ₋₁ ← cr1*old_wₖ + conj(sr1)*z
        kaxpby!(n, -cr1, vbar, sr1, wₖ)              # wₖ   ← -cr1*z + sr1*old_wₖ
      else
        kcopy!(n, p, wₖ₋₁)                   # p ← old wₖ₋₁
        kcopy!(n, wₖ₋₂, vbar)
        kaxpby!(n, cr2, p, conj(sr2), wₖ₋₂)          # wₖ₋₂ ← cr2*old_wₖ₋₁ + conj(sr2)*z  (final)
        kaxpby!(n, sr2, p, zero(FC), wₖ₋₁)
        kaxpy!(n, -cr2, vbar, wₖ₋₁)                  # wₖ₋₁ ← sr2*old_wₖ₋₁ - cr2*z        (= wₐᵤₓ, temp)
        kcopy!(n, p, wₖ)                             # p ← old wₖ
        kaxpby!(n, -cr1, wₖ₋₁, sr1, wₖ)               # wₖ   ← sr1*old_wₖ - cr1*wₐᵤₓ        (final)
        kaxpby!(n, cr1, p, conj(sr1), wₖ₋₁)          # wₖ₋₁ ← cr1*old_wₖ + conj(sr1)*wₐᵤₓ  (final)
      end

      # Settle the k-2 direction into x permanently; wₖ₋₁, wₖ remain tentative
      # until either the next iteration settles wₖ₋₂ again or the post-loop finalize.
      kaxpy!(n, uk2, wₖ₋₂, x)
      kcopy!(n, p, x)
      kaxpy!(n, uk, wₖ₋₁, p)
      kaxpy!(n, u, wₖ, p)
      cr2, sr2, reflected = sym_givens(conj(gamal), conj(eplnn))
      gamal = conj(reflected)

      # rNorm = ‖b - (A + λI)*p‖ where p holds the tentative solution this iteration.
      kmul!(vbar, A, p)
      (λ ≠ 0) && kaxpy!(n, λ, p, vbar)
      kaxpby!(n, one(FC), b, -one(FC), vbar)
      rNorm = knorm(n, vbar)
      history && push!(rNorms, rNorm)
      solved = rNorm ≤ ε
      tired = iter ≥ itmax
      user_requested_exit = callback(workspace)::Bool
      timer = time_ns() - start_time
      overtimed = timer > timemax_ns

      if closed
        status = solved ? "found approximate zero-residual solution" : "found approximate minimum least-squares solution"
      elseif solved
        status = "solution good enough given atol and rtol"
      elseif pivot_small && !solved
        inconsistent = true
        status = "found approximate minimum least-squares solution"
      end
      kdisplay(iter, verbose) && @printf(iostream, "%5d  %7.1e  %.2fs\n", iter, rNorm, start_time |> ktimer)
    end
    (verbose > 0) && @printf(iostream, "\n")
    # uk and u are already zero whenever their own pivot was judged too small
    # to trust (see the delayed lower-triangular solve above), so no extra
    # guard is needed here.
    kaxpy!(n, uk, wₖ₋₁, x)
    kaxpy!(n, u, wₖ, x)

    tired               && (status = "maximum number of iterations exceeded")
    user_requested_exit && (status = "user-requested exit")
    overtimed           && (status = "time limit exceeded")

    warm_start && kaxpy!(n, one(FC), Δx, x)
    workspace.warm_start = false

    stats.niter = iter
    stats.solved = solved
    stats.inconsistent = inconsistent
    stats.timer = start_time |> ktimer
    stats.status = status
    return workspace
  end
end
