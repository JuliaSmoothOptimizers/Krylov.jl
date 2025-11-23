"""
    solution(workspace)

Return the solution(s) stored in the `workspace`.

Optionally you can specify which solution you want to recover,
`solution(workspace, 1)` returns `x` and `solution(workspace, 2)` returns `y`.
"""
function solution end

"""
    statistics(workspace)

Return the statistics stored in `workspace`.
"""
function statistics end

"""
    results(workspace)

Return a tuple containing the solution(s) and the statistics associated with `workspace`.
This allows retrieving the output arguments of an out-of-place method from an in-place method.

For example, instead of `x, stats = cg(A, b)`, you can use:
```julia
workspace = CgWorkspace(A, b)
cg!(workspace, A, b)
x, stats = results(workspace)
```
"""
function results end

"""
    issolved(workspace)

Return a boolean indicating whether the Krylov method associated with `workspace` has succeeded.

For the Krylov methods [`bilqr`](@ref) and [`trilqr`](@ref), you can use `issolved_primal(workspace)`
and `issolved_dual(workspace)` to separately check whether the solver has converged on the primal or dual system.
"""
function issolved end

"""
    elapsed_time(workspace)

Return the elapsed time (in seconds) during the last call to the Krylov method associated with `workspace`.
"""
function elapsed_time end

"""
    solution_count(workspace)

Return the number of outputs of `solution(workspace)`.
"""
function solution_count end

"""
    iteration_count(workspace)

Return the number of iterations performed by the Krylov method associated with `workspace`.

The number of iterations alone is not a reliable basis for comparing different Krylov methods,
since the work performed in each iteration can vary significantly.
For a fairer performance comparison, use the total number of operator-vector products with `A` and `A'` (see [Aprod_count](@ref) and [Atprod_count](@ref)).
"""
function iteration_count end

"""
    Aprod_count(workspace)

Return the number of operator-vector products with `A` performed by the Krylov method associated with `workspace`.

This function can also be used to determine the number of operator-vector products with `B` in [`gpmr`](@ref), since it is the same as for `A`.
"""
function Aprod_count end

"""
    Atprod_count(workspace)

Return the number of operator-vector products with `A'` performed by the Krylov method associated with `workspace`.
"""
function Atprod_count end

"""
  warm_start!(workspace, x0)
  warm_start!(workspace, x0, y0)

Warm-start the Krylov method associated with `workspace` using a user-provided initial guess, instead of starting from zero.
"""
function warm_start! end

# Krylov methods
for (KS, fun, nsol, nA, nAt, warm_start) in [
  (:CarWorkspace      , :car!       , 1, 1, 0, true )
  (:LsmrWorkspace     , :lsmr!      , 1, 1, 1, false)
  (:CgsWorkspace      , :cgs!       , 1, 2, 0, true )
  (:UsymlqWorkspace   , :usymlq!    , 1, 1, 1, true )
  (:LnlqWorkspace     , :lnlq!      , 2, 1, 1, false)
  (:BicgstabWorkspace , :bicgstab!  , 1, 2, 0, true )
  (:CrlsWorkspace     , :crls!      , 1, 1, 1, false)
  (:LsqrWorkspace     , :lsqr!      , 1, 1, 1, false)
  (:MinresWorkspace   , :minres!    , 1, 1, 0, true )
  (:MinaresWorkspace  , :minares!   , 1, 1, 0, true )
  (:CgneWorkspace     , :cgne!      , 1, 1, 1, false)
  (:DqgmresWorkspace  , :dqgmres!   , 1, 1, 0, true )
  (:SymmlqWorkspace   , :symmlq!    , 1, 1, 0, true )
  (:TrimrWorkspace    , :trimr!     , 2, 1, 1, true )
  (:UsymqrWorkspace   , :usymqr!    , 1, 1, 1, true )
  (:BilqrWorkspace    , :bilqr!     , 2, 1, 1, true )
  (:CrWorkspace       , :cr!        , 1, 1, 0, true )
  (:CraigmrWorkspace  , :craigmr!   , 2, 1, 1, false)
  (:TricgWorkspace    , :tricg!     , 2, 1, 1, true )
  (:CraigWorkspace    , :craig!     , 2, 1, 1, false)
  (:DiomWorkspace     , :diom!      , 1, 1, 0, true )
  (:LslqWorkspace     , :lslq!      , 1, 1, 1, false)
  (:TrilqrWorkspace   , :trilqr!    , 2, 1, 1, true )
  (:CrmrWorkspace     , :crmr!      , 1, 1, 1, false)
  (:CgWorkspace       , :cg!        , 1, 1, 0, true )
  (:CglsWorkspace     , :cgls!      , 1, 1, 1, false)
  (:CgLanczosWorkspace, :cg_lanczos!, 1, 1, 0, true )
  (:BilqWorkspace     , :bilq!      , 1, 1, 1, true )
  (:MinresQlpWorkspace, :minres_qlp!, 1, 1, 0, true )
  (:QmrWorkspace      , :qmr!       , 1, 1, 1, true )
  (:GmresWorkspace    , :gmres!     , 1, 1, 0, true )
  (:FgmresWorkspace   , :fgmres!    , 1, 1, 0, true )
  (:FomWorkspace      , :fom!       , 1, 1, 0, true )
  (:GpmrWorkspace     , :gpmr!      , 2, 1, 0, true )
  (:CgLanczosShiftWorkspace  , :cg_lanczos_shift!  , 1, 1, 0, false)
  (:CglsLanczosShiftWorkspace, :cgls_lanczos_shift!, 1, 1, 1, false)
]
  @eval begin
    elapsed_time(workspace :: $KS) = workspace.stats.timer
    statistics(workspace :: $KS) = workspace.stats
    solution_count(workspace :: $KS) = $nsol
    iteration_count(workspace :: $KS) = workspace.stats.niter
    Aprod_count(workspace :: $KS) = $nA * workspace.stats.niter
    Atprod_count(workspace :: $KS) = $nAt * workspace.stats.niter
    if $nsol == 1
      solution(workspace :: $KS) = workspace.x
      solution(workspace :: $KS, p :: Integer) = (p == 1) ? solution(workspace) : error("solution(workspace) has only one output.")
      results(workspace :: $KS) = (workspace.x, workspace.stats)
    end
    if $nsol == 2
      solution(workspace :: $KS) = (workspace.x, workspace.y)
      solution(workspace :: $KS, p :: Integer) = (1 ≤ p ≤ 2) ? solution(workspace)[p] : error("solution(workspace) has only two outputs.")
      results(workspace :: $KS) = (workspace.x, workspace.y, workspace.stats)
    end
    if $KS ∈ (BilqrWorkspace, TrilqrWorkspace)
      issolved_primal(workspace :: $KS) = workspace.stats.solved_primal
      issolved_dual(workspace :: $KS) = workspace.stats.solved_dual
      issolved(workspace :: $KS) = issolved_primal(workspace) && issolved_dual(workspace)
    else
      issolved(workspace :: $KS) = workspace.stats.solved
    end
    if $warm_start
      if $KS in (BilqrWorkspace, TrilqrWorkspace)
        function warm_start!(workspace :: $KS, x0, y0)
          length(x0) == workspace.n || error("x0 should have size $(workspace.n)")
          length(y0) == workspace.m || error("y0 should have size $(workspace.m)")
          S = typeof(workspace.x)
          allocate_if(true, workspace, :Δx, S, workspace.x)  # The length of Δx is n
          allocate_if(true, workspace, :Δy, S, workspace.y)  # The length of Δy is m
          kcopy!(workspace.n, workspace.Δx, x0)
          kcopy!(workspace.m, workspace.Δy, y0)
          workspace.warm_start = true
          return workspace
        end
      elseif $KS in (TricgWorkspace, TrimrWorkspace, GpmrWorkspace)
        function warm_start!(workspace :: $KS, x0, y0)
          length(x0) == workspace.m || error("x0 should have size $(workspace.m)")
          length(y0) == workspace.n || error("y0 should have size $(workspace.n)")
          S = typeof(workspace.x)
          allocate_if(true, workspace, :Δx, S, workspace.x)  # The length of Δx is m
          allocate_if(true, workspace, :Δy, S, workspace.y)  # The length of Δy is n
          kcopy!(workspace.m, workspace.Δx, x0)
          kcopy!(workspace.n, workspace.Δy, y0)
          workspace.warm_start = true
          return workspace
        end
      else
        function warm_start!(workspace :: $KS, x0)
          S = typeof(workspace.x)
          length(x0) == workspace.n || error("x0 should have size $n")
          allocate_if(true, workspace, :Δx, S, workspace.x)  # The length of Δx is n
          kcopy!(workspace.n, workspace.Δx, x0)
          workspace.warm_start = true
          return workspace
        end
      end
    end
  end
end

# Block Krylov methods
for (KS, fun, nsol, nA, nAt, warm_start) in [
  (:BlockMinresWorkspace, :block_minres!, 1, 1, 0, true)
  (:BlockGmresWorkspace , :block_gmres! , 1, 1, 0, true)
]
  @eval begin
    elapsed_time(workspace :: $KS) = workspace.stats.timer
    statistics(workspace :: $KS) = workspace.stats
    solution_count(workspace :: $KS) = $nsol
    iteration_count(workspace :: $KS) = workspace.stats.niter
    Aprod_count(workspace :: $KS) = $nA * workspace.stats.niter
    Atprod_count(workspace :: $KS) = $nAt * workspace.stats.niter
    if $nsol == 1
      solution(workspace :: $KS) = workspace.X
      solution(workspace :: $KS, p :: Integer) = (p == 1) ? solution(workspace) : error("solution(workspace) has only one output.")
      results(workspace :: $KS) = (workspace.X, workspace.stats)
    end
    issolved(workspace :: $KS) = workspace.stats.solved
    if $warm_start
      function warm_start!(workspace :: $KS, X0)
        n2, p2 = size(X0)
        SM = typeof(workspace.X)
        (workspace.n == n2 && workspace.p == p2) || error("X0 should have size ($n, $p)")
        allocate_if(true, workspace, :ΔX, SM, workspace.n, workspace.p)
        copyto!(workspace.ΔX, X0)
        workspace.warm_start = true
        return workspace
      end
    end
  end
end
