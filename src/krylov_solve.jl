"""
    solve!(solver, args...; kwargs...)

Use the in-place Krylov method associated to `solver`.
"""
function solve! end

for (KS, fun, args, def_args, optargs, def_optargs, kwargs, def_kwargs) in [
  (:LsmrSolver          , :lsmr!            , args_lsmr            , def_args_lsmr            , ()                , ()                    , kwargs_lsmr            , def_kwargs_lsmr            )
  (:CgsSolver           , :cgs!             , args_cgs             , def_args_cgs             , optargs_cgs       , def_optargs_cgs       , kwargs_cgs             , def_kwargs_cgs             )
  (:UsymlqSolver        , :usymlq!          , args_usymlq          , def_args_usymlq          , optargs_usymlq    , def_optargs_usymlq    , kwargs_usymlq          , def_kwargs_usymlq          )
  (:LnlqSolver          , :lnlq!            , args_lnlq            , def_args_lnlq            , ()                , ()                    , kwargs_lnlq            , def_kwargs_lnlq            )
  (:BicgstabSolver      , :bicgstab!        , args_bicgstab        , def_args_bicgstab        , optargs_bicgstab  , def_optargs_bicgstab  , kwargs_bicgstab        , def_kwargs_bicgstab        )
  (:CrlsSolver          , :crls!            , args_crls            , def_args_crls            , ()                , ()                    , kwargs_crls            , def_kwargs_crls            )
  (:LsqrSolver          , :lsqr!            , args_lsqr            , def_args_lsqr            , ()                , ()                    , kwargs_lsqr            , def_kwargs_lsqr            )
  (:MinresSolver        , :minres!          , args_minres          , def_args_minres          , optargs_minres    , def_optargs_minres    , kwargs_minres          , def_kwargs_minres          )
  (:MinaresSolver       , :minares!         , args_minares         , def_args_minares         , optargs_minares   , def_optargs_minares   , kwargs_minares         , def_kwargs_minares         )
  (:CgneSolver          , :cgne!            , args_cgne            , def_args_cgne            , ()                , ()                    , kwargs_cgne            , def_kwargs_cgne            )
  (:DqgmresSolver       , :dqgmres!         , args_dqgmres         , def_args_dqgmres         , optargs_dqgmres   , def_optargs_dqgmres   , kwargs_dqgmres         , def_kwargs_dqgmres         )
  (:SymmlqSolver        , :symmlq!          , args_symmlq          , def_args_symmlq          , optargs_symmlq    , def_optargs_symmlq    , kwargs_symmlq          , def_kwargs_symmlq          )
  (:TrimrSolver         , :trimr!           , args_trimr           , def_args_trimr           , optargs_trimr     , def_optargs_trimr     , kwargs_trimr           , def_kwargs_trimr           )
  (:UsymqrSolver        , :usymqr!          , args_usymqr          , def_args_usymqr          , optargs_usymqr    , def_optargs_usymqr    , kwargs_usymqr          , def_kwargs_usymqr          )
  (:BilqrSolver         , :bilqr!           , args_bilqr           , def_args_bilqr           , optargs_bilqr     , def_optargs_bilqr     , kwargs_bilqr           , def_kwargs_bilqr           )
  (:CrSolver            , :cr!              , args_cr              , def_args_cr              , optargs_cr        , def_optargs_cr        , kwargs_cr              , def_kwargs_cr              )
  (:CarSolver           , :car!             , args_car             , def_args_car             , optargs_car       , def_optargs_car       , kwargs_car             , def_kwargs_car             )
  (:CraigmrSolver       , :craigmr!         , args_craigmr         , def_args_craigmr         , ()                , ()                    , kwargs_craigmr         , def_kwargs_craigmr         )
  (:TricgSolver         , :tricg!           , args_tricg           , def_args_tricg           , optargs_tricg     , def_optargs_tricg     , kwargs_tricg           , def_kwargs_tricg           )
  (:CraigSolver         , :craig!           , args_craig           , def_args_craig           , ()                , ()                    , kwargs_craig           , def_kwargs_craig           )
  (:DiomSolver          , :diom!            , args_diom            , def_args_diom            , optargs_diom      , def_optargs_diom      , kwargs_diom            , def_kwargs_diom            )
  (:LslqSolver          , :lslq!            , args_lslq            , def_args_lslq            , ()                , ()                    , kwargs_lslq            , def_kwargs_lslq            )
  (:TrilqrSolver        , :trilqr!          , args_trilqr          , def_args_trilqr          , optargs_trilqr    , def_optargs_trilqr    , kwargs_trilqr          , def_kwargs_trilqr          )
  (:CrmrSolver          , :crmr!            , args_crmr            , def_args_crmr            , ()                , ()                    , kwargs_crmr            , def_kwargs_crmr            )
  (:CgSolver            , :cg!              , args_cg              , def_args_cg              , optargs_cg        , def_optargs_cg        , kwargs_cg              , def_kwargs_cg              )
  (:CgLanczosShiftSolver, :cg_lanczos_shift!, args_cg_lanczos_shift, def_args_cg_lanczos_shift, ()                , ()                    , kwargs_cg_lanczos_shift, def_kwargs_cg_lanczos_shift)
  (:CglsSolver          , :cgls!            , args_cgls            , def_args_cgls            , ()                , ()                    , kwargs_cgls            , def_kwargs_cgls            )
  (:CgLanczosSolver     , :cg_lanczos!      , args_cg_lanczos      , def_args_cg_lanczos      , optargs_cg_lanczos, def_optargs_cg_lanczos, kwargs_cg_lanczos      , def_kwargs_cg_lanczos      )
  (:BilqSolver          , :bilq!            , args_bilq            , def_args_bilq            , optargs_bilq      , def_optargs_bilq      , kwargs_bilq            , def_kwargs_bilq            )
  (:MinresQlpSolver     , :minres_qlp!      , args_minres_qlp      , def_args_minres_qlp      , optargs_minres_qlp, def_optargs_minres_qlp, kwargs_minres_qlp      , def_kwargs_minres_qlp      )
  (:QmrSolver           , :qmr!             , args_qmr             , def_args_qmr             , optargs_qmr       , def_optargs_qmr       , kwargs_qmr             , def_kwargs_qmr             )
  (:GmresSolver         , :gmres!           , args_gmres           , def_args_gmres           , optargs_gmres     , def_optargs_gmres     , kwargs_gmres           , def_kwargs_gmres           )
  (:FgmresSolver        , :fgmres!          , args_fgmres          , def_args_fgmres          , optargs_fgmres    , def_optargs_fgmres    , kwargs_fgmres          , def_kwargs_fgmres          )
  (:FomSolver           , :fom!             , args_fom             , def_args_fom             , optargs_fom       , def_optargs_fom       , kwargs_fom             , def_kwargs_fom             )
  (:GpmrSolver          , :gpmr!            , args_gpmr            , def_args_gpmr            , optargs_gpmr      , def_optargs_gpmr      , kwargs_gpmr            , def_kwargs_gpmr            )
]
  @eval begin
    solve!(solver :: $KS{T,FC,S}, $(def_args...); $(def_kwargs...)) where {T <: AbstractFloat, FC <: FloatOrComplex{T}, S <: AbstractVector{FC}} = $(fun)(solver, $(args...); $(kwargs...))

    if !isempty($optargs)
      function $(fun)(solver :: $KS{T,FC,S}, $(def_args...), $(def_optargs...); $(def_kwargs...)) where {T <: AbstractFloat, FC <: FloatOrComplex{T}, S <: AbstractVector{FC}}
        start_time = time_ns()
        warm_start!(solver, $(optargs...))
        elapsed_time = ktimer(start_time)
        timemax -= elapsed_time
        $(fun)(solver, $(args...); $(kwargs...))
        solver.stats.timer += elapsed_time
        return solver
      end

      solve!(solver :: $KS{T,FC,S}, $(def_args...), $(def_optargs...); $(def_kwargs...)) where {T <: AbstractFloat, FC <: FloatOrComplex{T}, S <: AbstractVector{FC}} = $(fun)(solver, $(args...), $(optargs...); $(kwargs...))
    end
  end
end
