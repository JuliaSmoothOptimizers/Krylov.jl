"""
    solve!(solver, args...; kwargs...)

Use the in-place Krylov method associated to `solver`.
"""
function solve! end

for (KS, fun, args, def_args, kwargs, def_kwargs) in [
  (:LsmrSolver          , :lsmr!            , args_lsmr            , def_args_lsmr            , kwargs_lsmr            , def_kwargs_lsmr            )
  (:CgsSolver           , :cgs!             , args_cgs             , def_args_cgs             , kwargs_cgs             , def_kwargs_cgs             )
  (:UsymlqSolver        , :usymlq!          , args_usymlq          , def_args_usymlq          , kwargs_usymlq          , def_kwargs_usymlq          )
  (:LnlqSolver          , :lnlq!            , args_lnlq            , def_args_lnlq            , kwargs_lnlq            , def_kwargs_lnlq            )
  (:BicgstabSolver      , :bicgstab!        , args_bicgstab        , def_args_bicgstab        , kwargs_bicgstab        , def_kwargs_bicgstab        )
  (:CrlsSolver          , :crls!            , args_crls            , def_args_crls            , kwargs_crls            , def_kwargs_crls            )
  (:LsqrSolver          , :lsqr!            , args_lsqr            , def_args_lsqr            , kwargs_lsqr            , def_kwargs_lsqr            )
  (:MinresSolver        , :minres!          , args_minres          , def_args_minres          , kwargs_minres          , def_kwargs_minres          )
  (:CgneSolver          , :cgne!            , args_cgne            , def_args_cgne            , kwargs_cgne            , def_kwargs_cgne            )
  (:DqgmresSolver       , :dqgmres!         , args_dqgmres         , def_args_dqgmres         , kwargs_dqgmres         , def_kwargs_dqgmres         )
  (:SymmlqSolver        , :symmlq!          , args_symmlq          , def_args_symmlq          , kwargs_symmlq          , def_kwargs_symmlq          )
  (:TrimrSolver         , :trimr!           , args_trimr           , def_args_trimr           , kwargs_trimr           , def_kwargs_trimr           )
  (:UsymqrSolver        , :usymqr!          , args_usymqr          , def_args_usymqr          , kwargs_usymqr          , def_kwargs_usymqr          )
  (:BilqrSolver         , :bilqr!           , args_bilqr           , def_args_bilqr           , kwargs_bilqr           , def_kwargs_bilqr           )
  (:CrSolver            , :cr!              , args_cr              , def_args_cr              , kwargs_cr              , def_kwargs_cr              )
  (:CraigmrSolver       , :craigmr!         , args_craigmr         , def_args_craigmr         , kwargs_craigmr         , def_kwargs_craigmr         )
  (:TricgSolver         , :tricg!           , args_tricg           , def_args_tricg           , kwargs_tricg           , def_kwargs_tricg           )
  (:CraigSolver         , :craig!           , args_craig           , def_args_craig           , kwargs_craig           , def_kwargs_craig           )
  (:DiomSolver          , :diom!            , args_diom            , def_args_diom            , kwargs_diom            , def_kwargs_diom            )
  (:LslqSolver          , :lslq!            , args_lslq            , def_args_lslq            , kwargs_lslq            , def_kwargs_lslq            )
  (:TrilqrSolver        , :trilqr!          , args_trilqr          , def_args_trilqr          , kwargs_trilqr          , def_kwargs_trilqr          )
  (:CrmrSolver          , :crmr!            , args_crmr            , def_args_crmr            , kwargs_crmr            , def_kwargs_crmr            )
  (:CgSolver            , :cg!              , args_cg              , def_args_cg              , kwargs_cg              , def_kwargs_cg              )
  (:CgLanczosShiftSolver, :cg_lanczos_shift!, args_cg_lanczos_shift, def_args_cg_lanczos_shift, kwargs_cg_lanczos_shift, def_kwargs_cg_lanczos_shift)
  (:CglsSolver          , :cgls!            , args_cgls            , def_args_cgls            , kwargs_cgls            , def_kwargs_cgls            )
  (:CgLanczosSolver     , :cg_lanczos!      , args_cg_lanczos      , def_args_cg_lanczos      , kwargs_cg_lanczos      , def_kwargs_cg_lanczos      )
  (:BilqSolver          , :bilq!            , args_bilq            , def_args_bilq            , kwargs_bilq            , def_kwargs_bilq            )
  (:MinresQlpSolver     , :minres_qlp!      , args_minres_qlp      , def_args_minres_qlp      , kwargs_minres_qlp      , def_kwargs_minres_qlp      )
  (:QmrSolver           , :qmr!             , args_qmr             , def_args_qmr             , kwargs_qmr             , def_kwargs_qmr             )
  (:GmresSolver         , :gmres!           , args_gmres           , def_args_gmres           , kwargs_gmres           , def_kwargs_gmres           )
  (:FgmresSolver        , :fgmres!          , args_fgmres          , def_args_fgmres          , kwargs_fgmres          , def_kwargs_fgmres          )
  (:FomSolver           , :fom!             , args_fom             , def_args_fom             , kwargs_fom             , def_kwargs_fom             )
  (:GpmrSolver          , :gpmr!            , args_gpmr            , def_args_gpmr            , kwargs_gpmr            , def_kwargs_gpmr            )
]
  @eval begin
    solve!(solver :: $KS{T,FC,S}, $(def_args...); $(def_kwargs...)) where {T <: AbstractFloat, FC <: FloatOrComplex{T}, S <: AbstractVector{FC}} = $(fun)(solver, $(args...); $(kwargs...))
  end
end
