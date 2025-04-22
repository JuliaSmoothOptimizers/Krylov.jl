# The purpose of this file is to automatically define all variants of in-place and out-of-place methods for Krylov solvers using metaprogramming.
# For example, the file `cg.jl` only implements the in-place method `cg!(workspace, A, b; kwargs...)`.
# This file generates additional variants of this method automatically, such as:
#
# - `cg(A, b; kwargs...)`
# - `krylov_solve(Val(:cg), A, b; kwargs...)`
# - `krylov_solve!(workspace, A, b; kwargs...)`
#
# Since the conjugate gradient method also supports an optional argument `x0`, additional methods are generated:
#
# - `cg(A, b, x0; kwargs...)`
# - `cg!(workspace, A, b, x0; kwargs...)`
# - `krylov_solve(Val(:cg), A, b, x0; kwargs...)`
# - `krylov_solve!(workspace, A, b, x0; kwargs...)`
#
# Generic constructors for each KrylovWorkspace and BlockKrylovWorkspace are also defined using metaprogramming.
# The function `krylov_workspace` uses the first argument, `Val(method)`, where `method` is a symbol used to dispatch to the appropriate method.
# For example, the three constructors for `CgWorkspace` can be called using the following syntax:
#
# - `krylov_workspace(Val(:cg), kc)`
# - `krylov_workspace(Val(:cg), m, n, S)`
# - `krylov_workspace(Val(:cg), A, b)`
#
# Alexis Montoison, <alexis.montoison@polymtl.ca> -- <amontoison@anl.gov>
# Chicago, October 2024 -- Chicago, April 2025.

export krylov_workspace, krylov_solve, krylov_solve!

"""
    workspace = krylov_workspace(Val(method), args...; kwargs...)

Generic function that dispatches to the appropriate workspace constructor for each subtype of [`KrylovWorkspace`](@ref) and [`BlockKrylovWorkspace`](@ref).
The first argument `Val(method)`, where `method` is a symbol (such as `:cg`, `:gmres` or `:block_minres`), specifies the (block) Krylov method for which a workspace is desired.
The returned workspace can later be used by [`krylov_solve!`](@ref) to execute the (block) Krylov method in-place.
"""
function krylov_workspace end

"""
    krylov_solve(Val(method), args...; kwargs...)

Generic function that dispatches to the appropriate out-of-place (block) Krylov method specified by symbol `method` (such as `:cg`, `:gmres` or `:block_minres`).
"""
function krylov_solve end

"""
    krylov_solve!(workspace, args...; kwargs...)

Generic function that dispatches to the appropriate in-place (block) Krylov method based on the type of `workspace`.
The argument `workspace` must be a subtype of [`KrylovWorkspace`](@ref) or [`BlockKrylovWorkspace`](@ref) (such as `CgWorkspace`, `GmresWorkspace` or `BlockMinresWorkspace`).
"""
function krylov_solve! end

# Krylov methods
for (workspace, krylov, args, def_args, optargs, def_optargs, kwargs, def_kwargs) in [
  (:LsmrWorkspace     , :lsmr      , args_lsmr      , def_args_lsmr      , ()                , ()                    , kwargs_lsmr      , def_kwargs_lsmr      )
  (:CgsWorkspace      , :cgs       , args_cgs       , def_args_cgs       , optargs_cgs       , def_optargs_cgs       , kwargs_cgs       , def_kwargs_cgs       )
  (:UsymlqWorkspace   , :usymlq    , args_usymlq    , def_args_usymlq    , optargs_usymlq    , def_optargs_usymlq    , kwargs_usymlq    , def_kwargs_usymlq    )
  (:LnlqWorkspace     , :lnlq      , args_lnlq      , def_args_lnlq      , ()                , ()                    , kwargs_lnlq      , def_kwargs_lnlq      )
  (:BicgstabWorkspace , :bicgstab  , args_bicgstab  , def_args_bicgstab  , optargs_bicgstab  , def_optargs_bicgstab  , kwargs_bicgstab  , def_kwargs_bicgstab  )
  (:CrlsWorkspace     , :crls      , args_crls      , def_args_crls      , ()                , ()                    , kwargs_crls      , def_kwargs_crls      )
  (:LsqrWorkspace     , :lsqr      , args_lsqr      , def_args_lsqr      , ()                , ()                    , kwargs_lsqr      , def_kwargs_lsqr      )
  (:MinresWorkspace   , :minres    , args_minres    , def_args_minres    , optargs_minres    , def_optargs_minres    , kwargs_minres    , def_kwargs_minres    )
  (:MinaresWorkspace  , :minares   , args_minares   , def_args_minares   , optargs_minares   , def_optargs_minares   , kwargs_minares   , def_kwargs_minares   )
  (:CgneWorkspace     , :cgne      , args_cgne      , def_args_cgne      , ()                , ()                    , kwargs_cgne      , def_kwargs_cgne      )
  (:DqgmresWorkspace  , :dqgmres   , args_dqgmres   , def_args_dqgmres   , optargs_dqgmres   , def_optargs_dqgmres   , kwargs_dqgmres   , def_kwargs_dqgmres   )
  (:SymmlqWorkspace   , :symmlq    , args_symmlq    , def_args_symmlq    , optargs_symmlq    , def_optargs_symmlq    , kwargs_symmlq    , def_kwargs_symmlq    )
  (:TrimrWorkspace    , :trimr     , args_trimr     , def_args_trimr     , optargs_trimr     , def_optargs_trimr     , kwargs_trimr     , def_kwargs_trimr     )
  (:UsymqrWorkspace   , :usymqr    , args_usymqr    , def_args_usymqr    , optargs_usymqr    , def_optargs_usymqr    , kwargs_usymqr    , def_kwargs_usymqr    )
  (:BilqrWorkspace    , :bilqr     , args_bilqr     , def_args_bilqr     , optargs_bilqr     , def_optargs_bilqr     , kwargs_bilqr     , def_kwargs_bilqr     )
  (:CrWorkspace       , :cr        , args_cr        , def_args_cr        , optargs_cr        , def_optargs_cr        , kwargs_cr        , def_kwargs_cr        )
  (:CarWorkspace      , :car       , args_car       , def_args_car       , optargs_car       , def_optargs_car       , kwargs_car       , def_kwargs_car       )
  (:CraigmrWorkspace  , :craigmr   , args_craigmr   , def_args_craigmr   , ()                , ()                    , kwargs_craigmr   , def_kwargs_craigmr   )
  (:TricgWorkspace    , :tricg     , args_tricg     , def_args_tricg     , optargs_tricg     , def_optargs_tricg     , kwargs_tricg     , def_kwargs_tricg     )
  (:CraigWorkspace    , :craig     , args_craig     , def_args_craig     , ()                , ()                    , kwargs_craig     , def_kwargs_craig     )
  (:DiomWorkspace     , :diom      , args_diom      , def_args_diom      , optargs_diom      , def_optargs_diom      , kwargs_diom      , def_kwargs_diom      )
  (:LslqWorkspace     , :lslq      , args_lslq      , def_args_lslq      , ()                , ()                    , kwargs_lslq      , def_kwargs_lslq      )
  (:TrilqrWorkspace   , :trilqr    , args_trilqr    , def_args_trilqr    , optargs_trilqr    , def_optargs_trilqr    , kwargs_trilqr    , def_kwargs_trilqr    )
  (:CrmrWorkspace     , :crmr      , args_crmr      , def_args_crmr      , ()                , ()                    , kwargs_crmr      , def_kwargs_crmr      )
  (:CgWorkspace       , :cg        , args_cg        , def_args_cg        , optargs_cg        , def_optargs_cg        , kwargs_cg        , def_kwargs_cg        )
  (:CglsWorkspace     , :cgls      , args_cgls      , def_args_cgls      , ()                , ()                    , kwargs_cgls      , def_kwargs_cgls      )
  (:CgLanczosWorkspace, :cg_lanczos, args_cg_lanczos, def_args_cg_lanczos, optargs_cg_lanczos, def_optargs_cg_lanczos, kwargs_cg_lanczos, def_kwargs_cg_lanczos)
  (:BilqWorkspace     , :bilq      , args_bilq      , def_args_bilq      , optargs_bilq      , def_optargs_bilq      , kwargs_bilq      , def_kwargs_bilq      )
  (:MinresQlpWorkspace, :minres_qlp, args_minres_qlp, def_args_minres_qlp, optargs_minres_qlp, def_optargs_minres_qlp, kwargs_minres_qlp, def_kwargs_minres_qlp)
  (:QmrWorkspace      , :qmr       , args_qmr       , def_args_qmr       , optargs_qmr       , def_optargs_qmr       , kwargs_qmr       , def_kwargs_qmr       )
  (:GmresWorkspace    , :gmres     , args_gmres     , def_args_gmres     , optargs_gmres     , def_optargs_gmres     , kwargs_gmres     , def_kwargs_gmres     )
  (:FgmresWorkspace   , :fgmres    , args_fgmres    , def_args_fgmres    , optargs_fgmres    , def_optargs_fgmres    , kwargs_fgmres    , def_kwargs_fgmres    )
  (:FomWorkspace      , :fom       , args_fom       , def_args_fom       , optargs_fom       , def_optargs_fom       , kwargs_fom       , def_kwargs_fom       )
  (:GpmrWorkspace     , :gpmr      , args_gpmr      , def_args_gpmr      , optargs_gpmr      , def_optargs_gpmr      , kwargs_gpmr      , def_kwargs_gpmr      )
  (:CgLanczosShiftWorkspace  , :cg_lanczos_shift  , args_cg_lanczos_shift  , def_args_cg_lanczos_shift  , (), (), kwargs_cg_lanczos_shift  , def_kwargs_cg_lanczos_shift  )
  (:CglsLanczosShiftWorkspace, :cgls_lanczos_shift, args_cgls_lanczos_shift, def_args_cgls_lanczos_shift, (), (), kwargs_cgls_lanczos_shift, def_kwargs_cgls_lanczos_shift)
]
  # Create the symbol for the in-place method
  krylov! = Symbol(krylov, :!)

  ## Generic constructors for each subtype of KrylovWorkspace
  if krylov in (:cg_lanczos_shift, :cgls_lanczos_shift)
    @eval krylov_workspace(::Val{Symbol($krylov)}, kc::KrylovConstructor, nshifts::Integer) = $workspace(kc, nshifts)
    @eval krylov_workspace(::Val{Symbol($krylov)}, m::Integer, n::Integer, nshifts::Integer, S::Type) = $workspace(m, n, nshifts, S)
    @eval krylov_workspace(::Val{Symbol($krylov)}, A, b, nshifts::Integer) = $workspace(A, b, nshifts)
  elseif krylov in (:diom, :dqgmres, :fom, :gmres, :fgmres, :gpmr)
    @eval krylov_workspace(::Val{Symbol($krylov)}, kc::KrylovConstructor; memory::Integer = 20) = $workspace(kc; memory)
    @eval krylov_workspace(::Val{Symbol($krylov)}, m::Integer, n::Integer, S::Type; memory::Integer = 20) = $workspace(m, n, S; memory)
    @eval krylov_workspace(::Val{Symbol($krylov)}, A, b; memory::Integer = 20) = $workspace(A, b; memory)
  elseif krylov in (:lslq, :lsmr, :lsqr, :minres, :symmlq)
    @eval krylov_workspace(::Val{Symbol($krylov)}, kc::KrylovConstructor; window::Integer = 5) = $workspace(kc; window)
    @eval krylov_workspace(::Val{Symbol($krylov)}, m::Integer, n::Integer, S::Type; window::Integer = 5) = $workspace(m, n, S; window)
    @eval krylov_workspace(::Val{Symbol($krylov)}, A, b; window::Integer = 5) = $workspace(A, b; window)
  else
    @eval krylov_workspace(::Val{Symbol($krylov)}, kc::KrylovConstructor) = $workspace(kc)
    @eval krylov_workspace(::Val{Symbol($krylov)}, m::Integer, n::Integer, S::Type) = $workspace(m, n, S)
    @eval krylov_workspace(::Val{Symbol($krylov)}, A, b) = $workspace(A, b)
  end

  ## Out-of-place
  if krylov in (:cg_lanczos_shift, :cgls_lanczos_shift)
    @eval begin
      function $(krylov)($(def_args...); $(def_kwargs...)) where {T <: AbstractFloat, FC <: FloatOrComplex{T}}
        start_time = time_ns()
        nshifts = length(shifts)
        workspace = $workspace(A, b, nshifts)
        elapsed_time = start_time |> ktimer
        timemax -= elapsed_time
        $(krylov!)(workspace, $(args...); $(kwargs...))
        workspace.stats.timer += elapsed_time
        return results(workspace)
      end

      krylov_solve(::Val{Symbol($krylov)}, $(def_args...); $(def_kwargs...)) where {T <: AbstractFloat, FC <: FloatOrComplex{T}} = $(krylov)($(args...); $(kwargs...))
    end
  elseif krylov in (:diom, :dqgmres, :fom, :gmres, :fgmres, :gpmr)
    @eval begin
      function $(krylov)($(def_args...); memory::Int = 20, $(def_kwargs...)) where {T <: AbstractFloat, FC <: FloatOrComplex{T}}
        start_time = time_ns()
        workspace = $workspace(A, b; memory)
        elapsed_time = start_time |> ktimer
        timemax -= elapsed_time
        $(krylov!)(workspace, $(args...); $(kwargs...))
        workspace.stats.timer += elapsed_time
        return results(workspace)
      end

      krylov_solve(::Val{Symbol($krylov)}, $(def_args...); memory::Int = 20, $(def_kwargs...)) where {T <: AbstractFloat, FC <: FloatOrComplex{T}} = $(krylov)($(args...); memory, $(kwargs...))

      if !isempty($optargs)
        function $(krylov)($(def_args...), $(def_optargs...); memory::Int = 20, $(def_kwargs...)) where {T <: AbstractFloat, FC <: FloatOrComplex{T}}
          start_time = time_ns()
          workspace = $workspace(A, b; memory)
          warm_start!(workspace, $(optargs...))
          elapsed_time = start_time |> ktimer
          timemax -= elapsed_time
          $(krylov!)(workspace, $(args...); $(kwargs...))
          workspace.stats.timer += elapsed_time
          return results(workspace)
        end

        krylov_solve(::Val{Symbol($krylov)}, $(def_args...), $(def_optargs...); memory::Int = 20, $(def_kwargs...)) where {T <: AbstractFloat, FC <: FloatOrComplex{T}} = $(krylov)($(args...), $(optargs...); memory, $(kwargs...))
      end
    end
  elseif krylov in (:lslq, :lsmr, :lsqr, :minres, :symmlq)
    @eval begin
      function $(krylov)($(def_args...); window::Int = 5, $(def_kwargs...)) where {T <: AbstractFloat, FC <: FloatOrComplex{T}}
        start_time = time_ns()
        workspace = $workspace(A, b; window)
        elapsed_time = start_time |> ktimer
        timemax -= elapsed_time
        $(krylov!)(workspace, $(args...); $(kwargs...))
        workspace.stats.timer += elapsed_time
        return results(workspace)
      end

      krylov_solve(::Val{Symbol($krylov)}, $(def_args...); window::Int = 5, $(def_kwargs...)) where {T <: AbstractFloat, FC <: FloatOrComplex{T}} = $(krylov)($(args...); window, $(kwargs...))

      if !isempty($optargs)
        function $(krylov)($(def_args...), $(def_optargs...); window::Int = 5, $(def_kwargs...)) where {T <: AbstractFloat, FC <: FloatOrComplex{T}}
          start_time = time_ns()
          workspace = $workspace(A, b; window)
          warm_start!(workspace, $(optargs...))
          elapsed_time = start_time |> ktimer
          timemax -= elapsed_time
          $(krylov!)(workspace, $(args...); $(kwargs...))
          workspace.stats.timer += elapsed_time
          return results(workspace)
        end

        krylov_solve(::Val{Symbol($krylov)}, $(def_args...), $(def_optargs...); window::Int = 5, $(def_kwargs...)) where {T <: AbstractFloat, FC <: FloatOrComplex{T}} = $(krylov)($(args...), $(optargs...); window, $(kwargs...))
      end
    end
  else
    @eval begin
      function $(krylov)($(def_args...); $(def_kwargs...)) where {T <: AbstractFloat, FC <: FloatOrComplex{T}}
        start_time = time_ns()
        workspace = $workspace(A, b)
        elapsed_time = start_time |> ktimer
        timemax -= elapsed_time
        $(krylov!)(workspace, $(args...); $(kwargs...))
        workspace.stats.timer += elapsed_time
        return results(workspace)
      end

      krylov_solve(::Val{Symbol($krylov)}, $(def_args...); $(def_kwargs...)) where {T <: AbstractFloat, FC <: FloatOrComplex{T}} = $(krylov)($(args...); $(kwargs...))

      if !isempty($optargs)
        function $(krylov)($(def_args...), $(def_optargs...); $(def_kwargs...)) where {T <: AbstractFloat, FC <: FloatOrComplex{T}}
          start_time = time_ns()
          workspace = $workspace(A, b)
          warm_start!(workspace, $(optargs...))
          elapsed_time = start_time |> ktimer
          timemax -= elapsed_time
          $(krylov!)(workspace, $(args...); $(kwargs...))
          workspace.stats.timer += elapsed_time
          return results(workspace)
        end

        krylov_solve(::Val{Symbol($krylov)}, $(def_args...), $(def_optargs...); $(def_kwargs...)) where {T <: AbstractFloat, FC <: FloatOrComplex{T}} = $(krylov)($(args...), $(optargs...); $(kwargs...))
      end
    end
  end

  ## In-place
  @eval krylov_solve!(workspace :: $workspace{T,FC,S}, $(def_args...); $(def_kwargs...)) where {T <: AbstractFloat, FC <: FloatOrComplex{T}, S <: AbstractVector{FC}} = $(krylov!)(workspace, $(args...); $(kwargs...))

  for krylov_ip in (:krylov_solve!, krylov!)
    @eval begin
      if !isempty($optargs)
        function $(krylov_ip)(workspace :: $workspace{T,FC,S}, $(def_args...), $(def_optargs...); $(def_kwargs...)) where {T <: AbstractFloat, FC <: FloatOrComplex{T}, S <: AbstractVector{FC}}
          start_time = time_ns()
          warm_start!(workspace, $(optargs...))
          elapsed_time = start_time |> ktimer
          timemax -= elapsed_time
          $(krylov!)(workspace, $(args...); $(kwargs...))
          workspace.stats.timer += elapsed_time
          return workspace
        end
      end
    end
  end
end

# Block Krylov methods
for (workspace, krylov, args, def_args, optargs, def_optargs, kwargs, def_kwargs) in [
  (:BlockMinresWorkspace, :block_minres, args_block_minres, def_args_block_minres, optargs_block_minres, def_optargs_block_minres, kwargs_block_minres, def_kwargs_block_minres)
  (:BlockGmresWorkspace , :block_gmres , args_block_gmres , def_args_block_gmres , optargs_block_gmres , def_optargs_block_gmres , kwargs_block_gmres , def_kwargs_block_gmres )
]
  # Create the symbol for the in-place method
  krylov! = Symbol(krylov, :!)

  ## Generic constructors for each subtype of BlockKrylovWorkspace
  if krylov == :block_gmres
      @eval krylov_workspace(::Val{Symbol($krylov)}, m::Integer, n::Integer, p::Integer, SV::Type, SM::Type; memory::Integer = 5) = $workspace(m, n, p, SV, SM; memory)
      @eval krylov_workspace(::Val{Symbol($krylov)}, A, B; memory::Integer = 5) = $workspace(A, B; memory)
  else
      @eval krylov_workspace(::Val{Symbol($krylov)}, m::Integer, n::Integer, p::Integer, SV::Type, SM::Type) = $workspace(m, n, p, SV, SM)
      @eval krylov_workspace(::Val{Symbol($krylov)}, A, B) = $workspace(A, B)
  end

  ## Out-of-place
  if krylov == :block_gmres
    @eval begin
      function $(krylov)($(def_args...); memory::Int = 5, $(def_kwargs...)) where {T <: AbstractFloat, FC <: FloatOrComplex{T}}
        start_time = time_ns()
        workspace = $workspace(A, B; memory)
        elapsed_time = ktimer(start_time)
        timemax -= elapsed_time
        $(krylov!)(workspace, $(args...); $(kwargs...))
        workspace.stats.timer += elapsed_time
        return results(workspace)
      end

      krylov_solve(::Val{Symbol($krylov)}, $(def_args...); memory::Int = 5, $(def_kwargs...)) where {T <: AbstractFloat, FC <: FloatOrComplex{T}} = $(krylov)($(args...); memory, $(kwargs...))

      if !isempty($optargs)
        function $(krylov)($(def_args...), $(def_optargs...); memory::Int = 5, $(def_kwargs...)) where {T <: AbstractFloat, FC <: FloatOrComplex{T}}
          start_time = time_ns()
          workspace = $workspace(A, B; memory)
          warm_start!(workspace, $(optargs...))
          elapsed_time = ktimer(start_time)
          timemax -= elapsed_time
          $(krylov!)(workspace, $(args...); $(kwargs...))
          workspace.stats.timer += elapsed_time
          return results(workspace)
        end

        krylov_solve(::Val{Symbol($krylov)}, $(def_args...), $(def_optargs...); memory::Int = 5, $(def_kwargs...)) where {T <: AbstractFloat, FC <: FloatOrComplex{T}} = $(krylov)($(args...), $(optargs...); memory, $(kwargs...))
      end
    end
  else
    @eval begin
      function $(krylov)($(def_args...); $(def_kwargs...)) where {T <: AbstractFloat, FC <: FloatOrComplex{T}}
        start_time = time_ns()
        workspace = $workspace(A, B)
        elapsed_time = ktimer(start_time)
        timemax -= elapsed_time
        $(krylov!)(workspace, $(args...); $(kwargs...))
        workspace.stats.timer += elapsed_time
        return results(workspace)
      end

      krylov_solve(::Val{Symbol($krylov)}, $(def_args...); $(def_kwargs...)) where {T <: AbstractFloat, FC <: FloatOrComplex{T}} = $(krylov)($(args...); $(kwargs...))

      if !isempty($optargs)
        function $(krylov)($(def_args...), $(def_optargs...); $(def_kwargs...)) where {T <: AbstractFloat, FC <: FloatOrComplex{T}}
          start_time = time_ns()
          workspace = $workspace(A, B)
          warm_start!(workspace, $(optargs...))
          elapsed_time = ktimer(start_time)
          timemax -= elapsed_time
          $(krylov!)(workspace, $(args...); $(kwargs...))
          workspace.stats.timer += elapsed_time
          return results(workspace)
        end

        krylov_solve(::Val{Symbol($krylov)}, $(def_args...), $(def_optargs...); $(def_kwargs...)) where {T <: AbstractFloat, FC <: FloatOrComplex{T}} = $(krylov)($(args...), $(optargs...); $(kwargs...))
      end
    end
  end

  ## In-place
  @eval krylov_solve!(workspace :: $workspace{T,FC,SV,SM}, $(def_args...); $(def_kwargs...)) where {T <: AbstractFloat, FC <: FloatOrComplex{T}, SV <: AbstractVector{FC}, SM <: AbstractMatrix{FC}} = $(krylov!)(workspace, $(args...); $(kwargs...))

  for krylov_ip in (:krylov_solve!, krylov!)
    @eval begin
      if !isempty($optargs)
        function $(krylov_ip)(workspace :: $workspace{T,FC,SV,SM}, $(def_args...), $(def_optargs...); $(def_kwargs...)) where {T <: AbstractFloat, FC <: FloatOrComplex{T}, SV <: AbstractVector{FC}, SM <: AbstractMatrix{FC}}
          start_time = time_ns()
          warm_start!(workspace, $(optargs...))
          elapsed_time = start_time |> ktimer
          timemax -= elapsed_time
          $(krylov!)(workspace, $(args...); $(kwargs...))
          workspace.stats.timer += elapsed_time
          return workspace
        end
      end
    end
  end
end
