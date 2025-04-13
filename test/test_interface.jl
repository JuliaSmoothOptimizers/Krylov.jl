function test_krylov_solvers(FC; krylov_constructor::Bool=false)
  A   = FC.(get_div_grad(4, 4, 4))  # Dimension n x n
  n   = size(A, 1)
  m   = div(n, 2)
  Au  = A[1:m,:]  # Dimension m x n
  Ao  = A[:,1:m]  # Dimension n x m
  b   = Ao * ones(FC, m) # Dimension n
  c   = Au * ones(FC, n) # Dimension m
  mem = 10
  shifts = [1.0; 2.0; 3.0; 4.0; 5.0]
  nshifts = 5
  T = real(FC)
  S = Vector{FC}
  solvers = Dict{Symbol, KrylovWorkspace}()

  if krylov_constructor
    kc_nn = KrylovConstructor(b)
    kc_mn = KrylovConstructor(c, b)
    kc_nm = KrylovConstructor(b, c)
    solvers[:cg] = @inferred krylov_workspace(Val{:cg}(), kc_nn)
    solvers[:car] = @inferred krylov_workspace(Val{:car}(), kc_nn)
    solvers[:symmlq] = @inferred krylov_workspace(Val{:symmlq}(), kc_nn)
    solvers[:minres] = @inferred krylov_workspace(Val{:minres}(), kc_nn)
    solvers[:minares] = @inferred krylov_workspace(Val{:minares}(), kc_nn)
    solvers[:cg_lanczos] = @inferred krylov_workspace(Val{:cg_lanczos}(), kc_nn)
    solvers[:diom] = @inferred krylov_workspace(Val{:diom}(), kc_nn; memory=mem)
    solvers[:fom] = @inferred krylov_workspace(Val{:fom}(), kc_nn; memory=mem)
    solvers[:dqgmres] = @inferred krylov_workspace(Val{:dqgmres}(), kc_nn; memory=mem)
    solvers[:gmres] = @inferred krylov_workspace(Val{:gmres}(), kc_nn; memory=mem)
    solvers[:gpmr] = @inferred krylov_workspace(Val{:gpmr}(), kc_nm; memory=mem)
    solvers[:fgmres] = @inferred krylov_workspace(Val{:fgmres}(), kc_nn; memory=mem)
    solvers[:cr] = @inferred krylov_workspace(Val{:cr}(), kc_nn)
    solvers[:crmr] = @inferred krylov_workspace(Val{:crmr}(), kc_mn)
    solvers[:cgs] = @inferred krylov_workspace(Val{:cgs}(), kc_nn)
    solvers[:bicgstab] = @inferred krylov_workspace(Val{:bicgstab}(), kc_nn)
    solvers[:craigmr] = @inferred krylov_workspace(Val{:craigmr}(), kc_mn)
    solvers[:cgne] = @inferred krylov_workspace(Val{:cgne}(), kc_mn)
    solvers[:lnlq] = @inferred krylov_workspace(Val{:lnlq}(), kc_mn)
    solvers[:craig] = @inferred krylov_workspace(Val{:craig}(), kc_mn)
    solvers[:lslq] = @inferred krylov_workspace(Val{:lslq}(), kc_nm)
    solvers[:cgls] = @inferred krylov_workspace(Val{:cgls}(), kc_nm)
    solvers[:lsqr] = @inferred krylov_workspace(Val{:lsqr}(), kc_nm)
    solvers[:crls] = @inferred krylov_workspace(Val{:crls}(), kc_nm)
    solvers[:lsmr] = @inferred krylov_workspace(Val{:lsmr}(), kc_nm)
    solvers[:usymqr] = @inferred krylov_workspace(Val{:usymqr}(), kc_nm)
    solvers[:trilqr] = @inferred krylov_workspace(Val{:trilqr}(), kc_nn)
    solvers[:bilq] = @inferred krylov_workspace(Val{:bilq}(), kc_nn)
    solvers[:bilqr] = @inferred krylov_workspace(Val{:bilqr}(), kc_nn)
    solvers[:minres_qlp] = @inferred krylov_workspace(Val{:minres_qlp}(), kc_nn)
    solvers[:qmr] = @inferred krylov_workspace(Val{:qmr}(), kc_nn)
    solvers[:usymlq] = @inferred krylov_workspace(Val{:usymlq}(), kc_mn)
    solvers[:tricg] = @inferred krylov_workspace(Val{:tricg}(), kc_mn)
    solvers[:trimr] = @inferred krylov_workspace(Val{:trimr}(), kc_mn)
    solvers[:cg_lanczos_shift] = @inferred krylov_workspace(Val{:cg_lanczos_shift}(), kc_nn, nshifts)
    solvers[:cgls_lanczos_shift] = @inferred krylov_workspace(Val{:cgls_lanczos_shift}(), kc_nm, nshifts)
  else
    solvers[:cg] = @inferred krylov_workspace(Val{:cg}(), n, n, S)
    solvers[:car] = @inferred krylov_workspace(Val{:car}(), n, n, S)
    solvers[:symmlq] = @inferred krylov_workspace(Val{:symmlq}(), n, n, S)
    solvers[:minres] = @inferred krylov_workspace(Val{:minres}(), n, n, S)
    solvers[:minares] = @inferred krylov_workspace(Val{:minares}(), n, n, S)
    solvers[:cg_lanczos] = @inferred krylov_workspace(Val{:cg_lanczos}(), n, n, S)
    solvers[:diom] = @inferred krylov_workspace(Val{:diom}(), n, n, S; memory=mem)
    solvers[:fom] = @inferred krylov_workspace(Val{:fom}(), n, n, S; memory=mem)
    solvers[:dqgmres] = @inferred krylov_workspace(Val{:dqgmres}(), n, n, S; memory=mem)
    solvers[:gmres] = @inferred krylov_workspace(Val{:gmres}(), n, n, S; memory=mem)
    solvers[:gpmr] = @inferred krylov_workspace(Val{:gpmr}(), n, m, S; memory=mem)
    solvers[:fgmres] = @inferred krylov_workspace(Val{:fgmres}(), n, n, S; memory=mem)
    solvers[:cr] = @inferred krylov_workspace(Val{:cr}(), n, n, S)
    solvers[:crmr] = @inferred krylov_workspace(Val{:crmr}(), m, n, S)
    solvers[:cgs] = @inferred krylov_workspace(Val{:cgs}(), n, n, S)
    solvers[:bicgstab] = @inferred krylov_workspace(Val{:bicgstab}(), n, n, S)
    solvers[:craigmr] = @inferred krylov_workspace(Val{:craigmr}(), m, n, S)
    solvers[:cgne] = @inferred krylov_workspace(Val{:cgne}(), m, n, S)
    solvers[:lnlq] = @inferred krylov_workspace(Val{:lnlq}(), m, n, S)
    solvers[:craig] = @inferred krylov_workspace(Val{:craig}(), m, n, S)
    solvers[:lslq] = @inferred krylov_workspace(Val{:lslq}(), n, m, S)
    solvers[:cgls] = @inferred krylov_workspace(Val{:cgls}(), n, m, S)
    solvers[:lsqr] = @inferred krylov_workspace(Val{:lsqr}(), n, m, S)
    solvers[:crls] = @inferred krylov_workspace(Val{:crls}(), n, m, S)
    solvers[:lsmr] = @inferred krylov_workspace(Val{:lsmr}(), n, m, S)
    solvers[:usymqr] = @inferred krylov_workspace(Val{:usymqr}(), n, m, S)
    solvers[:trilqr] = @inferred krylov_workspace(Val{:trilqr}(), n, n, S)
    solvers[:bilq] = @inferred krylov_workspace(Val{:bilq}(), n, n, S)
    solvers[:bilqr] = @inferred krylov_workspace(Val{:bilqr}(), n, n, S)
    solvers[:minres_qlp] = @inferred krylov_workspace(Val{:minres_qlp}(), n, n, S)
    solvers[:qmr] = @inferred krylov_workspace(Val{:qmr}(), n, n, S)
    solvers[:usymlq] = @inferred krylov_workspace(Val{:usymlq}(), m, n, S)
    solvers[:tricg] = @inferred krylov_workspace(Val{:tricg}(), m, n, S)
    solvers[:trimr] = @inferred krylov_workspace(Val{:trimr}(), m, n, S)
    solvers[:cg_lanczos_shift] = @inferred krylov_workspace(Val{:cg_lanczos_shift}(), n, n, nshifts, S)
    solvers[:cgls_lanczos_shift] = @inferred krylov_workspace(Val{:cgls_lanczos_shift}(), n, m, nshifts, S)
  end

  @testset "Check compatibility between each KrylovWorkspace and the dimension of the linear problem" begin
    A2  = FC.(get_div_grad(2, 2, 2))
    n2  = size(A2, 1)
    m2  = div(n2, 2)
    Au2 = A2[1:m2,:]
    Ao2 = A2[:,1:m2]
    b2  = Ao2 * ones(FC, m2)
    c2  = Au2 * ones(FC, n2)
    shifts2 = [1.0; 2.0; 3.0; 4.0; 5.0; 6.0]
    for (method, solver) in solvers
      if method ∈ (:cg, :cr, :car, :symmlq, :minares, :minres, :minres_qlp, :cg_lanczos, :diom, :fom, :dqgmres, :gmres, :fgmres, :cgs, :bicgstab, :bilq, :qmr)
        @test_throws ErrorException("(solver.m, solver.n) = ($(solver.m), $(solver.n)) is inconsistent with size(A) = ($n2, $n2)") krylov_solve!(solver, A2, b2)
      end
      method == :cg_lanczos_shift && @test_throws ErrorException("(solver.m, solver.n) = ($(solver.m), $(solver.n)) is inconsistent with size(A) = ($n2, $n2)") krylov_solve!(solver, A2, b2, shifts2)
      method == :cg_lanczos_shift && @test_throws ErrorException("solver.nshifts = $(solver.nshifts) is inconsistent with length(shifts) = $(length(shifts2))") krylov_solve!(solver, A, b, shifts2)
      method ∈ (:cgne, :crmr, :lnlq, :craig, :craigmr) && @test_throws ErrorException("(solver.m, solver.n) = ($(solver.m), $(solver.n)) is inconsistent with size(A) = ($m2, $n2)") krylov_solve!(solver, Au2, c2)
      method ∈ (:cgls, :crls, :lslq, :lsqr, :lsmr) && @test_throws ErrorException("(solver.m, solver.n) = ($(solver.m), $(solver.n)) is inconsistent with size(A) = ($n2, $m2)") krylov_solve!(solver, Ao2, b2)
      method == :cgls_lanczos_shift && @test_throws ErrorException("(solver.m, solver.n) = ($(solver.m), $(solver.n)) is inconsistent with size(A) = ($n2, $m2)") krylov_solve!(solver, Ao2, b2, shifts2)
      method == :cgls_lanczos_shift && @test_throws ErrorException("solver.nshifts = $(solver.nshifts) is inconsistent with length(shifts) = $(length(shifts2))") krylov_solve!(solver, Ao, b, shifts2)
      method ∈ (:bilqr, :trilqr) && @test_throws ErrorException("(solver.m, solver.n) = ($(solver.m), $(solver.n)) is inconsistent with size(A) = ($n2, $n2)") krylov_solve!(solver, A2, b2, b2)
      method == :gpmr && @test_throws ErrorException("(solver.m, solver.n) = ($(solver.m), $(solver.n)) is inconsistent with size(A) = ($n2, $m2)") krylov_solve!(solver, Ao2, Au2, b2, c2)
      method ∈ (:tricg, :trimr) && @test_throws ErrorException("(solver.m, solver.n) = ($(solver.m), $(solver.n)) is inconsistent with size(A) = ($n2, $m2)") krylov_solve!(solver, Ao2, b2, c2)
      method == :usymlq && @test_throws ErrorException("(solver.m, solver.n) = ($(solver.m), $(solver.n)) is inconsistent with size(A) = ($m2, $n2)") krylov_solve!(solver, Au2, c2, b2)
      method == :usymqr && @test_throws ErrorException("(solver.m, solver.n) = ($(solver.m), $(solver.n)) is inconsistent with size(A) = ($n2, $m2)") krylov_solve!(solver, Ao2, b2, c2)
    end
  end

  @testset "Test the keyword argument timemax" begin
    timemax = 0.0
    for (method, solver) in solvers
      method ∈ (:cg, :cr, :car, :symmlq, :minares, :minres, :minres_qlp, :cg_lanczos, :diom, :fom, :dqgmres, :gmres, :fgmres, :cgs, :bicgstab, :bilq, :qmr) && krylov_solve!(solver, A, b, timemax=timemax)
      method == :cg_lanczos_shift && krylov_solve!(solver, A, b, shifts, timemax=timemax)
      method ∈ (:cgne, :crmr, :lnlq, :craig, :craigmr) && krylov_solve!(solver, Au, c, timemax=timemax)
      method ∈ (:cgls, :crls, :lslq, :lsqr, :lsmr) && krylov_solve!(solver, Ao, b, timemax=timemax)
      method == :cgls_lanczos_shift && krylov_solve!(solver, Ao, b, shifts, timemax=timemax)
      method ∈ (:bilqr, :trilqr) && krylov_solve!(solver, A, b, b, timemax=timemax)
      method == :gpmr && krylov_solve!(solver, Ao, Au, b, c, timemax=timemax)
      method ∈ (:tricg, :trimr) && krylov_solve!(solver, Au, c, b, timemax=timemax)
      method == :usymlq && krylov_solve!(solver, Au, c, b, timemax=timemax)
      method == :usymqr && krylov_solve!(solver, Ao, b, c, timemax=timemax)
      @test solver.stats.status == "time limit exceeded"
    end
  end

  for (method, solver) in solvers
    @testset "$(method)" begin
      for i = 1 : 3
        A  = i * A
        Au = i * Au
        Ao = i * Ao
        b  = 5 * b
        c  = 3 * c

        if method ∈ (:cg, :cr, :car, :symmlq, :minares, :minres, :minres_qlp, :cg_lanczos, :diom, :fom,
                     :dqgmres, :gmres, :fgmres, :cgs, :bicgstab, :bilq, :qmr, :cg_lanczos_shift)
          if method == :cg_lanczos_shift
            @inferred krylov_solve(Val{method}(), A, b, shifts)
            @inferred krylov_solve!(solver, A, b, shifts)
          else
            @inferred krylov_solve(Val{method}(), A, b)
            @inferred krylov_solve!(solver, A, b)
          end
          niter = niterations(solver)
          @test Aprod(solver) == (method ∈ (:cgs, :bicgstab) ? 2 * niter : niter)
          @test Atprod(solver) == (method ∈ (:bilq, :qmr) ? niter : 0)
          @test solution(solver) === solver.x
          @test nsolution(solver) == 1
        end

        if method ∈ (:cgne, :crmr, :lnlq, :craig, :craigmr)
          @inferred krylov_solve(Val{method}(), Au, c)
          @inferred krylov_solve!(solver, Au, c)
          niter = niterations(solver)
          @test Aprod(solver) == niter
          @test Atprod(solver) == niter
          @test solution(solver, 1) === solver.x
          @test nsolution(solver) == (method ∈ (:cgne, :crmr) ? 1 : 2)
          (nsolution == 2) && (@test solution(solver, 2) == solver.y)
        end

        if method ∈ (:cgls, :crls, :lslq, :lsqr, :lsmr, :cgls_lanczos_shift)
          if method == :cgls_lanczos_shift
            @inferred krylov_solve(Val{method}(), Ao, b, shifts)
            @inferred krylov_solve!(solver, Ao, b, shifts)
          else
            @inferred krylov_solve(Val{method}(), Ao, b)
            @inferred krylov_solve!(solver, Ao, b)
          end
          niter = niterations(solver)
          @test Aprod(solver) == niter
          @test Atprod(solver) == niter
          @test solution(solver) === solver.x
          @test nsolution(solver) == 1
        end

        if method ∈ (:bilqr, :trilqr)
          @inferred krylov_solve(Val{method}(), A, b, b)
          @inferred krylov_solve!(solver, A, b, b)
          niter = niterations(solver)
          @test Aprod(solver) == niter
          @test Atprod(solver) == niter
          @test solution(solver, 1) === solver.x
          @test solution(solver, 2) === solver.y
          @test nsolution(solver) == 2
          @test issolved_primal(solver)
          @test issolved_dual(solver)
        end

        if method ∈ (:tricg, :trimr, :gpmr)
          if method == :gpmr
            @inferred krylov_solve(Val{method}(), Ao, Au, b, c)
            @inferred krylov_solve!(solver, Ao, Au, b, c)
          else
            @inferred krylov_solve(Val{method}(), Au, c, b)
            @inferred krylov_solve!(solver, Au, c, b)
          end
          niter = niterations(solver)
          @test Aprod(solver) == niter
          method != :gpmr && (@test Atprod(solver) == niter)
          method == :gpmr && (@test Bprod(solver) == niter)
          @test solution(solver, 1) === solver.x
          @test solution(solver, 2) === solver.y
          @test nsolution(solver) == 2
        end

        if method ∈ (:usymlq, :usymqr)
          if method == :usymlq
            @inferred krylov_solve(Val{method}(), Au, c, b)
            @inferred krylov_solve!(solver, Au, c, b)
          else
            @inferred krylov_solve(Val{method}(), Ao, b, c)
            @inferred krylov_solve!(solver, Ao, b, c)
          end
          niter = niterations(solver)
          @test Aprod(solver) == niter
          @test Atprod(solver) == niter
          @test solution(solver) === solver.x
          @test nsolution(solver) == 1
        end

        @test niter > 0
        @test statistics(solver) === solver.stats
        @test issolved(solver)
      end

      test_show(solver)
    end
  end
end

function test_block_krylov_solvers(FC)
  A = FC.(get_div_grad(4, 4, 4))  # Dimension n x n
  m, n = size(A)
  p = 8
  B = rand(FC, m, p)
  mem = 4
  T = real(FC)
  SV = Vector{FC}
  SM = Matrix{FC}
  solvers = Dict{Symbol, BlockKrylovWorkspace}()
  solvers[:block_minres] = @inferred krylov_workspace(Val{:block_minres}(), m, n, p, SV, SM)
  solvers[:block_gmres] = @inferred krylov_workspace(Val{:block_gmres}(), m, n, p, SV, SM; memory=mem)

  for (method, solver) in solvers
    @testset "$(method)" begin
      for i = 1 : 3
        B = 5 * B
        @inferred krylov_solve(Val{method}(), A, B)
        @inferred krylov_solve!(solver, A, B)
        niter = niterations(solver)
        @test Aprod(solver) == niter
        @test Atprod(solver) == 0
        @test solution(solver) === solver.X
        @test nsolution(solver) == 1
        @test niter > 0
        @test statistics(solver) === solver.stats
        @test issolved(solver)
      end

      test_show(solver)
    end
  end
end

function test_show(solver)
  io = IOBuffer()
  show(io, solver, show_stats=false)
  showed = String(take!(io))

  # Test that the lines have the same length
  str = split(showed, '\n', keepempty=false)
  len_row = length(str[1])
  @test mapreduce(x -> length(x) - mapreduce(y -> occursin(y, x), |, ["w̅","w̄","d̅"]) == len_row, &, str)

  # Test that the columns have the same length
  str2 = split(showed, ['│','┌','┬','┐','├','┼','┤','└','┴','┴','┘','\n'], keepempty=false)
  len_col1 = length(str2[1])
  len_col2 = length(str2[2])
  len_col3 = length(str2[3])
  @test mapreduce(x -> length(x) - mapreduce(y -> occursin(y, x), |, ["w̅","w̄","d̅"]) == len_col1, &, str2[1:3:end-2])
  @test mapreduce(x -> length(x) - mapreduce(y -> occursin(y, x), |, ["w̅","w̄","d̅"]) == len_col2, &, str2[2:3:end-1])
  @test mapreduce(x -> length(x) - mapreduce(y -> occursin(y, x), |, ["w̅","w̄","d̅"]) == len_col3, &, str2[3:3:end])

  # Code coverage
  show(io, solver, show_stats=true)
end

@testset "Krylov solvers" begin
  for FC in (Float64, ComplexF64)
    @testset "Data Type: FC" begin
      test_krylov_solvers(FC; krylov_constructor=false)
      test_krylov_solvers(FC; krylov_constructor=true)
    end
  end
end

@testset "Block Krylov solvers" begin
  for FC in (Float64, ComplexF64)
    @testset "Data Type: FC" begin
      test_block_krylov_solvers(FC)
    end
  end
end
