function test_solvers(FC)
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

  @eval begin
    cg_solver = $(KRYLOV_SOLVERS[:cg])($n, $n, $S)
    symmlq_solver = $(KRYLOV_SOLVERS[:symmlq])($n, $n, $S)
    minres_solver = $(KRYLOV_SOLVERS[:minres])($n, $n, $S)
    cg_lanczos_solver = $(KRYLOV_SOLVERS[:cg_lanczos])($n, $n, $S)
    diom_solver = $(KRYLOV_SOLVERS[:diom])($n, $n, $mem, $S)
    fom_solver = $(KRYLOV_SOLVERS[:fom])($n, $n, $mem, $S)
    dqgmres_solver = $(KRYLOV_SOLVERS[:dqgmres])($n, $n, $mem, $S)
    gmres_solver = $(KRYLOV_SOLVERS[:gmres])($n, $n, $mem, $S)
    cr_solver = $(KRYLOV_SOLVERS[:cr])($n, $n, $S)
    crmr_solver = $(KRYLOV_SOLVERS[:crmr])($m, $n, $S)
    cgs_solver = $(KRYLOV_SOLVERS[:cgs])($n, $n, $S)
    bicgstab_solver = $(KRYLOV_SOLVERS[:bicgstab])($n, $n, $S)
    craigmr_solver = $(KRYLOV_SOLVERS[:craigmr])($m, $n, $S)
    cgne_solver = $(KRYLOV_SOLVERS[:cgne])($m, $n, $S)
    lnlq_solver = $(KRYLOV_SOLVERS[:lnlq])($m, $n, $S)
    craig_solver = $(KRYLOV_SOLVERS[:craig])($m, $n, $S)
    lslq_solver = $(KRYLOV_SOLVERS[:lslq])($n, $m, $S)
    cgls_solver = $(KRYLOV_SOLVERS[:cgls])($n, $m, $S)
    lsqr_solver = $(KRYLOV_SOLVERS[:lsqr])($n, $m, $S)
    crls_solver = $(KRYLOV_SOLVERS[:crls])($n, $m, $S)
    lsmr_solver = $(KRYLOV_SOLVERS[:lsmr])($n, $m, $S)
    usymqr_solver = $(KRYLOV_SOLVERS[:usymqr])($n, $m, $S)
    trilqr_solver = $(KRYLOV_SOLVERS[:trilqr])($n, $n, $S)
    bilq_solver = $(KRYLOV_SOLVERS[:bilq])($n, $n, $S)
    bilqr_solver = $(KRYLOV_SOLVERS[:bilqr])($n, $n, $S)
    minres_qlp_solver = $(KRYLOV_SOLVERS[:minres_qlp])($n, $n, $S)
    qmr_solver = $(KRYLOV_SOLVERS[:qmr])($n, $n, $S)
    usymlq_solver = $(KRYLOV_SOLVERS[:usymlq])($m, $n, $S)
    tricg_solver = $(KRYLOV_SOLVERS[:tricg])($m, $n, $S)
    trimr_solver = $(KRYLOV_SOLVERS[:trimr])($m, $n, $S)
    gpmr_solver = $(KRYLOV_SOLVERS[:gpmr])($n, $m, $mem, $S)
    cg_lanczos_shift_solver = $(KRYLOV_SOLVERS[:cg_lanczos_shift])($n, $m, $nshifts, $S)
  end

  for i = 1 : 3
    A  = i * A
    Au = i * Au
    Ao = i * Ao
    b  = 5 * b
    c  = 3 * c

    solver = solve!(cg_solver, A, b)
    niter = niterations(solver)
    @test niter > 0
    @test Aprod(solver) == niter
    @test Atprod(solver) == 0
    @test statistics(solver) === solver.stats
    @test solution(solver, 1) === solver.x
    @test nsolution(solver) == 1
    @test issolved(solver)

    solver = solve!(symmlq_solver, A, b)
    niter = niterations(solver)
    @test niter > 0
    @test Aprod(solver) == niter
    @test Atprod(solver) == 0
    @test statistics(solver) === solver.stats
    @test solution(solver, 1) === solver.x
    @test nsolution(solver) == 1
    @test issolved(solver)

    solver = solve!(minres_solver, A, b)
    niter = niterations(solver)
    @test niter > 0
    @test Aprod(solver) == niter
    @test Atprod(solver) == 0
    @test statistics(solver) === solver.stats
    @test solution(solver, 1) === solver.x
    @test nsolution(solver) == 1
    @test issolved(solver)

    solver = solve!(cg_lanczos_solver, A, b)
    niter = niterations(solver)
    @test niter > 0
    @test Aprod(solver) == niter
    @test Atprod(solver) == 0
    @test statistics(solver) === solver.stats
    @test solution(solver, 1) === solver.x
    @test nsolution(solver) == 1
    @test issolved(solver)

    solver = solve!(cg_lanczos_shift_solver, A, b, shifts)
    niter = niterations(solver)
    @test niter > 0
    @test Aprod(solver) == niter
    @test Atprod(solver) == 0
    @test statistics(solver) === solver.stats
    @test solution(solver, 1) === solver.x
    @test nsolution(solver) == 1
    @test issolved(solver)

    solver = solve!(diom_solver, A, b)
    niter = niterations(solver)
    @test niter > 0
    @test Aprod(solver) == niter
    @test Atprod(solver) == 0
    @test statistics(solver) === solver.stats
    @test solution(solver, 1) === solver.x
    @test nsolution(solver) == 1
    @test issolved(solver)

    solver = solve!(fom_solver, A, b)
    niter = niterations(solver)
    @test niter > 0
    @test Aprod(solver) == niter
    @test Atprod(solver) == 0
    @test statistics(solver) === solver.stats
    @test solution(solver, 1) === solver.x
    @test nsolution(solver) == 1
    @test issolved(solver)

    solver = solve!(dqgmres_solver, A, b)
    niter = niterations(solver)
    @test niter > 0
    @test Aprod(solver) == niter
    @test Atprod(solver) == 0
    @test statistics(solver) === solver.stats
    @test solution(solver, 1) === solver.x
    @test nsolution(solver) == 1
    @test issolved(solver)

    solver = solve!(gmres_solver, A, b)
    niter = niterations(solver)
    @test niter > 0
    @test Aprod(solver) == niter
    @test Atprod(solver) == 0
    @test statistics(solver) === solver.stats
    @test solution(solver, 1) === solver.x
    @test nsolution(solver) == 1
    @test issolved(solver)

    solver = solve!(cr_solver, A, b)
    niter = niterations(solver)
    @test niter > 0
    @test Aprod(solver) == niter
    @test Atprod(solver) == 0
    @test statistics(solver) === solver.stats
    @test solution(solver, 1) === solver.x
    @test nsolution(solver) == 1
    # @test issolved(solver)

    solver = solve!(crmr_solver, Au, c)
    niter = niterations(solver)
    @test niter > 0
    @test Aprod(solver) == niter
    @test Atprod(solver) == niter
    @test statistics(solver) === solver.stats
    @test solution(solver, 1) === solver.x
    @test nsolution(solver) == 1
    @test issolved(solver)

    solver = solve!(cgs_solver, A, b)
    niter = niterations(solver)
    @test niter > 0
    @test Aprod(solver) == 2 * niter
    @test Atprod(solver) == 0
    @test statistics(solver) === solver.stats
    @test solution(solver, 1) === solver.x
    @test nsolution(solver) == 1
    @test issolved(solver)

    solver = solve!(bicgstab_solver, A, b)
    niter = niterations(solver)
    @test niter > 0
    @test Aprod(solver) == 2 * niter
    @test Atprod(solver) == 0
    @test statistics(solver) === solver.stats
    @test solution(solver, 1) === solver.x
    @test nsolution(solver) == 1
    @test issolved(solver)

    solver = solve!(craigmr_solver, Au, c)
    niter = niterations(solver)
    @test niter > 0
    @test Aprod(solver) == niter
    @test Atprod(solver) == niter
    @test statistics(solver) === solver.stats
    @test solution(solver, 1) === solver.x
    @test nsolution(solver) == 2
    @test issolved(solver)

    solver = solve!(cgne_solver, Au, c)
    niter = niterations(solver)
    @test niter > 0
    @test Aprod(solver) == niter
    @test Atprod(solver) == niter
    @test statistics(solver) === solver.stats
    @test solution(solver, 1) === solver.x
    @test nsolution(solver) == 1
    @test issolved(solver)

    solver = solve!(lnlq_solver, Au, c)
    niter = niterations(solver)
    @test niter > 0
    @test Aprod(solver) == niter
    @test Atprod(solver) == niter
    @test statistics(solver) === solver.stats
    @test solution(solver, 1) === solver.x
    @test solution(solver, 2) === solver.y
    @test nsolution(solver) == 2
    @test issolved(solver)

    solver = solve!(craig_solver, Au, c)
    niter = niterations(solver)
    @test niter > 0
    @test Aprod(solver) == niter
    @test Atprod(solver) == niter
    @test statistics(solver) === solver.stats
    @test solution(solver, 1) === solver.x
    @test solution(solver, 2) === solver.y
    @test nsolution(solver) == 2
    @test issolved(solver)

    solver = solve!(lslq_solver, Ao, b)
    niter = niterations(solver)
    @test niter > 0
    @test Aprod(solver) == niter
    @test Atprod(solver) == niter
    @test statistics(solver) === solver.stats
    @test solution(solver, 1) === solver.x
    @test nsolution(solver) == 1
    @test issolved(solver)

    solver = solve!(cgls_solver, Ao, b)
    niter = niterations(solver)
    @test niter > 0
    @test Aprod(solver) == niter
    @test Atprod(solver) == niter
    @test statistics(solver) === solver.stats
    @test solution(solver, 1) === solver.x
    @test nsolution(solver) == 1
    @test issolved(solver)

    solver = solve!(lsqr_solver, Ao, b)
    niter = niterations(solver)
    @test niter > 0
    @test Aprod(solver) == niter
    @test Atprod(solver) == niter
    @test statistics(solver) === solver.stats
    @test solution(solver, 1) === solver.x
    @test nsolution(solver) == 1
    @test issolved(solver)

    solver = solve!(crls_solver, Ao, b)
    niter = niterations(solver)
    @test niter > 0
    @test Aprod(solver) == niter
    @test Atprod(solver) == niter
    @test statistics(solver) === solver.stats
    @test solution(solver, 1) === solver.x
    @test nsolution(solver) == 1
    @test issolved(solver)

    solver = solve!(lsmr_solver, Ao, b)
    niter = niterations(solver)
    @test niter > 0
    @test Aprod(solver) == niter
    @test Atprod(solver) == niter
    @test statistics(solver) === solver.stats
    @test solution(solver, 1) === solver.x
    @test nsolution(solver) == 1
    @test issolved(solver)

    solver = solve!(usymqr_solver, Ao, b, c)
    niter = niterations(solver)
    @test niter > 0
    @test Aprod(solver) == niter
    @test Atprod(solver) == niter
    @test statistics(solver) === solver.stats
    @test solution(solver, 1) === solver.x
    @test nsolution(solver) == 1
    @test issolved(solver)

    solver = solve!(trilqr_solver, A, b, b)
    niter = niterations(solver)
    @test niter > 0
    @test Aprod(solver) == niter
    @test Atprod(solver) == niter
    @test statistics(solver) === solver.stats
    @test solution(solver, 1) === solver.x
    @test solution(solver, 2) === solver.y
    @test nsolution(solver) == 2
    @test issolved_primal(solver)
    @test issolved_dual(solver)
    @test issolved(solver)

    solver = solve!(bilq_solver, A, b)
    niter = niterations(solver)
    @test niter > 0
    @test Aprod(solver) == niter
    @test Atprod(solver) == niter
    @test statistics(solver) === solver.stats
    @test solution(solver, 1) === solver.x
    @test nsolution(solver) == 1
    @test issolved(solver)

    solver = solve!(bilqr_solver, A, b, b)
    niter = niterations(solver)
    @test niter > 0
    @test Aprod(solver) == niter
    @test Atprod(solver) == niter
    @test statistics(solver) === solver.stats
    @test solution(solver, 1) === solver.x
    @test solution(solver, 2) === solver.y
    @test nsolution(solver) == 2
    @test issolved_primal(solver)
    @test issolved_dual(solver)
    @test issolved(solver)

    solver = solve!(minres_qlp_solver, A, b)
    niter = niterations(solver)
    @test niter > 0
    @test Aprod(solver) == niter
    @test Atprod(solver) == 0
    @test statistics(solver) === solver.stats
    @test solution(solver, 1) === solver.x
    @test nsolution(solver) == 1
    @test issolved(solver)

    solver = solve!(qmr_solver, A, b)
    niter = niterations(solver)
    @test niter > 0
    @test Aprod(solver) == niter
    @test Atprod(solver) == niter
    @test statistics(solver) === solver.stats
    @test solution(solver, 1) === solver.x
    @test nsolution(solver) == 1
    @test issolved(solver)

    solver = solve!(usymlq_solver, Au, c, b)
    niter = niterations(solver)
    @test niter > 0
    @test Aprod(solver) == niter
    @test Atprod(solver) == niter
    @test statistics(solver) === solver.stats
    @test solution(solver, 1) === solver.x
    @test nsolution(solver) == 1
    @test issolved(solver)

    solver = solve!(tricg_solver, Au, c, b)
    niter = niterations(solver)
    @test niter > 0
    @test Aprod(solver) == niter
    @test Atprod(solver) == niter
    @test statistics(solver) === solver.stats
    @test solution(solver, 1) === solver.x
    @test solution(solver, 2) === solver.y
    @test nsolution(solver) == 2
    @test issolved(solver)

    solver = solve!(trimr_solver, Au, c, b)
    niter = niterations(solver)
    @test niter > 0
    @test Aprod(solver) == niter
    @test Atprod(solver) == niter
    @test statistics(solver) === solver.stats
    @test solution(solver, 1) === solver.x
    @test solution(solver, 2) === solver.y
    @test nsolution(solver) == 2
    @test issolved(solver)

    solver = solve!(gpmr_solver, Ao, Au, b, c)
    niter = niterations(solver)
    @test niter > 0
    @test Aprod(solver) == niter
    @test Atprod(solver) == 0
    @test Bprod(solver) == niter
    @test statistics(solver) === solver.stats
    @test solution(solver, 1) === solver.x
    @test solution(solver, 2) === solver.y
    @test nsolution(solver) == 2
    @test issolved(solver)
  end

  io = IOBuffer()
  show(io, cg_solver, show_stats=false)
  showed = String(take!(io))
  expected = """
  ┌──────────┬───────────────┬─────────────────┐
  │  CgSolver│Precision: $FC │Architecture: CPU│
  ├──────────┼───────────────┼─────────────────┤
  │ Attribute│           Type│             Size│
  ├──────────┼───────────────┼─────────────────┤
  │        Δx│    Vector{$FC}│                0│
  │         x│    Vector{$FC}│               64│
  │         r│    Vector{$FC}│               64│
  │         p│    Vector{$FC}│               64│
  │        Ap│    Vector{$FC}│               64│
  │         z│    Vector{$FC}│                0│
  │warm_start│           Bool│                0│
  └──────────┴───────────────┴─────────────────┘
  """
  @test reduce(replace, [" " => "", "\n" => "", "─" => ""], init=showed) == reduce(replace, [" " => "", "\n" => "", "─" => ""], init=expected)

  io = IOBuffer()
  show(io, symmlq_solver, show_stats=false)
  showed = String(take!(io))
  expected = """
  ┌────────────┬───────────────┬─────────────────┐
  │SymmlqSolver│Precision: $FC │Architecture: CPU│
  ├────────────┼───────────────┼─────────────────┤
  │   Attribute│           Type│             Size│
  ├────────────┼───────────────┼─────────────────┤
  │          Δx│    Vector{$FC}│                0│
  │           x│    Vector{$FC}│               64│
  │       Mvold│    Vector{$FC}│               64│
  │          Mv│    Vector{$FC}│               64│
  │     Mv_next│    Vector{$FC}│               64│
  │           w̅│    Vector{$FC}│               64│
  │           v│    Vector{$FC}│                0│
  │       clist│     Vector{$T}│                5│
  │       zlist│     Vector{$T}│                5│
  │       sprod│     Vector{$T}│                5│
  │  warm_start│           Bool│                0│
  └────────────┴───────────────┴─────────────────┘
  """
  @test reduce(replace, [" " => "", "\n" => "", "─" => ""], init=showed) == reduce(replace, [" " => "", "\n" => "", "─" => ""], init=expected)

  io = IOBuffer()
  show(io, minres_solver, show_stats=false)
  showed = String(take!(io))
  expected = """
  ┌────────────┬───────────────┬─────────────────┐
  │MinresSolver│Precision: $FC │Architecture: CPU│
  ├────────────┼───────────────┼─────────────────┤
  │   Attribute│           Type│             Size│
  ├────────────┼───────────────┼─────────────────┤
  │          Δx│    Vector{$FC}│                0│
  │           x│    Vector{$FC}│               64│
  │          r1│    Vector{$FC}│               64│
  │          r2│    Vector{$FC}│               64│
  │          w1│    Vector{$FC}│               64│
  │          w2│    Vector{$FC}│               64│
  │           y│    Vector{$FC}│               64│
  │           v│    Vector{$FC}│                0│
  │     err_vec│     Vector{$T}│                5│
  │  warm_start│           Bool│                0│
  └────────────┴───────────────┴─────────────────┘
  """
  @test reduce(replace, [" " => "", "\n" => "", "─" => ""], init=showed) == reduce(replace, [" " => "", "\n" => "", "─" => ""], init=expected)

  io = IOBuffer()
  show(io, cg_lanczos_solver, show_stats=false)
  showed = String(take!(io))
  expected = """
  ┌───────────────┬───────────────┬─────────────────┐
  │CgLanczosSolver│Precision: $FC │Architecture: CPU│
  ├───────────────┼───────────────┼─────────────────┤
  │      Attribute│           Type│             Size│
  ├───────────────┼───────────────┼─────────────────┤
  │             Δx│    Vector{$FC}│                0│
  │              x│    Vector{$FC}│               64│
  │             Mv│    Vector{$FC}│               64│
  │        Mv_prev│    Vector{$FC}│               64│
  │              p│    Vector{$FC}│               64│
  │        Mv_next│    Vector{$FC}│               64│
  │              v│    Vector{$FC}│                0│
  │     warm_start│           Bool│                0│
  └───────────────┴───────────────┴─────────────────┘
  """
  @test reduce(replace, [" " => "", "\n" => "", "─" => ""], init=showed) == reduce(replace, [" " => "", "\n" => "", "─" => ""], init=expected)

  io = IOBuffer()
  show(io, cg_lanczos_shift_solver, show_stats=false)
  showed = String(take!(io))
  expected = """
  ┌────────────────────┬───────────────────┬─────────────────┐
  │CgLanczosShiftSolver│    Precision: $FC │Architecture: CPU│
  ├────────────────────┼───────────────────┼─────────────────┤
  │           Attribute│               Type│             Size│
  ├────────────────────┼───────────────────┼─────────────────┤
  │                  Mv│        Vector{$FC}│               64│
  │             Mv_prev│        Vector{$FC}│               64│
  │             Mv_next│        Vector{$FC}│               64│
  │                   v│        Vector{$FC}│                0│
  │                   x│Vector{Vector{$FC}}│           5 x 64│
  │                   p│Vector{Vector{$FC}}│           5 x 64│
  │                   σ│         Vector{$T}│                5│
  │                δhat│         Vector{$T}│                5│
  │                   ω│         Vector{$T}│                5│
  │                   γ│         Vector{$T}│                5│
  │              rNorms│         Vector{$T}│                5│
  │           converged│          BitVector│                5│
  │              not_cv│          BitVector│                5│
  └────────────────────┴───────────────────┴─────────────────┘
  """
  @test reduce(replace, [" " => "", "\n" => "", "─" => ""], init=showed) == reduce(replace, [" " => "", "\n" => "", "─" => ""], init=expected)

  io = IOBuffer()
  show(io, diom_solver, show_stats=false)
  showed = String(take!(io))
  expected = """
  ┌──────────┬───────────────────┬─────────────────┐
  │DiomSolver│    Precision: $FC │Architecture: CPU│
  ├──────────┼───────────────────┼─────────────────┤
  │ Attribute│               Type│             Size│
  ├──────────┼───────────────────┼─────────────────┤
  │        Δx│        Vector{$FC}│                0│
  │         x│        Vector{$FC}│               64│
  │         t│        Vector{$FC}│               64│
  │         z│        Vector{$FC}│                0│
  │         w│        Vector{$FC}│                0│
  │         P│Vector{Vector{$FC}}│          10 x 64│
  │         V│Vector{Vector{$FC}}│          10 x 64│
  │         L│        Vector{$FC}│               10│
  │         H│        Vector{$FC}│               12│
  │warm_start│               Bool│                0│
  └──────────┴───────────────────┴─────────────────┘
  """
  @test reduce(replace, [" " => "", "\n" => "", "─" => ""], init=showed) == reduce(replace, [" " => "", "\n" => "", "─" => ""], init=expected)

  io = IOBuffer()
  show(io, fom_solver, show_stats=false)
  showed = String(take!(io))
  expected = """
  ┌──────────┬───────────────────┬─────────────────┐
  │ FomSolver│    Precision: $FC │Architecture: CPU│
  ├──────────┼───────────────────┼─────────────────┤
  │ Attribute│               Type│             Size│
  ├──────────┼───────────────────┼─────────────────┤
  │        Δx│        Vector{$FC}│                0│
  │         x│        Vector{$FC}│               64│
  │         w│        Vector{$FC}│               64│
  │         p│        Vector{$FC}│                0│
  │         q│        Vector{$FC}│                0│
  │         V│Vector{Vector{$FC}}│          10 x 64│
  │         l│        Vector{$FC}│               10│
  │         z│        Vector{$FC}│               10│
  │         U│        Vector{$FC}│               55│
  │warm_start│               Bool│                0│
  └──────────┴───────────────────┴─────────────────┘
  """
  @test reduce(replace, [" " => "", "\n" => "", "─" => ""], init=showed) == reduce(replace, [" " => "", "\n" => "", "─" => ""], init=expected)

  io = IOBuffer()
  show(io, dqgmres_solver, show_stats=false)
  showed = String(take!(io))
  expected = """
  ┌─────────────┬───────────────────┬─────────────────┐
  │DqgmresSolver│    Precision: $FC │Architecture: CPU│
  ├─────────────┼───────────────────┼─────────────────┤
  │    Attribute│               Type│             Size│
  ├─────────────┼───────────────────┼─────────────────┤
  │           Δx│        Vector{$FC}│                0│
  │            x│        Vector{$FC}│               64│
  │            t│        Vector{$FC}│               64│
  │            z│        Vector{$FC}│                0│
  │            w│        Vector{$FC}│                0│
  │            P│Vector{Vector{$FC}}│          10 x 64│
  │            V│Vector{Vector{$FC}}│          10 x 64│
  │            c│         Vector{$T}│               10│
  │            s│        Vector{$FC}│               10│
  │            H│        Vector{$FC}│               12│
  │   warm_start│               Bool│                0│
  └─────────────┴───────────────────┴─────────────────┘
  """
  @test reduce(replace, [" " => "", "\n" => "", "─" => ""], init=showed) == reduce(replace, [" " => "", "\n" => "", "─" => ""], init=expected)

  io = IOBuffer()
  show(io, gmres_solver, show_stats=false)
  showed = String(take!(io))
  expected = """
  ┌───────────┬───────────────────┬─────────────────┐
  │GmresSolver│    Precision: $FC │Architecture: CPU│
  ├───────────┼───────────────────┼─────────────────┤
  │  Attribute│               Type│             Size│
  ├───────────┼───────────────────┼─────────────────┤
  │         Δx│        Vector{$FC}│                0│
  │          x│        Vector{$FC}│               64│
  │          w│        Vector{$FC}│               64│
  │          p│        Vector{$FC}│                0│
  │          q│        Vector{$FC}│                0│
  │          V│Vector{Vector{$FC}}│          10 x 64│
  │          c│         Vector{$T}│               10│
  │          s│        Vector{$FC}│               10│
  │          z│        Vector{$FC}│               10│
  │          R│        Vector{$FC}│               55│
  │ warm_start│               Bool│                0│
  │ inner_iter│              Int64│                0│
  └───────────┴───────────────────┴─────────────────┘
  """
  @test reduce(replace, [" " => "", "\n" => "", "─" => ""], init=showed) == reduce(replace, [" " => "", "\n" => "", "─" => ""], init=expected)

  io = IOBuffer()
  show(io, cr_solver, show_stats=false)
  showed = String(take!(io))
  expected = """
  ┌──────────┬───────────────┬─────────────────┐
  │  CrSolver│Precision: $FC │Architecture: CPU│
  ├──────────┼───────────────┼─────────────────┤
  │ Attribute│           Type│             Size│
  ├──────────┼───────────────┼─────────────────┤
  │        Δx│    Vector{$FC}│                0│
  │         x│    Vector{$FC}│               64│
  │         r│    Vector{$FC}│               64│
  │         p│    Vector{$FC}│               64│
  │         q│    Vector{$FC}│               64│
  │        Ar│    Vector{$FC}│               64│
  │        Mq│    Vector{$FC}│                0│
  │warm_start│           Bool│                0│
  └──────────┴───────────────┴─────────────────┘
  """
  @test reduce(replace, [" " => "", "\n" => "", "─" => ""], init=showed) == reduce(replace, [" " => "", "\n" => "", "─" => ""], init=expected)

  io = IOBuffer()
  show(io, crmr_solver, show_stats=false)
  showed = String(take!(io))
  expected = """
  ┌──────────┬───────────────┬─────────────────┐
  │CrmrSolver│Precision: $FC │Architecture: CPU│
  ├──────────┼───────────────┼─────────────────┤
  │ Attribute│           Type│             Size│
  ├──────────┼───────────────┼─────────────────┤
  │         x│    Vector{$FC}│               64│
  │         p│    Vector{$FC}│               64│
  │       Aᵀr│    Vector{$FC}│               64│
  │         r│    Vector{$FC}│               32│
  │         q│    Vector{$FC}│               32│
  │        Mq│    Vector{$FC}│                0│
  │         s│    Vector{$FC}│                0│
  └──────────┴───────────────┴─────────────────┘
  """
  @test reduce(replace, [" " => "", "\n" => "", "─" => ""], init=showed) == reduce(replace, [" " => "", "\n" => "", "─" => ""], init=expected)

  io = IOBuffer()
  show(io, cgs_solver, show_stats=false)
  showed = String(take!(io))
  expected = """
  ┌──────────┬───────────────┬─────────────────┐
  │ CgsSolver│Precision: $FC │Architecture: CPU│
  ├──────────┼───────────────┼─────────────────┤
  │Attribute │           Type│             Size│
  ├──────────┼───────────────┼─────────────────┤
  │        Δx│    Vector{$FC}│                0│
  │         x│    Vector{$FC}│               64│
  │         r│    Vector{$FC}│               64│
  │         u│    Vector{$FC}│               64│
  │         p│    Vector{$FC}│               64│
  │         q│    Vector{$FC}│               64│
  │        ts│    Vector{$FC}│               64│
  │        yz│    Vector{$FC}│                0│
  │        vw│    Vector{$FC}│                0│
  │warm_start│           Bool│                0│
  └──────────┴───────────────┴─────────────────┘
  """
  @test reduce(replace, [" " => "", "\n" => "", "─" => ""], init=showed) == reduce(replace, [" " => "", "\n" => "", "─" => ""], init=expected)

  io = IOBuffer()
  show(io, bicgstab_solver, show_stats=false)
  showed = String(take!(io))
  expected = """
  ┌──────────────┬───────────────┬─────────────────┐
  │BicgstabSolver│Precision: $FC │Architecture: CPU│
  ├──────────────┼───────────────┼─────────────────┤
  │     Attribute│           Type│             Size│
  ├──────────────┼───────────────┼─────────────────┤
  │            Δx│    Vector{$FC}│                0│
  │             x│    Vector{$FC}│               64│
  │             r│    Vector{$FC}│               64│
  │             p│    Vector{$FC}│               64│
  │             v│    Vector{$FC}│               64│
  │             s│    Vector{$FC}│               64│
  │            qd│    Vector{$FC}│               64│
  │            yz│    Vector{$FC}│                0│
  │             t│    Vector{$FC}│                0│
  │    warm_start│           Bool│                0│
  └──────────────┴───────────────┴─────────────────┘
  """
  @test reduce(replace, [" " => "", "\n" => "", "─" => ""], init=showed) == reduce(replace, [" " => "", "\n" => "", "─" => ""], init=expected)

  io = IOBuffer()
  show(io, craigmr_solver, show_stats=false)
  showed = String(take!(io))
  expected = """
  ┌─────────────┬───────────────┬─────────────────┐
  │CraigmrSolver│Precision: $FC │Architecture: CPU│
  ├─────────────┼───────────────┼─────────────────┤
  │    Attribute│           Type│             Size│
  ├─────────────┼───────────────┼─────────────────┤
  │            x│    Vector{$FC}│               64│
  │           Nv│    Vector{$FC}│               64│
  │          Aᵀu│    Vector{$FC}│               64│
  │            d│    Vector{$FC}│               64│
  │            y│    Vector{$FC}│               32│
  │           Mu│    Vector{$FC}│               32│
  │            w│    Vector{$FC}│               32│
  │         wbar│    Vector{$FC}│               32│
  │           Av│    Vector{$FC}│               32│
  │            u│    Vector{$FC}│                0│
  │            v│    Vector{$FC}│                0│
  │            q│    Vector{$FC}│                0│
  └─────────────┴───────────────┴─────────────────┘
  """
  @test reduce(replace, [" " => "", "\n" => "", "─" => ""], init=showed) == reduce(replace, [" " => "", "\n" => "", "─" => ""], init=expected)

  io = IOBuffer()
  show(io, cgne_solver, show_stats=false)
  showed = String(take!(io))
  expected = """
  ┌──────────┬───────────────┬─────────────────┐
  │CgneSolver│Precision: $FC │Architecture: CPU│
  ├──────────┼───────────────┼─────────────────┤
  │ Attribute│           Type│             Size│
  ├──────────┼───────────────┼─────────────────┤
  │         x│    Vector{$FC}│               64│
  │         p│    Vector{$FC}│               64│
  │       Aᵀz│    Vector{$FC}│               64│
  │         r│    Vector{$FC}│               32│
  │         q│    Vector{$FC}│               32│
  │         s│    Vector{$FC}│                0│
  │         z│    Vector{$FC}│                0│
  └──────────┴───────────────┴─────────────────┘
  """
  @test reduce(replace, [" " => "", "\n" => "", "─" => ""], init=showed) == reduce(replace, [" " => "", "\n" => "", "─" => ""], init=expected)

  io = IOBuffer()
  show(io, lnlq_solver, show_stats=false)
  showed = String(take!(io))
  expected = """
  ┌──────────┬───────────────┬─────────────────┐
  │LnlqSolver│Precision: $FC │Architecture: CPU│
  ├──────────┼───────────────┼─────────────────┤
  │ Attribute│           Type│             Size│
  ├──────────┼───────────────┼─────────────────┤
  │         x│    Vector{$FC}│               64│
  │        Nv│    Vector{$FC}│               64│
  │       Aᵀu│    Vector{$FC}│               64│
  │         y│    Vector{$FC}│               32│
  │         w̄│    Vector{$FC}│               32│
  │        Mu│    Vector{$FC}│               32│
  │        Av│    Vector{$FC}│               32│
  │         u│    Vector{$FC}│                0│
  │         v│    Vector{$FC}│                0│
  │         q│    Vector{$FC}│                0│
  └──────────┴───────────────┴─────────────────┘
  """
  @test reduce(replace, [" " => "", "\n" => "", "─" => ""], init=showed) == reduce(replace, [" " => "", "\n" => "", "─" => ""], init=expected)

  io = IOBuffer()
  show(io, craig_solver, show_stats=false)
  showed = String(take!(io))
  expected = """
  ┌───────────┬───────────────┬─────────────────┐
  │CraigSolver│Precision: $FC │Architecture: CPU│
  ├───────────┼───────────────┼─────────────────┤
  │  Attribute│           Type│             Size│
  ├───────────┼───────────────┼─────────────────┤
  │          x│    Vector{$FC}│               64│
  │         Nv│    Vector{$FC}│               64│
  │        Aᵀu│    Vector{$FC}│               64│
  │          y│    Vector{$FC}│               32│
  │          w│    Vector{$FC}│               32│
  │         Mu│    Vector{$FC}│               32│
  │         Av│    Vector{$FC}│               32│
  │          u│    Vector{$FC}│                0│
  │          v│    Vector{$FC}│                0│
  │         w2│    Vector{$FC}│                0│
  └───────────┴───────────────┴─────────────────┘
  """
  @test reduce(replace, [" " => "", "\n" => "", "─" => ""], init=showed) == reduce(replace, [" " => "", "\n" => "", "─" => ""], init=expected)

  io = IOBuffer()
  show(io, lslq_solver, show_stats=false)
  showed = String(take!(io))
  expected = """
  ┌──────────┬───────────────┬─────────────────┐
  │LslqSolver│Precision: $FC │Architecture: CPU│
  ├──────────┼───────────────┼─────────────────┤
  │ Attribute│           Type│             Size│
  ├──────────┼───────────────┼─────────────────┤
  │         x│    Vector{$FC}│               32│
  │        Nv│    Vector{$FC}│               32│
  │       Aᵀu│    Vector{$FC}│               32│
  │         w̄│    Vector{$FC}│               32│
  │        Mu│    Vector{$FC}│               64│
  │        Av│    Vector{$FC}│               64│
  │         u│    Vector{$FC}│                0│
  │         v│    Vector{$FC}│                0│
  │   err_vec│     Vector{$T}│                5│
  └──────────┴───────────────┴─────────────────┘
  """
  @test reduce(replace, [" " => "", "\n" => "", "─" => ""], init=showed) == reduce(replace, [" " => "", "\n" => "", "─" => ""], init=expected)

  io = IOBuffer()
  show(io, cgls_solver, show_stats=false)
  showed = String(take!(io))
  expected = """
  ┌──────────┬───────────────┬─────────────────┐
  │CglsSolver│Precision: $FC │Architecture: CPU│
  ├──────────┼───────────────┼─────────────────┤
  │ Attribute│           Type│             Size│
  ├──────────┼───────────────┼─────────────────┤
  │         x│    Vector{$FC}│               32│
  │         p│    Vector{$FC}│               32│
  │         s│    Vector{$FC}│               32│
  │         r│    Vector{$FC}│               64│
  │         q│    Vector{$FC}│               64│
  │        Mr│    Vector{$FC}│                0│
  └──────────┴───────────────┴─────────────────┘
  """
  @test reduce(replace, [" " => "", "\n" => "", "─" => ""], init=showed) == reduce(replace, [" " => "", "\n" => "", "─" => ""], init=expected)

  io = IOBuffer()
  show(io, lsqr_solver, show_stats=false)
  showed = String(take!(io))
  expected = """
  ┌──────────┬───────────────┬─────────────────┐
  │LsqrSolver│Precision: $FC │Architecture: CPU│
  ├──────────┼───────────────┼─────────────────┤
  │ Attribute│           Type│             Size│
  ├──────────┼───────────────┼─────────────────┤
  │         x│    Vector{$FC}│               32│
  │        Nv│    Vector{$FC}│               32│
  │       Aᵀu│    Vector{$FC}│               32│
  │         w│    Vector{$FC}│               32│
  │        Mu│    Vector{$FC}│               64│
  │        Av│    Vector{$FC}│               64│
  │         u│    Vector{$FC}│                0│
  │         v│    Vector{$FC}│                0│
  │   err_vec│     Vector{$T}│                5│
  └──────────┴───────────────┴─────────────────┘
  """
  @test reduce(replace, [" " => "", "\n" => "", "─" => ""], init=showed) == reduce(replace, [" " => "", "\n" => "", "─" => ""], init=expected)

  io = IOBuffer()
  show(io, crls_solver, show_stats=false)
  showed = String(take!(io))
  expected = """
  ┌──────────┬───────────────┬─────────────────┐
  │CrlsSolver│Precision: $FC │Architecture: CPU│
  ├──────────┼───────────────┼─────────────────┤
  │ Attribute│           Type│             Size│
  ├──────────┼───────────────┼─────────────────┤
  │         x│    Vector{$FC}│               32│
  │         p│    Vector{$FC}│               32│
  │        Ar│    Vector{$FC}│               32│
  │         q│    Vector{$FC}│               32│
  │         r│    Vector{$FC}│               64│
  │        Ap│    Vector{$FC}│               64│
  │         s│    Vector{$FC}│               64│
  │        Ms│    Vector{$FC}│                0│
  └──────────┴───────────────┴─────────────────┘
  """
  @test reduce(replace, [" " => "", "\n" => "", "─" => ""], init=showed) == reduce(replace, [" " => "", "\n" => "", "─" => ""], init=expected)

  io = IOBuffer()
  show(io, lsmr_solver, show_stats=false)
  showed = String(take!(io))
  expected = """
  ┌──────────┬───────────────┬─────────────────┐
  │LsmrSolver│Precision: $FC │Architecture: CPU│
  ├──────────┼───────────────┼─────────────────┤
  │ Attribute│           Type│             Size│
  ├──────────┼───────────────┼─────────────────┤
  │         x│    Vector{$FC}│               32│
  │        Nv│    Vector{$FC}│               32│
  │       Aᵀu│    Vector{$FC}│               32│
  │         h│    Vector{$FC}│               32│
  │      hbar│    Vector{$FC}│               32│
  │        Mu│    Vector{$FC}│               64│
  │        Av│    Vector{$FC}│               64│
  │         u│    Vector{$FC}│                0│
  │         v│    Vector{$FC}│                0│
  │   err_vec│     Vector{$T}│                5│
  └──────────┴───────────────┴─────────────────┘
  """
  @test reduce(replace, [" " => "", "\n" => "", "─" => ""], init=showed) == reduce(replace, [" " => "", "\n" => "", "─" => ""], init=expected)

  io = IOBuffer()
  show(io, usymqr_solver, show_stats=false)
  showed = String(take!(io))
  expected = """
  ┌────────────┬───────────────┬─────────────────┐
  │UsymqrSolver│Precision: $FC │Architecture: CPU│
  ├────────────┼───────────────┼─────────────────┤
  │   Attribute│           Type│             Size│
  ├────────────┼───────────────┼─────────────────┤
  │        vₖ₋₁│    Vector{$FC}│               64│
  │          vₖ│    Vector{$FC}│               64│
  │           q│    Vector{$FC}│               64│
  │          Δx│    Vector{$FC}│                0│
  │           x│    Vector{$FC}│               32│
  │        wₖ₋₂│    Vector{$FC}│               32│
  │        wₖ₋₁│    Vector{$FC}│               32│
  │        uₖ₋₁│    Vector{$FC}│               32│
  │          uₖ│    Vector{$FC}│               32│
  │           p│    Vector{$FC}│               32│
  │  warm_start│           Bool│                0│
  └────────────┴───────────────┴─────────────────┘
  """
  @test reduce(replace, [" " => "", "\n" => "", "─" => ""], init=showed) == reduce(replace, [" " => "", "\n" => "", "─" => ""], init=expected)

  io = IOBuffer()
  show(io, trilqr_solver, show_stats=false)
  showed = String(take!(io))
  expected = """
  ┌────────────┬───────────────┬─────────────────┐
  │TrilqrSolver│Precision: $FC │Architecture: CPU│
  ├────────────┼───────────────┼─────────────────┤
  │   Attribute│           Type│             Size│
  ├────────────┼───────────────┼─────────────────┤
  │        uₖ₋₁│    Vector{$FC}│               64│
  │          uₖ│    Vector{$FC}│               64│
  │           p│    Vector{$FC}│               64│
  │           d̅│    Vector{$FC}│               64│
  │          Δx│    Vector{$FC}│                0│
  │           x│    Vector{$FC}│               64│
  │        vₖ₋₁│    Vector{$FC}│               64│
  │          vₖ│    Vector{$FC}│               64│
  │           q│    Vector{$FC}│               64│
  │          Δy│    Vector{$FC}│                0│
  │           y│    Vector{$FC}│               64│
  │        wₖ₋₃│    Vector{$FC}│               64│
  │        wₖ₋₂│    Vector{$FC}│               64│
  │  warm_start│           Bool│                0│
  └────────────┴───────────────┴─────────────────┘
  """
  @test reduce(replace, [" " => "", "\n" => "", "─" => ""], init=showed) == reduce(replace, [" " => "", "\n" => "", "─" => ""], init=expected)

  io = IOBuffer()
  show(io, bilq_solver, show_stats=false)
  showed = String(take!(io))
  expected = """
  ┌──────────┬───────────────┬─────────────────┐
  │BilqSolver│Precision: $FC │Architecture: CPU│
  ├──────────┼───────────────┼─────────────────┤
  │ Attribute│           Type│             Size│
  ├──────────┼───────────────┼─────────────────┤
  │      uₖ₋₁│    Vector{$FC}│               64│
  │        uₖ│    Vector{$FC}│               64│
  │         q│    Vector{$FC}│               64│
  │      vₖ₋₁│    Vector{$FC}│               64│
  │        vₖ│    Vector{$FC}│               64│
  │         p│    Vector{$FC}│               64│
  │        Δx│    Vector{$FC}│                0│
  │         x│    Vector{$FC}│               64│
  │         d̅│    Vector{$FC}│               64│
  │warm_start│           Bool│                0│
  └──────────┴───────────────┴─────────────────┘
  """
  @test reduce(replace, [" " => "", "\n" => "", "─" => ""], init=showed) == reduce(replace, [" " => "", "\n" => "", "─" => ""], init=expected)

  io = IOBuffer()
  show(io, bilqr_solver, show_stats=false)
  showed = String(take!(io))
  expected = """
  ┌───────────┬───────────────┬─────────────────┐
  │BilqrSolver│Precision: $FC │Architecture: CPU│
  ├───────────┼───────────────┼─────────────────┤
  │  Attribute│           Type│             Size│
  ├───────────┼───────────────┼─────────────────┤
  │       uₖ₋₁│    Vector{$FC}│               64│
  │         uₖ│    Vector{$FC}│               64│
  │          q│    Vector{$FC}│               64│
  │       vₖ₋₁│    Vector{$FC}│               64│
  │         vₖ│    Vector{$FC}│               64│
  │          p│    Vector{$FC}│               64│
  │         Δx│    Vector{$FC}│                0│
  │          x│    Vector{$FC}│               64│
  │         Δy│    Vector{$FC}│                0│
  │          y│    Vector{$FC}│               64│
  │          d̅│    Vector{$FC}│               64│
  │       wₖ₋₃│    Vector{$FC}│               64│
  │       wₖ₋₂│    Vector{$FC}│               64│
  │ warm_start│           Bool│                0│
  └───────────┴───────────────┴─────────────────┘
  """
  @test reduce(replace, [" " => "", "\n" => "", "─" => ""], init=showed) == reduce(replace, [" " => "", "\n" => "", "─" => ""], init=expected)

  io = IOBuffer()
  show(io, minres_qlp_solver, show_stats=false)
  showed = String(take!(io))
  expected = """
  ┌───────────────┬───────────────┬─────────────────┐
  │MinresQlpSolver│Precision: $FC │Architecture: CPU│
  ├───────────────┼───────────────┼─────────────────┤
  │      Attribute│           Type│             Size│
  ├───────────────┼───────────────┼─────────────────┤
  │             Δx│    Vector{$FC}│                0│
  │           wₖ₋₁│    Vector{$FC}│               64│
  │             wₖ│    Vector{$FC}│               64│
  │        M⁻¹vₖ₋₁│    Vector{$FC}│               64│
  │          M⁻¹vₖ│    Vector{$FC}│               64│
  │              x│    Vector{$FC}│               64│
  │              p│    Vector{$FC}│               64│
  │             vₖ│    Vector{$FC}│                0│
  │     warm_start│           Bool│                0│
  └───────────────┴───────────────┴─────────────────┘
  """
  @test reduce(replace, [" " => "", "\n" => "", "─" => ""], init=showed) == reduce(replace, [" " => "", "\n" => "", "─" => ""], init=expected)

  io = IOBuffer()
  show(io, qmr_solver, show_stats=false)
  showed = String(take!(io))
  expected = """
  ┌──────────┬───────────────┬─────────────────┐
  │ QmrSolver│Precision: $FC │Architecture: CPU│
  ├──────────┼───────────────┼─────────────────┤
  │ Attribute│           Type│             Size│
  ├──────────┼───────────────┼─────────────────┤
  │      uₖ₋₁│    Vector{$FC}│               64│
  │        uₖ│    Vector{$FC}│               64│
  │         q│    Vector{$FC}│               64│
  │      vₖ₋₁│    Vector{$FC}│               64│
  │        vₖ│    Vector{$FC}│               64│
  │         p│    Vector{$FC}│               64│
  │        Δx│    Vector{$FC}│                0│
  │         x│    Vector{$FC}│               64│
  │      wₖ₋₂│    Vector{$FC}│               64│
  │      wₖ₋₁│    Vector{$FC}│               64│
  │warm_start│           Bool│                0│
  └──────────┴───────────────┴─────────────────┘
  """
  @test reduce(replace, [" " => "", "\n" => "", "─" => ""], init=showed) == reduce(replace, [" " => "", "\n" => "", "─" => ""], init=expected)

  io = IOBuffer()
  show(io, usymlq_solver, show_stats=false)
  showed = String(take!(io))
  expected = """
  ┌────────────┬───────────────┬─────────────────┐
  │UsymlqSolver│Precision: $FC │Architecture: CPU│
  ├────────────┼───────────────┼─────────────────┤
  │   Attribute│           Type│             Size│
  ├────────────┼───────────────┼─────────────────┤
  │        uₖ₋₁│    Vector{$FC}│               64│
  │          uₖ│    Vector{$FC}│               64│
  │           p│    Vector{$FC}│               64│
  │          Δx│    Vector{$FC}│                0│
  │           x│    Vector{$FC}│               64│
  │           d̅│    Vector{$FC}│               64│
  │        vₖ₋₁│    Vector{$FC}│               32│
  │          vₖ│    Vector{$FC}│               32│
  │           q│    Vector{$FC}│               32│
  │  warm_start│           Bool│                0│
  └────────────┴───────────────┴─────────────────┘
  """
  @test reduce(replace, [" " => "", "\n" => "", "─" => ""], init=showed) == reduce(replace, [" " => "", "\n" => "", "─" => ""], init=expected)

  io = IOBuffer()
  show(io, tricg_solver, show_stats=false)
  showed = String(take!(io))
  expected = """
  ┌───────────┬───────────────┬─────────────────┐
  │TricgSolver│Precision: $FC │Architecture: CPU│
  ├───────────┼───────────────┼─────────────────┤
  │  Attribute│           Type│             Size│
  ├───────────┼───────────────┼─────────────────┤
  │          y│    Vector{$FC}│               64│
  │    N⁻¹uₖ₋₁│    Vector{$FC}│               64│
  │      N⁻¹uₖ│    Vector{$FC}│               64│
  │          p│    Vector{$FC}│               64│
  │     gy₂ₖ₋₁│    Vector{$FC}│               64│
  │       gy₂ₖ│    Vector{$FC}│               64│
  │          x│    Vector{$FC}│               32│
  │    M⁻¹vₖ₋₁│    Vector{$FC}│               32│
  │      M⁻¹vₖ│    Vector{$FC}│               32│
  │          q│    Vector{$FC}│               32│
  │     gx₂ₖ₋₁│    Vector{$FC}│               32│
  │       gx₂ₖ│    Vector{$FC}│               32│
  │         Δx│    Vector{$FC}│                0│
  │         Δy│    Vector{$FC}│                0│
  │         uₖ│    Vector{$FC}│                0│
  │         vₖ│    Vector{$FC}│                0│
  │ warm_start│           Bool│                0│
  └───────────┴───────────────┴─────────────────┘
  """
  @test reduce(replace, [" " => "", "\n" => "", "─" => ""], init=showed) == reduce(replace, [" " => "", "\n" => "", "─" => ""], init=expected)

  io = IOBuffer()
  show(io, trimr_solver, show_stats=false)
  showed = String(take!(io))
  expected = """
  ┌───────────┬───────────────┬─────────────────┐
  │TrimrSolver│Precision: $FC │Architecture: CPU│
  ├───────────┼───────────────┼─────────────────┤
  │  Attribute│           Type│             Size│
  ├───────────┼───────────────┼─────────────────┤
  │          y│    Vector{$FC}│               64│
  │    N⁻¹uₖ₋₁│    Vector{$FC}│               64│
  │      N⁻¹uₖ│    Vector{$FC}│               64│
  │          p│    Vector{$FC}│               64│
  │     gy₂ₖ₋₃│    Vector{$FC}│               64│
  │     gy₂ₖ₋₂│    Vector{$FC}│               64│
  │     gy₂ₖ₋₁│    Vector{$FC}│               64│
  │       gy₂ₖ│    Vector{$FC}│               64│
  │          x│    Vector{$FC}│               32│
  │    M⁻¹vₖ₋₁│    Vector{$FC}│               32│
  │      M⁻¹vₖ│    Vector{$FC}│               32│
  │          q│    Vector{$FC}│               32│
  │     gx₂ₖ₋₃│    Vector{$FC}│               32│
  │     gx₂ₖ₋₂│    Vector{$FC}│               32│
  │     gx₂ₖ₋₁│    Vector{$FC}│               32│
  │       gx₂ₖ│    Vector{$FC}│               32│
  │         Δx│    Vector{$FC}│                0│
  │         Δy│    Vector{$FC}│                0│
  │         uₖ│    Vector{$FC}│                0│
  │         vₖ│    Vector{$FC}│                0│
  │ warm_start│           Bool│                0│
  └───────────┴───────────────┴─────────────────┘
  """
  @test reduce(replace, [" " => "", "\n" => "", "─" => ""], init=showed) == reduce(replace, [" " => "", "\n" => "", "─" => ""], init=expected)

  io = IOBuffer()
  show(io, gpmr_solver, show_stats=false)
  showed = String(take!(io))
  expected = """
  ┌──────────┬───────────────────┬─────────────────┐
  │GpmrSolver│    Precision: $FC │Architecture: CPU│
  ├──────────┼───────────────────┼─────────────────┤
  │ Attribute│               Type│             Size│
  ├──────────┼───────────────────┼─────────────────┤
  │        wA│        Vector{$FC}│                0│
  │        wB│        Vector{$FC}│                0│
  │        dA│        Vector{$FC}│               64│
  │        dB│        Vector{$FC}│               32│
  │        Δx│        Vector{$FC}│                0│
  │        Δy│        Vector{$FC}│                0│
  │         x│        Vector{$FC}│               64│
  │         y│        Vector{$FC}│               32│
  │         q│        Vector{$FC}│                0│
  │         p│        Vector{$FC}│                0│
  │         V│Vector{Vector{$FC}}│          10 x 64│
  │         U│Vector{Vector{$FC}}│          10 x 32│
  │        gs│        Vector{$FC}│               40│
  │        gc│         Vector{$T}│               40│
  │        zt│        Vector{$FC}│               20│
  │         R│        Vector{$FC}│              210│
  │warm_start│               Bool│                0│
  └──────────┴───────────────────┴─────────────────┘
  """
  @test reduce(replace, [" " => "", "\n" => "", "─" => ""], init=showed) == reduce(replace, [" " => "", "\n" => "", "─" => ""], init=expected)
end

@testset "solvers" begin
  for FC in (Float64, ComplexF64)
    @testset "Data Type: $FC" begin
      test_solvers(FC)
    end
  end
end
