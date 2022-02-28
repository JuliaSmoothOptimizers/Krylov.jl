@testset "solvers" begin
  A   = get_div_grad(4, 4, 4)  # Dimension n x n
  n   = size(A, 1)
  m   = div(n, 2)
  Au  = A[1:m,:]  # Dimension m x n
  Ao  = A[:,1:m]  # Dimension n x m
  b   = Ao * ones(m) # Dimension n
  c   = Au * ones(n) # Dimension m
  mem = 10
  shifts = [1.0; 2.0; 3.0; 4.0; 5.0]
  nshifts = 5
  S = Vector{Float64}

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
  end
  cg_lanczos_shift_solver = CgLanczosShiftSolver(n, n, nshifts, S)

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
    @test issolved(solver)

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

  if VERSION < v"1.8.0-DEV.1090"

    io = IOBuffer()
    show(io, cg_solver, show_stats=false)
    showed = String(take!(io))
    expected = """
    ┌────────────────────┬──────────────────────────┬──────────────────┐
    │            CgSolver│        Precision: Float64│ Architecture: CPU│
    ├────────────────────┼──────────────────────────┼──────────────────┤
    │           Attribute│                      Type│              Size│
    ├────────────────────┼──────────────────────────┼──────────────────┤
    │                  Δx│           Vector{Float64}│                 0│
    │                   x│           Vector{Float64}│                64│
    │                   r│           Vector{Float64}│                64│
    │                   p│           Vector{Float64}│                64│
    │                  Ap│           Vector{Float64}│                64│
    │                   z│           Vector{Float64}│                 0│
    └────────────────────┴──────────────────────────┴──────────────────┘
    """
    @test strip.(split(chomp(showed), "\n")) == strip.(split(chomp(expected), "\n"))

    io = IOBuffer()
    show(io, symmlq_solver, show_stats=false)
    showed = String(take!(io))
    expected = """
    ┌────────────────────┬──────────────────────────┬──────────────────┐
    │        SymmlqSolver│        Precision: Float64│ Architecture: CPU│
    ├────────────────────┼──────────────────────────┼──────────────────┤
    │           Attribute│                      Type│              Size│
    ├────────────────────┼──────────────────────────┼──────────────────┤
    │                  Δx│           Vector{Float64}│                 0│
    │                   x│           Vector{Float64}│                64│
    │               Mvold│           Vector{Float64}│                64│
    │                  Mv│           Vector{Float64}│                64│
    │             Mv_next│           Vector{Float64}│                64│
    │                   w̅│           Vector{Float64}│                64│
    │                   v│           Vector{Float64}│                 0│
    │               clist│           Vector{Float64}│                 5│
    │               zlist│           Vector{Float64}│                 5│
    │               sprod│           Vector{Float64}│                 5│
    └────────────────────┴──────────────────────────┴──────────────────┘
    """
    @test strip.(split(chomp(showed), "\n")) == strip.(split(chomp(expected), "\n"))

    io = IOBuffer()
    show(io, minres_solver, show_stats=false)
    showed = String(take!(io))
    expected = """
    ┌────────────────────┬──────────────────────────┬──────────────────┐
    │        MinresSolver│        Precision: Float64│ Architecture: CPU│
    ├────────────────────┼──────────────────────────┼──────────────────┤
    │           Attribute│                      Type│              Size│
    ├────────────────────┼──────────────────────────┼──────────────────┤
    │                  Δx│           Vector{Float64}│                 0│
    │                   x│           Vector{Float64}│                64│
    │                  r1│           Vector{Float64}│                64│
    │                  r2│           Vector{Float64}│                64│
    │                  w1│           Vector{Float64}│                64│
    │                  w2│           Vector{Float64}│                64│
    │                   y│           Vector{Float64}│                64│
    │                   v│           Vector{Float64}│                 0│
    │             err_vec│           Vector{Float64}│                 5│
    └────────────────────┴──────────────────────────┴──────────────────┘
    """
    @test strip.(split(chomp(showed), "\n")) == strip.(split(chomp(expected), "\n"))

    io = IOBuffer()
    show(io, cg_lanczos_solver, show_stats=false)
    showed = String(take!(io))
    expected = """
    ┌────────────────────┬──────────────────────────┬──────────────────┐
    │     CgLanczosSolver│        Precision: Float64│ Architecture: CPU│
    ├────────────────────┼──────────────────────────┼──────────────────┤
    │           Attribute│                      Type│              Size│
    ├────────────────────┼──────────────────────────┼──────────────────┤
    │                   x│           Vector{Float64}│                64│
    │                  Mv│           Vector{Float64}│                64│
    │             Mv_prev│           Vector{Float64}│                64│
    │                   p│           Vector{Float64}│                64│
    │             Mv_next│           Vector{Float64}│                64│
    │                   v│           Vector{Float64}│                 0│
    └────────────────────┴──────────────────────────┴──────────────────┘
    """
    @test strip.(split(chomp(showed), "\n")) == strip.(split(chomp(expected), "\n"))

    io = IOBuffer()
    show(io, cg_lanczos_shift_solver, show_stats=false)
    showed = String(take!(io))
    expected = """
    ┌────────────────────┬──────────────────────────┬──────────────────┐
    │CgLanczosShiftSolver│        Precision: Float64│ Architecture: CPU│
    ├────────────────────┼──────────────────────────┼──────────────────┤
    │           Attribute│                      Type│              Size│
    ├────────────────────┼──────────────────────────┼──────────────────┤
    │                  Mv│           Vector{Float64}│                64│
    │             Mv_prev│           Vector{Float64}│                64│
    │             Mv_next│           Vector{Float64}│                64│
    │                   v│           Vector{Float64}│                 0│
    │                   x│   Vector{Vector{Float64}}│            5 x 64│
    │                   p│   Vector{Vector{Float64}}│            5 x 64│
    │                   σ│           Vector{Float64}│                 5│
    │                δhat│           Vector{Float64}│                 5│
    │                   ω│           Vector{Float64}│                 5│
    │                   γ│           Vector{Float64}│                 5│
    │              rNorms│           Vector{Float64}│                 5│
    │           converged│                 BitVector│                 5│
    │              not_cv│                 BitVector│                 5│
    └────────────────────┴──────────────────────────┴──────────────────┘
    """
    @test strip.(split(chomp(showed), "\n")) == strip.(split(chomp(expected), "\n"))

    io = IOBuffer()
    show(io, diom_solver, show_stats=false)
    showed = String(take!(io))
    expected = """
    ┌────────────────────┬──────────────────────────┬──────────────────┐
    │          DiomSolver│        Precision: Float64│ Architecture: CPU│
    ├────────────────────┼──────────────────────────┼──────────────────┤
    │           Attribute│                      Type│              Size│
    ├────────────────────┼──────────────────────────┼──────────────────┤
    │                  Δx│           Vector{Float64}│                 0│
    │                   x│           Vector{Float64}│                64│
    │                   t│           Vector{Float64}│                64│
    │                   z│           Vector{Float64}│                 0│
    │                   w│           Vector{Float64}│                 0│
    │                   P│   Vector{Vector{Float64}}│           10 x 64│
    │                   V│   Vector{Vector{Float64}}│           10 x 64│
    │                   L│           Vector{Float64}│                10│
    │                   H│           Vector{Float64}│                12│
    └────────────────────┴──────────────────────────┴──────────────────┘
    """
    @test strip.(split(chomp(showed), "\n")) == strip.(split(chomp(expected), "\n"))

    io = IOBuffer()
    show(io, fom_solver, show_stats=false)
    showed = String(take!(io))
    expected = """
    ┌────────────────────┬──────────────────────────┬──────────────────┐
    │           FomSolver│        Precision: Float64│ Architecture: CPU│
    ├────────────────────┼──────────────────────────┼──────────────────┤
    │           Attribute│                      Type│              Size│
    ├────────────────────┼──────────────────────────┼──────────────────┤
    │                  Δx│           Vector{Float64}│                 0│
    │                   x│           Vector{Float64}│                64│
    │                   w│           Vector{Float64}│                64│
    │                   p│           Vector{Float64}│                 0│
    │                   q│           Vector{Float64}│                 0│
    │                   V│   Vector{Vector{Float64}}│           10 x 64│
    │                   l│           Vector{Float64}│                10│
    │                   z│           Vector{Float64}│                10│
    │                   U│           Vector{Float64}│                55│
    └────────────────────┴──────────────────────────┴──────────────────┘
    """
    @test strip.(split(chomp(showed), "\n")) == strip.(split(chomp(expected), "\n"))

    io = IOBuffer()
    show(io, dqgmres_solver, show_stats=false)
    showed = String(take!(io))
    expected = """
    ┌────────────────────┬──────────────────────────┬──────────────────┐
    │       DqgmresSolver│        Precision: Float64│ Architecture: CPU│
    ├────────────────────┼──────────────────────────┼──────────────────┤
    │           Attribute│                      Type│              Size│
    ├────────────────────┼──────────────────────────┼──────────────────┤
    │                  Δx│           Vector{Float64}│                 0│
    │                   x│           Vector{Float64}│                64│
    │                   t│           Vector{Float64}│                64│
    │                   z│           Vector{Float64}│                 0│
    │                   w│           Vector{Float64}│                 0│
    │                   P│   Vector{Vector{Float64}}│           10 x 64│
    │                   V│   Vector{Vector{Float64}}│           10 x 64│
    │                   c│           Vector{Float64}│                10│
    │                   s│           Vector{Float64}│                10│
    │                   H│           Vector{Float64}│                12│
    └────────────────────┴──────────────────────────┴──────────────────┘
    """
    @test strip.(split(chomp(showed), "\n")) == strip.(split(chomp(expected), "\n"))

    io = IOBuffer()
    show(io, gmres_solver, show_stats=false)
    showed = String(take!(io))
    expected = """
    ┌────────────────────┬──────────────────────────┬──────────────────┐
    │         GmresSolver│        Precision: Float64│ Architecture: CPU│
    ├────────────────────┼──────────────────────────┼──────────────────┤
    │           Attribute│                      Type│              Size│
    ├────────────────────┼──────────────────────────┼──────────────────┤
    │                  Δx│           Vector{Float64}│                 0│
    │                   x│           Vector{Float64}│                64│
    │                   w│           Vector{Float64}│                64│
    │                   p│           Vector{Float64}│                 0│
    │                   q│           Vector{Float64}│                 0│
    │                   V│   Vector{Vector{Float64}}│           10 x 64│
    │                   c│           Vector{Float64}│                10│
    │                   s│           Vector{Float64}│                10│
    │                   z│           Vector{Float64}│                10│
    │                   R│           Vector{Float64}│                55│
    └────────────────────┴──────────────────────────┴──────────────────┘
    """
    @test strip.(split(chomp(showed), "\n")) == strip.(split(chomp(expected), "\n"))

    io = IOBuffer()
    show(io, cr_solver, show_stats=false)
    showed = String(take!(io))
    expected = """
    ┌────────────────────┬──────────────────────────┬──────────────────┐
    │            CrSolver│        Precision: Float64│ Architecture: CPU│
    ├────────────────────┼──────────────────────────┼──────────────────┤
    │           Attribute│                      Type│              Size│
    ├────────────────────┼──────────────────────────┼──────────────────┤
    │                   x│           Vector{Float64}│                64│
    │                   r│           Vector{Float64}│                64│
    │                   p│           Vector{Float64}│                64│
    │                   q│           Vector{Float64}│                64│
    │                  Ar│           Vector{Float64}│                64│
    │                  Mq│           Vector{Float64}│                 0│
    └────────────────────┴──────────────────────────┴──────────────────┘
    """
    @test strip.(split(chomp(showed), "\n")) == strip.(split(chomp(expected), "\n"))

    io = IOBuffer()
    show(io, crmr_solver, show_stats=false)
    showed = String(take!(io))
    expected = """
    ┌────────────────────┬──────────────────────────┬──────────────────┐
    │          CrmrSolver│        Precision: Float64│ Architecture: CPU│
    ├────────────────────┼──────────────────────────┼──────────────────┤
    │           Attribute│                      Type│              Size│
    ├────────────────────┼──────────────────────────┼──────────────────┤
    │                   x│           Vector{Float64}│                64│
    │                   p│           Vector{Float64}│                64│
    │                 Aᵀr│           Vector{Float64}│                64│
    │                   r│           Vector{Float64}│                32│
    │                   q│           Vector{Float64}│                32│
    │                  Mq│           Vector{Float64}│                 0│
    │                   s│           Vector{Float64}│                 0│
    └────────────────────┴──────────────────────────┴──────────────────┘
    """
    @test strip.(split(chomp(showed), "\n")) == strip.(split(chomp(expected), "\n"))

    io = IOBuffer()
    show(io, cgs_solver, show_stats=false)
    showed = String(take!(io))
    expected = """
    ┌────────────────────┬──────────────────────────┬──────────────────┐
    │           CgsSolver│        Precision: Float64│ Architecture: CPU│
    ├────────────────────┼──────────────────────────┼──────────────────┤
    │           Attribute│                      Type│              Size│
    ├────────────────────┼──────────────────────────┼──────────────────┤
    │                   x│           Vector{Float64}│                64│
    │                   r│           Vector{Float64}│                64│
    │                   u│           Vector{Float64}│                64│
    │                   p│           Vector{Float64}│                64│
    │                   q│           Vector{Float64}│                64│
    │                  ts│           Vector{Float64}│                64│
    │                  yz│           Vector{Float64}│                 0│
    │                  vw│           Vector{Float64}│                 0│
    └────────────────────┴──────────────────────────┴──────────────────┘
    """
    @test strip.(split(chomp(showed), "\n")) == strip.(split(chomp(expected), "\n"))

    io = IOBuffer()
    show(io, bicgstab_solver, show_stats=false)
    showed = String(take!(io))
    expected = """
    ┌────────────────────┬──────────────────────────┬──────────────────┐
    │      BicgstabSolver│        Precision: Float64│ Architecture: CPU│
    ├────────────────────┼──────────────────────────┼──────────────────┤
    │           Attribute│                      Type│              Size│
    ├────────────────────┼──────────────────────────┼──────────────────┤
    │                   x│           Vector{Float64}│                64│
    │                   r│           Vector{Float64}│                64│
    │                   p│           Vector{Float64}│                64│
    │                   v│           Vector{Float64}│                64│
    │                   s│           Vector{Float64}│                64│
    │                  qd│           Vector{Float64}│                64│
    │                  yz│           Vector{Float64}│                 0│
    │                   t│           Vector{Float64}│                 0│
    └────────────────────┴──────────────────────────┴──────────────────┘
    """
    @test strip.(split(chomp(showed), "\n")) == strip.(split(chomp(expected), "\n"))

    io = IOBuffer()
    show(io, craigmr_solver, show_stats=false)
    showed = String(take!(io))
    expected = """
    ┌────────────────────┬──────────────────────────┬──────────────────┐
    │       CraigmrSolver│        Precision: Float64│ Architecture: CPU│
    ├────────────────────┼──────────────────────────┼──────────────────┤
    │           Attribute│                      Type│              Size│
    ├────────────────────┼──────────────────────────┼──────────────────┤
    │                   x│           Vector{Float64}│                64│
    │                  Nv│           Vector{Float64}│                64│
    │                 Aᵀu│           Vector{Float64}│                64│
    │                   d│           Vector{Float64}│                64│
    │                   y│           Vector{Float64}│                32│
    │                  Mu│           Vector{Float64}│                32│
    │                   w│           Vector{Float64}│                32│
    │                wbar│           Vector{Float64}│                32│
    │                  Av│           Vector{Float64}│                32│
    │                   u│           Vector{Float64}│                 0│
    │                   v│           Vector{Float64}│                 0│
    │                   q│           Vector{Float64}│                 0│
    └────────────────────┴──────────────────────────┴──────────────────┘
    """
    @test strip.(split(chomp(showed), "\n")) == strip.(split(chomp(expected), "\n"))

    io = IOBuffer()
    show(io, cgne_solver, show_stats=false)
    showed = String(take!(io))
    expected = """
    ┌────────────────────┬──────────────────────────┬──────────────────┐
    │          CgneSolver│        Precision: Float64│ Architecture: CPU│
    ├────────────────────┼──────────────────────────┼──────────────────┤
    │           Attribute│                      Type│              Size│
    ├────────────────────┼──────────────────────────┼──────────────────┤
    │                   x│           Vector{Float64}│                64│
    │                   p│           Vector{Float64}│                64│
    │                 Aᵀz│           Vector{Float64}│                64│
    │                   r│           Vector{Float64}│                32│
    │                   q│           Vector{Float64}│                32│
    │                   s│           Vector{Float64}│                 0│
    │                   z│           Vector{Float64}│                 0│
    └────────────────────┴──────────────────────────┴──────────────────┘
    """
    @test strip.(split(chomp(showed), "\n")) == strip.(split(chomp(expected), "\n"))

    io = IOBuffer()
    show(io, lnlq_solver, show_stats=false)
    showed = String(take!(io))
    expected = """
    ┌────────────────────┬──────────────────────────┬──────────────────┐
    │          LnlqSolver│        Precision: Float64│ Architecture: CPU│
    ├────────────────────┼──────────────────────────┼──────────────────┤
    │           Attribute│                      Type│              Size│
    ├────────────────────┼──────────────────────────┼──────────────────┤
    │                   x│           Vector{Float64}│                64│
    │                  Nv│           Vector{Float64}│                64│
    │                 Aᵀu│           Vector{Float64}│                64│
    │                   y│           Vector{Float64}│                32│
    │                   w̄│           Vector{Float64}│                32│
    │                  Mu│           Vector{Float64}│                32│
    │                  Av│           Vector{Float64}│                32│
    │                   u│           Vector{Float64}│                 0│
    │                   v│           Vector{Float64}│                 0│
    │                   q│           Vector{Float64}│                 0│
    └────────────────────┴──────────────────────────┴──────────────────┘
    """
    @test strip.(split(chomp(showed), "\n")) == strip.(split(chomp(expected), "\n"))

    io = IOBuffer()
    show(io, craig_solver, show_stats=false)
    showed = String(take!(io))
    expected = """
    ┌────────────────────┬──────────────────────────┬──────────────────┐
    │         CraigSolver│        Precision: Float64│ Architecture: CPU│
    ├────────────────────┼──────────────────────────┼──────────────────┤
    │           Attribute│                      Type│              Size│
    ├────────────────────┼──────────────────────────┼──────────────────┤
    │                   x│           Vector{Float64}│                64│
    │                  Nv│           Vector{Float64}│                64│
    │                 Aᵀu│           Vector{Float64}│                64│
    │                   y│           Vector{Float64}│                32│
    │                   w│           Vector{Float64}│                32│
    │                  Mu│           Vector{Float64}│                32│
    │                  Av│           Vector{Float64}│                32│
    │                   u│           Vector{Float64}│                 0│
    │                   v│           Vector{Float64}│                 0│
    │                  w2│           Vector{Float64}│                 0│
    └────────────────────┴──────────────────────────┴──────────────────┘
    """
    @test strip.(split(chomp(showed), "\n")) == strip.(split(chomp(expected), "\n"))

    io = IOBuffer()
    show(io, lslq_solver, show_stats=false)
    showed = String(take!(io))
    expected = """
    ┌────────────────────┬──────────────────────────┬──────────────────┐
    │          LslqSolver│        Precision: Float64│ Architecture: CPU│
    ├────────────────────┼──────────────────────────┼──────────────────┤
    │           Attribute│                      Type│              Size│
    ├────────────────────┼──────────────────────────┼──────────────────┤
    │                   x│           Vector{Float64}│                32│
    │                  Nv│           Vector{Float64}│                32│
    │                 Aᵀu│           Vector{Float64}│                32│
    │                   w̄│           Vector{Float64}│                32│
    │                  Mu│           Vector{Float64}│                64│
    │                  Av│           Vector{Float64}│                64│
    │                   u│           Vector{Float64}│                 0│
    │                   v│           Vector{Float64}│                 0│
    │             err_vec│           Vector{Float64}│                 5│
    └────────────────────┴──────────────────────────┴──────────────────┘
    """
    @test strip.(split(chomp(showed), "\n")) == strip.(split(chomp(expected), "\n"))

    io = IOBuffer()
    show(io, cgls_solver, show_stats=false)
    showed = String(take!(io))
    expected = """
    ┌────────────────────┬──────────────────────────┬──────────────────┐
    │          CglsSolver│        Precision: Float64│ Architecture: CPU│
    ├────────────────────┼──────────────────────────┼──────────────────┤
    │           Attribute│                      Type│              Size│
    ├────────────────────┼──────────────────────────┼──────────────────┤
    │                   x│           Vector{Float64}│                32│
    │                   p│           Vector{Float64}│                32│
    │                   s│           Vector{Float64}│                32│
    │                   r│           Vector{Float64}│                64│
    │                   q│           Vector{Float64}│                64│
    │                  Mr│           Vector{Float64}│                 0│
    └────────────────────┴──────────────────────────┴──────────────────┘
    """
    @test strip.(split(chomp(showed), "\n")) == strip.(split(chomp(expected), "\n"))

    io = IOBuffer()
    show(io, lsqr_solver, show_stats=false)
    showed = String(take!(io))
    expected = """
    ┌────────────────────┬──────────────────────────┬──────────────────┐
    │          LsqrSolver│        Precision: Float64│ Architecture: CPU│
    ├────────────────────┼──────────────────────────┼──────────────────┤
    │           Attribute│                      Type│              Size│
    ├────────────────────┼──────────────────────────┼──────────────────┤
    │                   x│           Vector{Float64}│                32│
    │                  Nv│           Vector{Float64}│                32│
    │                 Aᵀu│           Vector{Float64}│                32│
    │                   w│           Vector{Float64}│                32│
    │                  Mu│           Vector{Float64}│                64│
    │                  Av│           Vector{Float64}│                64│
    │                   u│           Vector{Float64}│                 0│
    │                   v│           Vector{Float64}│                 0│
    │             err_vec│           Vector{Float64}│                 5│
    └────────────────────┴──────────────────────────┴──────────────────┘
    """
    @test strip.(split(chomp(showed), "\n")) == strip.(split(chomp(expected), "\n"))

    io = IOBuffer()
    show(io, crls_solver, show_stats=false)
    showed = String(take!(io))
    expected = """
    ┌────────────────────┬──────────────────────────┬──────────────────┐
    │          CrlsSolver│        Precision: Float64│ Architecture: CPU│
    ├────────────────────┼──────────────────────────┼──────────────────┤
    │           Attribute│                      Type│              Size│
    ├────────────────────┼──────────────────────────┼──────────────────┤
    │                   x│           Vector{Float64}│                32│
    │                   p│           Vector{Float64}│                32│
    │                  Ar│           Vector{Float64}│                32│
    │                   q│           Vector{Float64}│                32│
    │                   r│           Vector{Float64}│                64│
    │                  Ap│           Vector{Float64}│                64│
    │                   s│           Vector{Float64}│                64│
    │                  Ms│           Vector{Float64}│                 0│
    └────────────────────┴──────────────────────────┴──────────────────┘
    """
    @test strip.(split(chomp(showed), "\n")) == strip.(split(chomp(expected), "\n"))

    io = IOBuffer()
    show(io, lsmr_solver, show_stats=false)
    showed = String(take!(io))
    expected = """
    ┌────────────────────┬──────────────────────────┬──────────────────┐
    │          LsmrSolver│        Precision: Float64│ Architecture: CPU│
    ├────────────────────┼──────────────────────────┼──────────────────┤
    │           Attribute│                      Type│              Size│
    ├────────────────────┼──────────────────────────┼──────────────────┤
    │                   x│           Vector{Float64}│                32│
    │                  Nv│           Vector{Float64}│                32│
    │                 Aᵀu│           Vector{Float64}│                32│
    │                   h│           Vector{Float64}│                32│
    │                hbar│           Vector{Float64}│                32│
    │                  Mu│           Vector{Float64}│                64│
    │                  Av│           Vector{Float64}│                64│
    │                   u│           Vector{Float64}│                 0│
    │                   v│           Vector{Float64}│                 0│
    │             err_vec│           Vector{Float64}│                 5│
    └────────────────────┴──────────────────────────┴──────────────────┘
    """
    @test strip.(split(chomp(showed), "\n")) == strip.(split(chomp(expected), "\n"))

    io = IOBuffer()
    show(io, usymqr_solver, show_stats=false)
    showed = String(take!(io))
    expected = """
    ┌────────────────────┬──────────────────────────┬──────────────────┐
    │        UsymqrSolver│        Precision: Float64│ Architecture: CPU│
    ├────────────────────┼──────────────────────────┼──────────────────┤
    │           Attribute│                      Type│              Size│
    ├────────────────────┼──────────────────────────┼──────────────────┤
    │                vₖ₋₁│           Vector{Float64}│                64│
    │                  vₖ│           Vector{Float64}│                64│
    │                   q│           Vector{Float64}│                64│
    │                   x│           Vector{Float64}│                32│
    │                wₖ₋₂│           Vector{Float64}│                32│
    │                wₖ₋₁│           Vector{Float64}│                32│
    │                uₖ₋₁│           Vector{Float64}│                32│
    │                  uₖ│           Vector{Float64}│                32│
    │                   p│           Vector{Float64}│                32│
    └────────────────────┴──────────────────────────┴──────────────────┘
    """
    @test strip.(split(chomp(showed), "\n")) == strip.(split(chomp(expected), "\n"))

    io = IOBuffer()
    show(io, trilqr_solver, show_stats=false)
    showed = String(take!(io))
    expected = """
    ┌────────────────────┬──────────────────────────┬──────────────────┐
    │        TrilqrSolver│        Precision: Float64│ Architecture: CPU│
    ├────────────────────┼──────────────────────────┼──────────────────┤
    │           Attribute│                      Type│              Size│
    ├────────────────────┼──────────────────────────┼──────────────────┤
    │                uₖ₋₁│           Vector{Float64}│                64│
    │                  uₖ│           Vector{Float64}│                64│
    │                   p│           Vector{Float64}│                64│
    │                   d̅│           Vector{Float64}│                64│
    │                   x│           Vector{Float64}│                64│
    │                vₖ₋₁│           Vector{Float64}│                64│
    │                  vₖ│           Vector{Float64}│                64│
    │                   q│           Vector{Float64}│                64│
    │                   y│           Vector{Float64}│                64│
    │                wₖ₋₃│           Vector{Float64}│                64│
    │                wₖ₋₂│           Vector{Float64}│                64│
    └────────────────────┴──────────────────────────┴──────────────────┘
    """
    @test strip.(split(chomp(showed), "\n")) == strip.(split(chomp(expected), "\n"))

    io = IOBuffer()
    show(io, bilq_solver, show_stats=false)
    showed = String(take!(io))
    expected = """
    ┌────────────────────┬──────────────────────────┬──────────────────┐
    │          BilqSolver│        Precision: Float64│ Architecture: CPU│
    ├────────────────────┼──────────────────────────┼──────────────────┤
    │           Attribute│                      Type│              Size│
    ├────────────────────┼──────────────────────────┼──────────────────┤
    │                uₖ₋₁│           Vector{Float64}│                64│
    │                  uₖ│           Vector{Float64}│                64│
    │                   q│           Vector{Float64}│                64│
    │                vₖ₋₁│           Vector{Float64}│                64│
    │                  vₖ│           Vector{Float64}│                64│
    │                   p│           Vector{Float64}│                64│
    │                   x│           Vector{Float64}│                64│
    │                   d̅│           Vector{Float64}│                64│
    └────────────────────┴──────────────────────────┴──────────────────┘
    """
    @test strip.(split(chomp(showed), "\n")) == strip.(split(chomp(expected), "\n"))

    io = IOBuffer()
    show(io, bilqr_solver, show_stats=false)
    showed = String(take!(io))
    expected = """
    ┌────────────────────┬──────────────────────────┬──────────────────┐
    │         BilqrSolver│        Precision: Float64│ Architecture: CPU│
    ├────────────────────┼──────────────────────────┼──────────────────┤
    │           Attribute│                      Type│              Size│
    ├────────────────────┼──────────────────────────┼──────────────────┤
    │                uₖ₋₁│           Vector{Float64}│                64│
    │                  uₖ│           Vector{Float64}│                64│
    │                   q│           Vector{Float64}│                64│
    │                vₖ₋₁│           Vector{Float64}│                64│
    │                  vₖ│           Vector{Float64}│                64│
    │                   p│           Vector{Float64}│                64│
    │                   x│           Vector{Float64}│                64│
    │                   y│           Vector{Float64}│                64│
    │                   d̅│           Vector{Float64}│                64│
    │                wₖ₋₃│           Vector{Float64}│                64│
    │                wₖ₋₂│           Vector{Float64}│                64│
    └────────────────────┴──────────────────────────┴──────────────────┘
    """
    @test strip.(split(chomp(showed), "\n")) == strip.(split(chomp(expected), "\n"))

    io = IOBuffer()
    show(io, minres_qlp_solver, show_stats=false)
    showed = String(take!(io))
    expected = """
    ┌────────────────────┬──────────────────────────┬──────────────────┐
    │     MinresQlpSolver│        Precision: Float64│ Architecture: CPU│
    ├────────────────────┼──────────────────────────┼──────────────────┤
    │           Attribute│                      Type│              Size│
    ├────────────────────┼──────────────────────────┼──────────────────┤
    │                  Δx│           Vector{Float64}│                 0│
    │                wₖ₋₁│           Vector{Float64}│                64│
    │                  wₖ│           Vector{Float64}│                64│
    │             M⁻¹vₖ₋₁│           Vector{Float64}│                64│
    │               M⁻¹vₖ│           Vector{Float64}│                64│
    │                   x│           Vector{Float64}│                64│
    │                   p│           Vector{Float64}│                64│
    │                  vₖ│           Vector{Float64}│                 0│
    └────────────────────┴──────────────────────────┴──────────────────┘
    """
    @test strip.(split(chomp(showed), "\n")) == strip.(split(chomp(expected), "\n"))

    io = IOBuffer()
    show(io, qmr_solver, show_stats=false)
    showed = String(take!(io))
    expected = """
    ┌────────────────────┬──────────────────────────┬──────────────────┐
    │           QmrSolver│        Precision: Float64│ Architecture: CPU│
    ├────────────────────┼──────────────────────────┼──────────────────┤
    │           Attribute│                      Type│              Size│
    ├────────────────────┼──────────────────────────┼──────────────────┤
    │                uₖ₋₁│           Vector{Float64}│                64│
    │                  uₖ│           Vector{Float64}│                64│
    │                   q│           Vector{Float64}│                64│
    │                vₖ₋₁│           Vector{Float64}│                64│
    │                  vₖ│           Vector{Float64}│                64│
    │                   p│           Vector{Float64}│                64│
    │                   x│           Vector{Float64}│                64│
    │                wₖ₋₂│           Vector{Float64}│                64│
    │                wₖ₋₁│           Vector{Float64}│                64│
    └────────────────────┴──────────────────────────┴──────────────────┘
    """
    @test strip.(split(chomp(showed), "\n")) == strip.(split(chomp(expected), "\n"))

    io = IOBuffer()
    show(io, usymlq_solver, show_stats=false)
    showed = String(take!(io))
    expected = """
    ┌────────────────────┬──────────────────────────┬──────────────────┐
    │        UsymlqSolver│        Precision: Float64│ Architecture: CPU│
    ├────────────────────┼──────────────────────────┼──────────────────┤
    │           Attribute│                      Type│              Size│
    ├────────────────────┼──────────────────────────┼──────────────────┤
    │                uₖ₋₁│           Vector{Float64}│                64│
    │                  uₖ│           Vector{Float64}│                64│
    │                   p│           Vector{Float64}│                64│
    │                   x│           Vector{Float64}│                64│
    │                   d̅│           Vector{Float64}│                64│
    │                vₖ₋₁│           Vector{Float64}│                32│
    │                  vₖ│           Vector{Float64}│                32│
    │                   q│           Vector{Float64}│                32│
    └────────────────────┴──────────────────────────┴──────────────────┘
    """
    @test strip.(split(chomp(showed), "\n")) == strip.(split(chomp(expected), "\n"))

    io = IOBuffer()
    show(io, tricg_solver, show_stats=false)
    showed = String(take!(io))
    expected = """
    ┌────────────────────┬──────────────────────────┬──────────────────┐
    │         TricgSolver│        Precision: Float64│ Architecture: CPU│
    ├────────────────────┼──────────────────────────┼──────────────────┤
    │           Attribute│                      Type│              Size│
    ├────────────────────┼──────────────────────────┼──────────────────┤
    │                   y│           Vector{Float64}│                64│
    │             N⁻¹uₖ₋₁│           Vector{Float64}│                64│
    │               N⁻¹uₖ│           Vector{Float64}│                64│
    │                   p│           Vector{Float64}│                64│
    │              gy₂ₖ₋₁│           Vector{Float64}│                64│
    │                gy₂ₖ│           Vector{Float64}│                64│
    │                   x│           Vector{Float64}│                32│
    │             M⁻¹vₖ₋₁│           Vector{Float64}│                32│
    │               M⁻¹vₖ│           Vector{Float64}│                32│
    │                   q│           Vector{Float64}│                32│
    │              gx₂ₖ₋₁│           Vector{Float64}│                32│
    │                gx₂ₖ│           Vector{Float64}│                32│
    │                  Δx│           Vector{Float64}│                 0│
    │                  Δy│           Vector{Float64}│                 0│
    │                  uₖ│           Vector{Float64}│                 0│
    │                  vₖ│           Vector{Float64}│                 0│
    └────────────────────┴──────────────────────────┴──────────────────┘
    """
    @test strip.(split(chomp(showed), "\n")) == strip.(split(chomp(expected), "\n"))

    io = IOBuffer()
    show(io, trimr_solver, show_stats=false)
    showed = String(take!(io))
    expected = """
    ┌────────────────────┬──────────────────────────┬──────────────────┐
    │         TrimrSolver│        Precision: Float64│ Architecture: CPU│
    ├────────────────────┼──────────────────────────┼──────────────────┤
    │           Attribute│                      Type│              Size│
    ├────────────────────┼──────────────────────────┼──────────────────┤
    │                   y│           Vector{Float64}│                64│
    │             N⁻¹uₖ₋₁│           Vector{Float64}│                64│
    │               N⁻¹uₖ│           Vector{Float64}│                64│
    │                   p│           Vector{Float64}│                64│
    │              gy₂ₖ₋₃│           Vector{Float64}│                64│
    │              gy₂ₖ₋₂│           Vector{Float64}│                64│
    │              gy₂ₖ₋₁│           Vector{Float64}│                64│
    │                gy₂ₖ│           Vector{Float64}│                64│
    │                   x│           Vector{Float64}│                32│
    │             M⁻¹vₖ₋₁│           Vector{Float64}│                32│
    │               M⁻¹vₖ│           Vector{Float64}│                32│
    │                   q│           Vector{Float64}│                32│
    │              gx₂ₖ₋₃│           Vector{Float64}│                32│
    │              gx₂ₖ₋₂│           Vector{Float64}│                32│
    │              gx₂ₖ₋₁│           Vector{Float64}│                32│
    │                gx₂ₖ│           Vector{Float64}│                32│
    │                  Δx│           Vector{Float64}│                 0│
    │                  Δy│           Vector{Float64}│                 0│
    │                  uₖ│           Vector{Float64}│                 0│
    │                  vₖ│           Vector{Float64}│                 0│
    └────────────────────┴──────────────────────────┴──────────────────┘
    """
    @test strip.(split(chomp(showed), "\n")) == strip.(split(chomp(expected), "\n"))

    io = IOBuffer()
    show(io, gpmr_solver, show_stats=false)
    showed = String(take!(io))
    expected = """
    ┌────────────────────┬──────────────────────────┬──────────────────┐
    │          GpmrSolver│        Precision: Float64│ Architecture: CPU│
    ├────────────────────┼──────────────────────────┼──────────────────┤
    │           Attribute│                      Type│              Size│
    ├────────────────────┼──────────────────────────┼──────────────────┤
    │                  wA│           Vector{Float64}│                 0│
    │                  wB│           Vector{Float64}│                 0│
    │                  dA│           Vector{Float64}│                64│
    │                  dB│           Vector{Float64}│                32│
    │                  Δx│           Vector{Float64}│                 0│
    │                  Δy│           Vector{Float64}│                 0│
    │                   x│           Vector{Float64}│                64│
    │                   y│           Vector{Float64}│                32│
    │                   q│           Vector{Float64}│                 0│
    │                   p│           Vector{Float64}│                 0│
    │                   V│   Vector{Vector{Float64}}│           10 x 64│
    │                   U│   Vector{Vector{Float64}}│           10 x 32│
    │                  gs│           Vector{Float64}│                40│
    │                  gc│           Vector{Float64}│                40│
    │                  zt│           Vector{Float64}│                20│
    │                   R│           Vector{Float64}│               210│
    └────────────────────┴──────────────────────────┴──────────────────┘
    """
    @test strip.(split(chomp(showed), "\n")) == strip.(split(chomp(expected), "\n"))
  end
end
