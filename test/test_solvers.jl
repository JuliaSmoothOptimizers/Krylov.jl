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

  cg_solver = CgSolver(n, n, Vector{Float64})
  symmlq_solver = SymmlqSolver(n, n, Vector{Float64})
  minres_solver = MinresSolver(n, n, Vector{Float64})
  cg_lanczos_solver = CgLanczosSolver(n, n, Vector{Float64})
  cg_lanczos_shift_solver = CgLanczosShiftSolver(n, n, nshifts, Vector{Float64})
  diom_solver = DiomSolver(n, n, mem, Vector{Float64})
  dqgmres_solver = DqgmresSolver(n, n, mem, Vector{Float64})
  gmres_solver = GmresSolver(n, n, mem, Vector{Float64})
  cr_solver = CrSolver(n, n, Vector{Float64})
  crmr_solver = CrmrSolver(m, n, Vector{Float64})
  cgs_solver = CgsSolver(n, n, Vector{Float64})
  bicgstab_solver = BicgstabSolver(n, n, Vector{Float64})
  craigmr_solver = CraigmrSolver(m, n, Vector{Float64})
  cgne_solver = CgneSolver(m, n, Vector{Float64})
  lnlq_solver = LnlqSolver(m, n, Vector{Float64})
  craig_solver = CraigSolver(m, n, Vector{Float64})
  lslq_solver = LslqSolver(n, m, Vector{Float64})
  cgls_solver = CglsSolver(n, m, Vector{Float64})
  lsqr_solver = LsqrSolver(n, m, Vector{Float64})
  crls_solver = CrlsSolver(n, m, Vector{Float64})
  lsmr_solver = LsmrSolver(n, m, Vector{Float64})
  usymqr_solver = UsymqrSolver(n, m, Vector{Float64})
  trilqr_solver = TrilqrSolver(n, n, Vector{Float64})
  bilq_solver = BilqSolver(n, n, Vector{Float64})
  bilqr_solver = BilqrSolver(n, n, Vector{Float64})
  minres_qlp_solver = MinresQlpSolver(n, n, Vector{Float64})
  qmr_solver = QmrSolver(n, n, Vector{Float64})
  usymlq_solver = UsymlqSolver(m, n, Vector{Float64})
  tricg_solver = TricgSolver(m, n, Vector{Float64})
  trimr_solver = TrimrSolver(m, n, Vector{Float64})
  
  for i = 1 : 3
    A  = i * A
    Au = i * Au
    Ao = i * Ao
    b  = 5 * b
    c  = 3 * c

    x, stats = solve!(cg_solver, A, b)
    @test stats.solved

    x, stats = solve!(symmlq_solver, A, b)
    @test stats.solved

    x, stats = solve!(minres_solver, A, b)
    @test stats.solved

    x, stats = solve!(cg_lanczos_solver, A, b)
    @test stats.solved

    x, stats = solve!(cg_lanczos_shift_solver, A, b, shifts)
    @test stats.solved

    x, stats = solve!(diom_solver, A, b)
    @test stats.solved

    x, stats = solve!(dqgmres_solver, A, b)
    @test stats.solved

    x, stats = solve!(gmres_solver, A, b)
    @test stats.solved

    x, stats = solve!(cr_solver, A, b)
    @test stats.solved

    x, stats = solve!(crmr_solver, Au, c)
    @test stats.solved

    x, stats = solve!(cgs_solver, A, b)
    @test stats.solved

    x, stats = solve!(bicgstab_solver, A, b)
    @test stats.solved

    x, y, stats = solve!(craigmr_solver, Au, c)
    @test stats.solved

    x, stats = solve!(cgne_solver, Au, c)
    @test stats.solved

    x, y, stats = solve!(lnlq_solver, Au, c)
    @test stats.solved

    x, y, stats = solve!(craig_solver, Au, c)
    @test stats.solved

    x, stats = solve!(lslq_solver, Ao, b)
    @test stats.solved

    x, stats = solve!(cgls_solver, Ao, b)
    @test stats.solved

    x, stats = solve!(lsqr_solver, Ao, b)
    @test stats.solved

    x, stats = solve!(crls_solver, Ao, b)
    @test stats.solved

    x, stats = solve!(lsmr_solver, Ao, b)
    @test stats.solved

    x, stats = solve!(usymqr_solver, Ao, b, c)
    @test stats.solved

    x, t, stats = solve!(trilqr_solver, A, b, b)
    @test stats.solved_primal
    @test stats.solved_dual

    x, stats = solve!(bilq_solver, A, b)
    @test stats.solved

    x, t, stats = solve!(bilqr_solver, A, b, b)
    @test stats.solved_primal
    @test stats.solved_dual

    x, stats = solve!(minres_qlp_solver, A, b)
    @test stats.solved

    x, stats = solve!(qmr_solver, A, b)
    @test stats.solved

    x, stats = solve!(usymlq_solver, Au, c, b)
    @test stats.solved

    x, y, stats = solve!(tricg_solver, Au, c, b)
    @test stats.solved

    x, y, stats = solve!(trimr_solver, Au, c, b)
    @test stats.solved
  end
end
