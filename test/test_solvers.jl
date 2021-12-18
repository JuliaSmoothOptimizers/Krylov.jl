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
  fom_solver = FomSolver(n, n, mem, Vector{Float64})
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
  gpmr_solver = GpmrSolver(n, m, mem, Vector{Float64})
  
  for i = 1 : 3
    A  = i * A
    Au = i * Au
    Ao = i * Ao
    b  = 5 * b
    c  = 3 * c

    solver = solve!(cg_solver, A, b)
    @test statistics(solver) === solver.stats
    @test solution(solver, 1) === solver.x
    @test nsolution(solver) == 1
    @test issolved(solver)

    solver = solve!(symmlq_solver, A, b)
    @test statistics(solver) === solver.stats
    @test solution(solver, 1) === solver.x
    @test nsolution(solver) == 1
    @test issolved(solver)

    solver = solve!(minres_solver, A, b)
    @test statistics(solver) === solver.stats
    @test solution(solver, 1) === solver.x
    @test nsolution(solver) == 1
    @test issolved(solver)

    solver = solve!(cg_lanczos_solver, A, b)
    @test statistics(solver) === solver.stats
    @test solution(solver, 1) === solver.x
    @test nsolution(solver) == 1
    @test issolved(solver)

    solver = solve!(cg_lanczos_shift_solver, A, b, shifts)
    @test statistics(solver) === solver.stats
    @test solution(solver, 1) === solver.x
    @test nsolution(solver) == 1
    @test issolved(solver)

    solver = solve!(diom_solver, A, b)
    @test statistics(solver) === solver.stats
    @test solution(solver, 1) === solver.x
    @test nsolution(solver) == 1
    @test issolved(solver)

    solver = solve!(fom_solver, A, b)
    @test statistics(solver) === solver.stats
    @test solution(solver, 1) === solver.x
    @test nsolution(solver) == 1
    @test issolved(solver)

    solver = solve!(dqgmres_solver, A, b)
    @test statistics(solver) === solver.stats
    @test solution(solver, 1) === solver.x
    @test nsolution(solver) == 1
    @test issolved(solver)

    solver = solve!(gmres_solver, A, b)
    @test statistics(solver) === solver.stats
    @test solution(solver, 1) === solver.x
    @test nsolution(solver) == 1
    @test issolved(solver)

    solver = solve!(cr_solver, A, b)
    @test statistics(solver) === solver.stats
    @test solution(solver, 1) === solver.x
    @test nsolution(solver) == 1
    @test issolved(solver)

    solver = solve!(crmr_solver, Au, c)
    @test statistics(solver) === solver.stats
    @test solution(solver, 1) === solver.x
    @test nsolution(solver) == 1
    @test issolved(solver)

    solver = solve!(cgs_solver, A, b)
    @test statistics(solver) === solver.stats
    @test solution(solver, 1) === solver.x
    @test nsolution(solver) == 1
    @test issolved(solver)

    solver = solve!(bicgstab_solver, A, b)
    @test statistics(solver) === solver.stats
    @test solution(solver, 1) === solver.x
    @test nsolution(solver) == 1
    @test issolved(solver)

    solver = solve!(craigmr_solver, Au, c)
    @test statistics(solver) === solver.stats
    @test solution(solver, 1) === solver.x
    @test nsolution(solver) == 2
    @test issolved(solver)

    solver = solve!(cgne_solver, Au, c)
    @test statistics(solver) === solver.stats
    @test solution(solver, 1) === solver.x
    @test nsolution(solver) == 1
    @test issolved(solver)

    solver = solve!(lnlq_solver, Au, c)
    @test statistics(solver) === solver.stats
    @test solution(solver, 1) === solver.x
    @test solution(solver, 2) === solver.y
    @test nsolution(solver) == 2
    @test issolved(solver)

    solver = solve!(craig_solver, Au, c)
    @test statistics(solver) === solver.stats
    @test solution(solver, 1) === solver.x
    @test solution(solver, 2) === solver.y
    @test nsolution(solver) == 2
    @test issolved(solver)

    solver = solve!(lslq_solver, Ao, b)
    @test statistics(solver) === solver.stats
    @test solution(solver, 1) === solver.x
    @test nsolution(solver) == 1
    @test issolved(solver)

    solver = solve!(cgls_solver, Ao, b)
    @test statistics(solver) === solver.stats
    @test solution(solver, 1) === solver.x
    @test nsolution(solver) == 1
    @test issolved(solver)

    solver = solve!(lsqr_solver, Ao, b)
    @test statistics(solver) === solver.stats
    @test solution(solver, 1) === solver.x
    @test nsolution(solver) == 1
    @test issolved(solver)

    solver = solve!(crls_solver, Ao, b)
    @test statistics(solver) === solver.stats
    @test solution(solver, 1) === solver.x
    @test nsolution(solver) == 1
    @test issolved(solver)

    solver = solve!(lsmr_solver, Ao, b)
    @test statistics(solver) === solver.stats
    @test solution(solver, 1) === solver.x
    @test nsolution(solver) == 1
    @test issolved(solver)

    solver = solve!(usymqr_solver, Ao, b, c)
    @test statistics(solver) === solver.stats
    @test solution(solver, 1) === solver.x
    @test nsolution(solver) == 1
    @test issolved(solver)

    solver = solve!(trilqr_solver, A, b, b)
    @test statistics(solver) === solver.stats
    @test solution(solver, 1) === solver.x
    @test solution(solver, 2) === solver.y
    @test nsolution(solver) == 2
    @test issolved_primal(solver)
    @test issolved_dual(solver)
    @test issolved(solver)

    solver = solve!(bilq_solver, A, b)
    @test statistics(solver) === solver.stats
    @test solution(solver, 1) === solver.x
    @test nsolution(solver) == 1
    @test issolved(solver)

    solver = solve!(bilqr_solver, A, b, b)
    @test statistics(solver) === solver.stats
    @test solution(solver, 1) === solver.x
    @test solution(solver, 2) === solver.y
    @test nsolution(solver) == 2
    @test issolved_primal(solver)
    @test issolved_dual(solver)
    @test issolved(solver)

    solver = solve!(minres_qlp_solver, A, b)
    @test statistics(solver) === solver.stats
    @test solution(solver, 1) === solver.x
    @test nsolution(solver) == 1
    @test issolved(solver)

    solver = solve!(qmr_solver, A, b)
    @test statistics(solver) === solver.stats
    @test solution(solver, 1) === solver.x
    @test nsolution(solver) == 1
    @test issolved(solver)

    solver = solve!(usymlq_solver, Au, c, b)
    @test statistics(solver) === solver.stats
    @test solution(solver, 1) === solver.x
    @test nsolution(solver) == 1
    @test issolved(solver)

    solver = solve!(tricg_solver, Au, c, b)
    @test statistics(solver) === solver.stats
    @test solution(solver, 1) === solver.x
    @test solution(solver, 2) === solver.y
    @test nsolution(solver) == 2
    @test issolved(solver)

    solver = solve!(trimr_solver, Au, c, b)
    @test statistics(solver) === solver.stats
    @test solution(solver, 1) === solver.x
    @test solution(solver, 2) === solver.y
    @test nsolution(solver) == 2
    @test issolved(solver)

    solver = solve!(gpmr_solver, Ao, Au, b, c)
    @test statistics(solver) === solver.stats
    @test solution(solver, 1) === solver.x
    @test solution(solver, 2) === solver.y
    @test nsolution(solver) == 2
    @test issolved(solver)
  end

  if VERSION < v"1.8.0-DEV.1090"

    io = IOBuffer()
    show(io, cg_solver)
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
    Simple stats
    solved: true
    inconsistent: false
    residuals: []
    Aresiduals: []
    κ₂(A): []
    status: solution good enough given atol and rtol"""
    @test strip.(split(chomp(showed), "\n")) == strip.(split(chomp(expected), "\n"))

    io = IOBuffer()
    show(io, symmlq_solver)
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
    Symmlq stats
    solved: true
    residuals: []
    residuals (cg): []
    errors: []
    errors (cg): []
    ‖A‖F: $(symmlq_solver.stats.Anorm)
    κ₂(A): $(symmlq_solver.stats.Acond)
    status: solution xᶜ good enough given atol and rtol"""
    @test strip.(split(chomp(showed), "\n")) == strip.(split(chomp(expected), "\n"))

    io = IOBuffer()
    show(io, minres_solver)
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
    Simple stats
    solved: true
    inconsistent: false
    residuals: []
    Aresiduals: []
    κ₂(A): []
    status: found approximate zero-residual solution"""
    @test strip.(split(chomp(showed), "\n")) == strip.(split(chomp(expected), "\n"))

    io = IOBuffer()
    show(io, cg_lanczos_solver)
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
    Lanczos stats
    solved: true
    residuals: []
    indefinite: false
    ‖A‖F: $(cg_lanczos_solver.stats.Anorm)
    κ₂(A): NaN
    status: solution good enough given atol and rtol"""
    @test strip.(split(chomp(showed), "\n")) == strip.(split(chomp(expected), "\n"))

    io = IOBuffer()
    show(io, cg_lanczos_shift_solver)
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
    LanczosShift stats
    solved: true
    residuals: [Float64[], Float64[], Float64[], Float64[], Float64[]]
    indefinite: Bool[0, 0, 0, 0, 0]
    ‖A‖F: NaN
    κ₂(A): NaN
    status: solution good enough given atol and rtol"""
    @test strip.(split(chomp(showed), "\n")) == strip.(split(chomp(expected), "\n"))

    io = IOBuffer()
    show(io, diom_solver)
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
    Simple stats
    solved: true
    inconsistent: false
    residuals: []
    Aresiduals: []
    κ₂(A): []
    status: solution good enough given atol and rtol"""
    @test strip.(split(chomp(showed), "\n")) == strip.(split(chomp(expected), "\n"))

    io = IOBuffer()
    show(io, fom_solver)
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
    Simple stats
    solved: true
    inconsistent: false
    residuals: []
    Aresiduals: []
    κ₂(A): []
    status: solution good enough given atol and rtol"""
    @test strip.(split(chomp(showed), "\n")) == strip.(split(chomp(expected), "\n"))

    io = IOBuffer()
    show(io, dqgmres_solver)
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
    Simple stats
    solved: true
    inconsistent: false
    residuals: []
    Aresiduals: []
    κ₂(A): []
    status: solution good enough given atol and rtol"""
    @test strip.(split(chomp(showed), "\n")) == strip.(split(chomp(expected), "\n"))

    io = IOBuffer()
    show(io, gmres_solver)
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
    Simple stats
    solved: true
    inconsistent: false
    residuals: []
    Aresiduals: []
    κ₂(A): []
    status: solution good enough given atol and rtol"""
    @test strip.(split(chomp(showed), "\n")) == strip.(split(chomp(expected), "\n"))

    io = IOBuffer()
    show(io, cr_solver)
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
    Simple stats
    solved: true
    inconsistent: false
    residuals: []
    Aresiduals: []
    κ₂(A): []
    status: solution good enough given atol and rtol"""
    @test strip.(split(chomp(showed), "\n")) == strip.(split(chomp(expected), "\n"))

    io = IOBuffer()
    show(io, crmr_solver)
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
    Simple stats
    solved: true
    inconsistent: false
    residuals: []
    Aresiduals: []
    κ₂(A): []
    status: solution good enough given atol and rtol"""
    @test strip.(split(chomp(showed), "\n")) == strip.(split(chomp(expected), "\n"))

    io = IOBuffer()
    show(io, cgs_solver)
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
    Simple stats
    solved: true
    inconsistent: false
    residuals: []
    Aresiduals: []
    κ₂(A): []
    status: solution good enough given atol and rtol"""
    @test strip.(split(chomp(showed), "\n")) == strip.(split(chomp(expected), "\n"))

    io = IOBuffer()
    show(io, bicgstab_solver)
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
    Simple stats
    solved: true
    inconsistent: false
    residuals: []
    Aresiduals: []
    κ₂(A): []
    status: solution good enough given atol and rtol"""
    @test strip.(split(chomp(showed), "\n")) == strip.(split(chomp(expected), "\n"))

    io = IOBuffer()
    show(io, craigmr_solver)
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
    Simple stats
    solved: true
    inconsistent: false
    residuals: []
    Aresiduals: []
    κ₂(A): []
    status: found approximate minimum-norm solution"""
    @test strip.(split(chomp(showed), "\n")) == strip.(split(chomp(expected), "\n"))

    io = IOBuffer()
    show(io, cgne_solver)
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
    Simple stats
    solved: true
    inconsistent: false
    residuals: []
    Aresiduals: []
    κ₂(A): []
    status: solution good enough given atol and rtol"""
    @test strip.(split(chomp(showed), "\n")) == strip.(split(chomp(expected), "\n"))

    io = IOBuffer()
    show(io, lnlq_solver)
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
    LNLQ stats
    solved: true
    residuals: []
    error with bnd: false
    error bnd x: []
    error bnd y: []
    status: solutions (xᶜ, yᶜ) good enough for the tolerances given"""
    @test strip.(split(chomp(showed), "\n")) == strip.(split(chomp(expected), "\n"))

    io = IOBuffer()
    show(io, craig_solver)
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
    Simple stats
    solved: true
    inconsistent: false
    residuals: []
    Aresiduals: []
    κ₂(A): []
    status: solution good enough for the tolerances given"""
    @test strip.(split(chomp(showed), "\n")) == strip.(split(chomp(expected), "\n"))

    io = IOBuffer()
    show(io, lslq_solver)
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
    LSLQ stats
    solved: true
    inconsistent: true
    residuals: []
    Aresiduals: []
    err lbnds: []
    error with bnd: false
    error bound LQ: []
    error bound CG: []
    status: found approximate minimum least-squares solution"""
    @test strip.(split(chomp(showed), "\n")) == strip.(split(chomp(expected), "\n"))

    io = IOBuffer()
    show(io, cgls_solver)
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
    Simple stats
    solved: true
    inconsistent: false
    residuals: []
    Aresiduals: []
    κ₂(A): []
    status: solution good enough given atol and rtol"""
    @test strip.(split(chomp(showed), "\n")) == strip.(split(chomp(expected), "\n"))

    io = IOBuffer()
    show(io, lsqr_solver)
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
    Simple stats
    solved: true
    inconsistent: false
    residuals: []
    Aresiduals: []
    κ₂(A): []
    status: found approximate zero-residual solution"""
    @test strip.(split(chomp(showed), "\n")) == strip.(split(chomp(expected), "\n"))

    io = IOBuffer()
    show(io, crls_solver)
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
    Simple stats
    solved: true
    inconsistent: false
    residuals: []
    Aresiduals: []
    κ₂(A): []
    status: solution good enough given atol and rtol"""
    @test strip.(split(chomp(showed), "\n")) == strip.(split(chomp(expected), "\n"))

    io = IOBuffer()
    show(io, lsmr_solver)
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
    Simple stats
    solved: true
    inconsistent: false
    residuals: []
    Aresiduals: []
    κ₂(A): []
    status: found approximate zero-residual solution"""
    @test strip.(split(chomp(showed), "\n")) == strip.(split(chomp(expected), "\n"))

    io = IOBuffer()
    show(io, usymqr_solver)
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
    Simple stats
    solved: true
    inconsistent: false
    residuals: []
    Aresiduals: []
    κ₂(A): []
    status: solution good enough given atol and rtol"""
    @test strip.(split(chomp(showed), "\n")) == strip.(split(chomp(expected), "\n"))

    io = IOBuffer()
    show(io, trilqr_solver)
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
    Adjoint stats
    solved primal: true
    solved dual: true
    residuals primal: []
    residuals dual: []
    status: Both primal and dual solutions (xᶜ, t) are good enough given atol and rtol"""
    @test strip.(split(chomp(showed), "\n")) == strip.(split(chomp(expected), "\n"))

    io = IOBuffer()
    show(io, bilq_solver)
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
    Simple stats
    solved: true
    inconsistent: false
    residuals: []
    Aresiduals: []
    κ₂(A): []
    status: solution xᶜ good enough given atol and rtol"""
    @test strip.(split(chomp(showed), "\n")) == strip.(split(chomp(expected), "\n"))

    io = IOBuffer()
    show(io, bilqr_solver)
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
    Adjoint stats
    solved primal: true
    solved dual: true
    residuals primal: []
    residuals dual: []
    status: Both primal and dual solutions (xᶜ, t) are good enough given atol and rtol"""
    @test strip.(split(chomp(showed), "\n")) == strip.(split(chomp(expected), "\n"))

    io = IOBuffer()
    show(io, minres_qlp_solver)
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
    Simple stats
    solved: true
    inconsistent: false
    residuals: []
    Aresiduals: []
    κ₂(A): []
    status: solution good enough given atol and rtol"""
    @test strip.(split(chomp(showed), "\n")) == strip.(split(chomp(expected), "\n"))

    io = IOBuffer()
    show(io, qmr_solver)
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
    Simple stats
    solved: true
    inconsistent: false
    residuals: []
    Aresiduals: []
    κ₂(A): []
    status: solution good enough given atol and rtol"""
    @test strip.(split(chomp(showed), "\n")) == strip.(split(chomp(expected), "\n"))

    io = IOBuffer()
    show(io, usymlq_solver)
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
    Simple stats
    solved: true
    inconsistent: false
    residuals: []
    Aresiduals: []
    κ₂(A): []
    status: solution xᶜ good enough given atol and rtol"""
    @test strip.(split(chomp(showed), "\n")) == strip.(split(chomp(expected), "\n"))

    io = IOBuffer()
    show(io, tricg_solver)
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
    Simple stats
    solved: true
    inconsistent: false
    residuals: []
    Aresiduals: []
    κ₂(A): []
    status: solution good enough given atol and rtol"""
    @test strip.(split(chomp(showed), "\n")) == strip.(split(chomp(expected), "\n"))

    io = IOBuffer()
    show(io, trimr_solver)
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
    Simple stats
    solved: true
    inconsistent: false
    residuals: []
    Aresiduals: []
    κ₂(A): []
    status: solution good enough given atol and rtol"""
    @test strip.(split(chomp(showed), "\n")) == strip.(split(chomp(expected), "\n"))

    io = IOBuffer()
    show(io, gpmr_solver)
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
    Simple stats
    solved: true
    inconsistent: false
    residuals: []
    Aresiduals: []
    κ₂(A): []
    status: solution good enough given atol and rtol"""
    @test strip.(split(chomp(showed), "\n")) == strip.(split(chomp(expected), "\n"))
  end
end
