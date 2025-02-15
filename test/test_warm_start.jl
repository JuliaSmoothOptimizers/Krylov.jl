function test_warm_start(FC)
  T = real(FC)
  A, b = warm_start(FC=FC)
  c = copy(b)
  n, m = size(A)
  x0 = 1.2 * ones(FC, n)
  y0 = 0.8 * ones(FC, n)
  shifts = [1.0; 2.0; 3.0; 4.0; 5.0]
  nshifts = 5
  tol = 1.0e-6

  J = 5 * Matrix{FC}(I, n, n)
  z0 = -2 * ones(FC, n)
  d = -10 * ones(FC, n)

  # BILQR
  @testset "bilqr" begin
    x, y, stats = bilqr(A, b, c, x0, y0)
    r = b - A * x
    resid = norm(r) / norm(b)
    @test(resid ≤ tol)
    s = c - A' * y
    resid = norm(s) / norm(c)
    @test(resid ≤ tol)

    solver = BilqrSolver(A, b)
    solve!(solver, A, b, c, x0, y0)
    r = b - A * solver.x
    resid = norm(r) / norm(b)
    @test(resid ≤ tol)
    s = c - A' * solver.y
    resid = norm(s) / norm(c)
    @test(resid ≤ tol)

    solve!(solver, J, d, d, z0, z0)
    r = d - J * solver.x
    resid = norm(r) / norm(d)
    @test(resid ≤ tol)
    s = d - J' * solver.y
    resid = norm(s) / norm(d)
    @test(resid ≤ tol)
  end

  # TRILQR
  @testset "trilqr" begin
    x, y, stats = trilqr(A, b, c, x0, y0)
    r = b - A * x
    resid = norm(r) / norm(b)
    @test(resid ≤ tol)
    s = c - A' * y
    resid = norm(s) / norm(c)
    @test(resid ≤ tol)

    solver = TrilqrSolver(A, b)
    solve!(solver, A, b, c, x0, y0)
    r = b - A * solver.x
    resid = norm(r) / norm(b)
    @test(resid ≤ tol)
    s = c - A' * solver.y
    resid = norm(s) / norm(c)
    @test(resid ≤ tol)

    solve!(solver, J, d, d, z0, z0)
    r = d - J * solver.x
    resid = norm(r) / norm(d)
    @test(resid ≤ tol)
    s = d - J' * solver.y
    resid = norm(s) / norm(d)
    @test(resid ≤ tol)
  end

  # TRICG
  @testset "tricg" begin
    x, y, stats = tricg(A, b, b, x0, y0)
    r = [b - x - A * y; b - A' * x + y]
    resid = norm(r) / norm([b; b])
    @test(resid ≤ tol)

    solver = TricgSolver(A, b)
    solve!(solver, A, b, b, x0, y0)
    r = [b - solver.x - A * solver.y; b - A' * solver.x + solver.y]
    resid = norm(r) / norm([b; b])
    @test(resid ≤ tol)

    τ = zero(T)
    ν = zero(T)
    solve!(solver, J, d, d, z0, z0; τ, ν)
    r = [d - τ * solver.x - J * solver.y; d - J' * solver.x - ν * solver.y]
    resid = norm(r) / norm([d; d])
    @test(resid ≤ tol)
  end

  # TRIMR
  @testset "trimr" begin
    x, y, stats = trimr(A, b, b, x0, y0)
    r = [b - x - A * y; b - A' * x + y]
    resid = norm(r) / norm([b; b])
    @test(resid ≤ tol)

    solver = TrimrSolver(A, b)
    solve!(solver, A, b, b, x0, y0)
    r = [b - solver.x - A * solver.y; b - A' * solver.x + solver.y]
    resid = norm(r) / norm([b; b])
    @test(resid ≤ tol)

    τ = zero(T)
    ν = zero(T)
    solve!(solver, J, d, d, z0, z0; τ, ν)
    r = [d - τ * solver.x - J * solver.y; d - J' * solver.x - ν * solver.y]
    resid = norm(r) / norm([d; d])
    @test(resid ≤ tol)
  end

  # GPMR
  @testset "gpmr" begin
    x, y, stats = gpmr(A, A', b, b, x0, y0)
    r = [b - x - A * y; b - A' * x - y]
    resid = norm(r) / norm([b; b])
    @test(resid ≤ tol)

    solver = GpmrSolver(A, b)
    solve!(solver, A, A', b, b, x0, y0)
    r = [b - solver.x - A * solver.y; b - A' * solver.x - solver.y]
    resid = norm(r) / norm([b; b])
    @test(resid ≤ tol)

    λ = zero(FC)
    μ = zero(FC)
    solve!(solver, J, J', d, d, z0, z0; λ, μ)
    r = [d - λ * solver.x - J * solver.y; d - J' * solver.x - μ * solver.y]
    resid = norm(r) / norm([d; d])
    @test(resid ≤ tol)
  end

  # MINRES-QLP
  @testset "minres_qlp" begin
    x, stats = minres_qlp(A, b, x0)
    r = b - A * x
    resid = norm(r) / norm(b)
    @test(resid ≤ tol)

    solver = MinresQlpSolver(A, b)
    solve!(solver, A, b, x0)
    r = b - A * solver.x
    resid = norm(r) / norm(b)
    @test(resid ≤ tol)

    solve!(solver, J, d, z0)
    r = d - J * solver.x
    resid = norm(r) / norm(d)
    @test(resid ≤ tol)
  end

  # SYMMLQ
  @testset "symmlq" begin
    x, stats = symmlq(A, b, x0)
    r = b - A * x
    resid = norm(r) / norm(b)
    @test(resid ≤ tol)

    solver = SymmlqSolver(A, b)
    solve!(solver, A, b, x0)
    r = b - A * solver.x
    resid = norm(r) / norm(b)
    @test(resid ≤ tol)

    solve!(solver, J, d, z0)
    r = d - J * solver.x
    resid = norm(r) / norm(d)
    @test(resid ≤ tol)
  end

  # CG
  @testset "cg" begin
    x, stats = cg(A, b, x0)
    r = b - A * x
    resid = norm(r) / norm(b)
    @test(resid ≤ tol)

    solver = CgSolver(A, b)
    solve!(solver, A, b, x0)
    r = b - A * solver.x
    resid = norm(r) / norm(b)
    @test(resid ≤ tol)

    solve!(solver, J, d, z0)
    r = d - J * solver.x
    resid = norm(r) / norm(d)
    @test(resid ≤ tol)
  end

  # CR
  @testset "cr" begin
    x, stats = cr(A, b, x0)
    r = b - A * x
    resid = norm(r) / norm(b)
    @test(resid ≤ tol)

    solver = CrSolver(A, b)
    solve!(solver, A, b, x0)
    r = b - A * solver.x
    resid = norm(r) / norm(b)
    @test(resid ≤ tol)

    solve!(solver, J, d, z0)
    r = d - J * solver.x
    resid = norm(r) / norm(d)
    @test(resid ≤ tol)
  end

  # CAR
  @testset "car" begin
    x, stats = car(A, b, x0)
    r = b - A * x
    resid = norm(r) / norm(b)
    @test(resid ≤ tol)

    solver = CarSolver(A, b)
    solve!(solver, A, b, x0)
    r = b - A * solver.x
    resid = norm(r) / norm(b)
    @test(resid ≤ tol)

    solve!(solver, J, d, z0)
    r = d - J * solver.x
    resid = norm(r) / norm(d)
    @test(resid ≤ tol)
  end

  # CG-LANCZOS
  @testset "cg_lanczos" begin
    x, stats = cg_lanczos(A, b, x0)
    r = b - A * x
    resid = norm(r) / norm(b)
    @test(resid ≤ tol)

    solver = CgLanczosSolver(A, b)
    solve!(solver, A, b, x0)
    r = b - A * solver.x
    resid = norm(r) / norm(b)
    @test(resid ≤ tol)

    solve!(solver, J, d, z0)
    r = d - J * solver.x
    resid = norm(r) / norm(d)
    @test(resid ≤ tol)
  end

  # MINRES
  @testset "minres" begin
    x, stats = minres(A, b, x0)
    r = b - A * x
    resid = norm(r) / norm(b)
    @test(resid ≤ tol)

    solver = MinresSolver(A, b)
    solve!(solver, A, b, x0)
    r = b - A * solver.x
    resid = norm(r) / norm(b)
    @test(resid ≤ tol)

    solve!(solver, J, d, z0)
    r = d - J * solver.x
    resid = norm(r) / norm(d)
    @test(resid ≤ tol)
  end

  # MINARES
  @testset "minares" begin
    x, stats = minares(A, b, x0)
    r = b - A * x
    resid = norm(r) / norm(b)
    @test(resid ≤ tol)

    solver = MinaresSolver(A, b)
    solve!(solver, A, b, x0)
    r = b - A * solver.x
    resid = norm(r) / norm(b)
    @test(resid ≤ tol)

    solve!(solver, J, d, z0)
    r = d - J * solver.x
    resid = norm(r) / norm(d)
    @test(resid ≤ tol)
  end

  # DIOM
  @testset "diom" begin
    x, stats = diom(A, b, x0)
    r = b - A * x
    resid = norm(r) / norm(b)
    @test(resid ≤ tol)

    solver = DiomSolver(A, b)
    solve!(solver, A, b, x0)
    r = b - A * solver.x
    resid = norm(r) / norm(b)
    @test(resid ≤ tol)

    solve!(solver, J, d, z0)
    r = d - J * solver.x
    resid = norm(r) / norm(d)
    @test(resid ≤ tol)
  end

  # DQGMRES
  @testset "dqgmres" begin
    x, stats = dqgmres(A, b, x0)
    r = b - A * x
    resid = norm(r) / norm(b)
    @test(resid ≤ tol)

    solver = DqgmresSolver(A, b)
    solve!(solver, A, b, x0)
    r = b - A * solver.x
    resid = norm(r) / norm(b)
    @test(resid ≤ tol)

    solve!(solver, J, d, z0)
    r = d - J * solver.x
    resid = norm(r) / norm(d)
    @test(resid ≤ tol)
  end

  # FOM
  @testset "fom" begin
    x, stats = fom(A, b, x0)
    r = b - A * x
    resid = norm(r) / norm(b)
    @test(resid ≤ tol)

    solver = FomSolver(A, b)
    solve!(solver, A, b, x0)
    r = b - A * solver.x
    resid = norm(r) / norm(b)
    @test(resid ≤ tol)

    solve!(solver, J, d, z0)
    r = d - J * solver.x
    resid = norm(r) / norm(d)
    @test(resid ≤ tol)
  end

  # GMRES
  @testset "gmres" begin
    x, stats = gmres(A, b, x0)
    r = b - A * x
    resid = norm(r) / norm(b)
    @test(resid ≤ tol)

    solver = GmresSolver(A, b)
    solve!(solver, A, b, x0)
    r = b - A * solver.x
    resid = norm(r) / norm(b)
    @test(resid ≤ tol)

    solve!(solver, J, d, z0)
    r = d - J * solver.x
    resid = norm(r) / norm(d)
    @test(resid ≤ tol)
  end

  # FGMRES
  @testset "fgmres" begin
    x, stats = fgmres(A, b, x0)
    r = b - A * x
    resid = norm(r) / norm(b)
    @test(resid ≤ tol)

    solver = FgmresSolver(A, b)
    solve!(solver, A, b, x0)
    r = b - A * solver.x
    resid = norm(r) / norm(b)
    @test(resid ≤ tol)

    solve!(solver, J, d, z0)
    r = d - J * solver.x
    resid = norm(r) / norm(d)
    @test(resid ≤ tol)
  end

  # BICGSTAB
  @testset "bicgstab" begin
    x, stats = bicgstab(A, b, x0)
    r = b - A * x
    resid = norm(r) / norm(b)
    @test(resid ≤ tol)

    solver = BicgstabSolver(A, b)
    solve!(solver, A, b, x0)
    r = b - A * solver.x
    resid = norm(r) / norm(b)
    @test(resid ≤ tol)

    solve!(solver, J, d, z0)
    r = d - J * solver.x
    resid = norm(r) / norm(d)
    @test(resid ≤ tol)
  end

  # CGS
  @testset "cgs" begin
    x, stats = cgs(A, b, x0)
    r = b - A * x
    resid = norm(r) / norm(b)
    @test(resid ≤ tol)

    solver = CgsSolver(A, b)
    solve!(solver, A, b, x0)
    r = b - A * solver.x
    resid = norm(r) / norm(b)
    @test(resid ≤ tol)

    solve!(solver, J, d, z0)
    r = d - J * solver.x
    resid = norm(r) / norm(d)
    @test(resid ≤ tol)
  end

  # BILQ
  @testset "bilq" begin
    x, stats = bilq(A, b, x0)
    r = b - A * x
    resid = norm(r) / norm(b)
    @test(resid ≤ tol)

    solver = BilqSolver(A, b)
    solve!(solver, A, b, x0)
    r = b - A * solver.x
    resid = norm(r) / norm(b)
    @test(resid ≤ tol)

    solve!(solver, J, d, z0)
    r = d - J * solver.x
    resid = norm(r) / norm(d)
    @test(resid ≤ tol)
  end

  # QMR
  @testset "qmr" begin
    x, stats = qmr(A, b, x0)
    r = b - A * x
    resid = norm(r) / norm(b)
    @test(resid ≤ tol)

    solver = QmrSolver(A, b)
    solve!(solver, A, b, x0)
    r = b - A * solver.x
    resid = norm(r) / norm(b)
    @test(resid ≤ tol)

    solve!(solver, J, d, z0)
    r = d - J * solver.x
    resid = norm(r) / norm(d)
    @test(resid ≤ tol)
  end

  # USYMLQ
  @testset "usymlq" begin
    x, stats = usymlq(A, b, c, x0)
    r = b - A * x
    resid = norm(r) / norm(b)
    @test(resid ≤ tol)

    solver = UsymlqSolver(A, b)
    solve!(solver, A, b, c, x0)
    r = b - A * solver.x
    resid = norm(r) / norm(b)
    @test(resid ≤ tol)

    solve!(solver, J, d, d, z0)
    r = d - J * solver.x
    resid = norm(r) / norm(d)
    @test(resid ≤ tol)
  end

  # USYMQR
  @testset "usymqr" begin
    x, stats = usymqr(A, b, c, x0)
    r = b - A * x
    resid = norm(r) / norm(b)
    @test(resid ≤ tol)

    solver = UsymqrSolver(A, b)
    solve!(solver, A, b, c, x0)
    r = b - A * solver.x
    resid = norm(r) / norm(b)
    @test(resid ≤ tol)

    solve!(solver, J, d, d, z0)
    r = d - J * solver.x
    resid = norm(r) / norm(d)
    @test(resid ≤ tol)
  end
end

@testset "warm-start" begin
  for FC in (Float64, ComplexF64)
    @testset "Data Type: $FC" begin
      test_warm_start(FC)
    end
  end
end
