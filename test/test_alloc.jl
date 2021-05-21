function test_alloc()
  L   = get_div_grad(32, 32, 32)
  n   = size(L, 1)
  m   = div(n, 2)
  Lu  = L[1:m,:]
  Lo  = L[:,1:m]
  A   = PreallocatedLinearOperator(L)   # Dimension n x n
  Au  = PreallocatedLinearOperator(Lu)  # Dimension m x n
  Ao  = PreallocatedLinearOperator(Lo)  # Dimension n x m
  b   = Ao * ones(m) # Dimension n
  c   = Au * ones(n) # Dimension m
  mem = 10

  shifts  = [1.0; 2.0; 3.0; 4.0; 5.0]
  nshifts = 5

  # UniformScaling preconditioners I should work as OpEye()
  M1 = opEye()
  M2 = I
  cg(L, b, M=M1) # warmup
  cg(L, b, M=M2) # warmup
  opEye_bytes = @allocated cg(L, b, M=M1)
  UniformScaling_bytes = @allocated cg(L, b, M=M2)
  @test 0.99 * UniformScaling_bytes ≤ opEye_bytes ≤ 1.01 * UniformScaling_bytes

  # without preconditioner and with Ap preallocated, SYMMLQ needs 4 n-vectors: x_lq, vold, v, w̅ (= x_cg)
  storage_symmlq(n) = 4 * n
  storage_symmlq_bytes(n) = 8 * storage_symmlq(n)

  expected_symmlq_bytes = storage_symmlq_bytes(n)
  symmlq(A, b)  # warmup
  actual_symmlq_bytes = @allocated symmlq(A, b)
  @test actual_symmlq_bytes ≤ 1.1 * expected_symmlq_bytes

  solver = SymmlqSolver(A, b)
  symmlq!(solver, A, b)  # warmup
  inplace_symmlq_bytes = @allocated symmlq!(solver, A, b)
  @test (VERSION < v"1.5") || (inplace_symmlq_bytes == 672)

  # CG needs:
  # 4 n-vectors: x, r, p, Ap
  storage_cg(n) = 4 * n
  storage_cg_bytes(n) = 8 * storage_cg(n)

  expected_cg_bytes = storage_cg_bytes(n)
  cg(L, b)  # warmup
  actual_cg_bytes = @allocated cg(L, b)
  @test actual_cg_bytes ≤ 1.1 * expected_cg_bytes

  solver = CgSolver(L, b)
  cg!(solver, L, b)  # warmup
  inplace_cg_bytes = @allocated cg!(solver, L, b)
  @test (VERSION < v"1.5") || (inplace_cg_bytes == 208)

  # MINRES needs:
  # 6 n-vectors: x, r1, r2, w1, w2, y
  storage_minres(n) = 6 * n
  storage_minres_bytes(n) = 8 * storage_minres(n)

  expected_minres_bytes = storage_minres_bytes(n)
  minres(L, b)  # warmup
  actual_minres_bytes = @allocated minres(L, b)
  @test actual_minres_bytes ≤ 1.1 * expected_minres_bytes

  solver = MinresSolver(L, b)
  minres!(solver, L, b)  # warmup
  inplace_minres_bytes = @allocated minres!(solver, L, b)
  @test (VERSION < v"1.5") || (inplace_minres_bytes == 0)

  # without preconditioner and with Ap preallocated, DIOM needs:
  # - 2 n-vectors: x, x_old
  # - 2 (n*mem)-matrices: P, V
  # - 1 mem-vector: L
  # - 1 (mem+2)-vector: H
  # - 1 mem-bitArray: p
  storage_diom(mem, n) = (2 * n) + (2 * n * mem) + (mem) + (mem + 2) + (mem / 64)
  storage_diom_bytes(mem, n) = 8 * storage_diom(mem, n)

  expected_diom_bytes = storage_diom_bytes(mem, n)
  diom(A, b, memory=mem)  # warmup
  actual_diom_bytes = @allocated diom(A, b, memory=mem)
  @test actual_diom_bytes ≤ 1.05 * expected_diom_bytes

  solver = DiomSolver(A, b)
  diom!(solver, A, b)  # warmup
  inplace_diom_bytes = @allocated diom!(solver, A, b)
  @test (VERSION < v"1.5") || (inplace_diom_bytes ≤ 240)

  # with Ap preallocated, CG_LANCZOS needs 4 n-vectors: x, v, v_prev, p
  storage_cg_lanczos(n) = 4 * n
  storage_cg_lanczos_bytes(n) = 8 * storage_cg_lanczos(n)

  expected_cg_lanczos_bytes = storage_cg_lanczos_bytes(n)
  cg_lanczos(A, b)  # warmup
  actual_cg_lanczos_bytes = @allocated cg_lanczos(A, b)
  @test actual_cg_lanczos_bytes ≤ 1.1 * expected_cg_lanczos_bytes

  solver = CgLanczosSolver(A, b)
  cg_lanczos!(solver, A, b)  # warmup
  inplace_cg_lanczos_bytes = @allocated cg_lanczos!(solver, A, b)
  @test (VERSION < v"1.5") || (inplace_cg_lanczos_bytes == 144)

  # with Ap preallocated, CG_LANCZOS_SHIFT_SEQ needs:
  # - 2 n-vectors: v, v_prev
  # - 2 (n*nshifts)-matrices: x, p
  # - 5 nshifts-vectors: σ, δhat, ω, γ, rNorms
  # - 3 nshifts-bitArray: indefinite, converged, not_cv
  storage_cg_lanczos_shift_seq(n, nshifts) = (2 * n) + (2 * n * nshifts) + (5 * nshifts) + (3 * nshifts / 64)
  storage_cg_lanczos_shift_seq_bytes(n, nshifts) = 8 * storage_cg_lanczos_shift_seq(n, nshifts)

  expected_cg_lanczos_shift_seq_bytes = storage_cg_lanczos_shift_seq_bytes(n, nshifts)
  cg_lanczos_shift_seq(A, b, shifts)  # warmup
  actual_cg_lanczos_shift_seq_bytes = @allocated cg_lanczos_shift_seq(A, b, shifts)
  @test actual_cg_lanczos_shift_seq_bytes ≤ 1.1 * expected_cg_lanczos_shift_seq_bytes

  solver = CgLanczosShiftSolver(A, b, shifts)
  cg_lanczos_shift_seq!(solver, A, b, shifts)  # warmup
  inplace_cg_lanczos_shift_seq_bytes = @allocated cg_lanczos_shift_seq!(solver, A, b, shifts)
  @test (VERSION < v"1.5") || (inplace_cg_lanczos_shift_seq_bytes ≤ 356)

  # without preconditioner and with Ap preallocated, DQGMRES needs:
  # - 1 n-vector: x
  # - 2 (n*mem)-matrices: P, V
  # - 2 mem-vectors: c, s
  # - 1 (mem+2)-vector: H
  storage_dqgmres(mem, n) = (n) + (2 * n * mem) + (2 * mem) + (mem + 2)
  storage_dqgmres_bytes(mem, n) = 8 * storage_dqgmres(mem, n)

  expected_dqgmres_bytes = storage_dqgmres_bytes(mem, n)
  dqgmres(A, b, memory=mem)  # warmup
  actual_dqgmres_bytes = @allocated dqgmres(A, b, memory=mem)
  @test actual_dqgmres_bytes ≤ 1.05 * expected_dqgmres_bytes

  solver = DqgmresSolver(A, b)
  dqgmres!(solver, A, b)  # warmup
  inplace_dqgmres_bytes = @allocated dqgmres!(solver, A, b)
  @test (VERSION < v"1.5") || (inplace_dqgmres_bytes == 208)

  # without preconditioner and with Ap preallocated, CR needs 4 n-vectors: x, r, p, q
  storage_cr(n) = 4 * n
  storage_cr_bytes(n) = 8 * storage_cr(n)

  expected_cr_bytes = storage_cr_bytes(n)
  cr(A, b, rtol=1e-6)  # warmup
  actual_cr_bytes = @allocated cr(A, b, rtol=1e-6)
  @test actual_cr_bytes ≤ 1.1 * expected_cr_bytes

  solver = CrSolver(A, b)
  cr!(solver, A, b, rtol=1e-6)  # warmup
  inplace_cr_bytes = @allocated cr!(solver, A, b, rtol=1e-6)
  @test (VERSION < v"1.5") || (inplace_cr_bytes == 208)

  # without preconditioner and with (Ap, Aᵀq) preallocated, CRMR needs:
  # - 2 n-vectors: x, p
  # - 1 m-vector: r
  storage_crmr(n, m) = 2 * n + m
  storage_crmr_bytes(n, m) = 8 * storage_crmr(n, m)

  expected_crmr_bytes = storage_crmr_bytes(n, m)
  (x, stats) = crmr(Au, c)  # warmup
  actual_crmr_bytes = @allocated crmr(Au, c)
  @test actual_crmr_bytes ≤ 1.1 * expected_crmr_bytes

  solver = CrmrSolver(Au, c)
  crmr!(solver, Au, c)  # warmup
  inplace_crmr_bytes = @allocated crmr!(solver, Au, c)
  @test (VERSION < v"1.5") || (inplace_crmr_bytes == 208)

  # without preconditioner and with Ap preallocated, CGS needs 5 n-vectors: x, r, u, p, q
  storage_cgs(n) = 5 * n
  storage_cgs_bytes(n) = 8 * storage_cgs(n)

  expected_cgs_bytes = storage_cgs_bytes(n)
  cgs(A, b)  # warmup
  actual_cgs_bytes = @allocated cgs(A, b)
  @test actual_cgs_bytes ≤ 1.1 * expected_cgs_bytes

  solver = CgsSolver(A, b)
  cgs!(solver, A, b)  # warmup
  inplace_cgs_bytes = @allocated cgs!(solver, A, b)
  @test (VERSION < v"1.5") || (inplace_cgs_bytes == 208)

  # without preconditioner and with Ap preallocated, BICGSTAB needs 5 n-vectors: x, r, p, v, s
  storage_bicgstab(n) = 5 * n
  storage_bicgstab_bytes(n) = 8 * storage_bicgstab(n)

  expected_bicgstab_bytes = storage_bicgstab_bytes(n)
  bicgstab(A, b)  # warmup
  actual_bicgstab_bytes = @allocated bicgstab(A, b)
  @test actual_bicgstab_bytes ≤ 1.1 * expected_bicgstab_bytes

  solver = BicgstabSolver(A, b)
  bicgstab!(solver, A, b)  # warmup
  inplace_bicgstab_bytes = @allocated bicgstab!(solver, A, b)
  @test (VERSION < v"1.5") || (inplace_bicgstab_bytes == 208)

  # with (Ap, Aᵀq) preallocated, CRAIGMR needs:
  # - 2 n-vector: x, v
  # - 4 m-vectors: y, u, w, wbar
  storage_craigmr(n, m) = 2 * n + 4 * m
  storage_craigmr_bytes(n, m) = 8 * storage_craigmr(n, m)

  expected_craigmr_bytes = storage_craigmr_bytes(n, m)
  craigmr(Au, c)  # warmup
  actual_craigmr_bytes = @allocated craigmr(Au, c)
  @test actual_craigmr_bytes ≤ 1.1 * expected_craigmr_bytes

  solver = CraigmrSolver(Au, c)
  craigmr!(solver, Au, c)  # warmup
  inplace_craigmr_bytes = @allocated craigmr!(solver, Au, c)
  @test (VERSION < v"1.5") || (inplace_craigmr_bytes == 208)

  # without preconditioner and with (Ap, Aᵀq) preallocated, CGNE needs:
  # - 2 n-vectors: x, p
  # - 1 m-vector: r
  storage_cgne(n, m) = 2 * n + m
  storage_cgne_bytes(n, m) = 8 * storage_cgne(n, m)

  expected_cgne_bytes = storage_cgne_bytes(n, m)
  (x, stats) = cgne(Au, c)  # warmup
  actual_cgne_bytes = @allocated cgne(Au, c)
  @test actual_cgne_bytes ≤ 1.1 * expected_cgne_bytes

  solver = CgneSolver(Au, c)
  cgne!(solver, Au, c)  # warmup
  inplace_cgne_bytes = @allocated cgne!(solver, Au, c)
  @test (VERSION < v"1.5") || (inplace_cgne_bytes == 208)

  # with (Ap, Aᵀq) preallocated, LNLQ needs:
  # - 2 n-vector: x, v
  # - 3 m-vectors: y, w̄, u
  storage_lnlq(n, m) = 2 * n + 3 * m
  storage_lnlq_bytes(n, m) = 8 * storage_lnlq(n, m)

  expected_lnlq_bytes = storage_lnlq_bytes(n, m)
  lnlq(Au, c)  # warmup
  actual_lnlq_bytes = @allocated lnlq(Au, c)
  @test actual_lnlq_bytes ≤ 1.1 * expected_lnlq_bytes

  solver = LnlqSolver(Au, c)
  lnlq!(solver, Au, c)  # warmup
  inplace_lnlq_bytes = @allocated lnlq!(solver, Au, c)
  @test (VERSION < v"1.5") || (inplace_lnlq_bytes == 208)

  # with (Ap, Aᵀq) preallocated, CRAIG needs:
  # - 2 n-vector: x, v
  # - 3 m-vectors: y, w, u
  storage_craig(n, m) = 2 * n + 3 * m
  storage_craig_bytes(n, m) = 8 * storage_craig(n, m)

  expected_craig_bytes = storage_craig_bytes(n, m)
  craig(Au, c)  # warmup
  actual_craig_bytes = @allocated craig(Au, c)
  @test actual_craig_bytes ≤ 1.1 * expected_craig_bytes

  solver = CraigSolver(Au, c)
  craig!(solver, Au, c)  # warmup
  inplace_craig_bytes = @allocated craig!(solver, Au, c)
  @test (VERSION < v"1.5") || (inplace_craig_bytes == 208)

  # without preconditioner and with (Ap, Aᵀq) preallocated, LSLQ needs:
  # - 3 m-vectors: x_lq, v, w̄ (= x_cg)
  # - 1 n-vector: u
  storage_lslq(n, m) = 3 * m + n
  storage_lslq_bytes(n, m) = 8 * storage_lslq(n, m)

  expected_lslq_bytes = storage_lslq_bytes(n, m)
  (x, stats) = lslq(Ao, b)  # warmup
  actual_lslq_bytes = @allocated lslq(Ao, b)
  @test actual_lslq_bytes ≤ 1.1 * expected_lslq_bytes

  solver = LslqSolver(Ao, b)
  lslq!(solver, Ao, b)  # warmup
  inplace_lslq_bytes = @allocated lslq!(solver, Ao, b)
  @test (VERSION < v"1.5") || (inplace_lslq_bytes == 576)

  # without preconditioner and with (Ap, Aᵀq) preallocated, CGLS needs:
  # - 2 m-vectors: x, p
  # - 1 n-vector: r
  storage_cgls(n, m) = n + 2 * m
  storage_cgls_bytes(n, m) = 8 * storage_cgls(n, m)

  expected_cgls_bytes = storage_cgls_bytes(n, m)
  (x, stats) = cgls(Ao, b)  # warmup
  actual_cgls_bytes = @allocated cgls(Ao, b)
  @test actual_cgls_bytes ≤ 1.1 * expected_cgls_bytes

  solver = CglsSolver(Ao, b)
  cgls!(solver, Ao, b)  # warmup
  inplace_cgls_bytes = @allocated cgls!(solver, Ao, b)
  @test (VERSION < v"1.5") || (inplace_cgls_bytes == 208)

  # without preconditioner and with (Ap, Aᵀq) preallocated, LSQR needs:
  # - 3 m-vectors: x, v, w
  # - 1 n-vector: u
  storage_lsqr(n, m) = 3 * m + n
  storage_lsqr_bytes(n, m) = 8 * storage_lsqr(n, m)

  expected_lsqr_bytes = storage_lsqr_bytes(n, m)
  (x, stats) = lsqr(Ao, b)  # warmup
  actual_lsqr_bytes = @allocated lsqr(Ao, b)
  @test actual_lsqr_bytes ≤ 1.1 * expected_lsqr_bytes

  solver = LsqrSolver(Ao, b)
  lsqr!(solver, Ao, b)  # warmup
  inplace_lsqr_bytes = @allocated lsqr!(solver, Ao, b)
  @test (VERSION < v"1.5") || (inplace_lsqr_bytes == 432)

  # without preconditioner and with (Ap, Aᵀq) preallocated, CRLS needs:
  # - 3 m-vectors: x, p, Ar
  # - 2 n-vector: r, Ap
  storage_crls(n, m) = 3 * m + 2 * n
  storage_crls_bytes(n, m) = 8 * storage_crls(n, m)

  expected_crls_bytes = storage_crls_bytes(n, m)
  (x, stats) = crls(Ao, b)  # warmup
  actual_crls_bytes = @allocated crls(Ao, b)
  @test actual_crls_bytes ≤ 1.1 * expected_crls_bytes

  solver = CrlsSolver(Ao, b)
  crls!(solver, Ao, b)  # warmup
  inplace_crls_bytes = @allocated crls!(solver, Ao, b)
  @test (VERSION < v"1.5") || (inplace_crls_bytes == 208)

  # without preconditioner and with (Ap, Aᵀq) preallocated, LSMR needs:
  # - 4 m-vectors: x, v, h, hbar
  # - 1 n-vector: u
  storage_lsmr(n, m) = 4 * m + n
  storage_lsmr_bytes(n, m) = 8 * storage_lsmr(n, m)

  expected_lsmr_bytes = storage_lsmr_bytes(n, m)
  (x, stats) = lsmr(Ao, b)  # warmup
  actual_lsmr_bytes = @allocated lsmr(Ao, b)
  @test actual_lsmr_bytes ≤ 1.1 * expected_lsmr_bytes

  solver = LsmrSolver(Ao, b)
  lsmr!(solver, Ao, b)  # warmup
  inplace_lsmr_bytes = @allocated lsmr!(solver, Ao, b)
  @test (VERSION < v"1.5") || (inplace_lsmr_bytes == 336)

  # USYMQR needs:
  # - 6 m-vectors: vₖ₋₁, vₖ, x, wₖ₋₁, wₖ, p 
  # - 3 n-vectors: uₖ₋₁, uₖ, q
  storage_usymqr(n, m) = 6 * m + 3 * n
  storage_usymqr_bytes(n, m) = 8 * storage_usymqr(n, m)

  expected_usymqr_bytes = storage_usymqr_bytes(n, m)
  (x, stats) = usymqr(Lo, b, c) # warmup
  actual_usymqr_bytes = @allocated usymqr(Lo, b, c)
  @test actual_usymqr_bytes ≤ 1.1 * expected_usymqr_bytes

  solver = UsymqrSolver(Lo, b)
  usymqr!(solver, Lo, b, c)  # warmup
  inplace_usymqr_bytes = @allocated usymqr!(solver, Lo, b, c)
  @test (VERSION < v"1.5") || (inplace_usymqr_bytes == 208)

  # TRILQR needs:
  # - 6 m-vectors: vₖ₋₁, vₖ, t, wₖ₋₁, wₖ, q
  # - 5 n-vectors: uₖ₋₁, uₖ, x, d̅, p
  storage_trilqr(n, m) = 6 * m + 5 * n
  storage_trilqr_bytes(n, m) = 8 * storage_trilqr(n, m)

  expected_trilqr_bytes = storage_trilqr_bytes(n, n)
  trilqr(L, b, b)  # warmup
  actual_trilqr_bytes = @allocated trilqr(L, b, b)
  @test actual_trilqr_bytes ≤ 1.1 * expected_trilqr_bytes

  solver = TrilqrSolver(L, b)
  trilqr!(solver, L, b, b)  # warmup
  inplace_trilqr_bytes = @allocated trilqr!(solver, L, b, b)
  @test (VERSION < v"1.5") || (inplace_trilqr_bytes == 208)

  # BILQ needs:
  # - 8 n-vectors: uₖ₋₁, uₖ, vₖ₋₁, vₖ, x, d̅, p, q
  storage_bilq(n) = 8 * n
  storage_bilq_bytes(n) = 8 * storage_bilq(n)

  expected_bilq_bytes = storage_bilq_bytes(n)
  bilq(L, b)  # warmup
  actual_bilq_bytes = @allocated bilq(L, b)
  @test actual_bilq_bytes ≤ 1.1 * expected_bilq_bytes

  solver = BilqSolver(L, b)
  bilq!(solver, L, b)  # warmup
  inplace_bilq_bytes = @allocated bilq!(solver, L, b)
  @test (VERSION < v"1.5") || (inplace_bilq_bytes == 208)

  # BILQR needs:
  # - 11 n-vectors: uₖ₋₁, uₖ, vₖ₋₁, vₖ, x, t, d̅, wₖ₋₁, wₖ, p, q
  storage_bilqr(n) = 11 * n
  storage_bilqr_bytes(n) = 8 * storage_bilqr(n)

  expected_bilqr_bytes = storage_bilqr_bytes(n)
  bilqr(L, b, b)  # warmup
  actual_bilqr_bytes = @allocated bilqr(L, b, b)
  @test actual_bilqr_bytes ≤ 1.1 * expected_bilqr_bytes

  solver = BilqrSolver(L, b)
  bilqr!(solver, L, b, b)  # warmup
  inplace_bilqr_bytes = @allocated bilqr!(solver, L, b, b)
  @test (VERSION < v"1.5") || (inplace_bilqr_bytes == 208)

  # with Ap preallocated, MINRES-QLP needs:
  # - 5 n-vectors: wₖ₋₁, wₖ, vₖ₋₁, vₖ, x
  storage_minres_qlp(n) = 5 * n
  storage_minres_qlp_bytes(n) = 8 * storage_minres_qlp(n)

  expected_minres_qlp_bytes = storage_minres_qlp_bytes(n)
  minres_qlp(A, b)  # warmup
  actual_minres_qlp_bytes = @allocated minres_qlp(A, b)
  @test actual_minres_qlp_bytes ≤ 1.1 * expected_minres_qlp_bytes

  solver = MinresQlpSolver(A, b)
  minres_qlp!(solver, A, b)  # warmup
  inplace_minres_qlp_bytes = @allocated minres_qlp!(solver, A, b)
  @test (VERSION < v"1.5") || (inplace_minres_qlp_bytes == 208)

  # QMR needs:
  # - 9 n-vectors: uₖ₋₁, uₖ, vₖ₋₁, vₖ, x, wₖ₋₁, wₖ, p, q
  storage_qmr(n) = 9 * n
  storage_qmr_bytes(n) = 8 * storage_qmr(n)

  expected_qmr_bytes = storage_qmr_bytes(n)
  qmr(L, b)  # warmup
  actual_qmr_bytes = @allocated qmr(L, b)
  @test actual_qmr_bytes ≤ 1.1 * expected_qmr_bytes

  solver = QmrSolver(L, b)
  qmr!(solver, L, b)  # warmup
  inplace_qmr_bytes = @allocated qmr!(solver, L, b)
  @test (VERSION < v"1.5") || (inplace_qmr_bytes == 208)

  # USYMLQ needs:
  # - 5 n-vectors: uₖ₋₁, uₖ, x, d̅, p
  # - 3 m-vectors: vₖ₋₁, vₖ, q
  storage_usymlq(n, m) = 5 * n + 3 * m
  storage_usymlq_bytes(n, m) = 8 * storage_usymlq(n, m)

  expected_usymlq_bytes = storage_usymlq_bytes(n, m)
  usymlq(Lu, c, b)  # warmup
  actual_usymlq_bytes = @allocated usymlq(Lu, c, b)
  @test actual_usymlq_bytes ≤ 1.1 * expected_usymlq_bytes

  solver = UsymlqSolver(Lu, c)
  usymlq!(solver, Lu, c, b)  # warmup
  inplace_usymlq_bytes = @allocated usymlq!(solver, Lu, c, b)
  @test (VERSION < v"1.5") || (inplace_usymlq_bytes == 208)

  # TriCG needs:
  # - 6 n-vectors: yₖ, uₖ₋₁, uₖ, gy₂ₖ₋₁, gy₂ₖ, p
  # - 6 m-vectors: xₖ, vₖ₋₁, vₖ, gx₂ₖ₋₁, gx₂ₖ, q
  storage_tricg(n, m) = 6 * n + 6 * m
  storage_tricg_bytes(n, m) = 8 * storage_tricg(n, m)

  expected_tricg_bytes = storage_tricg_bytes(n, m)
  tricg(Lu, c, b)  # warmup
  actual_tricg_bytes = @allocated tricg(Lu, c, b)
  @test actual_tricg_bytes ≤ 1.1 * expected_tricg_bytes

  solver = TricgSolver(Lu, c)
  tricg!(solver, Lu, c, b)  # warmup
  inplace_tricg_bytes = @allocated tricg!(solver, Lu, c, b)
  @test (VERSION < v"1.5") || (inplace_tricg_bytes == 208)

  # TriMR needs:
  # - 8 n-vectors: yₖ, uₖ₋₁, uₖ, gy₂ₖ₋₃, gy₂ₖ₋₂, gy₂ₖ₋₁, gy₂ₖ, p
  # - 8 m-vectors: xₖ, vₖ₋₁, vₖ, gx₂ₖ₋₃, gx₂ₖ₋₂, gx₂ₖ₋₁, gx₂ₖ, q
  storage_trimr(n, m) = 8 * n + 8 * m
  storage_trimr_bytes(n, m) = 8 * storage_trimr(n, m)

  expected_trimr_bytes = storage_trimr_bytes(n, m)
  trimr(Lu, c, b)  # warmup
  actual_trimr_bytes = @allocated trimr(Lu, c, b)
  @test actual_trimr_bytes ≤ 1.1 * expected_trimr_bytes

  solver = TrimrSolver(Lu, c)
  trimr!(solver, Lu, c, b)  # warmup
  inplace_trimr_bytes = @allocated trimr!(solver, Lu, c, b)
  @test (VERSION < v"1.5") || (inplace_trimr_bytes == 208)
end

@testset "alloc" begin
  test_alloc()
end
