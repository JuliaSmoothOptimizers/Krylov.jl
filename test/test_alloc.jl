L   = get_div_grad(32, 32, 32)
n   = size(L, 1)
m   = div(n, 2)
A   = PreallocatedLinearOperator(L) # Dimension n x n
Au  = PreallocatedLinearOperator(L[1:m,:]) # Dimension m x n
Ao  = PreallocatedLinearOperator(L[:,1:m]) # Dimension n x m
b   = ones(n)
c   = ones(m)
mem = 10

shifts  = [1:5;]
nshifts = 5

# without preconditioner and with Ap preallocated, SYMMLQ needs 4 n-vectors: x_lq, vold, v, w̅ (= x_cg)
storage_symmlq(n) = 4 * n
storage_symmlq_bytes(n) = 8 * storage_symmlq(n)

expected_symmlq_bytes = storage_symmlq_bytes(n)
symmlq(A, b)  # warmup
actual_symmlq_bytes = @allocated symmlq(A, b)
@test actual_symmlq_bytes ≤ 1.1 * expected_symmlq_bytes

# without preconditioner and with Ap preallocated, CG needs 3 n-vectors: x, r, p
storage_cg(n) = 3 * n
storage_cg_bytes(n) = 8 * storage_cg(n)

expected_cg_bytes = storage_cg_bytes(n)
cg(A, b)  # warmup
actual_cg_bytes = @allocated cg(A, b)
@test actual_cg_bytes ≤ 1.1 * expected_cg_bytes

# without preconditioner and with Ap preallocated, MINRES needs 5 n-vectors: x, r1, r2, w1, w2
storage_minres(n) = 5 * n
storage_minres_bytes(n) = 8 * storage_minres(n)

expected_minres_bytes = storage_minres_bytes(n)
minres(A, b)  # warmup
actual_minres_bytes = @allocated minres(A, b)
@test actual_minres_bytes ≤ 1.1 * expected_minres_bytes

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

# with Ap preallocated, CG_LANCZOS needs 4 n-vectors: x, v, v_prev, p
storage_cg_lanczos(n) = 4 * n
storage_cg_lanczos_bytes(n) = 8 * storage_cg_lanczos(n)

expected_cg_lanczos_bytes = storage_cg_lanczos_bytes(n)
cg_lanczos(A, b)  # warmup
actual_cg_lanczos_bytes = @allocated cg_lanczos(A, b)
@test actual_cg_lanczos_bytes ≤ 1.1 * expected_cg_lanczos_bytes

# with Ap preallocated, CG_LANCZOS_SHIFT_SEQ needs:
# - 2 n-vectors: v, v_prev
# - 2 (n*nshifts)-matrices: x, p
# - 5 nshifts-vectors: σ, δhat, ω, γ, rNorms
# - 2 nshifts-bitArray: indefinite, converged
storage_cg_lanczos_shift_seq(n, nshifts) = (2 * n) + (2 * n * nshifts) + (5 * nshifts) + (2 * nshifts / 64)
storage_cg_lanczos_shift_seq_bytes(n, nshifts) = 8 * storage_cg_lanczos_shift_seq(n, nshifts)

expected_cg_lanczos_shift_seq_bytes = storage_cg_lanczos_shift_seq_bytes(n, nshifts)
cg_lanczos_shift_seq(A, b, shifts)  # warmup
actual_cg_lanczos_shift_seq_bytes = @allocated cg_lanczos_shift_seq(A, b, shifts)
@test actual_cg_lanczos_shift_seq_bytes ≤ 1.1 * expected_cg_lanczos_shift_seq_bytes

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

# without preconditioner and with Ap preallocated, CR needs 4 n-vectors: x, r, p, q
storage_cr(n) = 4 * n
storage_cr_bytes(n) = 8 * storage_cr(n)

expected_cr_bytes = storage_cr_bytes(n)
cr(A, b)  # warmup
actual_cr_bytes = @allocated cr(A, b)
@test actual_cr_bytes ≤ 1.1 * expected_cr_bytes

# without preconditioner and with (Ap, Aᵀq) preallocated, CRMR needs:
# - 2 n-vectors: x, p
# - 1 m-vector: r
storage_crmr(n, m) = 2 * n + m
storage_crmr_bytes(n, m) = 8 * storage_crmr(n, m)
expected_crmr_bytes = storage_crmr_bytes(n, m)
(x, stats) = crmr(Au, c)  # warmup
actual_crmr_bytes = @allocated crmr(Au, c)
@test actual_crmr_bytes ≤ 1.1 * expected_crmr_bytes

# without preconditioner and with Ap preallocated, CGS needs 5 n-vectors: x, r, u, p, q
storage_cgs(n) = 5 * n
storage_cgs_bytes(n) = 8 * storage_cgs(n)
expected_cgs_bytes = storage_cgs_bytes(n)
cgs(A, b)  # warmup
actual_cgs_bytes = @allocated cgs(A, b)
@test actual_cgs_bytes ≤ 1.1 * expected_cgs_bytes

# with (Ap, Aᵀq) preallocated, CRAIGMR needs:
# - 2 n-vector: x, v
# - 4 m-vectors: y, u, w, wbar
storage_craigmr(n, m) = 2 * n + 4 * m
storage_craigmr_bytes(n, m) = 8 * storage_craigmr(n, m)

expected_craigmr_bytes = storage_craigmr_bytes(n, m)
craigmr(Au, c)  # warmup
actual_craigmr_bytes = @allocated craigmr(Au, c)
@test actual_craigmr_bytes ≤ 1.1 * expected_craigmr_bytes

# without preconditioner and with (Ap, Aᵀq) preallocated, CGNE needs:
# - 2 n-vectors: x, p
# - 1 m-vector: r
storage_cgne(n, m) = 2*n + m
storage_cgne_bytes(n, m) = 8 * storage_cgne(n, m)
expected_cgne_bytes = storage_cgne_bytes(n, m)
(x, stats) = cgne(Au, c)  # warmup
actual_cgne_bytes = @allocated cgne(Au, c)
@test actual_cgne_bytes ≤ 1.1 * expected_cgne_bytes

# with (Ap, Aᵀq) preallocated, CRAIG needs:
# - 2 n-vector: x, v
# - 3 m-vectors: y, w, u
storage_craig(n, m) = 2 * n + 3 * m
storage_craig_bytes(n, m) = 8 * storage_craig(n, m)

expected_craig_bytes = storage_craig_bytes(n, m)
craig(Au, c)  # warmup
actual_craig_bytes = @allocated craig(Au, c)
@test actual_craig_bytes ≤ 1.1 * expected_craig_bytes

# without preconditioner and with (Ap, Aᵀq) preallocated, LSLQ needs:
# - 3 m-vectors: x_lq, v, w̄ (= x_cg)
# - 1 n-vector: u
storage_lslq(n, m) = 3 * m + n
storage_lslq_bytes(n, m) = 8 * storage_lslq(n, m)
expected_lslq_bytes = storage_lslq_bytes(n, m)
(x, stats) = lslq(Ao, b)  # warmup
actual_lslq_bytes = @allocated lslq(Ao, b)
@test actual_lslq_bytes ≤ 1.1 * expected_lslq_bytes

# without preconditioner and with (Ap, Aᵀq) preallocated, CGLS needs:
# - 2 m-vectors: x, p
# - 1 n-vector: r
storage_cgls(n, m) = 2*m + n
storage_cgls_bytes(n, m) = 8 * storage_cgls(n, m)
expected_cgls_bytes = storage_cgls_bytes(n, m)
(x, stats) = cgls(Ao, b)  # warmup
actual_cgls_bytes = @allocated cgls(Ao, b)
@test actual_cgls_bytes ≤ 1.1 * expected_cgls_bytes

# without preconditioner and with (Ap, Aᵀq) preallocated, LSQR needs:
# - 3 m-vectors: x, v, w
# - 1 n-vector: u
storage_lsqr(n, m) = 3 * m + n
storage_lsqr_bytes(n, m) = 8 * storage_lsqr(n, m)
expected_lsqr_bytes = storage_lsqr_bytes(n, m)
(x, stats) = lsqr(Ao, b)  # warmup
actual_lsqr_bytes = @allocated lsqr(Ao, b)
@test actual_lsqr_bytes ≤ 1.1 * expected_lsqr_bytes

# without preconditioner and with (Ap, Aᵀq) preallocated, CRLS needs:
# - 3 m-vectors: x, p, Ar
# - 2 n-vector: r, Ap
storage_crls(n, m) = 3 * m + 2 * n
storage_crls_bytes(n, m) = 8 * storage_crls(n, m)
expected_crls_bytes = storage_crls_bytes(n, m)
(x, stats) = crls(Ao, b)  # warmup
actual_crls_bytes = @allocated crls(Ao, b)
@test actual_crls_bytes ≤ 1.1 * expected_crls_bytes

# without preconditioner and with (Ap, Aᵀq) preallocated, LSMR needs:
# - 4 m-vectors: x, v, h, hbar
# - 1 n-vector: u
storage_lsmr(n, m) = 4 * m + n
storage_lsmr_bytes(n, m) = 8 * storage_lsmr(n, m)
expected_lsmr_bytes = storage_lsmr_bytes(n, m)
(x, stats) = lsmr(Ao, b)  # warmup
actual_lsmr_bytes = @allocated lsmr(Ao, b)
@test actual_lsmr_bytes ≤ 1.1 * expected_lsmr_bytes
