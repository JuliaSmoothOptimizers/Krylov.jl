include("test_utils.jl")

L   = get_div_grad(32, 32, 32)
n   = size(L, 1)
m   = div(n, 2)
A   = preallocated_LinearOperator(L) # Dimension n x n
Au  = preallocated_LinearOperator(L[1:m,:]) # Dimension m x n
Ao  = preallocated_LinearOperator(L[:,1:m]) # Dimension n x m
b   = ones(n)
c   = ones(m)
M   = nonallocating_opEye(n)
N   = nonallocating_opEye(m)
mem = 10

# without preconditioner and with Ap preallocated, CG needs 3 n-vectors: x, r, p
storage_cg(n) = 3 * n
storage_cg_bytes(n) = 8 * storage_cg(n)

expected_cg_bytes = storage_cg_bytes(n)
cg(A, b, M=M)  # warmup
actual_cg_bytes = @allocated cg(A, b, M=M)
@test actual_cg_bytes ≤ 1.1 * expected_cg_bytes

# without preconditioner and with Ap preallocated, DIOM needs:
# - 2 n-vectors: x, x_old
# - 2 (n*mem)-matrices: P, V
# - 1 mem-vector: L
# - 1 (mem+2)-vector: H
# - 1 mem-bitArray: p
storage_diom(mem, n) = (2 * n) + (2 * n * mem) + (mem) + (mem + 2) + (mem / 64)
storage_diom_bytes(mem, n) = 8 * storage_diom(mem, n)

expected_diom_bytes = storage_diom_bytes(mem, n)
diom(A, b, M=M, memory=mem)  # warmup
actual_diom_bytes = @allocated diom(A, b, M=M, memory=mem)
@test actual_diom_bytes ≤ 1.05 * expected_diom_bytes

# with Ap preallocated, CG-Lanczos needs 4 n-vectors: x, v, v_prev, p
storage_cg_lanczos(n) = 4 * n
storage_cg_lanczos_bytes(n) = 8 * storage_cg_lanczos(n)

expected_cg_lanczos_bytes = storage_cg_lanczos_bytes(n)
cg_lanczos(A, b)  # warmup
actual_cg_lanczos_bytes = @allocated cg_lanczos(A, b)
@test actual_cg_lanczos_bytes ≤ 1.1 * expected_cg_lanczos_bytes

# without preconditioner and with Ap preallocated, DQGMRES needs:
# - 1 n-vector: x
# - 2 (n*mem)-matrices: P, V
# - 2 mem-vectors: c, s
# - 1 (mem+2)-vector: H
storage_dqgmres(mem, n) = (n) + (2 * n * mem) + (2 * mem) + (mem + 2)
storage_dqgmres_bytes(mem, n) = 8 * storage_dqgmres(mem, n)

expected_dqgmres_bytes = storage_dqgmres_bytes(mem, n)
dqgmres(A, b, M=M, memory=mem)  # warmup
actual_dqgmres_bytes = @allocated dqgmres(A, b, M=M, memory=mem)
@test actual_dqgmres_bytes ≤ 1.05 * expected_dqgmres_bytes

# without preconditioner and with Ap preallocated, CR needs 4 n-vectors: x, r, p, q
storage_cr(n) = 4 * n
storage_cr_bytes(n) = 8 * storage_cr(n)

expected_cr_bytes = storage_cr_bytes(n)
cr(A, b, M=M)  # warmup
actual_cr_bytes = @allocated cr(A, b, M=M)
@test actual_cr_bytes ≤ 1.1 * expected_cr_bytes

# without preconditioner and with Ap preallocated, CGS needs 5 n-vectors: x, r, u, p, q
storage_cgs(n) = 5 * n
storage_cgs_bytes(n) = 8 * storage_cgs(n)
expected_cgs_bytes = storage_cgs_bytes(n)
cgs(A, b, M=M)  # warmup
actual_cgs_bytes = @allocated cgs(A, b, M=M)
@test actual_cgs_bytes ≤ 1.1 * expected_cgs_bytes

# without preconditioner and with (Ap, Aᵀq) preallocated, CGNE needs:
# - 2 n-vectors: x, p
# - 1 m-vector: r
storage_cgne(n, m) = 2*n + m
storage_cgne_bytes(n, m) = 8 * storage_cgne(n, m)
expected_cgne_bytes = storage_cgne_bytes(n, m)
(x, stats) = cgne(Au, c, M=N)  # warmup
actual_cgne_bytes = @allocated cgne(Au, c, M=N)
@test actual_cgne_bytes ≤ 1.1 * expected_cgne_bytes
