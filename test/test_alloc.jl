include("test_utils.jl")

A   = preallocated_LinearOperator(get_div_grad(32, 32, 32))
n   = size(A, 1)
b   = ones(n)
M   = nonallocating_opEye(n)
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

# without preconditioner and with Ap preallocated, CR needs 4 n-vectors: x, r, p, q
storage_cr(n) = 4 * n
storage_cr_bytes(n) = 8 * storage_cr(n)

expected_cr_bytes = storage_cr_bytes(n)
cr(A, b, M=M)  # warmup
actual_cr_bytes = @allocated cr(A, b, M=M)
@test actual_cr_bytes ≤ 1.1 * expected_cr_bytes
