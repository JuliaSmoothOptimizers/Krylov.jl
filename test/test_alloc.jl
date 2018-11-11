include("test_utils.jl")

A   = preallocated_LinearOperator(get_div_grad(32, 32, 32))
n   = size(A, 1)
b   = ones(n)
M   = nonallocating_opEye(n)

# without preconditioner and with Ap preallocated, CG needs 3 n-vectors: x, r, p
storage_cg(n) = 3 * n
storage_cg_bytes(n) = 8 * storage_cg(n)

expected_cg_bytes = storage_cg_bytes(n)
cg(A, b, M=M)  # warmup
actual_cg_bytes = @allocated cg(A, b, M=M)
@test actual_cg_bytes ≤ 1.1 * expected_cg_bytes

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

# without preconditioner and with Ap preallocated, CGS needs 5 n-vectors: x, r, u, p, q
storage_cgs(n) = 5 * n
storage_cgs_bytes(n) = 8 * storage_cgs(n)
expected_cgs_bytes = storage_cgs_bytes(n)
cgs(A, b, M=M)  # warmup
actual_cgs_bytes = @allocated cgs(A, b, M=M)
@test actual_cgs_bytes ≤ 1.1 * expected_cgs_bytes
