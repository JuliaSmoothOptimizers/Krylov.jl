using Krylov, LinearOperators, LDLFactorizations
using LinearAlgebra, Printf

include("../test/test_utils.jl")

# Adjoint systems
A, b, c = adjoint_ode()
m, n = size(A)

# [0   A] [x] = [b]
# [Aᵀ  0] [y]   [c]
(x, y, stats) = trimr(A, b, c, τ=0.0, ν=0.0)
K = [zeros(m,m) A; A' zeros(n,n)]
B = [b; c]
r = B - K * [x; y]
resid = norm(r)
@printf("TriMR: Relative residual: %8.1e\n", resid)

# Saddle-point systems
A, b, D = saddle_point()
m, n = size(A)
c = -b

# [D   A] [x] = [b]
# [Aᵀ  0] [y]   [c]
llt_D = cholesky(D)
vD = similar(b)
opD⁻¹ = LinearOperator(Float64, size(D,1), size(D,2), true, true, v -> ldiv!(vD, llt_D, v))
opH⁻¹ = BlockDiagonalOperator(opD⁻¹, eye(n))
(x, y, stats) = trimr(A, b, c, M=opD⁻¹, sp=true)
K = [D A; A' zeros(n,n)]
B = [b; c]
r = B - K * [x; y]
resid = sqrt(dot(r, opH⁻¹ * r))
@printf("TriMR: Relative residual: %8.1e\n", resid)

# Symmetric quasi-definite systems
A, b, M, N = sqd()
m, n = size(A)
c = -b

# [I   A] [x] = [b]
# [Aᵀ -I] [y]   [c]
(x, y, stats) = trimr(A, b, c)
K = [eye(m) A; A' -eye(n)]
B = [b; c]
r = B - K * [x; y]
resid = norm(r)
@printf("TriMR: Relative residual: %8.1e\n", resid)

# [M   A] [x] = [b]
# [Aᵀ -N] [y]   [c]
ldlt_M = ldl(M)
ldlt_N = ldl(N)
vM = similar(b)
vN = similar(c)
opM⁻¹ = LinearOperator(Float64, size(M,1), size(M,2), true, true, v -> ldiv!(vM, ldlt_M, v))
opN⁻¹ = LinearOperator(Float64, size(N,1), size(N,2), true, true, v -> ldiv!(vN, ldlt_N, v))
opH⁻¹ = BlockDiagonalOperator(opM⁻¹, opN⁻¹)
(x, y, stats) = trimr(A, b, c, M=opM⁻¹, N=opN⁻¹, verbose=true)
K = [M A; A' -N]
B = [b; c]
r = B - K * [x; y]
resid = sqrt(dot(r, opH⁻¹ * r))
@printf("TriMR: Relative residual: %8.1e\n", resid)
