## TriMR

```@example trimr
using Krylov, LinearOperators, LDLFactorizations
using LinearAlgebra, Printf, SparseArrays

# Identity matrix.
eye(n::Int) = sparse(1.0 * I, n, n)

# Saddle-point systems
n = m = 5
A = [2^(i/j)*j + (-1)^(i-j) * n*(i-1) for i = 1:n, j = 1:n]
b = ones(n)
D = diagm(0 => [2.0 * i for i = 1:n])
m, n = size(A)
c = -b

# [D   A] [x] = [b]
# [Aᵀ  0] [y]   [c]
llt_D = cholesky(D)
opD⁻¹ = LinearOperator(Float64, 5, 5, true, true, (y, v) -> ldiv!(y, llt_D, v))
opH⁻¹ = BlockDiagonalOperator(opD⁻¹, eye(n))
(x, y, stats) = trimr(A, b, c, M=opD⁻¹, sp=true)
K = [D A; A' zeros(n,n)]
B = [b; c]
r = B - K * [x; y]
resid = sqrt(dot(r, opH⁻¹ * r))
@printf("TriMR: Relative residual: %8.1e\n", resid)

# Symmetric quasi-definite systems
n = m = 5
A = [2^(i/j)*j + (-1)^(i-j) * n*(i-1) for i = 1:n, j = 1:n]
b = ones(n)
M = diagm(0 => [3.0 * i for i = 1:n])
N = diagm(0 => [5.0 * i for i = 1:n])
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
opM⁻¹ = LinearOperator(Float64, size(M,1), size(M,2), true, true, (y, v) -> ldiv!(y, ldlt_M, v))
opN⁻¹ = LinearOperator(Float64, size(N,1), size(N,2), true, true, (y, v) -> ldiv!(y, ldlt_N, v))
opH⁻¹ = BlockDiagonalOperator(opM⁻¹, opN⁻¹)
(x, y, stats) = trimr(A, b, c, M=opM⁻¹, N=opN⁻¹, verbose=1)
K = [M A; A' -N]
B = [b; c]
r = B - K * [x; y]
resid = sqrt(dot(r, opH⁻¹ * r))
@printf("TriMR: Relative residual: %8.1e\n", resid)
```
