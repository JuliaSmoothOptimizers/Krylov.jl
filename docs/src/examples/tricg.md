## TriCG

```@example tricg
using Krylov, LinearOperators
using LinearAlgebra, Printf, SparseArrays

# Identity matrix.
eye(n::Int) = sparse(1.0 * I, n, n)

# Symmetric quasi-definite systems and variants
n = m = 5
A = [2^(i/j)*j + (-1)^(i-j) * n*(i-1) for i = 1:n, j = 1:n]
b = ones(n)
M = diagm(0 => [3.0 * i for i = 1:n])
N = diagm(0 => [5.0 * i for i = 1:n])
c = -b

# [I   A] [x] = [b]
# [Aᵀ -I] [y]   [c]
(x, y, stats) = tricg(A, b, c)
K = [eye(m) A; A' -eye(n)]
B = [b; c]
r = B - K * [x; y]
resid = norm(r)
@printf("TriCG: Relative residual: %8.1e\n", resid)

# [-I   A] [x] = [b]
# [ Aᵀ  I] [y]   [c]
(x, y, stats) = tricg(A, b, c, flip=true)
K = [-eye(m) A; A' eye(n)]
B = [b; c]
r = B - K * [x; y]
resid = norm(r)
@printf("TriCG: Relative residual: %8.1e\n", resid)

# [I   A] [x] = [b]
# [Aᵀ  I] [y]   [c]
(x, y, stats) = tricg(A, b, c, spd=true)
K = [eye(m) A; A' eye(n)]
B = [b; c]
r = B - K * [x; y]
resid = norm(r)
@printf("TriCG: Relative residual: %8.1e\n", resid)

# [-I    A] [x] = [b]
# [ Aᵀ  -I] [y]   [c]
(x, y, stats) = tricg(A, b, c, snd=true)
K = [-eye(m) A; A' -eye(n)]
B = [b; c]
r = B - K * [x; y]
resid = norm(r)
@printf("TriCG: Relative residual: %8.1e\n", resid)

# [τI    A] [x] = [b]
# [ Aᵀ  νI] [y]   [c]
(τ, ν) = (1e-4, 1e2)
(x, y, stats) = tricg(A, b, c, τ=τ, ν=ν)
K = [τ*eye(m) A; A' ν*eye(n)]
B = [b; c]
r = B - K * [x; y]
resid = norm(r)
@printf("TriCG: Relative residual: %8.1e\n", resid)

# [M⁻¹  A  ] [x] = [b]
# [Aᵀ  -N⁻¹] [y]   [c]
(x, y, stats) = tricg(A, b, c, M=M, N=N, verbose=1)
K = [inv(M) A; A' -inv(N)]
H = BlockDiagonalOperator(M, N)
B = [b; c]
r = B - K * [x; y]
resid = sqrt(dot(r, H * r))
@printf("TriCG: Relative residual: %8.1e\n", resid)
```
