```@example usymlqr
using LinearAlgebra, Printf, SparseArrays
using Krylov

# Identity matrix.
eye(n::Int) = sparse(1.0 * I, n, n)

# Saddle-point systems
n = m = 5
A = [2^(i/j)*j + (-1)^(i-j) * n*(i-1) for i = 1:n, j = 1:n]
b = ones(n)
D = diagm(0 => [2.0 * i for i = 1:n])
m, n = size(A)
c = -3*b

# [I   A] [x] = [b]
# [Aᴴ  0] [y]   [c]
(x, y, stats) = usymlqr(A, b, c)
K = [I A; A' zeros(n,n)]
d = [b; c]
r = d - K * [x; y]
resid = norm(r)
@printf("USYMLQR: Relative residual: %8.1e\n", resid)

# [I   A] [x] = [b]
# [Aᴴ  0] [y]   [0]
(x, y, stats) = usymlqr(A, b, c, ln=false)
K = [I A; A' zeros(n,n)]
d = [b; 0*c]
r = d - K * [x; y]
resid = norm(r)
@printf("USYMLQR: Relative residual: %8.1e\n", resid)

# [I   A] [x] = [0]
# [Aᴴ  0] [y]   [c]
(x, y, stats) = usymlqr(A, b, c, ls=false)
K = [I A; A' zeros(n,n)]
d = [0*b; c]
r = d - K * [x; y]
resid = norm(r)
@printf("USYMLQR: Relative residual: %8.1e\n", resid)
```
