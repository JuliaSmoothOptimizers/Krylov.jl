using Krylov, LinearOperators
using LinearAlgebra, Printf

include("../test/test_utils.jl")

# Symmetric quasi-definite systems and variants
A, b, M, N = sqd()
m, n = size(A)
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
(x, y, stats) = tricg(A, b, c, M=M, N=N, verbose=true)
K = [inv(M) A; A' -inv(N)]
H = BlockDiagonalOperator(M, N)
B = [b; c]
r = B - K * [x; y]
resid = sqrt(dot(r, H * r))
@printf("TriCG: Relative residual: %8.1e\n", resid)
