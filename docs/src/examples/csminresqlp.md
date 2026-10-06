```@example csminresqlp
using Krylov, LinearAlgebra, Printf

# A random complex symmetric matrix: transpose(A) == A, but A is not Hermitian.
n = 100
rng_A = randn(ComplexF64, n, n)
A = rng_A + transpose(rng_A)  # complex symmetric
@assert transpose(A) == A

b = A * ones(ComplexF64, n)  # consistent right-hand side

# Solve Ax = b.
x, stats = csminresqlp(A, b)
show(stats)
println()
r = b - A * x
@printf("Relative residual: %8.1e\n", norm(r) / norm(b))
```
