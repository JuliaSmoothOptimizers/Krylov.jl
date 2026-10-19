```@example minares
using Krylov, MatrixMarket, SuiteSparseMatrixCollection
using LinearAlgebra, Printf

ssmc = ssmc_db(verbose=false)
matrix = ssmc_matrices(ssmc, "GHS_indef", "laser")
path = fetch_ssmc(matrix, format="MM")

n = matrix.nrows[1]
A = MatrixMarket.mmread(joinpath(path[1], "$(matrix.name[1]).mtx"))
b = ones(n)

# Solve Ax = b.
x, stats = minares(A, b)
show(stats)
r = b - A * x
Ar = A * r
@printf("Relative A-residual: %8.1e\n", norm(A * r) / norm(A * b))
```

## Complex symmetric systems

A complex symmetric matrix satisfies `transpose(A) == A` without being Hermitian; such systems arise, for instance, from damped wave problems.
With `complex_symmetric = true`, MINARES runs on the conjugate Bunse-Gerstner-Stover (BGS) process instead of the Hermitian Lanczos process and minimizes ‖Aᴴrₖ‖₂.
[`minres_qlp`](@ref) accepts the same keyword.

```@example minares_cs
using Krylov, SparseArrays, LinearAlgebra, Printf

# Indefinite 1D Helmholtz operator with a variable damping term.
n = 100
L = spdiagm(-1 => -ones(n - 1), 0 => 2 * ones(n), 1 => -ones(n - 1))
A = L - I + im * Diagonal(range(0.5, 1.0, length = n))
b = ones(ComplexF64, n)
@printf("transpose(A) == A: %s, A' == A: %s\n", transpose(A) == A, A' == A)

x, stats = minares(A, b; complex_symmetric = true)
show(stats)
r = b - A * x
@printf("Relative residual: %8.1e\n", norm(r) / norm(b))
@printf("Relative Aᴴ-residual: %8.1e\n", norm(A' * r) / norm(A' * b))
```
