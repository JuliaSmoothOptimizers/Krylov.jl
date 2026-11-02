```@example bicgstabl
using Krylov, LinearOperators, KrylovPreconditioners, HarwellRutherfordBoeing
using LinearAlgebra, Printf, SuiteSparseMatrixCollection, SparseArrays

ssmc = ssmc_db(verbose=false)
matrix = ssmc_matrices(ssmc, "HB", "sherman5")
path = fetch_ssmc(matrix, format="RB")

n = matrix.nrows[1]
A = RutherfordBoeingData(joinpath(path[1], "$(matrix.name[1]).rb")).data
b = A * ones(n)

F = ilu(A, τ = 0.05)

@printf("nnz(ILU) / nnz(A): %7.1e\n", nnz(F) / nnz(A))

# Solve Ax = b with BiCGstab(ℓ) and an incomplete LU factorization
opM = LinearOperator(Float64, n, n, false, false, (y, v) -> forward_substitution!(y, F, v))
opN = LinearOperator(Float64, n, n, false, false, (y, v) -> backward_substitution!(y, F, v))
opP = LinearOperator(Float64, n, n, false, false, (y, v) -> ldiv!(y, F, v))

ℓ = 2

# Without preconditioning
x, stats = bicgstabl(A, b, ℓ=ℓ, history=true)
r = b - A * x
@printf("[Without preconditioning] Residual norm: %8.1e\n", norm(r))
@printf("[Without preconditioning] Number of iterations: %3d\n", stats.niter)

# Split preconditioning
x, stats = bicgstabl(A, b, ℓ=ℓ, history=true, M=opM, N=opN)
r = b - A * x
@printf("[Split preconditioning] Residual norm: %8.1e\n", norm(r))
@printf("[Split preconditioning] Number of iterations: %3d\n", stats.niter)

# Left preconditioning
x, stats = bicgstabl(A, b, ℓ=ℓ, history=true, M=opP)
r = b - A * x
@printf("[Left preconditioning] Residual norm: %8.1e\n", norm(r))
@printf("[Left preconditioning] Number of iterations: %3d\n", stats.niter)

# Right preconditioning
x, stats = bicgstabl(A, b, ℓ=ℓ, history=true, N=opP)
r = b - A * x
@printf("[Right preconditioning] Residual norm: %8.1e\n", norm(r))
@printf("[Right preconditioning] Number of iterations: %3d\n", stats.niter)
```
