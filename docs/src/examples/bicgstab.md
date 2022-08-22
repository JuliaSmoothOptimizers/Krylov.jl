```@example bicgstab
using Krylov, LinearOperators, IncompleteLU, HarwellRutherfordBoeing
using LinearAlgebra, Printf, SuiteSparseMatrixCollection, SparseArrays

ssmc = ssmc_db(verbose=false)
matrix = ssmc_matrices(ssmc, "HB", "sherman5")
path = fetch_ssmc(matrix, format="RB")

n = matrix.nrows[1]
A = RutherfordBoeingData(joinpath(path[1], "$(matrix.name[1]).rb")).data
b = A * ones(n)

F = ilu(A, τ = 0.05)

@printf("nnz(ILU) / nnz(A): %7.1e\n", nnz(F) / nnz(A))

# Solve Ax = b with BICGSTAB and an incomplete LU factorization
# Remark: CGS can be used in the same way
opM = LinearOperator(Float64, n, n, false, false, (y, v) -> forward_substitution!(y, F, v))
opN = LinearOperator(Float64, n, n, false, false, (y, v) -> backward_substitution!(y, F, v))
opP = LinearOperator(Float64, n, n, false, false, (y, v) -> ldiv!(y, F, v))

# Without preconditioning
x, stats = bicgstab(A, b, history=true)
r = b - A * x
@printf("[Without preconditioning] Residual norm: %8.1e\n", norm(r))
@printf("[Without preconditioning] Number of iterations: %3d\n", length(stats.residuals) - 1)

# Split preconditioning
x, stats = bicgstab(A, b, history=true, M=opM, N=opN)
r = b - A * x
@printf("[Split preconditioning] Residual norm: %8.1e\n", norm(r))
@printf("[Split preconditioning] Number of iterations: %3d\n", length(stats.residuals) - 1)

# Left preconditioning
x, stats = bicgstab(A, b, history=true, M=opP)
r = b - A * x
@printf("[Left preconditioning] Residual norm: %8.1e\n", norm(r))
@printf("[Left preconditioning] Number of iterations: %3d\n", length(stats.residuals) - 1)

# Right preconditioning
x, stats = bicgstab(A, b, history=true, N=opP)
r = b - A * x
@printf("[Right preconditioning] Residual norm: %8.1e\n", norm(r))
@printf("[Right preconditioning] Number of iterations: %3d\n", length(stats.residuals) - 1)
```
