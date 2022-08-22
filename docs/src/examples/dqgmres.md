```@example dqgmres
using Krylov, LinearOperators, ILUZero, MatrixMarket
using LinearAlgebra, Printf, SuiteSparseMatrixCollection

ssmc = ssmc_db(verbose=false)
matrix = ssmc_matrices(ssmc, "Simon", "raefsky1")
path = fetch_ssmc(matrix, format="MM")

n = matrix.nrows[1]
A = MatrixMarket.mmread(joinpath(path[1], "$(matrix.name[1]).mtx"))
b = A * ones(n)

F = ilu0(A)

@printf("nnz(ILU) / nnz(A): %7.1e\n", nnz(F) / nnz(A))

# Solve Ax = b with DQGMRES and an ILU(0) preconditioner
# Remark: DIOM, FOM and GMRES can be used in the same way
opM = LinearOperator(Float64, n, n, false, false, (y, v) -> forward_substitution!(y, F, v))
opN = LinearOperator(Float64, n, n, false, false, (y, v) -> backward_substitution!(y, F, v))
opP = LinearOperator(Float64, n, n, false, false, (y, v) -> ldiv!(y, F, v))

# Without preconditioning
x, stats = dqgmres(A, b, memory=50, history=true)
r = b - A * x
@printf("[Without preconditioning] Residual norm: %8.1e\n", norm(r))
@printf("[Without preconditioning] Number of iterations: %3d\n", length(stats.residuals) - 1)

# Split preconditioning
x, stats = dqgmres(A, b, memory=50, history=true, M=opM, N=opN)
r = b - A * x
@printf("[Split preconditioning] Residual norm: %8.1e\n", norm(r))
@printf("[Split preconditioning] Number of iterations: %3d\n", length(stats.residuals) - 1)

# Left preconditioning
x, stats = dqgmres(A, b, memory=50, history=true, M=opP)
r = b - A * x
@printf("[Left preconditioning] Residual norm: %8.1e\n", norm(r))
@printf("[Left preconditioning] Number of iterations: %3d\n", length(stats.residuals) - 1)

# Right preconditioning
x, stats = dqgmres(A, b, memory=50, history=true, N=opP)
r = b - A * x
@printf("[Right preconditioning] Residual norm: %8.1e\n", norm(r))
@printf("[Right preconditioning] Number of iterations: %3d\n", length(stats.residuals) - 1)
```
