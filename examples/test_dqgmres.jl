using Krylov, LinearOperators, ILUZero
using LinearAlgebra, Printf

include("../test/test_utils.jl")

A, b = polar_poisson()
# A, b = kron_unsymmetric()
n = length(b)
F = ilu0(A)

@printf("nnz(ILU) / nnz(A): %7.1e\n", nnz(F) / nnz(A))

# Solve Ax = b with DQGMRES and an ILU(0) preconditioner
# Remark: DIOM can be used in the same way
yM = zeros(n)
yN = zeros(n)
yP = zeros(n)
opM = LinearOperator(Float64, n, n, false, false, y -> forward_substitution!(yM, F, y))
opN = LinearOperator(Float64, n, n, false, false, y -> backward_substitution!(yN, F, y))
opP = LinearOperator(Float64, n, n, false, false, y -> ldiv!(yP, F, y))

# Without preconditioning
x, stats = dqgmres(A, b)
r = b - A * x
@printf("[Without preconditioning] Residual norm: %8.1e\n", norm(r))
@printf("[Without preconditioning] Number of iterations: %3d\n", length(stats.residuals) - 1)

# Split preconditioning
x, stats = dqgmres(A, b, M=opM, N=opN)
r = b - A * x
@printf("[Split preconditioning] Residual norm: %8.1e\n", norm(r))
@printf("[Split preconditioning] Number of iterations: %3d\n", length(stats.residuals) - 1)

# Left preconditioning
x, stats = dqgmres(A, b, M=opP)
r = b - A * x
@printf("[Left preconditioning] Residual norm: %8.1e\n", norm(r))
@printf("[Left preconditioning] Number of iterations: %3d\n", length(stats.residuals) - 1)

# Right preconditioning
x, stats = dqgmres(A, b, N=opP)
r = b - A * x
@printf("[Right preconditioning] Residual norm: %8.1e\n", norm(r))
@printf("[Right preconditioning] Number of iterations: %3d\n", length(stats.residuals) - 1)
