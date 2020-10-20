using Krylov, LinearOperators, ILU0
using LinearAlgebra, Printf

n = 64
N = n^3
A = spdiagm(
      -1 => fill(-1.0, n - 1), 
       0 => fill(3.0, n), 
       1 => fill(-2.0, n - 1)
    )
Id = sparse(1.0I, n, n)
A = kron(A, Id) + kron(Id, A)
A = kron(A, Id) + kron(Id, A)
x = ones(N)
b = A * x
F = ilu0(A)

@printf("nnz(ILU) / nnz(A): %7.1e\n", nnz(F) / nnz(A))

# Solve Ax = b with DQGMRES and an ILU(0) preconditioner
# Remark: DIOM can be used in the same way
yM = zeros(N)
yN = zeros(N)
yP = zeros(N)
opM = LinearOperator(Float64, N, N, false, false, y -> forward_substitution(yM, F, y))
opN = LinearOperator(Float64, N, N, false, false, y -> backward_substitution(yN, F, y))
opP = LinearOperator(Float64, N, N, false, false, y -> ldiv!(yP, F, y))

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
