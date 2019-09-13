using HarwellRutherfordBoeing
using Krylov, LinearOperators
using LinearAlgebra, Printf

# M = HarwellBoeingMatrix("data/illc1033.rra")
M = HarwellBoeingMatrix("data/illc1850.rra")
A = M.matrix
(m, n) = size(A)
@printf("System size: %d rows and %d columns\n", m, n)

# Define a linear operator with preallocation.
Ap = zeros(m)
Aᵀq = zeros(n)
op = LinearOperator(m, n, false, false,
                    p -> mul!(Ap, A, p),
                    q -> mul!(Aᵀq, transpose(A), q),
                    q -> mul!(Aᵀq, transpose(A), q))
λ = 1.0e-3

for nrhs = 1 : size(M.rhs, 2)
  b = M.rhs[:,nrhs]
  (x, stats) = crls(op, b, λ=λ)
  show(stats)
  resid = norm(A' * (A * x - b) + λ * x) / norm(b)
  @printf("CRLS: Relative residual: %8.1e\n", resid)
  @printf("CRLS: ‖x‖: %8.1e\n", norm(x))
end
