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
λ > 0.0 && (N = 1.0 / λ * opEye(n))

for nrhs = 1 : size(M.rhs, 2)
  b = M.rhs[:,nrhs]
  (x, stats) = lsmr(op, b, λ=λ, sqd=λ > 0, atol=0.0, btol=0.0, N=N)
  show(stats)
  resid = norm(A' * (A * x - b) + λ * x) / norm(b)
  @printf("LSMR: Relative residual: %8.1e\n", resid)
  @printf("LSMR: ‖x‖: %8.1e\n", norm(x))
end
