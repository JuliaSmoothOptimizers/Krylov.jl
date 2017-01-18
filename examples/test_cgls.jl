using HarwellRutherfordBoeing
using Krylov
using LinearOperators
# using ProfileView

# M = HarwellBoeingMatrix("data/illc1033.rra");
M = HarwellBoeingMatrix("data/illc1850.rra");
A = M.matrix;
(m, n) = size(A);
@printf("System size: %d rows and %d columns\n", m, n);

# Define a linear operator with preallocation.
Ap = zeros(m);
Atq = zeros(n);
op = LinearOperator(m, n, false, false,
                    p -> A_mul_B!(1.0,  A, p, 0.0, Ap),
                    q -> At_mul_B!(1.0, A, q, 0.0, Atq),
                    q -> At_mul_B!(1.0, A, q, 0.0, Atq));
λ = 1.0e-3;

for nrhs = 1 : size(M.rhs, 2)
  b = M.rhs[:,nrhs];
  (x, stats) = cgls(op, b, λ=λ);
  #   @profile (x, stats) = cgls(op, b, λ=λ);
  @time (x, stats) = cgls(op, b, λ=λ);
  show(stats);
  resid = norm(A' * (A * x - b) + λ * x) / norm(b);
  @printf("CGLS: Relative residual: %8.1e\n", resid);
  @printf("CGLS: ‖x‖: %8.1e\n", norm(x));
end

# ProfileView.view()
