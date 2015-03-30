using HarwellRutherfordBoeing
using Krylov
using LinearOperators
# using ProfileView

# M = HarwellBoeingMatrix("data/illc1033.rra");
M = HarwellBoeingMatrix("data/illc1850.rra");
A = M.matrix;
(m, n) = size(A);

# Define a linear operator with preallocation.
Ap = zeros(m);
Atq = zeros(n);
op = LinearOperator(m, n, Float64, false, false,
                    p -> A_mul_B!(1.0,  A, p, 0.0, Ap),
                    q -> Ac_mul_B!(1.0, A, q, 0.0, Atq),
                    q -> Ac_mul_B!(1.0, A, q, 0.0, Atq));

for nrhs = 1 : size(M.rhs, 2)
  b = M.rhs[:,nrhs];
  (x, stats) = cgls(op, b);
  #   @profile (x, stats) = cgls(op, b, verbose=false);
  @time (x, stats) = cgls(op, b);
  @printf("CGLS: Relative residual: %8.1e\n", norm(A' * (A * x - b)) / norm(b));
end

# ProfileView.view()
