using HarwellRutherfordBoeing
using Krylov
using LinearOperators
# using ProfileView

# M = HarwellBoeingMatrix("data/illc1033.rra");
M = HarwellBoeingMatrix("data/illc1850.rra");
A = M.matrix;
(m, n) = size(A);
@printf("System size: %d rows and %d columns\n", n, m);  # We use A'.

# Define a linear operator with preallocation.
Ap = zeros(m);
Atq = zeros(n);
op = LinearOperator(m, n, false, false,
                    p -> A_mul_B!(1.0,  A, p, 0.0, Ap),
                    q -> At_mul_B!(1.0, A, q, 0.0, Atq),
                    q -> At_mul_B!(1.0, A, q, 0.0, Atq));

x_exact = A * ones(n);  # y = ones(n);
x_exact_norm = norm(x_exact);
x_exact /= x_exact_norm;
b = A' * x_exact;
(x, stats) = cgne(op', b);
# @profile (x, stats) = cgne(op', b);
@time (x, stats) = cgne(op', b);
show(stats);
resid = norm(A' * x - b) / norm(b);
@printf("CGNE: Relative residual: %7.1e\n", resid);
@printf("CGNE: ‖x - x*‖₂: %7.1e\n", norm(x - x_exact));

# ProfileView.view()
