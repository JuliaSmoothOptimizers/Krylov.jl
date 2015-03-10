using HarwellRutherfordBoeing
using Krylov
# using ProfileView

# M = HarwellBoeingMatrix("data/illc1033.rra");
M = HarwellBoeingMatrix("data/illc1850.rra");
A = M.matrix;

for nrhs = 1 : size(M.rhs, 2)
  b = M.rhs[:,nrhs];
  x = crls(A, b);
  @time x = crls(A, b);
#   @profile x = crls(A, b);
  @printf("CRLS: Relative residual: %8.1e\n", norm(A' * (A * x - b)) / norm(b));
end

# ProfileView.view()
