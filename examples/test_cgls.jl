using HarwellRutherfordBoeing
using Krylov
# using ProfileView

# M = HarwellBoeingMatrix("data/illc1033.rra");
M = HarwellBoeingMatrix("data/illc1850.rra");
A = M.matrix;

for nrhs = 1 : size(M.rhs, 2)
  b = M.rhs[:,nrhs];
  x = cgls(A, b);
#   @profile x = cgls(A, b, verbose=false);
  @time x = cgls(A, b);
  @printf("CGLS: Relative residual: %8.1e\n", norm(A' * (A * x - b)) / norm(b));
end

# ProfileView.view()
