using HarwellRutherfordBoeing
using Krylov
# using ProfileView

# M = HarwellBoeingMatrix("data/illc1033.rra");
M = HarwellBoeingMatrix("data/illc1850.rra");
A = M.matrix;

for nrhs = 1 : size(M.rhs, 2)
  b = M.rhs[:,nrhs];
  (x, rNorms, ArNorms) = crls(A, b);
#   @profile (x, rNorms, ArNorms) = crls(A, b);
  @time (x, rNorms, ArNorms) = crls(A, b);
  resid = norm(A' * (A * x - b)) / norm(b);
  @printf("CRLS: Relative residual: %8.1e\n", resid);
end

# ProfileView.view()
