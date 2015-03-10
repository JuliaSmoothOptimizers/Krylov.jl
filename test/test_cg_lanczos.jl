function residuals(A, b, shifts, x)
  nshifts = size(shifts, 1);
  r = { (b - A * x[:,i] - shifts[i] * x[:,i]) for i = 1 : nshifts };
  return r;
end

n = 10;
A = spdiagm([1:n]); b = ones(n); shifts=[1:6];

x = cg_lanczos_shift_seq(A, b, shifts, itmax=10);
r = residuals(A, b, shifts, x);
b_norm = norm(b);
resids = map(norm, r) / b_norm;
@printf("Relative residuals with shifts:\n");
for resid in resids
  @printf(" %8.1e", resid);
end
@printf("\n");
