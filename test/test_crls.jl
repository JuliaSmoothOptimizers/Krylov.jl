crls_tol = 1.0e-6;

for npower = 1 : 4
  (b, A, D, HY, HZ, Acond, rnorm) = test(40, 40, 4, npower, 0);  # No regularization.
  x = crls(A, b);
  resid = norm(A' * (A*x - b)) / norm(b)
  @printf("CRLS: Relative residual: %8.1e\n", resid);
  @test(resid <= crls_tol);
end
