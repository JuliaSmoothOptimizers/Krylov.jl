for npower = 1 : 4
  (b, A, D, HY, HZ, Acond, rnorm) = test(40, 40, 4, npower, 0);  # No regularization.
  x = crls(A, b);
  @printf("CRLS: Relative residual: %8.1e\n", norm(A' * (A*x - b)) / norm(b));
end
