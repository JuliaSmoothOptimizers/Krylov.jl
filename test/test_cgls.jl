for npower = 1 : 4
  (b, A, D, HY, HZ, Acond, rnorm) = test(40, 40, 4, npower, 0);  # No regularization.
  x = cgls(A, b);
  @printf("CGLS: Relative residual: %8.1e\n", norm(A' * (A*x - b)) / norm(b));
end
