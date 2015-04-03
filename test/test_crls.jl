crls_tol = 1.0e-6;

for npower = 1 : 4
  (b, A, D, HY, HZ, Acond, rnorm) = test(40, 40, 4, npower, 0);  # No regularization.

  for mat in {sparse(full(A)), full(A), A}
    (x, stats) = crls(mat, b);
    show(stats);
    resid = norm(A' * (A*x - b)) / norm(b)
    @printf("CRLS: Relative residual: %8.1e\n", resid);
    @test(resid <= crls_tol);
    @test(stats.solved);
  end

  λ = 1.0e-3;
  for mat in {sparse(full(A)), full(A), A}
    (x, stats) = crls(mat, b, λ=λ);
    show(stats);
    resid = norm(A' * (A*x - b) + λ * x) / norm(b)
    @printf("CRLS: Relative residual: %8.1e\n", resid);
    @test(resid <= crls_tol);
    @test(stats.solved);
  end
end
