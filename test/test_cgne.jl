cgne_tol = 1.0e-6;

function test_cgne(nrow :: Int, ncol :: Int; λ :: Float64=0.0)
  A = rand(nrow, ncol); b = A * ones(ncol);
  stats = Nothing;
  for mat in {A, sparse(A), LinearOperator(A)}
    (x, stats) = cgne(A, b, λ=λ);
  end
  r = b - A * x;
  if λ > 0
    s = r / sqrt(λ);
    r = r - sqrt(λ) * s;
  end
  resid = norm(r) / norm(b);
  show(stats);
  @printf("CGNE: residual: %7.1e\n", resid);
  @test(resid <= cgne_tol);
  @test(stats.solved);

  # See if we have the minimum-norm solution.
  if λ > 0
    A = [A sqrt(λ)*eye(nrow)];
    x = [x ; s];
  end
  xmin = A' * ((A * A') \ b);
  xmin_norm = norm(xmin);
  @printf("CGNE: ‖x‖ = %7.1e, ‖xmin‖ = %7.1e\n", norm(x), xmin_norm);
  @test(norm(x - xmin) <= cgne_tol * xmin_norm);
end

for λ in {0.0, 1.0e-2}
  test_cgne(10, 25, λ=λ);  # Underdetermined.
  test_cgne(10, 10, λ=λ);  # Square.
  test_cgne(25, 10, λ=λ);  # Overdetermined.
end

