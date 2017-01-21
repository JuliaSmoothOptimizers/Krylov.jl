function check_min_norm(A, b, x; λ=0.0)
  (nrow, ncol) = size(A);
  if λ > 0.0
    AI = [A sqrt(λ)*eye(nrow)];
    xI = [x ; (b-A*x)/sqrt(λ)];
  else
    AI = A;
    xI = x;
  end
  xmin = AI' * ((AI * AI') \ b);
  xmin_norm = norm(xmin);
  @printf("‖x - xmin‖ = %7.1e\n", norm(xI - xmin));
  return (xI, xmin, xmin_norm)
end
