function check_min_norm(A, b, x; λ=0.0)
  (nrow, ncol) = size(A)
  if λ > 0.0
    AI = [A sqrt(λ)*I]
    xI = [x ; (b-A*x)/sqrt(λ)]
  else
    AI = A
    xI = x
  end
  QR = qr(AI')
  xmin = QR.Q * (QR.R' \ b)
  xmin_norm = norm(xmin)
  return (xI, xmin, xmin_norm)
end
