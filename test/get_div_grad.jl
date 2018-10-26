# Based on Lars Ruthotto's initial implementation.
function get_div_grad(n1 :: Int, n2 :: Int, n3 :: Int)

  # Divergence
  D1 = kron(Matrix(1.0I, n3, n3), kron(Matrix(1.0I, n2, n2), ddx(n1)))
  D2 = kron(Matrix(1.0I, n3, n3), kron(ddx(n2),   Matrix(1.0I, n1, n1)))
  D3 = kron(ddx(n3),   kron(Matrix(1.0I, n2, n2), Matrix(1.0I, n1, n1)))

  # DIV from faces to cell-centers
  Div = [D1 D2 D3]

  return Div * Div';
end

# 1D finite difference on staggered grid
function ddx(n :: Int)
  e = ones(n);
  return d = sparse([1:n; 1:n], [1:n; 2:n+1], [-e; e])
end

