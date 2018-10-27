using FastClosures

function preallocated_LinearOperator(A)
  (n, m) = size(A)
  Ap = zeros(n)
  prod = @closure p -> mul!(Ap, A, p)
  Atq = zeros(m)
  tprod = @closure q -> mul!(Atq, transpose(A), q)
  T = eltype(A)
  F1 = typeof(prod)
  F2 = typeof(tprod)
  LinearOperator{T,F1,F2,F2}(n, m, false, false, prod, tprod, tprod)
end

function nonallocating_opEye(n)
  prod = @closure v -> v
  F = typeof(prod)
  LinearOperator{Float64,F,F,F}(n, n, true, true, prod, prod, prod)
end
