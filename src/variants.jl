using LinearAlgebra

# Define a generic linear operator with preallocation
function preallocated_LinearOperator(A)
  (n, m) = size(A)
  Ap = zeros(n)
  Atq = zeros(m)
  return LinearOperator(n, m, false, false, p -> mul!(Ap, A, p),
                        q -> mul!(Atq, transpose(A), q), q -> mul!(Atq, transpose(A), q))
end

# Variants where A is a matrix without specific properties
for fn in (:cgls, :cgne, :craig, :craigmr, :crls, :crmr, :lslq, :lsmr, :lsqr, :dqgmres, :gmres, :diom, :cgs)
  @eval begin
    $fn(A :: AbstractMatrix{TA}, b :: AbstractVector{Tb}, args...; kwargs...) where {TA, Tb <: Number} =
      $fn(preallocated_LinearOperator(A), convert(Vector{Float64}, b), args...; kwargs...)
  end
end

# Define a symmetric linear operator with preallocation
function preallocated_symmetric_LinearOperator(A)
  (n, m) = size(A)
  Ap = zeros(n)
  return LinearOperator(n, m, true, true, p -> mul!(Ap, A, p))
end

# Variants where A must be symmetric
for fn in (:cg_lanczos, :cg_lanczos_shift_seq, :cg, :cr, :minres, :symmlq)
  @eval begin
    $fn(A :: AbstractMatrix{TA}, b :: AbstractVector{Tb}, args...; kwargs...) where {TA, Tb <: Number} =
      $fn(preallocated_symmetric_LinearOperator(A), convert(Vector{Float64}, b), args...; kwargs...)
  end
end
