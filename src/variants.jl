# Define a generic linear operator with preallocation
function preallocated_LinearOperator(A)
  (n, m) = size(A)
  Ap = zeros(n)
  Atq = zeros(m)
  return LinearOperator(n, m, false, false, p -> A_mul_B!(Ap, A, p), q -> At_mul_B!(Atq, A, q), q -> At_mul_B!(Atq, A, q))
end

# Variants where A is a matrix without specific properties
for fn in (:cgls, :cgne, :craig, :craigmr, :crls, :crmr, :lslq, :lsmr, :lsqr, :dqgmres, :diom, :cgs)
  @eval begin
    # Variant for A given as a dense array and b given as a dense vector
    $fn{TA <: Number, Tb <: Number}(A :: Array{TA,2}, b :: Vector{Tb}, args...; kwargs...) =
      $fn(preallocated_LinearOperator(A), convert(Vector{Float64}, b), args...; kwargs...)

    # Variant for A given as a sparse matrix and b given as a dense vector
    $fn{TA <: Number, Tb <: Number, IA <: Integer}(A :: SparseMatrixCSC{TA,IA}, b :: Array{Tb,1}, args...; kwargs...) =
      $fn(preallocated_LinearOperator(A), convert(Vector{Float64}, b), args...; kwargs...)

    # Variant for A given as a dense array and b given as a sparse vector
    $fn{TA <: Number, Tb <: Number, Ib <: Integer}(A :: Array{TA,2}, b :: SparseVector{Tb,Ib}, args...; kwargs...) =
      $fn(preallocated_LinearOperator(A), convert(Vector{Float64}, b), args...; kwargs...)

    # Variant for A given as a sparse matrix and b given as a sparse vector
    $fn{TA <: Number, Tb <: Number, IA <: Integer, Ib <: Integer}(A :: SparseMatrixCSC{TA,IA}, b :: SparseVector{Tb,Ib}, args...; kwargs...) =
      $fn(preallocated_LinearOperator(A), convert(Vector{Float64}, b), args...; kwargs...)
  end
end

# Define a symmetric linear operator with preallocation
function preallocated_symmetric_LinearOperator(A)
  (n, m) = size(A)
  Ap = zeros(n)
  return LinearOperator(n, m, true, true, p -> A_mul_B!(Ap, A, p))
end

# Variants where A must be symmetric
for fn in (:cg_lanczos, :cg_lanczos_shift_seq, :cg, :cr, :minres, :symmlq)
  @eval begin
    # Variant for A given as a dense array and b given as a dense vector
    $fn{TA <: Number, Tb <: Number}(A :: Array{TA,2}, b :: Vector{Tb}, args...; kwargs...) =
      $fn(preallocated_symmetric_LinearOperator(A), convert(Vector{Float64}, b), args...; kwargs...)

    # Variant for A given as a sparse matrix and b given as a dense vector
    $fn{TA <: Number, Tb <: Number, IA <: Integer}(A :: SparseMatrixCSC{TA,IA}, b :: Array{Tb,1}, args...; kwargs...) =
      $fn(preallocated_symmetric_LinearOperator(A), convert(Vector{Float64}, b), args...; kwargs...)

    # Variant for A given as a dense array and b given as a sparse vector
    $fn{TA <: Number, Tb <: Number, Ib <: Integer}(A :: Array{TA,2}, b :: SparseVector{Tb,Ib}, args...; kwargs...) =
      $fn(preallocated_symmetric_LinearOperator(A), convert(Vector{Float64}, b), args...; kwargs...)

    # Variant for A given as a sparse matrix and b given as a sparse vector
    $fn{TA <: Number, Tb <: Number, IA <: Integer, Ib <: Integer}(A :: SparseMatrixCSC{TA,IA}, b :: SparseVector{Tb,Ib}, args...; kwargs...) =
      $fn(preallocated_symmetric_LinearOperator(A), convert(Vector{Float64}, b), args...; kwargs...)
  end
end