# Tests of variants.jl
for fn in (:cg_lanczos, :cg, :cgls, :cgne, :cr, :craig, :craigmr, :crls, :crmr, :lslq, :lsmr, :lsqr, :minres, :symmlq)
  for TA in (Int32, Int64, Float32, Float64)
    for IA in (Int32, Int64)
      for Tb in (Int32, Int64, Float32, Float64)
        for Ib in (Int32, Int64)
          A_dense = eye(TA,5)
          A_sparse = convert(SparseMatrixCSC{TA,IA}, A_dense)
          b_dense = ones(Tb,5)
          b_sparse = convert(SparseVector{Tb,Ib}, b_dense)
          @eval $fn($A_dense, $b_dense, verbose=false)
          @eval $fn($A_dense, $b_sparse, verbose=false)
          @eval $fn($A_sparse, $b_dense, verbose=false)
          @eval $fn($A_sparse, $b_sparse, verbose=false)
        end
      end
    end
  end
end
