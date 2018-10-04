# Tests of variants.jl
for fn in (:cg_lanczos, :cg_lanczos_shift_seq, :cg, :cgls, :cgne, :cr, :craig, :craigmr, :crls, :crmr, :lslq, :lsmr, :lsqr, :minres, :symmlq, :dqgmres, :diom)
  for TA in (Int32, Int64, Float32, Float64)
    for IA in (Int32, Int64)
      for Tb in (Int32, Int64, Float32, Float64)
        for Ib in (Int32, Int64)
          A_dense = eye(TA,5)
          A_sparse = convert(SparseMatrixCSC{TA,IA}, A_dense)
          b_dense = ones(Tb,5)
          b_sparse = convert(SparseVector{Tb,Ib}, b_dense)
          if fn == :cg_lanczos_shift_seq
            shifts = [1:5;]
            @eval $fn($A_dense, $b_dense, $shifts, verbose=false)
            @eval $fn($A_dense, $b_sparse, $shifts, verbose=false)
            @eval $fn($A_sparse, $b_dense, $shifts, verbose=false)
            @eval $fn($A_sparse, $b_sparse, $shifts, verbose=false)
          else
            @eval $fn($A_dense, $b_dense, verbose=false)
            @eval $fn($A_dense, $b_sparse, verbose=false)
            @eval $fn($A_sparse, $b_dense, verbose=false)
            @eval $fn($A_sparse, $b_sparse, verbose=false)
          end
        end
      end
    end
  end
end