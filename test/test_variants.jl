# Tests of variants.jl
function test_variants()
  @printf("\nTests of variants:\n")
  for fn in (:cg_lanczos, :cg_lanczos_shift_seq, :cg, :cgls, :cgne,
             :cr, :craig, :craigmr, :crls, :crmr, :lslq, :lsmr, :bilq,
             :lsqr, :minres, :symmlq, :dqgmres, :diom, :cgs, :usymqr,
             :minres_qlp)
    @printf("%s ", string(fn))
    for TA in (Int32, Int64, Float32, Float64)
      for IA in (Int32, Int64)
        for Tb in (Int32, Int64, Float32, Float64)
          for Ib in (Int32, Int64)
            A_dense = Matrix{TA}(I, 5, 5)
            A_sparse = convert(SparseMatrixCSC{TA,IA}, A_dense)
            b_dense = ones(Tb,5)
            b_sparse = convert(SparseVector{Tb,Ib}, b_dense)
            if fn == :cg_lanczos_shift_seq
              shifts = [1:5;]
              @eval $fn($A_dense,  $b_dense,  $shifts)
              @eval $fn($A_dense,  $b_sparse, $shifts)
              @eval $fn($A_sparse, $b_dense,  $shifts)
              @eval $fn($A_sparse, $b_sparse, $shifts)
              @eval $fn(transpose($A_dense),  $b_dense,  $shifts)
              @eval $fn(transpose($A_dense),  $b_sparse, $shifts)
              @eval $fn(transpose($A_sparse), $b_dense,  $shifts)
              @eval $fn(transpose($A_sparse), $b_sparse, $shifts)
              @eval $fn(adjoint($A_dense),  $b_dense,  $shifts)
              @eval $fn(adjoint($A_dense),  $b_sparse, $shifts)
              @eval $fn(adjoint($A_sparse), $b_dense,  $shifts)
              @eval $fn(adjoint($A_sparse), $b_sparse, $shifts)
            elseif fn == :usymqr
              c_dense = ones(Tb,5)
              c_sparse = convert(SparseVector{Tb,Ib}, c_dense)
              @eval $fn($A_dense,  $b_dense,  $c_dense )
              @eval $fn($A_dense,  $b_dense,  $c_sparse)
              @eval $fn($A_dense,  $b_sparse, $c_dense )
              @eval $fn($A_dense,  $b_sparse, $c_sparse)
              @eval $fn($A_sparse, $b_dense,  $c_dense )
              @eval $fn($A_sparse, $b_dense,  $c_sparse)
              @eval $fn($A_sparse, $b_sparse, $c_dense )
              @eval $fn($A_sparse, $b_sparse, $c_sparse)
              @eval $fn(transpose($A_dense),  $b_dense,  $c_dense )
              @eval $fn(transpose($A_dense),  $b_dense,  $c_sparse)
              @eval $fn(transpose($A_dense),  $b_sparse, $c_dense )
              @eval $fn(transpose($A_dense),  $b_sparse, $c_sparse)
              @eval $fn(transpose($A_sparse), $b_dense,  $c_dense )
              @eval $fn(transpose($A_sparse), $b_dense,  $c_sparse)
              @eval $fn(transpose($A_sparse), $b_sparse, $c_dense )
              @eval $fn(transpose($A_sparse), $b_sparse, $c_sparse)
              @eval $fn(adjoint($A_dense),  $b_dense,  $c_dense )
              @eval $fn(adjoint($A_dense),  $b_dense,  $c_sparse)
              @eval $fn(adjoint($A_dense),  $b_sparse, $c_dense )
              @eval $fn(adjoint($A_dense),  $b_sparse, $c_sparse)
              @eval $fn(adjoint($A_sparse), $b_dense,  $c_dense )
              @eval $fn(adjoint($A_sparse), $b_dense,  $c_sparse)
              @eval $fn(adjoint($A_sparse), $b_sparse, $c_dense )
              @eval $fn(adjoint($A_sparse), $b_sparse, $c_sparse)
            else
              @eval $fn($A_dense,  $b_dense )
              @eval $fn($A_dense,  $b_sparse)
              @eval $fn($A_sparse, $b_dense )
              @eval $fn($A_sparse, $b_sparse)
              @eval $fn(transpose($A_dense),  $b_dense )
              @eval $fn(transpose($A_dense),  $b_sparse)
              @eval $fn(transpose($A_sparse), $b_dense )
              @eval $fn(transpose($A_sparse), $b_sparse)
              @eval $fn(adjoint($A_dense),  $b_dense )
              @eval $fn(adjoint($A_dense),  $b_sparse)
              @eval $fn(adjoint($A_sparse), $b_dense )
              @eval $fn(adjoint($A_sparse), $b_sparse)
            end
          end
        end
      end
    end
    @printf("✔\n")
  end
end

test_variants()
