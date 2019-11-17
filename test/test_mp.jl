function test_mp()
  @printf("Tests of multi-precision methods:\n")
  n = 5
  for fn in (:cg, :cgls, :usymqr, :cgne, :cgs, :crmr, :cg_lanczos,
             :dqgmres, :diom, :cr, :lslq, :lsqr, :lsmr, :craig,
             :craigmr, :crls, :symmlq, :minres, :cg_lanczos_shift_seq,
             :bilq, :minres_qlp, :qmr, :usymlq, :trilqr, :bilqr)
    @printf("%s ", string(fn))
    for T in (Float16, Float32, Float64, BigFloat)
      A = spdiagm(-1 => -ones(T,n-1), 0 => 3*ones(T,n), 1 => -ones(T,n-1))
      b = ones(T, n)
      c = - ones(T, n)
      λ = zero(T)
      if fn == :cg_lanczos_shift_seq
        shifts = [λ]
        xs = @eval $fn($A, $b, $shifts)[1]
        x = xs[1]
      elseif fn in (:usymlq, :usymqr)
        x = @eval $fn($A, $b, $c)[1]
      elseif fn in (:trilqr, :bilqr)
        x, t = @eval $fn($A, $b, $c)[1:2]
      else
        x = @eval $fn($A, $b)[1]
      end
      atol = √eps(T)
      rtol = √eps(T)
      if T == Float16
        @test norm(A * x - b) ≤ 10 * (atol + norm(b) * rtol)
        if fn in (:trilqr, :bilqr)
          @test norm(A' * t - c) ≤ 10 * (atol + norm(c) * rtol)
          @test eltype(t) == T
        end
      else
        @test norm(A * x - b) ≤ atol + norm(b) * rtol
        if fn in (:trilqr, :bilqr)
          @test norm(A' * t - c) ≤ atol + norm(c) * rtol
          @test eltype(t) == T
        end
      end
      @test eltype(x) == T
    end
    @printf("✔\n")
  end
  @printf("\n")
end

test_mp()
