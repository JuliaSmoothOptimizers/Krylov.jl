function test_mp()
  @printf("Tests of multi-precision methods:\n")
  n = 5
  for fn in (:cg, :cgls, :usymqr, :cgne, :cgs, :crmr, :cg_lanczos,
             :dqgmres, :diom, :cr, :lslq, :lsqr, :lsmr, :lnlq, :craig, :bicgstab,
             :craigmr, :crls, :symmlq, :minres, :cg_lanczos_shift_seq,
             :bilq, :minres_qlp, :qmr, :usymlq, :tricg, :trimr, :trilqr, :bilqr)
    @printf("%s ", string(fn))
    for T in (Float16, Float32, Float64, BigFloat)
      A = spdiagm(-1 => -ones(T,n-1), 0 => 3*ones(T,n), 1 => -ones(T,n-1))
      b = ones(T, n)
      c = - ones(T, n)
      λ = zero(T)
      if fn == :cg_lanczos_shift_seq
        shifts = [λ]
        xs = @eval $fn($A, $b, $shifts, history=true)[1]
        x = xs[1]
      elseif fn in (:usymlq, :usymqr)
        x = @eval $fn($A, $b, $c, history=true)[1]
      elseif fn in (:trilqr, :bilqr)
        x, t = @eval $fn($A, $b, $c, history=true)[1:2]
      elseif fn in (:tricg, :trimr)
        x, y = @eval $fn($A, $b, $c, history=true)[1:2]
      elseif fn in (:lnlq, :craig, :craigmr)
        x, y = @eval $fn($A, $b, history=true)[1:2]
      else
        x = @eval $fn($A, $b, history=true)[1]
      end
      atol = √eps(T)
      rtol = √eps(T)
      Κ = (T == Float16 ? 10 : 1)
      @test eltype(x) == T
      if fn in (:tricg, :trimr)
        @test norm(x + A * y - b) ≤ Κ * (atol + norm([b; c]) * rtol)
        @test norm(A' * x - y - c) ≤ Κ * (atol + norm([b; c]) * rtol)
        @test eltype(y) == T
      else
        @test norm(A * x - b) ≤ Κ * (atol + norm(b) * rtol)
      end
      if fn in (:trilqr, :bilqr)
        @test norm(A' * t - c) ≤ Κ * (atol + norm(c) * rtol)
        @test eltype(t) == T
      end
      if fn in (:lnlq, :craig, :craigmr)
        @test norm(A * A' * y - b) ≤ Κ * (atol + norm(b) * rtol)
        @test eltype(y) == T
      end
    end
    @printf("✔\n")
  end
  @printf("\n")
end

test_mp()
