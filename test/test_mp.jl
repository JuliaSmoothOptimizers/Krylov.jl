function test_mp()
  @printf("Tests of multi-precision methods:\n")
  n = 5
  for fn in (:cg, :cgls, :usymqr, :cgne, :cgs, :crmr, :cg_lanczos,
             :dqgmres, :diom, :cr, :lslq, :lsqr, :lsmr, :craig,
             :craigmr, :crls, :symmlq)
    @printf("%10s ", string(fn))
    for T in (Float16, Float32, Float64, BigFloat)
      M = spdiagm(-1 => -ones(T,n-1), 0 => 3*ones(T,n), 1 => -ones(T,n-1))
      A = LinearOperator(M)
      b = ones(T, n)
      c = - ones(T, n)
      atol = √eps(T)
      rtol = √eps(T)
      if fn == :usymqr
        (x, stats) = @eval $fn($A, $b, $c)
      else
        (x, stats) = @eval $fn($A, $b)
      end
      @test norm(A * x - b) ≤ atol + norm(b) * rtol
    end
    @printf("✔\n")
  end
end

test_mp()
