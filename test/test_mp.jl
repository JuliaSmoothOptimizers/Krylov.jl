function test_mp()
  @printf("Tests of multi-precision methods:\n")
  n = 5
  for fn in (:cg, :cgls, :fom)
    @printf("%s ", string(fn))
    for T in (Float16, Float32, Float64, BigFloat)
      M = spdiagm(-1 => ones(T,n-1), 0 => 4*ones(T,n), 1 => ones(T,n-1))
      A = LinearOperator(M)
      b = ones(T, n)
      (x, stats) = @eval $fn($A, $b)
      atol = √eps(T)
      rtol = √eps(T)
      @test norm(A*x - b) ≤ atol + norm(b) * rtol
    end  
    @printf("✔\n")
  end
end

test_mp()
