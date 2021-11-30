@testset "mp" begin
  n = 5
  for fn in (:cg, :cgls, :usymqr, :cgne, :cgs, :crmr, :cg_lanczos, :dqgmres, :diom, :cr, :gpmr,
             :lslq, :lsqr, :lsmr, :lnlq, :craig, :bicgstab, :craigmr, :crls, :symmlq, :minres,
             :bilq, :minres_qlp, :qmr, :usymlq, :tricg, :trimr, :trilqr, :bilqr, :gmres, :fom)
    for T in (Float16, Float32, Float64, BigFloat)
      A = spdiagm(-1 => -ones(T,n-1), 0 => 3*ones(T,n), 1 => -ones(T,n-1))
      B = spdiagm(-1 => -ones(T,n-1), 0 => 5*ones(T,n), 1 => -ones(T,n-1))
      b = ones(T, n)
      c = - ones(T, n)
      shifts = [-one(T), one(T)]
      if fn in (:usymlq, :usymqr)
        x, _ = @eval $fn($A, $b, $c)
      elseif fn in (:trilqr, :bilqr)
        x, t, _ = @eval $fn($A, $b, $c)
      elseif fn in (:tricg, :trimr)
        x, y, _ = @eval $fn($A, $b, $c)
      elseif fn == :gpmr
        x, y, _ = @eval $fn($A, $B, $b, $c)
      elseif fn in (:lnlq, :craig, :craigmr)
        x, y, _ = @eval $fn($A, $b)
      else
        x, _ = @eval $fn($A, $b)
        if fn == :cg_lanczos
          xs, _ = @eval $fn($A, $b, $shifts)
        end
      end
      atol = √eps(T)
      rtol = √eps(T)
      Κ = (T == Float16 ? 10 : 1)
      @test eltype(x) == T
      if fn in (:tricg, :trimr)
        @test norm(x + A * y - b) ≤ Κ * (atol + norm([b; c]) * rtol)
        @test norm(A' * x - y - c) ≤ Κ * (atol + norm([b; c]) * rtol)
        @test eltype(y) == T
      elseif fn == :gpmr
        @test norm(x + A * y - b) ≤ Κ * (atol + norm([b; c]) * rtol)
        @test norm(B * x + y - c) ≤ Κ * (atol + norm([b; c]) * rtol)
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
      if fn == :cg_lanczos
        @test norm((A - I) * xs[1] - b) ≤ Κ * (atol + norm(b) * rtol)
        @test norm((A + I) * xs[2] - b) ≤ Κ * (atol + norm(b) * rtol)
        @test eltype(xs) == Vector{T}
      end
    end
  end

  A = get_div_grad(4, 4, 4)
  B = A'
  b = ones(64)
  c = -ones(64)
  P = jacobi(A, T=Float32)
  @test eltype(A) ≠ eltype(P)
  @test promote_type(eltype(A), eltype(P)) == eltype(A)
  shifts = [-1.0, 1.0]
  for fn in (:cg, :cgls, :cgne, :cgs, :crmr, :cg_lanczos, :dqgmres, :diom, :cr, :gpmr, :lslq, :lsqr, :lsmr, :lnlq,
             :craig, :bicgstab, :craigmr, :crls, :symmlq, :minres, :minres_qlp, :tricg, :trimr, :gmres, :fom)
    if fn in (:tricg, :trimr)
      @eval $fn($A, $b, $c, M=$P, N=$P)
    elseif fn == :gpmr
      @eval $fn($A, $B, $b, $c, C=$P, D=$P, E=$P, F=$P)
    elseif fn in (:lnlq, :craig, :craigmr, :lnlq, :craig, :craigmr, :cgs, :bicgstab, :diom, :dqgmres, :fom, :gmres)
      @eval $fn($A, $b, M=$P, N=$P)
    else
      @eval $fn($A, $b, M=$P)
      if fn == :cg_lanczos
        @eval $fn($A, $b, $shifts, M=$P)
      end
    end
  end
end
