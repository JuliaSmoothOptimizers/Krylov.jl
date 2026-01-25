function test_mp(FC)
  n = 5
  T = real(FC)
  for method in (:cg, :cgls, :usymqr, :cgne, :cgs, :crmr, :cg_lanczos, :dqgmres, :diom, :cr, :gpmr,
                 :lslq, :lsqr, :lsmr, :lnlq, :craig, :bicgstab, :craigmr, :crls, :symmlq, :minres,
                 :bilq, :minres_qlp, :qmr, :usymlq, :tricg, :trimr, :trilqr, :bilqr, :gmres, :fom,
                 :car, :minares, :fgmres, :usymlqr, :cg_lanczos_shift, :cgls_lanczos_shift)

    @testset "$method" begin
      A = spdiagm(-1 => -ones(FC,n-1), 0 => 3*ones(FC,n), 1 => -ones(FC,n-1))
      B = spdiagm(-1 => -ones(FC,n-1), 0 => 5*ones(FC,n), 1 => -ones(FC,n-1))
      b = ones(FC, n)
      c = -ones(FC, n)
      shifts = [-one(T), one(T)]
      if method in (:usymlq, :usymqr)
        x, _ = krylov_solve(Val{method}(), A, b, c)
      elseif method in (:trilqr, :bilqr)
        x, t, _ = krylov_solve(Val{method}(), A, b, c)
      elseif method in (:tricg, :trimr, :usymlqr)
        x, y, _ = krylov_solve(Val{method}(), A, b, c)
      elseif method == :gpmr
        x, y, _ = krylov_solve(Val{method}(), A, B, b, c)
      elseif method in (:lnlq, :craig, :craigmr)
        x, y, _ = krylov_solve(Val{method}(), A, b)
      elseif method in (:cg_lanczos_shift, :cgls_lanczos_shift)
        x, _ = krylov_solve(Val{method}(), A, b, shifts)
      else
        x, _ = krylov_solve(Val{method}(), A, b)
      end
      atol = √eps(T)
      rtol = √eps(T)
      Κ = (T == Float16 ? 10 : 1)
      if method in (:tricg, :trimr)
        @test norm(x + A * y - b) ≤ Κ * (atol + norm([b; c]) * rtol)
        @test norm(A' * x - y - c) ≤ Κ * (atol + norm([b; c]) * rtol)
        @test eltype(y) == FC
      elseif method == :usymlqr
        @test norm(x + A * y - b) ≤ Κ * (atol + norm([b; c]) * rtol)
        @test norm(A' * x - c) ≤ Κ * (atol + norm([b; c]) * rtol)
        @test eltype(y) == FC
      elseif method == :gpmr
        @test norm(x + A * y - b) ≤ Κ * (atol + norm([b; c]) * rtol)
        @test norm(B * x + y - c) ≤ Κ * (atol + norm([b; c]) * rtol)
        @test eltype(y) == FC
      elseif method == :cg_lanczos_shift
        @test norm((A - I) * x[1] - b) ≤ Κ * (atol + norm(b) * rtol)
        @test norm((A + I) * x[2] - b) ≤ Κ * (atol + norm(b) * rtol)
        @test eltype(x) == Vector{FC}
      elseif method == :cgls_lanczos_shift
        @test norm(A' * (b - A * x[1]) + x[1]) ≤ Κ * (atol + norm(A' * b) * rtol)
        @test norm(A' * (b - A * x[2]) - x[2]) ≤ Κ * (atol + norm(A' * b) * rtol)
        @test eltype(x) == Vector{FC}
      else
        @test norm(A * x - b) ≤ Κ * (atol + norm(b) * rtol)
        @test eltype(x) == FC
      end
      if method in (:trilqr, :bilqr)
        @test norm(A' * t - c) ≤ Κ * (atol + norm(c) * rtol)
        @test eltype(t) == FC
      end
      if method in (:lnlq, :craig, :craigmr)
        @test norm(A * A' * y - b) ≤ Κ * (atol + norm(b) * rtol)
        @test eltype(y) == FC
      end
    end
  end
end

@testset "mp" begin
  for FC in (Float16, Float32, Float64, BigFloat, Complex{Float16}, ComplexF32, ComplexF64, Complex{BigFloat})
    @testset "Data Type: $FC" begin
      test_mp(FC)
    end
  end
end
