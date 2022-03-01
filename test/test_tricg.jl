@testset "tricg" begin
  tricg_tol = 1.0e-6

  for FC in (Float64, ComplexF64)
    @testset "Data Type: $FC" begin

      # Test saddle-point systems
      A, b, D = saddle_point(FC=FC)
      m, n = size(A)
      c = -b
      D⁻¹ = sparse(inv(D))
      N⁻¹ = eye(n)
      H⁻¹ = blockdiag(D⁻¹, N⁻¹)

      (x, y, stats) = tricg(A, b, c, τ=1.0, ν=0.0, M=D⁻¹)
      K = [D A; A' zeros(n, n)]
      B = [b; c]
      r =  B - K * [x; y]
      resid = sqrt(dot(r, H⁻¹ * r)) / sqrt(dot(B, H⁻¹ * B))
      @test(resid ≤ tricg_tol)

      (x, y, stats) = tricg(A, b, c, τ=1.0, ν=0.0)
      K = [eye(m) A; A' zeros(n, n)]
      B = [b; c]
      r =  B - K * [x; y]
      resid = norm(r) / norm(B)
      @test(resid ≤ tricg_tol)

      # Test symmetric and quasi-definite systems
      A, b, M, N = sqd(FC=FC)
      m, n = size(A)
      c = -b
      M⁻¹ = sparse(inv(M))
      N⁻¹ = sparse(inv(N))
      H⁻¹ = blockdiag(M⁻¹, N⁻¹)

      (x, y, stats) = tricg(A, b, c, M=M⁻¹, N=N⁻¹)
      K = [M A; A' -N]
      B = [b; c]
      r =  B - K * [x; y]
      resid = sqrt(dot(r, H⁻¹ * r)) / sqrt(dot(B, H⁻¹ * B))
      @test(resid ≤ tricg_tol)

      (x, y, stats) = tricg(A, b, c)
      K = [eye(m) A; A' -eye(n)]
      B = [b; c]
      r =  B - K * [x; y]
      resid = norm(r) / norm(B)
      @test(resid ≤ tricg_tol)

      (x, y, stats) = tricg(A, b, c, M=M⁻¹, N=N⁻¹, flip=true)
      K = [-M A; A' N]
      B = [b; c]
      r =  B - K * [x; y]
      resid = sqrt(dot(r, H⁻¹ * r)) / sqrt(dot(B, H⁻¹ * B))
      @test(resid ≤ tricg_tol)

      (x, y, stats) = tricg(A, b, c, flip=true)
      K = [-eye(m) A; A' eye(n)]
      B = [b; c]
      r =  B - K * [x; y]
      resid = norm(r) / norm(B)
      @test(resid ≤ tricg_tol)

      τ = 12.0; ν =-0.7
      (x, y, stats) = tricg(A, b, c, M=M⁻¹, N=N⁻¹, τ=τ, ν=ν)
      K = [τ*M A; A' ν*N]
      B = [b; c]
      r =  B - K * [x; y]
      resid = sqrt(dot(r, H⁻¹ * r)) / sqrt(dot(B, H⁻¹ * B))
      @test(resid ≤ tricg_tol)

      (x, y, stats) = tricg(A, b, c, τ=τ, ν=ν)
      K = [τ*eye(m) A; A' ν*eye(n)]
      B = [b; c]
      r =  B - K * [x; y]
      resid = norm(r) / norm(B)
      @test(resid ≤ tricg_tol)

      τ = -1e-6; ν =1e-8
      (x, y, stats) = tricg(A, b, c, M=M⁻¹, N=N⁻¹, τ=τ, ν=ν)
      K = [τ*M A; A' ν*N]
      B = [b; c]
      r =  B - K * [x; y]
      resid = sqrt(dot(r, H⁻¹ * r)) / sqrt(dot(B, H⁻¹ * B))
      @test(resid ≤ tricg_tol)

      (x, y, stats) = tricg(A, b, c, τ=τ, ν=ν)
      K = [τ*eye(m) A; A' ν*eye(n)]
      B = [b; c]
      r =  B - K * [x; y]
      resid = norm(r) / norm(B)
      @test(resid ≤ tricg_tol)

      # Test symmetric positive definite systems
      (x, y, stats) = tricg(A, b, c, M=M⁻¹, N=N⁻¹, spd=true)
      K = [M A; A' N]
      B = [b; c]
      r =  B - K * [x; y]
      resid = sqrt(dot(r, H⁻¹ * r)) / sqrt(dot(B, H⁻¹ * B))
      @test(resid ≤ tricg_tol)

      (x, y, stats) = tricg(A, b, c, spd=true)
      K = [eye(m) A; A' eye(n)]
      B = [b; c]
      r =  B - K * [x; y]
      resid = norm(r) / norm(B)
      @test(resid ≤ tricg_tol)

      # Test symmetric negative definite systems
      (x, y, stats) = tricg(A, b, c, M=M⁻¹, N=N⁻¹, snd=true)
      K = [-M A; A' -N]
      B = [b; c]
      r =  B - K * [x; y]
      resid = sqrt(dot(r, H⁻¹ * r)) / sqrt(dot(B, H⁻¹ * B))
      @test(resid ≤ tricg_tol)

      (x, y, stats) = tricg(A, b, c, snd=true)
      K = [-eye(m) A; A' -eye(n)]
      B = [b; c]
      r =  B - K * [x; y]
      resid = norm(r) / norm(B)
      @test(resid ≤ tricg_tol)

      # Test b=0 or c=0
      c .= 0
      @test_throws ErrorException("c must be nonzero") tricg(A, b, c)
      b .= 0
      @test_throws ErrorException("b must be nonzero") tricg(A, b, c)

      # Test dimension of additional vectors
      for transpose ∈ (false, true)
        A, b, c, M, N = small_sqd(transpose, FC=FC)
        M⁻¹ = inv(M)
        N⁻¹ = inv(N)
        (x, y, stats) = tricg(A, b, c, M=M⁻¹, N=N⁻¹)
      end

      # Test restart
      A, b = restart(FC=FC)
      solver = TricgSolver(A, b)
      tricg!(solver, A, b, b, itmax=20)
      @test !solver.stats.solved
      tricg!(solver, A, b, b, restart=true)
      r = [b - solver.x - A * solver.y; b - A' * solver.x + solver.y]
      resid = norm(r) / norm([b; b])
      @test(resid ≤ tricg_tol)
      @test solver.stats.solved

      for transpose ∈ (false, true)
        A, b, c = ssy_mo_breakdown(transpose)

        # Test breakdowns
        K = [I A; A' -I]
        d = [b; c]
        x, y, stats = tricg(A, b, c)
        r = d - K * [x; y]
        resid = norm(r) / norm(d)
        @test(resid ≤ tricg_tol)

        # Test inconsistent linear systems
        K = [I A; A' I]
        n, m = size(K)
        p = rank(K)
        @test(p < n)
        τ = ν = 1.0
        x, y, stats = tricg(A, b, c, τ=τ, ν=ν)
        @test(stats.inconsistent)
      end
    end
  end
end
