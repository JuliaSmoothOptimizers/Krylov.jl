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

      # Test b == 0 or c == 0
      b_zero = zeros(m)
      x, y, stats = tricg(A, b_zero, c)
      rhs = [b_zero; c]
      r =  rhs - [I A; A' -I] * [x; y]
      resid = norm(r) / norm(rhs)
      @test(resid ≤ tricg_tol)

      c_zero = zeros(n)
      x, y, stats = tricg(A, b, c_zero)
      rhs = [b; c_zero]
      r =  rhs - [I A; A' -I] * [x; y]
      resid = norm(r) / norm(rhs)
      @test(resid ≤ tricg_tol)

      # Test specific breakdowns
      A, b, c = ssy_mo_breakdown2(FC)
      K = [I A; A' -I]
      d = [b; c]
      x, y, stats = tricg(A, b, c)
      r = d - K * [x; y]
      resid = norm(r) / norm(d)
      @test(resid ≤ tricg_tol)

      A, b, c = ssy_mo_breakdown3(FC)
      K = [I A; A' -I]
      d = [b; c]
      x, y, stats = tricg(A, b, c)
      r = d - K * [x; y]
      resid = norm(r) / norm(d)
      @test(resid ≤ tricg_tol)

      # Test dimension of additional vectors
      for transpose ∈ (false, true)
        A, b, c, M, N = small_sqd(transpose, FC=FC)
        M⁻¹ = inv(M)
        N⁻¹ = inv(N)
        (x, y, stats) = tricg(A, b, c, M=M⁻¹, N=N⁻¹)
        K = [M A; A' -N]
        r = [b; c] - K * [x; y]
        H = [M zeros(size(M, 1), size(N, 2)); zeros(size(N, 1), size(M, 2)) N]
        @test sqrt(dot(r, inv(H) * r)) / sqrt(dot([b; c], inv(H) * [b; c])) ≤ tricg_tol
      end

      # Test different types for b and c
      A, b, _ = saddle_point(FC=FC)
      (x, y, stats) = tricg(A, b, -b)
      c = TestVector(-b)
      workspace = TricgWorkspace(KrylovConstructor(b, c))
      @test typeof(workspace.x) === typeof(b)
      @test typeof(workspace.y) === typeof(c)
      tricg!(workspace, A, b, c)
      xt, yt = workspace.x, workspace.y
      @test typeof(xt) === typeof(b)
      @test typeof(yt) === typeof(c)

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

        # test callback function
        A, b, D = saddle_point(FC=FC)
        c = -b
        workspace = TricgWorkspace(A, b)
        tol = 1.0e-1
        cb_n2 = TestCallbackN2SaddlePts(A, b, c, tol = tol)
        tricg!(workspace, A, b, c, callback = cb_n2)
        @test workspace.stats.status == "user-requested exit"
        @test cb_n2(workspace)

        @test_throws TypeError tricg(A, b, c, callback = workspace -> "string", history = true)
      end
    end
  end
end
