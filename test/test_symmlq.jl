@testset "symmlq" begin
  symmlq_tol = 1.0e-6

  for FC in (Float64, ComplexF64)
    @testset "Data Type: $FC" begin

      # Symmetric and positive definite system.
      n = 10
      A, b = symmetric_definite(FC=FC)
      (x, stats) = symmlq(A, b)
      r = b - A * x
      resid = norm(r) / norm(b)
      @test(resid ≤ symmlq_tol * norm(A) * norm(x))
      @test(stats.solved)

      # Symmetric indefinite variant.
      A, b = symmetric_indefinite(FC=FC)
      (x, stats) = symmlq(A, b)
      r = b - A * x
      resid = norm(r) / norm(b)
      @test(resid ≤ symmlq_tol * norm(A) * norm(x))
      @test(stats.solved)

      # Sparse Laplacian.
      A, b = sparse_laplacian(FC=FC)
      (x, stats) = symmlq(A, b)
      r = b - A * x
      resid = norm(r) / norm(b)
      @test(resid ≤ symmlq_tol * norm(A) * norm(x))
      @test(stats.solved)

      # System that cause a breakdown with the symmetric Lanczos process.
      A, b = symmetric_breakdown(FC=FC)
      (x, stats) = symmlq(A, b)
      r = b - A * x
      resid = norm(r) / norm(b)
      @test(resid ≤ symmlq_tol * norm(A) * norm(x))
      @test(stats.solved)

      # Test b == 0
      A, b = zero_rhs(FC=FC)
      (x, stats) = symmlq(A, b)
      @test norm(x) == 0
      @test stats.status == "x is a zero-residual solution"

      # Test error estimate
      A = Matrix(get_div_grad(8, 8, 8))
      b = ones(size(A, 1))
      λest = (1 - 1e-10) * eigmin(A)
      x_exact = A \ b
      (xlq, stats) = symmlq(A, b, λest=λest, transfer_to_cg=false, history=true)
      xcg = cg(A, b)[1]
      err = norm(x_exact - xlq)
      errcg = norm(x_exact - xcg)
      @test( err ≤ stats.errors[end] )
      @test( errcg ≤ stats.errorscg[end])
      (x, stats) = symmlq(A, b, λest=λest, window=1, history=true)
      @test( err ≤ stats.errors[end] )
      @test( errcg ≤ stats.errorscg[end])
      (x, stats) = symmlq(A, b, λest=λest, window=5, history=true)
      @test( err ≤ stats.errors[end] )
      @test( errcg ≤ stats.errorscg[end])

      # Test with Jacobi (or diagonal) preconditioner
      A, b, M = square_preconditioned(FC=FC)
      (x, stats) = symmlq(A, b, M=M)
      r = b - A * x
      resid = norm(r) / norm(b)
      @test(resid ≤ symmlq_tol * norm(A) * norm(x))
      @test(stats.solved)

      # test callback function
      workspace = SymmlqWorkspace(A, b)
      tol = 1.0
      cb_n2 = TestCallbackN2(A, b, tol = tol)
      symmlq!(workspace, A, b, atol = 0.0, rtol = 0.0, callback = cb_n2)
      @test workspace.stats.status == "user-requested exit"
      @test cb_n2(workspace)

      @test_throws TypeError symmlq(A, b, callback = workspace -> "string", history = true)
    end
  end

  # The Krylov subspace is invariant after one iteration.
  @testset "exact breakdown" begin
    for FC in (Float64, ComplexF64)
      n = 6
      A = Matrix{FC}(2I, n, n)
      b = FC[1:n;]
      for transfer_to_cg in (false, true)
        (x, stats) = symmlq(A, b, transfer_to_cg=transfer_to_cg)
        @test norm(x - b / 2) ≤ 1e-10 * norm(b / 2)
        @test stats.solved
      end

      # (A + λI)x = b
      (x, stats) = symmlq(A, b, λ=1.0)
      @test norm(x - b / 3) ≤ 1e-10 * norm(b / 3)
      @test stats.solved
      X = rand(FC, n, n)
      H = Matrix(Hermitian(X * X' + I))
      for λ in (-0.5, 0.5)
        (x, stats) = symmlq(H, b, λ=λ)
        @test norm(b - (H + λ * I) * x) ≤ symmlq_tol * norm(b)
        @test stats.solved
      end
    end
  end
end
