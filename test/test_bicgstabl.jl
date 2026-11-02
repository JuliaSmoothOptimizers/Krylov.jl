@testset "bicgstabl" begin
  bicgstabl_tol = 1.0e-6

  for FC in (Float64, ComplexF64)
    @testset "Data Type: $FC" begin

      for ℓ in (1, 2, 4)
        @testset "ℓ = $ℓ" begin

          # Symmetric and positive definite system.
          A, b = symmetric_definite(FC=FC)
          (x, stats) = bicgstabl(A, b, ℓ=ℓ)
          r = b - A * x
          resid = norm(r) / norm(b)
          @test(resid ≤ bicgstabl_tol)
          @test(stats.solved)

          # Symmetric indefinite variant.
          A, b = symmetric_indefinite(FC=FC)
          (x, stats) = bicgstabl(A, b, ℓ=ℓ)
          r = b - A * x
          resid = norm(r) / norm(b)
          @test(resid ≤ bicgstabl_tol)
          @test(stats.solved)

          # Nonsymmetric and positive definite systems.
          # A has only two distinct eigenvalues in the complex case and ℓ > 2 amplifies rounding errors.
          A, b = nonsymmetric_definite(FC=FC)
          (x, stats) = bicgstabl(A, b, ℓ=(FC <: Complex ? min(ℓ, 2) : ℓ))
          r = b - A * x
          resid = norm(r) / norm(b)
          @test(resid ≤ bicgstabl_tol)
          @test(stats.solved)

          # Nonsymmetric indefinite variant.
          A, b = nonsymmetric_indefinite(FC=FC)
          (x, stats) = bicgstabl(A, b, ℓ=ℓ)
          r = b - A * x
          resid = norm(r) / norm(b)
          @test(resid ≤ bicgstabl_tol)
          @test(stats.solved)

          # Sparse Laplacian.
          A, b = sparse_laplacian(FC=FC)
          (x, stats) = bicgstabl(A, b, ℓ=ℓ)
          r = b - A * x
          resid = norm(r) / norm(b)
          @test(resid ≤ bicgstabl_tol)
          @test(stats.solved)

          # Symmetric indefinite variant, almost singular.
          A, b = almost_singular(FC=FC)
          (x, stats) = bicgstabl(A, b, ℓ=ℓ)
          r = b - A * x
          resid = norm(r) / norm(b)
          @test(resid ≤ bicgstabl_tol)
          @test(stats.solved)

          # Test b == 0
          A, b = zero_rhs(FC=FC)
          (x, stats) = bicgstabl(A, b, ℓ=ℓ)
          @test norm(x) == 0
          @test stats.status == "x is a zero-residual solution"

          # Left preconditioning
          A, b, M = square_preconditioned(FC=FC)
          (x, stats) = bicgstabl(A, b, M=M, ℓ=min(ℓ, 2))
          r = b - A * x
          resid = norm(M * r) / norm(M * b)
          @test(resid ≤ bicgstabl_tol)
          @test(stats.solved)

          # Right preconditioning
          A, b, N = square_preconditioned(FC=FC)
          (x, stats) = bicgstabl(A, b, N=N, ℓ=min(ℓ, 2))
          r = b - A * x
          resid = norm(r) / norm(b)
          @test(resid ≤ bicgstabl_tol)
          @test(stats.solved)

          # Split preconditioning
          A, b, M, N = two_preconditioners(500, 32)
          (x, stats) = bicgstabl(A, b, M=M, N=N, ℓ=1)
          r = b - A * x
          resid = norm(M * r) / norm(M * b)
          @test(resid ≤ bicgstabl_tol)
          @test(stats.solved)

          # Test bᴴc == 0
          A, b, c = bc_breakdown(FC=FC)
          (x, stats) = bicgstabl(A, b, c=c, ℓ=ℓ)
          @test stats.status == "Breakdown bᴴc = 0"

          # Test successful termination after exact convergence in the Bi-CG part.
          A = 2 * Matrix{FC}(I, 4, 4)
          b = FC[1, 2, 3, 4]
          (x, stats) = bicgstabl(A, b, ℓ=ℓ)
          @test norm(b - A * x) == 0
          @test(stats.solved)

          # test callback function
          A, b = sparse_laplacian(FC=FC)
          workspace = BicgstablWorkspace(A, b, ℓ=ℓ)
          tol = 1.0e-1
          cb_n2 = TestCallbackN2(A, b, tol = tol)
          bicgstabl!(workspace, A, b, callback = cb_n2)
          @test workspace.stats.status == "user-requested exit"
          @test cb_n2(workspace)

          @test_throws TypeError bicgstabl(A, b, ℓ=ℓ, callback = workspace -> "string", history = true)
        end
      end

      # BiCGstab(1) is mathematically equivalent to BiCGSTAB.
      A, b = nonsymmetric_definite(FC=FC)
      (x, stats) = bicgstabl(A, b, ℓ=1)
      (x_bicgstab, stats_bicgstab) = bicgstab(A, b)
      @test stats.niter == stats_bicgstab.niter
      @test x ≈ x_bicgstab

      # Each iteration performs 2ℓ operator-vector products.
      A, b = sparse_laplacian(FC=FC)
      workspace = BicgstablWorkspace(A, b, ℓ=3)
      bicgstabl!(workspace, A, b)
      @test Aprod_count(workspace) == 2 * 3 * iteration_count(workspace)

      # ℓ is capped at n and must be at least 1.
      A, b = nonsymmetric_definite(FC=FC)
      n = size(A, 1)
      workspace = BicgstablWorkspace(A, b, ℓ=n+5)
      @test workspace.ℓ == n
      @test_throws ErrorException("ℓ must be at least 1") bicgstabl(A, b, ℓ=0)

      # Test ρ == 0
      A = FC[-2 2 1; -1 -2 -1; 1 -2 -2]
      b = FC[1, 1, 1]
      c = FC[0, -2, 0]
      (x, stats) = bicgstabl(A, b, c=c, ℓ=3)
      @test stats.status == "breakdown ρ = 0"

      # Test cᴴuⱼ₊₁ == 0
      A = FC[0 1; -1 0]
      b = FC[1, 0]
      (x, stats) = bicgstabl(A, b, ℓ=1)
      @test stats.status == "breakdown (c, uⱼ₊₁) = 0"

      # Test σⱼ == 0
      A = FC[-1 1 -1; 0 -2 2; -1 1 -1]
      b = FC[-2, 1, -1]
      c = FC[0, -2, 2]
      (x, stats) = bicgstabl(A, b, c=c, ℓ=1)
      @test stats.status == "breakdown σⱼ = 0"

      # Test ω == 0
      A = FC[0 1; -1 0]
      b = FC[1, 1]
      c = FC[1, 2]
      (x, stats) = bicgstabl(A, b, c=c, ℓ=1)
      @test stats.status == "breakdown ω = 0"
    end
  end
end
