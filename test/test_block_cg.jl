# Weighted 5-point graph Laplacian on an m × m grid: Hermitian positive semidefinite with the
# constants as null space, like the stiffness matrix of a pure Neumann problem.
function neumann_laplacian(m; FC=Float64, contrast=10.0)
  idx(i, j) = (j - 1) * m + i
  I, J, V = Int[], Int[], FC[]
  for j = 1:m, i = 1:m, (di, dj) in ((1, 0), (0, 1))
    (i + di > m || j + dj > m) && continue
    a, b = idx(i, j), idx(i + di, j + dj)
    w = 1 + (contrast - 1) * rand()
    append!(I, (a, b, a, b)); append!(J, (a, b, b, a)); append!(V, (w, w, -w, -w))
  end
  return sparse(I, J, V, m^2, m^2)
end

# Right-hand sides compatible with the Neumann problem: boundary sources with zero sum.
function neumann_rhs(m, p; FC=Float64)
  bd = [(j - 1) * m + i for j = 1:m, i = 1:m if i in (1, m) || j in (1, m)]
  B = zeros(FC, m^2, p)
  θ = range(0, 2π; length = length(bd) + 1)[1:end-1]
  for k = 1:p
    B[bd, k] .= isodd(k) ? cos.(((k + 1) ÷ 2) .* θ) : sin.((k ÷ 2) .* θ)
    FC <: Complex && (B[bd, k] .*= cis(k))
    B[:, k] .-= sum(B[:, k]) / m^2
  end
  return B, bd
end

@testset "block_cg" begin
  for FC in (Float64, ComplexF64)
    @testset "Data Type: $FC" begin
      T = real(FC)

      @testset "positive definite systems" begin
        A, b = symmetric_definite(FC=FC)
        B = hcat(b, -2b, FC.(sin.(1:10)))  # rank 2
        X, stats = block_cg(A, B)
        @test norm(B - A * X) ≤ √eps(T) * norm(B)
        @test stats.solved
        @test !stats.inconsistent
        @test stats.status == "solution good enough given atol and rtol"

        A, b = sparse_laplacian(FC=FC)
        B = hcat(b, A * ones(FC, size(A, 1)))
        X, stats = block_cg(A, B, rtol=1e-10, atol=0.0)
        @test norm(B - A * X) ≤ 1e-9 * norm(B)
        @test stats.solved

        # zero right-hand sides
        X, stats = block_cg(A, zeros(FC, size(A, 1), 2))
        @test norm(X) == 0
        @test stats.solved
        @test stats.niter == 0
      end

      m = 16
      n = m^2
      A = neumann_laplacian(m; FC)
      B, bd = neumann_rhs(m, 6; FC)
      Xref = pinv(Matrix(A)) * B  # minimum-norm solution

      @testset "null space: solution orthogonal to it" begin
        X, stats = block_cg(A, B; nullspace=ones(T, n), rtol=1e-10, atol=0.0)
        @test stats.solved
        @test norm(A * X - B) ≤ 1e-8 * norm(B)
        @test maximum(abs, sum(X; dims=1)) ≤ 1e-8 * norm(X)
        @test norm(X - Xref) ≤ 1e-7 * norm(Xref)
      end

      @testset "grounding" begin
        w = zeros(T, n)
        w[bd] .= 1  # the boundary values sum to zero
        X, stats = block_cg(A, B; nullspace=ones(T, n), grounding=w, rtol=1e-10, atol=0.0)
        @test stats.solved
        @test maximum(abs, w' * X) ≤ 1e-8 * norm(X)
        D = X - Xref  # the same solution up to a constant per column
        @test maximum(abs, D .- sum(D; dims=1) ./ n) ≤ 1e-7 * norm(Xref)
      end

      @testset "null space of dimension k" begin
        k = 3
        Q = Matrix(qr(randn(FC, 60, 60)).Q)
        λ = [zeros(T, k); 1 .+ 100 .* rand(T, 60 - k)]
        A2 = Q * Diagonal(λ) * Q'
        A2 = (A2 + A2') / 2
        V = Q[:, 1:k] * randn(FC, k, k)  # not orthonormal
        B2 = A2 * randn(FC, 60, 5)
        X2, stats = block_cg(A2, B2; nullspace=V, rtol=1e-12, atol=0.0, itmax=500)
        @test stats.solved
        @test norm(A2 * X2 - B2) ≤ 1e-9 * norm(B2)
        @test norm(Q[:, 1:k]' * X2) ≤ 1e-9 * norm(X2)
      end

      @testset "inconsistent right-hand sides are projected" begin
        workspace = BlockCgWorkspace(A, B; nullspace=ones(T, n))
        block_cg!(workspace, A, B .+ 0.3; rtol=1e-10, atol=0.0)
        X, stats = results(workspace)
        @test stats.solved
        @test stats.inconsistent
        @test workspace.defect > 0.1
        @test norm(A * X - B) ≤ 1e-8 * norm(B)  # the solution of the projected system
      end

      @testset "rank-deficient blocks" begin
        Bdep = hcat(B[:, 1], B[:, 1], 2 .* B[:, 2], zeros(FC, n), B[:, 1] + B[:, 2])
        X, stats = block_cg(A, Bdep; nullspace=ones(T, n), rtol=1e-10, atol=0.0)
        @test stats.solved
        @test norm(A * X - Bdep) ≤ 1e-8 * norm(Bdep)
        @test norm(X[:, 4]) ≤ 1e-12
      end

      @testset "preconditioner" begin
        d = diag(A)
        _, s0 = block_cg(A, B; nullspace=ones(T, n), rtol=1e-8, atol=0.0)
        X1, s1 = block_cg(A, B; nullspace=ones(T, n), M=inv(Diagonal(d)), rtol=1e-8, atol=0.0)
        X2, s2 = block_cg(A, B; nullspace=ones(T, n), M=Diagonal(d), ldiv=true, rtol=1e-8, atol=0.0)
        @test s1.solved && s2.solved
        @test s1.niter ≤ s0.niter
        @test s1.niter == s2.niter
        @test norm(X1 - X2) ≤ 1e-10 * norm(X1)
        @test norm(A * X1 - B) ≤ 1e-6 * norm(B)
      end

      @testset "warm start" begin
        X, stats = block_cg(A, B; nullspace=ones(T, n), rtol=1e-10, atol=0.0)
        X0 = X .+ ones(T, n) .* T(3)  # a component in the null space is removed
        X1, stats = block_cg(A, B, X0; nullspace=ones(T, n), rtol=1e-10, atol=0.0)
        @test stats.niter ≤ 1
        @test norm(X1 - X) ≤ 1e-8 * norm(X)
        workspace = BlockCgWorkspace(A, B; nullspace=ones(T, n))
        block_cg!(workspace, A, B, X0 .+ randn(FC, n, 6) .* T(1e-3); rtol=1e-10, atol=0.0)
        @test issolved(workspace)
        @test norm(solution(workspace) - X) ≤ 1e-7 * norm(X)
      end

      @testset "stopping, history, callback, verbose" begin
        X, stats = block_cg(A, B; nullspace=ones(T, n), itmax=3, history=true)
        @test stats.niter == 3
        @test !stats.solved
        @test stats.status == "maximum number of iterations exceeded"
        @test length(stats.residuals) == 4
        X, stats = block_cg(A, B; nullspace=ones(T, n), callback=workspace -> true)
        @test stats.niter == 1
        @test stats.status == "user-requested exit"
        io = IOBuffer()
        block_cg(A, B; nullspace=ones(T, n), verbose=1, iostream=io)
        @test occursin("BLOCK-CG", String(take!(io)))
      end

      @testset "generic interface" begin
        workspace = krylov_workspace(:block_cg, A, B; nullspace=ones(T, n))
        krylov_solve!(workspace, A, B; rtol=1e-10, atol=0.0)
        X, stats = krylov_solve(:block_cg, A, B; nullspace=ones(T, n), rtol=1e-10, atol=0.0)
        @test norm(solution(workspace) - X) ≤ 1e-12 * norm(X)
        @test iteration_count(workspace) == stats.niter
      end

      @testset "input checks" begin
        @test_throws ArgumentError BlockCgWorkspace(A, B; grounding=ones(T, n))
        @test_throws DimensionMismatch BlockCgWorkspace(A, B; nullspace=ones(T, n + 1))
        @test_throws ArgumentError BlockCgWorkspace(A, B; nullspace=ones(T, n), grounding=[1; zeros(T, n - 1)] .- [0; 1; zeros(T, n - 2)])
      end
    end
  end

  @testset "Float32" begin
    A = Float32.(neumann_laplacian(16))
    B = Float32.(neumann_rhs(16, 4)[1])
    X, stats = block_cg(A, B; nullspace=ones(Float32, 256), rtol=1f-5, atol=0f0)
    @test eltype(X) == Float32
    @test stats.solved
    @test norm(A * X - B) ≤ 1f-4 * norm(B)
  end

  @testset "precision: $FC" for FC in (BigFloat, Float16, ComplexF32)
    T = real(FC)
    A = neumann_laplacian(6; FC)                # assembled in FC (exact null space)
    B = FC.(neumann_rhs(6, 3)[1])
    B .-= sum(B; dims=1) ./ 36                  # compatible in the precision FC itself
    X, stats = block_cg(A, B; nullspace=ones(T, 36), atol=zero(T), rtol=T == Float16 ? T(0.05) : √eps(T))   # Float16: about 3 digits
    @test eltype(X) == FC
    @test stats.solved
    @test norm(A * X - B) ≤ (T == Float16 ? T(0.1) : 10 * √eps(T)) * norm(B)
    A2, b2 = symmetric_definite(FC=FC)
    X2, stats2 = block_cg(A2, hcat(b2, 2b2 .+ one(FC)))
    @test stats2.solved
  end

  @testset "allocations independent of n" begin
    function block_cg_bytes(m)
      A = neumann_laplacian(m)
      B, _ = neumann_rhs(m, 4)
      workspace = BlockCgWorkspace(A, B; nullspace=ones(m^2))
      block_cg!(workspace, A, B; itmax=20, rtol=0.0, atol=0.0)  # warm-up
      return @allocated block_cg!(workspace, A, B; itmax=20, rtol=0.0, atol=0.0)
    end
    @test block_cg_bytes(64) ≤ block_cg_bytes(16) + 4096  # only O(p²) host buffers per iteration
  end
end
