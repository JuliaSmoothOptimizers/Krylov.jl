struct ShiftInvertOperator{FC,F}
  F :: F
  n :: Int
end

ShiftInvertOperator(F, FC) = ShiftInvertOperator{FC,typeof(F)}(F, size(F, 1))
Base.size(op :: ShiftInvertOperator) = (op.n, op.n)
Base.eltype(op :: ShiftInvertOperator{FC}) where FC = FC
LinearAlgebra.mul!(y, op :: ShiftInvertOperator, x) = ldiv!(y, op.F, x)

eigen_residual(A, λ, X) = maximum(norm(A * X[:,i] - λ[i] * X[:,i]) for i in eachindex(λ))

# Distance between the computed eigenvalues λ and their nearest exact eigenvalues.
eigen_error(λ, ev) = maximum(minimum(abs.(λᵢ .- ev)) for λᵢ in λ)

function convection_diffusion_2d(N :: Int=20; cx=20.0, cy=5.0)
  h = 1 / (N+1)
  D = spdiagm(-1 => fill(1.0, N-1), 0 => fill(-2.0, N), 1 => fill(1.0, N-1)) / h^2
  C = spdiagm(-1 => fill(-1.0, N-1), 1 => fill(1.0, N-1)) / (2h)
  Iₙ = sparse(1.0I, N, N)
  return kron(Iₙ, D) + kron(D, Iₙ) + cx * kron(Iₙ, C) + cy * kron(C, Iₙ)
end

@testset "krylov_schur" begin
  n = 100
  A = randn(n, n) / sqrt(n) + Diagonal(range(0, 3, length=n))
  ev = eigvals(A)

  @testset "which = $which" for which in (:LM, :SM, :LR, :SR, :LI, :SI)
    nev = 4
    λ, X, stats = krylov_schur(A, ones(n); nev, memory=30, which, itmax=200)
    @test stats.solved
    @test length(λ) ≥ nev
    @test eltype(λ) == ComplexF64
    @test size(X) == (n, length(λ))
    @test eigen_residual(A, λ, X) ≤ 1e-6
    @test all(norm(X[:,i]) ≈ 1 for i in axes(X, 2))
    # In real arithmetic, both eigenvalues of a complex conjugate pair have the same score.
    score = which ∈ (:LI, :SI) ? θ -> abs(imag(θ)) : Krylov.eigs_score(which)
    ref = sort(ev, by=score, rev=true)
    @test eigen_error(λ[1:nev], ref[1:nev+1]) ≤ 1e-6
    @test issorted(score.(λ), rev=true)
    @test length(stats.residuals) == length(λ)
  end

  @testset "which as a function" begin
    target = 1.5 + 0.0im
    λ, X, stats = krylov_schur(A, ones(n); nev=3, memory=40, which=θ -> -abs(θ - target), itmax=500)
    @test stats.solved
    ref = sort(ev, by=θ -> abs(θ - target))
    @test eigen_error(λ[1:3], ref[1:4]) ≤ 1e-6
  end

  @testset "complex operator" begin
    Ac = A + im * randn(n, n) / sqrt(n)
    evc = eigvals(Ac)
    λ, X, stats = krylov_schur(Ac, ones(ComplexF64, n); nev=5, memory=30)
    @test stats.solved
    @test eigen_residual(Ac, λ, X) ≤ 1e-6
    @test eigen_error(λ[1:5], sort(evc, by=abs, rev=true)[1:5]) ≤ 1e-6
  end

  @testset "real operator with complex eigenvalues" begin
    # Rotation blocks: the eigenvalues are r ± iω.
    blocks = [[r ω; -ω r] for (r, ω) in zip(range(1, 2, length=20), range(0.5, 1.0, length=20))]
    R = cat(blocks..., dims=(1,2))
    Q, _ = qr(randn(40, 40))
    Ar = Matrix(Q) * R * Matrix(Q)'
    λ, X, stats = krylov_schur(Ar, ones(40); nev=4, memory=16, which=:LR)
    @test stats.solved
    @test eigen_residual(Ar, λ, X) ≤ 1e-6
    @test λ[1] ≈ conj(λ[2])
    @test real(λ[1]) ≈ 2 atol=1e-8
    @test abs(imag(λ[1])) ≈ 1 atol=1e-8
  end

  @testset "Hermitian operator" begin
    H = (A + A') / 2
    λ, X, stats = krylov_schur(H, ones(n); nev=3, memory=20)
    @test stats.solved
    @test maximum(abs.(imag.(λ))) ≤ 1e-10
    @test eigen_error(λ[1:3], sort(eigvals(H), by=abs, rev=true)[1:3]) ≤ 1e-6
  end

  @testset "shift-invert" begin
    L = convection_diffusion_2d(20)
    evL = eigvals(Matrix(L))
    for σ in (-100.0, -100.0 + 50.0im)
      FC = typeof(σ)
      op = ShiftInvertOperator(lu(L - σ * I), FC)
      μ, X, stats = krylov_schur(op, ones(FC, size(L, 1)); nev=3, memory=20)
      @test stats.solved
      λ = σ .+ 1 ./ μ
      ref = sort(evL, by=λᵢ -> abs(λᵢ - σ))
      @test eigen_error(λ[1:3], ref[1:4]) ≤ 1e-6 * abs(σ)
      @test eigen_residual(L, λ, X) ≤ 1e-6 * opnorm(Matrix(L), 1)
    end
  end

  @testset "Float32" begin
    A32 = Float32.(A)
    λ, X, stats = krylov_schur(A32, ones(Float32, n); nev=4, memory=30)
    @test stats.solved
    @test eltype(λ) == ComplexF32
    @test eltype(X) == ComplexF32
    @test eigen_residual(A32, λ, X) ≤ 1f-2
  end

  @testset "breakdown" begin
    # b lies in an invariant subspace of dimension 2.
    D = Diagonal(Float64.(1:50))
    b = zeros(50)
    b[1] = b[2] = 1
    λ, X, stats = krylov_schur(D, b; nev=2, memory=10)
    @test stats.solved
    @test λ[1:2] ≈ [50, 49]
  end

  @testset "in-place, accessors and stopping criteria" begin
    workspace = KrylovSchurWorkspace(A, ones(n); memory=30)
    @test workspace.memory == 30
    krylov_schur!(workspace, A, ones(n); nev=4, history=true)
    λ, X, stats = Krylov.results(workspace)
    @test stats.solved
    @test length(stats.history) == Krylov.iteration_count(workspace)
    @test Krylov.Aprod_count(workspace) == stats.nmatvec
    @test Krylov.eigenvalues(workspace) == λ
    @test Krylov.eigenvectors(workspace) == X
    λ₁ = copy(λ)
    krylov_schur!(workspace, A, ones(n); nev=4)
    @test Krylov.eigenvalues(workspace) == λ₁
    @test isempty(workspace.stats.history)

    krylov_schur!(workspace, A, ones(n); nev=4, itmax=1)
    @test !workspace.stats.solved
    @test workspace.stats.status == "maximum number of iterations exceeded"
    @test Krylov.iteration_count(workspace) == 1
    @test length(Krylov.eigenvalues(workspace)) == 4

    krylov_schur!(workspace, A, ones(n); nev=4, callback=workspace -> true)
    @test workspace.stats.status == "user-requested exit"

    io = IOBuffer()
    krylov_schur!(workspace, A, ones(n); nev=4, verbose=1, iostream=io)
    @test occursin("Krylov-Schur", String(take!(io)))
    show(io, workspace)
    @test occursin("KrylovSchurWorkspace", String(take!(io)))
    show(io, workspace.stats)
    @test occursin("EigenStats", String(take!(io)))

    @test_throws ErrorException krylov_schur!(workspace, A, zeros(n))
    @test_throws ErrorException krylov_schur!(workspace, A, ones(n); nev=30)
    @test_throws ErrorException krylov_schur!(workspace, A, ones(n); which=:XX)
  end
end
