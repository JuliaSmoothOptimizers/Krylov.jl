using SparseArrays, Random, Test
using LinearAlgebra, Krylov, KernelAbstractions

if VERSION < v"1.11"
  @kernel function copy_triangle_kernel!(dest, src)
    i, j = @index(Global, NTuple)
    if j >= i
      @inbounds dest[i, j] = src[i, j]
    end
  end

  function Krylov.copy_triangle(Q::AbstractMatrix{FC}, R::AbstractMatrix{FC}, k::Int) where FC <: Krylov.FloatOrComplex
    backend = get_backend(Q)
    ndrange = (k, k)
    copy_triangle_kernel!(backend)(R, Q; ndrange=ndrange)
    KernelAbstractions.synchronize(backend)
  end
end

Random.seed!(666)

include("../test_utils.jl")

function test_processes(S, M)
  m = 250
  n = 500
  k = 20
  FC = eltype(S)

  cpu_A, cpu_b = symmetric_indefinite(n, FC=FC)
  gpu_A, gpu_b = M(cpu_A), S(cpu_b)
  V, β, T = hermitian_lanczos(gpu_A, gpu_b, k)

  cpu_A, cpu_b = nonsymmetric_definite(n, FC=FC)
  cpu_c = -cpu_b
  gpu_A, gpu_b, gpu_c = M(cpu_A), S(cpu_b), S(cpu_c)
  V, β, T, U, γᴴ, Tᴴ = nonhermitian_lanczos(gpu_A, gpu_b, gpu_c, k)

  cpu_A, cpu_b = nonsymmetric_indefinite(n, FC=FC)
  gpu_A, gpu_b = M(cpu_A), S(cpu_b)
  V, β, H = arnoldi(gpu_A, gpu_b, k)

  cpu_A, cpu_b = under_consistent(m, n, FC=FC)
  gpu_A, gpu_b = M(cpu_A), S(cpu_b)
  V, U, β, L = golub_kahan(gpu_A, gpu_b, k)

  cpu_A, cpu_b = under_consistent(m, n, FC=FC)
  _, cpu_c = over_consistent(n, m, FC=FC)
  gpu_A, gpu_b, gpu_c = M(cpu_A), S(cpu_b), S(cpu_c)
  V, β, T, U, γᴴ, Tᴴ = saunders_simon_yip(gpu_A, gpu_b, gpu_c, k)

  cpu_A, cpu_b = under_consistent(m, n, FC=FC)
  cpu_B, cpu_c = over_consistent(n, m, FC=FC)
  gpu_A, gpu_B, gpu_b, gpu_c = M(cpu_A), M(cpu_B), S(cpu_b), S(cpu_c)
  V, β, H, U, γ, F = montoison_orban(gpu_A, gpu_B, gpu_b, gpu_c, k)

  for skew in (false, true)
    cpu_A = rand(FC, n, n)
    cpu_A = skew ? cpu_A - cpu_A' : cpu_A + cpu_A'
    cpu_b = rand(FC, n)
    gpu_A, gpu_b = M(cpu_A), S(cpu_b)
    V, β, T = hermitian_lanczos(gpu_A, gpu_b, k; skew_hermitian=skew)
    @test norm(cpu_A * Matrix(V[:,1:k]) - Matrix(V) * T) ≤ 1e-4 * norm(T)

    cpu_A = rand(FC, n, n)
    cpu_A = skew ? cpu_A - transpose(cpu_A) : cpu_A + transpose(cpu_A)
    gpu_A = M(cpu_A)
    V, β, T = bunse_gerstner_stover(gpu_A, gpu_b, k; skew_symmetric=skew)
    @test norm(cpu_A * conj(Matrix(V[:,1:k])) - Matrix(V) * T) ≤ 1e-4 * norm(T)
  end
end

function test_solver(S, M)
  n = 10
  memory = 5
  A = M(undef, n, n)
  b = S(undef, n)
  workspace = GmresWorkspace(n, n, S; memory)
  krylov_solve!(workspace, A, b)  # Test that we don't have errors
end

function test_eigensolver(S, M)
  FC = eltype(S)
  n = 50
  cpu_A = Matrix(Diagonal(FC.(1:n))) + FC.(triu(ones(n, n), 1)) / n
  cpu_b = ones(FC, n)
  workspace = KrylovSchurWorkspace(n, n, S, M; memory = 20)
  krylov_schur!(workspace, M(cpu_A), S(cpu_b); nev = 3)
  λ, X, stats = Krylov.results(workspace)
  @test stats.solved
  @test Array(λ)[1:3] ≈ [n, n-1, n-2]
  @test X isa AbstractMatrix{complex(FC)}
end

function test_conversion(S, M)
  @test Krylov.vector_to_matrix(S) <: M
  @test Krylov.matrix_to_vector(M) <: S
end
