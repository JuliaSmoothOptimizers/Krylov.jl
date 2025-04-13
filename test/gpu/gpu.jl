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
end

function test_solver(S, M)
  n = 10
  memory = 5
  A = M(undef, n, n)
  b = S(undef, n)
  solver = GmresWorkspace(n, n, S; memory)
  krylov_solve!(workspace, A, b)  # Test that we don't have errors
end

function test_conversion(S, M)
  @test Krylov.vector_to_matrix(S) <: M
  @test Krylov.matrix_to_vector(M) <: S
end
