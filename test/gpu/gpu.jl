using LinearAlgebra, SparseArrays, Test
using Krylov

include("../test_utils.jl")

function test_processes(S, M)
  n = 250
  m = 500
  k = 20
  FC = eltype(S)

  cpu_A, cpu_b = symmetric_indefinite(n, FC=FC)
  gpu_A, gpu_b = M(cpu_A), S(cpu_b)
  V, T = hermitian_lanczos(gpu_A, gpu_b, k)

  cpu_A, cpu_b = nonsymmetric_definite(n, FC=FC)
  cpu_c = -cpu_b
  gpu_A, gpu_b, gpu_c = M(cpu_A), S(cpu_b), S(cpu_c)
  V, T, U, Tᴴ = nonhermitian_lanczos(gpu_A, gpu_b, gpu_c, k)

  cpu_A, cpu_b = nonsymmetric_indefinite(n, FC=FC)
  gpu_A, gpu_b = M(cpu_A), S(cpu_b)
  V, H = arnoldi(gpu_A, gpu_b, k)

  cpu_A, cpu_b = under_consistent(n, m, FC=FC)
  gpu_A, gpu_b = M(cpu_A), S(cpu_b)
  V, U, L = golub_kahan(gpu_A, gpu_b, k)

  cpu_A, cpu_b = under_consistent(n, m, FC=FC)
  _, cpu_c = over_consistent(m, n, FC=FC)
  gpu_A, gpu_b, gpu_c = M(cpu_A), S(cpu_b), S(cpu_c)
  V, T, U, Tᴴ = saunders_simon_yip(gpu_A, gpu_b, gpu_c, k)

  cpu_A, cpu_b = under_consistent(n, m, FC=FC)
  cpu_B, cpu_c = over_consistent(m, n, FC=FC)
  gpu_A, gpu_B, gpu_b, gpu_c = M(cpu_A), M(gpu_B), S(cpu_b), S(cpu_c)
  V, H, U, F = montoison_orban(gpu_A, gpu_B, gpu_b, gpu_c, k)
end
