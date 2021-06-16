function test_operators()
  A, b, M, N = sqd()
  K = [-M A; A' N]
  d = [b; b]
  M⁻¹ = inv(M)
  N⁻¹ = inv(N)
  H⁻¹ = BlockDiagonalOperator(M⁻¹, N⁻¹)
  x, stats = minres_qlp(K, d, M=H⁻¹)
  @test stats.solved

  A, b = sparse_laplacian()
  op = LinearOperator(A)
  x, stats = cg(op, b)
  @test stats.solved
end

@testset "linear operators" begin
  test_operators()
end
