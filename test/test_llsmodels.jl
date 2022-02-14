using LLSModels
using Krylov

@testset "test solve LLSModel" begin
    A = rand(10,5)
    b = rand(10)
    lls = LLSModel(A, b)

    (x, stats) = cgls(lls)
    @test stats.solved

    (x, stats) = lsqr(lls)
    @test stats.solved
end
