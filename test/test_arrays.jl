@testset "Weird array types" begin
    @testset "ReshapedArrays" begin
        for T in (Float32, Float64)
            A = rand(T, 4, 4)
            b = reshape(sparse(rand(T, 2, 2)), 4)
            @test Krylov.ktypeof(b) == Vector{T}
            x, stats = gmres(A, b)
            @test stats.solved
        end
    end
end
