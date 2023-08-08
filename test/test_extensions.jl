@testset "extensions" begin
    @testset "KrylovStaticArraysExt" begin
        A = rand(Float32, 2, 2)
        b = SVector{2}(rand(Float32, 2))
        @test Krylov.ktypeof(b) == Vector{Float32}
        @test gmres(A, b)[2].solved

        A = rand(Float64, 3, 3)
        b = MVector{3}(rand(Float64, 3))
        @test Krylov.ktypeof(b) == Vector{Float64}
        @test gmres(A, b)[2].solved
    end
end
