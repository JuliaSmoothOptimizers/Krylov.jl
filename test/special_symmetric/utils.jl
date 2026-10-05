# Shared helpers and fixtures for the CS-MinAres tests.

const RNG = MersenneTwister(20261005)

# No elementwise access: this exercises the documented operator interface.
struct ProductOnly{T,M} <: AbstractMatrix{T}
    data::M
end

ProductOnly(A) = ProductOnly{eltype(A),typeof(A)}(A)
Base.size(A::ProductOnly) = size(A.data)
Base.getindex(::ProductOnly, i::Int, j::Int) = error("elementwise access unavailable")
Base.:*(A::ProductOnly, x::AbstractVector) = A.data * x
LinearAlgebra.mul!(y::AbstractVector, A::ProductOnly, x::AbstractVector) = mul!(y, A.data, x)

function cs_matrix(n, rank=n)
    U = Matrix(qr(randn(RNG, ComplexF64, n, n)).Q)
    s = [collect(range(0.5, 2.0; length=rank)); zeros(n-rank)]
    return U * Diagonal(s) * transpose(U)
end
