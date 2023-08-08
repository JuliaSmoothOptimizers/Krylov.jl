module KrylovStaticArraysExt

if isdefined(Base, :get_extension)
    using Krylov: Krylov
else
    using ..Krylov: Krylov
end

using StaticArrays: StaticVector

"""
    Krylov.ktypeof(::StaticVector{S,T}) where {S,T}

Return the corresponding (non-static) `Vector{T}` type.

This is useful because the type returned by `ktypeof(b)` is used to construct vectors with size different from `size(b)`.
Hence, if `b` is a `StaticVector`, this will fail.
"""
Krylov.ktypeof(::StaticVector{S,T}) where {S,T} = Vector{T}

end
