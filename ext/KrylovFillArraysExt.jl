module KrylovFillArraysExt

using Krylov: Krylov
using FillArrays: AbstractFill

"""
    Krylov.ktypeof(::AbstractFill{T,1}) where {T}

Return the corresponding (non-static) `Vector{T}` type.
"""
Krylov.ktypeof(::AbstractFill{T,1}) where {T} = Vector{T}

end
