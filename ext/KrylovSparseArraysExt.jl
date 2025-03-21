module KrylovSparseArraysExt

using Krylov, SparseArrays
using Krylov.LinearAlgebra
using Krylov.LinearAlgebra: NoPivot, LowerTriangular
using Krylov: FloatOrComplex, reduced_qr!, ktypeof, vector_to_matrix, knorm, kdot, kaxpy!, kdotr, kfill!

function Krylov.ktypeof(v::S) where S <: SparseVector
  T = eltype(S)
  return Vector{T}
end

function Krylov.ktypeof(v::S) where S <: AbstractSparseVector
  return S.types[2]  # return `CuVector` for a `CuSparseVector`
end

include("krylov_processes.jl")
include("block_krylov_processes.jl")

end
