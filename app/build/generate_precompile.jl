using CKrylov, LinearAlgebra, SparseArrays

n   = Int32(5)
m   = Int32(5)
nnz = Int32(15)
irn = Int32.([1, 1, 1, 1, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 5])
jcn = Int32.([1, 3, 4, 5, 2, 1, 3, 5, 1, 4, 5, 1, 3, 4, 5])
val = [53.0, 8.0, 4.0, 3.0, 10.0, 8.0, 6.0, 8.0, 4.0, 26.0, 5.0, 3.0, 8.0, 5.0, 14.0]
A   = Matrix(sparse(irn, jcn, val, n, m))
b   = [108.0, 20.0, 66.0, 133.0, 117.0]
x   = zeros(5)

cg_sparse(n, m, nnz, pointer(irn), pointer(jcn), pointer(val), pointer(b), pointer(x))
cg_dense(n, m, pointer(A), pointer(b), pointer(x))

n   = Int32(7)
m   = Int32(5)
nnz = Int32(13)
irn = Int32.([1, 1, 1, 2, 3, 3, 4, 4, 5, 5, 6, 7, 7])
jcn = Int32.([1, 3, 5, 2, 3, 5, 1, 4, 4, 5, 2, 1, 3])
val = [1.0, 2.0, 3.0, 1.0, 1.0, 2.0, 4.0, 1.0, 5.0, 1.0, 3.0, 6.0, 1.0]
A   = Matrix(sparse(irn, jcn, val, n, m))
b   = [22.0, 5.0, 13.0, 8.0, 25.0, 5.0, 9.0]
x   = zeros(5)

lsmr_sparse(n, m, nnz, pointer(irn), pointer(jcn), pointer(val), pointer(b), pointer(x))
lsmr_dense(n, m, pointer(A), pointer(b), pointer(x))

n   = Int32(5)
m   = Int32(7)
nnz = Int32(13)
irn = Int32.([1, 1, 1, 2, 2, 2, 3, 3, 4, 4, 5, 5, 5])
jcn = Int32.([3, 5, 7, 1, 4, 6, 2, 6, 5, 6, 3, 4, 7])
val = [2.0, 3.0, 5.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 1.0, 2.0, 2.0]
A   = Matrix(sparse(irn, jcn, val, n, m))
b   = [56.0, 21.0, 16.0, 22.0, 25.0]
x   = zeros(7)

craig_sparse(n, m, nnz, pointer(irn), pointer(jcn), pointer(val), pointer(b), pointer(x))
craig_dense(n, m, pointer(A), pointer(b), pointer(x))
