export craig_dense, craig_sparse

Base.@ccallable function craig_dense(n::Cint, m::Cint, A::Ptr{Cdouble}, b::Ptr{Cdouble}, x::Ptr{Cdouble})::Cint
  try
    cA = unsafe_wrap(Array, A, (n, m))
    cb = unsafe_wrap(Array, b, n)
    cx = unsafe_wrap(Array, x, m)
    sol, _, _ = Krylov.craig(cA, cb)
    cx .= sol
  catch
    Base.invokelatest(Base.display_error, Base.catch_stack())
    return 1
  end
  return 0
end

Base.@ccallable function craig_sparse(n::Cint, m::Cint, nnz::Cint, irn::Ptr{Cint}, jcn::Ptr{Cint}, val::Ptr{Cdouble}, b::Ptr{Cdouble}, x::Ptr{Cdouble})::Cint
  try
    cirn = unsafe_wrap(Array, irn, nnz)
    cjcn = unsafe_wrap(Array, jcn, nnz)
    cval = unsafe_wrap(Array, val, nnz)
    cA = sparse(cirn, cjcn, cval, n, m)
    cb = unsafe_wrap(Array, b, n)
    cx = unsafe_wrap(Array, x, m)
    sol, _, _ = Krylov.craig(cA, cb)
    cx .= sol
  catch
    Base.invokelatest(Base.display_error, Base.catch_stack())
    return 1
  end
  return 0
end
