# Wrappers that turn C function pointers into Julia linear operators.
#
# The C caller passes:
#   void matvec(const double *x, double *y, void *userdata)   -- computes y = A*x
#   void rmatvec(const double *x, double *y, void *userdata)  -- computes y = A'*x
#   void matvec(const double *x, double *y, void *userdata)   -- computes y = M\x  (preconditioner)
#
# We wrap these in a struct that satisfies the Krylov.jl operator protocol:
#   mul!(y, op, x),  mul!(y, op', x)  and  size(op) -> (m, n)
#
# Solvers that only apply A (CG, GMRES, MINRES, ...) only need fptr_A.
# Solvers that also apply A' (LSQR, LSMR, CGLS, ...) also need fptr_At.
# Pass C_NULL for fptr_At when the solver does not require it.

# ---------------------------------------------------------------------------
# C callback signatures (typedef'd in krylov.h)
# ---------------------------------------------------------------------------
# typedef void (*KrylovMatvec)(const void *x, void *y, void *userdata);

# ---------------------------------------------------------------------------
# COperator: wraps C matvec + rmatvec function pointers as a Julia operator
# ---------------------------------------------------------------------------
struct COperator{T}
  m        :: Int
  n        :: Int
  fptr     :: Ptr{Cvoid}   # y = A*x
  fptr_t   :: Ptr{Cvoid}   # y = A'*x  (C_NULL if not needed)
  userdata :: Ptr{Cvoid}
end

Base.size(op::COperator)         = (op.m, op.n)
Base.size(op::COperator, d::Int) = d == 1 ? op.m : op.n
Base.eltype(::COperator{T}) where T = T

function LinearAlgebra.mul!(y::AbstractVector{T}, op::COperator{T}, x::AbstractVector{T}) where T
  GC.@preserve x y begin
    xptr = Base.unsafe_convert(Ptr{Cvoid}, pointer(x))
    yptr = Base.unsafe_convert(Ptr{Cvoid}, pointer(y))
    ccall(op.fptr, Cvoid, (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}), xptr, yptr, op.userdata)
  end
  return y
end

# Adjoint wrapper — returned by adjoint(op), used by solvers that call A'*v
struct COperatorAdjoint{T}
  op :: COperator{T}
end

Base.size(ot::COperatorAdjoint)         = (ot.op.n, ot.op.m)
Base.size(ot::COperatorAdjoint, d::Int) = d == 1 ? ot.op.n : ot.op.m
Base.eltype(::COperatorAdjoint{T}) where T = T

LinearAlgebra.adjoint(op::COperator{T}) where T = COperatorAdjoint{T}(op)
LinearAlgebra.adjoint(ot::COperatorAdjoint{T}) where T = ot.op

function LinearAlgebra.mul!(y::AbstractVector{T}, ot::COperatorAdjoint{T}, x::AbstractVector{T}) where T
  ot.op.fptr_t != C_NULL || error("A' matvec requested but fptr_t is NULL — pass matvec_At to krylov_solve")
  GC.@preserve x y begin
    xptr = Base.unsafe_convert(Ptr{Cvoid}, pointer(x))
    yptr = Base.unsafe_convert(Ptr{Cvoid}, pointer(y))
    ccall(ot.op.fptr_t, Cvoid, (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}), xptr, yptr, ot.op.userdata)
  end
  return y
end

# Preconditioner variant: same layout, but invoked via ldiv! semantics.
# In Krylov.jl you pass ldiv=false and a normal mul! operator, so we use
# a plain mul! here too — the C side is responsible for applying M^{-1}.
struct CPreconditioner{T}
  n        :: Int
  fptr     :: Ptr{Cvoid}
  userdata :: Ptr{Cvoid}
end

Base.size(p::CPreconditioner)         = (p.n, p.n)
Base.size(p::CPreconditioner, d::Int) = p.n
Base.eltype(::CPreconditioner{T}) where T = T

function LinearAlgebra.mul!(y::AbstractVector{T}, p::CPreconditioner{T}, x::AbstractVector{T}) where T
  GC.@preserve x y begin
    xptr = Base.unsafe_convert(Ptr{Cvoid}, pointer(x))
    yptr = Base.unsafe_convert(Ptr{Cvoid}, pointer(y))
    ccall(p.fptr, Cvoid, (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}), xptr, yptr, p.userdata)
  end
  return y
end

# ---------------------------------------------------------------------------
# CBlockOperator: wraps a C block-matvec for the block Krylov solvers.
#
# The block solvers (block_gmres, block_minres) apply the operator to a whole
# block of p columns at once via mul!(Y, op, X) where X is n×p and Y is m×p.
# The C callback receives the column-major buffers and the block width p:
#
#   void KrylovBlockMatvec(const void *X, void *Y, int p, void *userdata);
#
# The same wrapper is reused for the preconditioner M (applied as Y = M⁻¹·X).
# ---------------------------------------------------------------------------
struct CBlockOperator{T}
  m        :: Int
  n        :: Int
  fptr     :: Ptr{Cvoid}
  userdata :: Ptr{Cvoid}
end

Base.size(op::CBlockOperator)         = (op.m, op.n)
Base.size(op::CBlockOperator, d::Int) = d == 1 ? op.m : op.n
Base.eltype(::CBlockOperator{T}) where T = T

function LinearAlgebra.mul!(Y::AbstractMatrix{T}, op::CBlockOperator{T}, X::AbstractMatrix{T}) where T
  p = size(X, 2)
  GC.@preserve X Y begin
    xptr = Base.unsafe_convert(Ptr{Cvoid}, pointer(X))
    yptr = Base.unsafe_convert(Ptr{Cvoid}, pointer(Y))
    ccall(op.fptr, Cvoid, (Ptr{Cvoid}, Ptr{Cvoid}, Cint, Ptr{Cvoid}), xptr, yptr, Cint(p), op.userdata)
  end
  return Y
end

# ---------------------------------------------------------------------------
# Helpers to build operators from C pointers
# ---------------------------------------------------------------------------
function make_operator(::Type{T}, m::Int, n::Int, fptr::Ptr{Cvoid}, fptr_t::Ptr{Cvoid}, userdata::Ptr{Cvoid}) where T
  return COperator{T}(m, n, fptr, fptr_t, userdata)
end

function make_preconditioner(::Type{T}, n::Int, fptr::Ptr{Cvoid}, userdata::Ptr{Cvoid}) where T
  return CPreconditioner{T}(n, fptr, userdata)
end
