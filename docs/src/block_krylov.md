# [Block Krylov methods](@id block-krylov-methods)

!!! note
    `block_minres`, `block_cg` and `block_gmres` work on GPUs with Julia 1.11. Version 11.2.0 or later of `GPUArrays.jl` is also required.

If you want to use `block_minres` and `block_gmres` on previous Julia versions, you can overload the function `Krylov.copy_triangle` with the following code:
```julia
using KernelAbstractions, Krylov

@kernel function copy_triangle_kernel!(dest, src)
  i, j = @index(Global, NTuple)
  if j >= i
    @inbounds dest[i, j] = src[i, j]
  end
end

function Krylov.copy_triangle(Q::AbstractMatrix{FC}, R::AbstractMatrix{FC}, k::Int) where FC <: Krylov.FloatOrComplex
  backend = get_backend(Q)
  ndrange = (k, k)
  copy_triangle_kernel!(backend)(R, Q; ndrange=ndrange)
  KernelAbstractions.synchronize(backend)
end
```

## Block-MINRES

```@docs
block_minres
block_minres!
BlockMinresWorkspace
```

## Block-CG

Block-CG solves Hermitian positive definite systems with several right-hand sides. It also
handles positive *semidefinite* systems with a known null space, such as the stiffness matrix of
a pure Neumann problem (null space: the constants): pass a basis of the null space as
`nullspace`, and `block_cg` solves the projected system `A X = Π B` on the range of `A`. A
right-hand side with a component in the null space is projected (`stats.inconsistent` is then
set), and the solution is selected by `Wᴴ X = 0` with `W = grounding` (by default the solution
orthogonal to the null space).

The search blocks are orthonormalized with a rank-revealing eigendecomposition of their Gram
matrix, so linearly dependent right-hand sides do not cause a breakdown, and converged columns
are removed from the block. Each column has its own stopping tolerance
`atol + rtol * ‖Π bⱼ‖`. The small dense computations (Gram matrices of the search block, their
eigendecomposition and the block solves) use preallocated buffers and generic Julia code, so
`block_cg!` allocates nothing and works with any floating-point type, e.g. `BigFloat`.

```@docs
block_cg
block_cg!
BlockCgWorkspace
```

## Block-GMRES

```@docs
block_gmres
block_gmres!
BlockGmresWorkspace
```
