## Block-GMRES

!!! note
    `block_gmres` works on GPUs
     with Julia 1.11.

If you want to use `block_gmres` on previous Julia versions, you can overload the function `Krylov.copy_triangle` with the following code:
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

```@docs
block_gmres
block_gmres!
```
