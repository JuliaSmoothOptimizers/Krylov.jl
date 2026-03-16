# [Multi-GPU support](@id multi-gpu)

Krylov.jl has an experimental support for executing on multi-GPU systems through the [BLAS package of XK.jl](https://github.com/anlsys/xk.jl).
Multi-GPU interfaces are analogous to single-GPU's --- relying on Julia's types dispatcher on `XKVector` and `XKMatrix` to target multi-GPUs.
XK.jl handles work distribution and communications automatically and lazily.

```julia
using Krylov, XK

# CPU Arrays
A_cpu, y_cpu = symmetric_definite(n)

# XK.jl Arrays
A_xk = XKMatrix(A_cpu)
y_xk = XKVector(y_cpu)

# Run a conjugate gradient
(x_cpu, stats) = cg(A_xk, b_xk)

# At this point of the execution, 'x' may be distributed across multiple memories.
# The next line triggers and wait for copies (e.g., D2H transfers) required so `x_cpu` holds a coherent replica on the host memory.
XK.memory_coherent_sync(x_cpu)
```

See [XK.jl repository](https://github.com/anlsys/XK.jl/tree/main/examples/Krylov) for examples.
