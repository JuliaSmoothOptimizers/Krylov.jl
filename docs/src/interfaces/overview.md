# [C and Fortran interfaces](@id overview-interfaces)

!!! warning "Prototype"
    This interface is still in early development. Not all Krylov.jl features are exposed yet (for example shifts and GPU devices). The API may change in future releases.

Krylov.jl ships `libkrylov`, a native shared library that exposes its solvers to languages that can call C code.

Pre-built artifacts for Linux (x86-64, aarch64), macOS (arm64, x86-64) and Windows (x86-64) are available on the [Releases](https://github.com/JuliaSmoothOptimizers/Krylov.jl/releases) page.
To build the library yourself, see [Building libkrylov](@ref building-libkrylov).

## How it works

`libkrylov` is the whole of Krylov.jl compiled ahead of time into a shared library, with the Julia runtime embedded in the bundle.
From the caller's point of view it is an ordinary C library.
There is no Julia process to start and no `.jl` file to ship, only `libkrylov.so` (or `.dylib` / `.dll`) and the two headers `krylov.h` / `krylov.f90`.

The design rests on three ideas:

- **Opaque workspace handle.** `krylov_workspace_create` allocates a *typed* Krylov workspace and hands back a `void *`. The workspace holds all the internal vectors a solver needs. You never dereference the pointer; you pass it to the other functions. The library keeps it alive (rooted against the Julia garbage collector) until you call `krylov_workspace_free`.

- **Matrix-free by callbacks.** Krylov methods only touch the operator through products `A * x` (and possibly `Aᴴ * x` or `M⁻¹ * x`). You never pass a matrix. You pass C function pointers that compute those products. The same interface drives a dense matrix, a sparse matrix, a stencil, or a fully matrix-free operator, in any language with a C ABI.

- **Concrete types, no dynamic dispatch.** A workspace is created for one `(solver, precision)` pair. Every product, dot, and axpy then runs through statically-typed, fully specialised Julia code.

The lifecycle is always the same, regardless of the solver:

1. **Create** a workspace with `krylov_workspace_create`.
2. **Solve** with `krylov_solve`, passing matrix-vector product callbacks.
3. **Retrieve** the solution with `krylov_get_x` (and `krylov_get_y` for two-solution solvers).
4. **Free** the workspace with `krylov_workspace_free`.

## Quickstart

A complete C program solving a small SPD system with CG (the Fortran equivalent is on the [Fortran interface](@ref fortran-interface) page):

```c
#include <stdio.h>
#include "krylov.h"

/* y = A*x for A = tridiag(-1, 2, -1); userdata carries the dimension n */
static void matvec_A(const void *xv, void *yv, void *userdata)
{
  const double *x = (const double *)xv;
  double       *y = (double *)yv;
  int n = *(const int *)userdata;
  for (int i = 0; i < n; i++) {
    y[i] = 2.0 * x[i];
    if (i > 0)     y[i] -= x[i-1];
    if (i < n - 1) y[i] -= x[i+1];
  }
}

int main(void)
{
  int n = 5;
  double b[5] = {1, 0, 0, 0, 1}, x[5];

  void *ws = NULL;
  krylov_workspace_create(KRYLOV_CG, n, n, KRYLOV_FLOAT64, KRYLOV_CPU, NULL, &ws);
  krylov_solve(ws, matvec_A, NULL, NULL, NULL, b, NULL, &n, NULL);  /* NULL opts = defaults */
  krylov_get_x(ws, x, n);

  printf("converged: %s   niter: %d\n",
         krylov_is_solved(ws) ? "yes" : "no", krylov_niter(ws));
  krylov_workspace_free(ws);
  return 0;
}
```

## Where to go next

The [Reference](@ref reference-interfaces) page documents the full API:

- [API overview](@ref ref-api-overview), the complete list of functions, and the [workspace lifetime](@ref ref-lifetime).
- [The matvec callbacks](@ref ref-callbacks), how the operator and preconditioners are passed.
- [Data types and precision](@ref ref-datatypes).
- [Choosing a solver](@ref ref-choosing): does it need the adjoint, a second right-hand side, or produce a second solution?
- [Options](@ref ref-options) and [return codes](@ref ref-return-codes).
- [Error handling and safety](@ref ref-errors), including thread safety.
- [Warm start](@ref ref-warm-start), [preconditioning](@ref ref-preconditioning), and [block Krylov solvers](@ref ref-block).

For runnable programs, see the [C interface](@ref c-interface) and [Fortran interface](@ref fortran-interface) pages.
