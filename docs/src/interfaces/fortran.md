# Fortran interface

Add `use iso_c_binding` and `include 'krylov.f90'` after `implicit none`, then link against `libkrylov` (see [Building libkrylov](building.md)). The concepts (callbacks, data types, options, return codes, block solvers) are described in the [Overview](overview.md); this page collects runnable Fortran programs.

## Example

The same 5×5 tridiagonal SPD system as the [C example](c.md), now in Fortran
([`examples/Fortran/basic_cg.f90`](https://github.com/JuliaSmoothOptimizers/Krylov.jl/blob/main/interfaces/examples/Fortran/basic_cg.f90)):

```fortran
program basic_cg
  use iso_c_binding
  implicit none
  include 'krylov.f90'

  integer, parameter    :: n = 5
  real(c_double), target :: diag(n), off(n-1), b(n), x(n)
  type(KrylovOptions), target :: opts
  type(c_ptr)           :: ws
  integer(c_int)        :: ret

  diag = 2.0_c_double ; off = -1.0_c_double
  b = 0.0_c_double ; b(1) = 1.0_c_double ; b(n) = 1.0_c_double

  ret = krylov_workspace_create(KRYLOV_CG, n, n, KRYLOV_FLOAT64, KRYLOV_CPU, &
                                c_null_ptr, ws)

  opts = krylov_default_options()
  opts%atol = 1d-10 ; opts%rtol = 1d-10
  ret = krylov_solve(ws, c_funloc(matvec_A), c_null_funptr, c_null_funptr, &
                     c_loc(b), c_null_ptr, c_loc(diag), c_loc(opts))
  ret = krylov_get_x(ws, c_loc(x), int(n, c_int))

  write(*,*) "Solved:", krylov_is_solved(ws) == 1, "  niter:", krylov_niter(ws)
  write(*,'(5F6.2)') x

  ret = krylov_workspace_free(ws)

contains
  subroutine matvec_A(x_ptr, y_ptr, userdata) bind(c)
    type(c_ptr), value :: x_ptr, y_ptr, userdata
    real(c_double), pointer :: xv(:), yv(:), dg(:)
    integer :: i
    call c_f_pointer(x_ptr, xv, [n]) ; call c_f_pointer(y_ptr, yv, [n])
    call c_f_pointer(userdata, dg, [n])
    do i = 1, n
      yv(i) = dg(i)*xv(i)
      if (i > 1) yv(i) = yv(i) - xv(i-1)
      if (i < n) yv(i) = yv(i) - xv(i+1)
    end do
  end subroutine
end program
```

## Fortran specifics

A few rules make the binding work:

- `include 'krylov.f90'` goes after `implicit none`. It declares the enum parameters, the `KrylovOptions` and `KrylovWorkspaceOptions` derived types (both `bind(c)`), and the `interface` blocks for every function.
- Anything whose address you take with `c_loc` must have the `target` attribute, including the option structs (`type(KrylovOptions), target :: opts`).
- Vectors and structs are passed as `c_loc(array)` and `c_loc(opts)`; pass `c_null_ptr` for an absent `b`, `c`, `opts` or `wopts`.
- Callbacks are passed as `c_funloc(my_sub)` and must be `bind(c)` subroutines with three `type(c_ptr), value` arguments (the block callback takes an extra `integer(c_int), value :: p`); pass `c_null_funptr` for an unused slot. Inside, recover Fortran arrays with `c_f_pointer`.
- Block right-hand sides and solutions are natural Fortran 2D arrays (column-major), passed with `c_loc`.

## More examples

The repository keeps complete Fortran programs, compiled and run in CI:

- [`test/Fortran/test_all_solvers.f90`](https://github.com/JuliaSmoothOptimizers/Krylov.jl/blob/main/interfaces/test/Fortran/test_all_solvers.f90), every solver.
- [`test/Fortran/test_block.f90`](https://github.com/JuliaSmoothOptimizers/Krylov.jl/blob/main/interfaces/test/Fortran/test_block.f90), `block_gmres` and `block_minres`.
