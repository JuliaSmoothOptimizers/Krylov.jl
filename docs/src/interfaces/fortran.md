# [Fortran interface](@id fortran-interface)

Add `use iso_c_binding` and `include 'krylov.f90'` after `implicit none`, then link against `libkrylov` (see [Building libkrylov](@ref building-libkrylov)).
The concepts (callbacks, data types, options, return codes, block solvers) are described in the [reference](@ref reference-interfaces).
This page collects runnable Fortran programs.

## Example

The same 5×5 tridiagonal SPD system as the [C example](@ref c-interface), now in Fortran ([`examples/Fortran/basic_cg.f90`](https://github.com/JuliaSmoothOptimizers/Krylov.jl/blob/main/interfaces/examples/Fortran/basic_cg.f90)):

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
                     c_null_funptr, c_loc(b), c_null_ptr, c_loc(diag), c_loc(opts))
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

## Block example

Solve `A X = B` with several right-hand sides at once using block GMRES. Blocks are natural Fortran 2D arrays (column-major); `B` must have full column rank (see [Block Krylov solvers](@ref block-krylov-methods)). Switch `KRYLOV_BLOCK_GMRES` to `KRYLOV_BLOCK_MINRES` for block MINRES
([`examples/Fortran/block_gmres.f90`](https://github.com/JuliaSmoothOptimizers/Krylov.jl/blob/main/interfaces/examples/Fortran/block_gmres.f90)):

```fortran
program block_gmres
  use iso_c_binding
  implicit none
  include 'krylov.f90'

  integer, parameter :: n = 16    ! operator dimension
  integer, parameter :: p = 3     ! number of right-hand sides (block width)

  real(c_double), target :: Xtrue(n,p), B(n,p), X(n,p)
  type(KrylovOptions), target :: opts
  type(c_ptr)    :: ws
  integer(c_int) :: ret
  integer        :: i, j
  real(c_double) :: t

  ! X_true with independent columns, then B = A * X_true.
  do j = 1, p
    do i = 1, n
      t = real(i, c_double) / n
      if (j == 1) then
        Xtrue(i,j) = 1.0_c_double
      else if (j == 2) then
        Xtrue(i,j) = t
      else
        Xtrue(i,j) = t*t
      end if
    end do
  end do
  call apply_A(Xtrue, B, p)

  ret = krylov_block_workspace_create(KRYLOV_BLOCK_GMRES, n, n, p,        &
                                      KRYLOV_FLOAT64, KRYLOV_CPU,          &
                                      c_null_ptr, ws)

  opts = krylov_default_options()
  opts%atol = 1.0d-10 ; opts%rtol = 1.0d-10
  ret = krylov_block_solve(ws, c_funloc(cb_block_A), c_null_funptr,       &
                           c_null_funptr, c_loc(B), c_null_ptr, c_loc(opts))
  ret = krylov_block_get_X(ws, c_loc(X), int(n, c_int), int(p, c_int))

  write(*,'(A,L1,A,I0,A,ES8.1)')                                          &
    "Block solved: ", (krylov_block_is_solved(ws) == 1),                  &
    "   niter: ", krylov_block_niter(ws),                                 &
    "   max error: ", maxval(abs(X - Xtrue))

  ret = krylov_block_workspace_free(ws)

contains

  ! Y = A * X for a p-column block, A = tridiag(-1, 8, -1).
  subroutine apply_A(Xin, Yout, pp)
    integer, intent(in)         :: pp
    real(c_double), intent(in)  :: Xin(:,:)
    real(c_double), intent(out) :: Yout(:,:)
    integer :: ii, jj
    do jj = 1, pp
      do ii = 1, n
        Yout(ii,jj) = 8.0_c_double * Xin(ii,jj)
        if (ii > 1) Yout(ii,jj) = Yout(ii,jj) - Xin(ii-1,jj)
        if (ii < n) Yout(ii,jj) = Yout(ii,jj) - Xin(ii+1,jj)
      end do
    end do
  end subroutine apply_A

  ! Block matvec callback: Y = A*X for a block of p columns.
  subroutine cb_block_A(x_ptr, y_ptr, pblk, userdata) bind(c)
    type(c_ptr),    value :: x_ptr, y_ptr, userdata
    integer(c_int), value :: pblk
    real(c_double), pointer :: xx(:,:), yy(:,:)
    call c_f_pointer(x_ptr, xx, [n, int(pblk)])
    call c_f_pointer(y_ptr, yy, [n, int(pblk)])
    call apply_A(xx, yy, int(pblk))
  end subroutine cb_block_A

end program block_gmres
```

## Fortran specifics

A few rules make the binding work:

- `include 'krylov.f90'` goes after `implicit none`. It declares the enum parameters, the `KrylovOptions` and `KrylovWorkspaceOptions` derived types (both `bind(c)`), and the `interface` blocks for every function.
- Anything whose address you take with `c_loc` must have the `target` attribute, including the option structs (`type(KrylovOptions), target :: opts`).
- Vectors and structs are passed as `c_loc(array)` and `c_loc(opts)`; pass `c_null_ptr` for an absent `b`, `c`, `opts` or `wopts`.
- Callbacks are passed as `c_funloc(my_sub)` and must be `bind(c)` subroutines with three `type(c_ptr), value` arguments (the block callback takes an extra `integer(c_int), value :: p`). Pass `c_null_funptr` for an unused slot. Inside, recover Fortran arrays with `c_f_pointer`.
- Block right-hand sides and solutions are natural Fortran 2D arrays (column-major), passed with `c_loc`.

## More examples

The repository keeps complete Fortran programs, compiled and run in CI:

- [`test/Fortran/test_all_solvers.f90`](https://github.com/JuliaSmoothOptimizers/Krylov.jl/blob/main/interfaces/test/Fortran/test_all_solvers.f90), every solver.
- [`test/Fortran/test_block.f90`](https://github.com/JuliaSmoothOptimizers/Krylov.jl/blob/main/interfaces/test/Fortran/test_block.f90), `block_gmres` and `block_minres`.
