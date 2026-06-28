! block_gmres.f90 — solve A X = B with several right-hand sides at once.
!
! A = tridiag(-1, 8, -1) (SPD),  B = A * X_true  with independent columns.
! Blocks are natural Fortran 2D arrays (column-major).  Switch
! KRYLOV_BLOCK_GMRES to KRYLOV_BLOCK_MINRES below to use block MINRES.
!
! Compile (after building libkrylov):
!   gfortran -o block_gmres interfaces/examples/Fortran/block_gmres.f90 \
!       interfaces/build/lib/libkrylov.so
!
! Expected output:
!   Block solved: T   niter: ...   max error: ...

program block_gmres
  use iso_c_binding
  implicit none
  include '../../include/krylov.f90'

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

  ! Create a block workspace, solve, retrieve the n x p solution block.
  ret = krylov_block_workspace_create(KRYLOV_BLOCK_GMRES, n, n, p,        &
                                      KRYLOV_FLOAT64, KRYLOV_CPU,          &
                                      c_null_ptr, ws)

  opts = krylov_default_options()
  opts%atol = 1.0d-10 ; opts%rtol = 1.0d-10
  ret = krylov_block_solve(ws, c_funloc(cb_block_A), c_null_funptr,       &
                           c_loc(B), c_null_ptr, c_loc(opts))
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
