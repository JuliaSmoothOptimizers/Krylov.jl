! test_block.f90 — tests the block Krylov interface (block_gmres / block_minres).
!
! Problem: A = tridiag(-1, 8, -1) (SPD, strongly diagonally dominant), n x n.
! The right-hand side is an n x p block B = A * X_true, where X_true has
! linearly independent columns (block Krylov methods require a full-rank block).
! Fortran arrays are column-major, matching the block layout expected by the API.
!
! Compile (after building libkrylov):
!   gfortran -O2 -o test_block interfaces/test/Fortran/test_block.f90 \
!       interfaces/build/lib/libkrylov.so
!
! Exit code: 0 if all tests pass, 1 otherwise.

program test_block
  use iso_c_binding
  implicit none
  include '../../include/krylov.f90'

  integer, parameter :: N = 20
  integer, parameter :: P = 3

  real(c_double), target :: Xt(N,P), B(N,P), X(N,P)
  type(KrylovOptions), target :: opts
  type(c_ptr)    :: ws
  integer(c_int) :: ret
  integer        :: n_pass, n_fail, i, j, k, niter_cold
  real(c_double) :: t

  integer(c_int)    :: solvers(2)
  character(len=12) :: names(2)

  ! Dimension shared with the bind(c) callbacks via host association.
  integer :: n_glob

  n_pass = 0; n_fail = 0
  n_glob = N

  ! -------------------------------------------------------------------------
  ! X_true with independent columns, then B = A * X_true.
  ! -------------------------------------------------------------------------
  do j = 1, P
    do i = 1, N
      t = real(i, c_double) / N
      if (j == 1) then
        Xt(i,j) = 1.0_c_double
      else if (j == 2) then
        Xt(i,j) = t
      else
        Xt(i,j) = t*t
      end if
    end do
  end do
  call apply_A(Xt, B, P)

  solvers = [KRYLOV_BLOCK_GMRES, KRYLOV_BLOCK_MINRES]
  names   = [character(len=12) :: "block_gmres", "block_minres"]

  ! -------------------------------------------------------------------------
  ! Solve A X = B with each block solver and check the solution.
  ! -------------------------------------------------------------------------
  do k = 1, 2
    write(*,'(2A)') names(k), " ..."
    ret = krylov_block_workspace_create(solvers(k), int(N,c_int), int(N,c_int), &
                                        int(P,c_int), KRYLOV_FLOAT64, KRYLOV_CPU, &
                                        c_null_ptr, ws)
    call check(ret == 0, "workspace created")

    opts = krylov_default_options()
    opts%atol = 1.0d-10 ; opts%rtol = 1.0d-10 ; opts%itmax = 200_c_int

    ret = krylov_block_solve(ws, c_funloc(cb_block_A), c_null_funptr, c_null_funptr, &
                             c_loc(B), c_null_ptr, c_loc(opts))
    call check(ret == 0,                          "block solve returns 0")
    call check(krylov_block_is_solved(ws) == 1,   "block solve converged")
    call check(krylov_block_niter(ws) > 0,        "block niter is positive")
    call check(krylov_block_elapsed_time(ws) >= 0.0_c_double, "elapsed_time non-negative")

    ret = krylov_block_get_X(ws, c_loc(X), int(N,c_int), int(P,c_int))
    call check(ret == 0,                          "block_get_X returns 0")
    call check(maxval(abs(X - Xt)) < 1.0d-6,      "block solution is correct")

    ret = krylov_block_workspace_free(ws)
  end do

  ! -------------------------------------------------------------------------
  ! Preconditioner (Jacobi: M = diag(A) = 8*I, so M^-1 X = X / 8).
  ! -------------------------------------------------------------------------
  write(*,'(A)') "block_gmres + Jacobi preconditioner ..."
  ret = krylov_block_workspace_create(KRYLOV_BLOCK_GMRES, int(N,c_int), int(N,c_int), &
                                      int(P,c_int), KRYLOV_FLOAT64, KRYLOV_CPU, c_null_ptr, ws)
  opts = krylov_default_options()
  opts%atol = 1.0d-10 ; opts%rtol = 1.0d-10 ; opts%itmax = 200_c_int
  ret = krylov_block_solve(ws, c_funloc(cb_block_A), c_funloc(cb_block_M), c_null_funptr, &
                           c_loc(B), c_null_ptr, c_loc(opts))
  call check(krylov_block_is_solved(ws) == 1, "preconditioned block solve converged")
  ret = krylov_block_get_X(ws, c_loc(X), int(N,c_int), int(P,c_int))
  call check(maxval(abs(X - Xt)) < 1.0d-6, "preconditioned block solution is correct")
  ret = krylov_block_workspace_free(ws)

  ! -------------------------------------------------------------------------
  ! Right preconditioner (block_gmres accepts matvec_N).
  ! -------------------------------------------------------------------------
  write(*,'(A)') "block_gmres + right Jacobi preconditioner ..."
  ret = krylov_block_workspace_create(KRYLOV_BLOCK_GMRES, int(N,c_int), int(N,c_int), &
                                      int(P,c_int), KRYLOV_FLOAT64, KRYLOV_CPU, c_null_ptr, ws)
  opts = krylov_default_options()
  opts%atol = 1.0d-10 ; opts%rtol = 1.0d-10 ; opts%itmax = 200_c_int
  ret = krylov_block_solve(ws, c_funloc(cb_block_A), c_null_funptr, c_funloc(cb_block_M), &
                           c_loc(B), c_null_ptr, c_loc(opts))
  call check(krylov_block_is_solved(ws) == 1, "right-preconditioned block solve converged")
  ret = krylov_block_get_X(ws, c_loc(X), int(N,c_int), int(P,c_int))
  call check(maxval(abs(X - Xt)) < 1.0d-6, "right-preconditioned block solution is correct")
  ret = krylov_block_workspace_free(ws)

  ! -------------------------------------------------------------------------
  ! Warm start (block_gmres): seeding with the exact solution cuts iterations.
  ! -------------------------------------------------------------------------
  write(*,'(A)') "block_gmres warm start ..."
  ret = krylov_block_workspace_create(KRYLOV_BLOCK_GMRES, int(N,c_int), int(N,c_int), &
                                      int(P,c_int), KRYLOV_FLOAT64, KRYLOV_CPU, c_null_ptr, ws)
  opts = krylov_default_options()
  opts%atol = 1.0d-10 ; opts%rtol = 1.0d-10 ; opts%itmax = 200_c_int
  ret = krylov_block_solve(ws, c_funloc(cb_block_A), c_null_funptr, c_null_funptr, &
                           c_loc(B), c_null_ptr, c_loc(opts))
  niter_cold = krylov_block_niter(ws)
  ret = krylov_block_workspace_free(ws)

  ret = krylov_block_workspace_create(KRYLOV_BLOCK_GMRES, int(N,c_int), int(N,c_int), &
                                      int(P,c_int), KRYLOV_FLOAT64, KRYLOV_CPU, c_null_ptr, ws)
  ret = krylov_block_warm_start(ws, c_loc(Xt), int(N,c_int), int(P,c_int))
  call check(ret == 0, "block warm_start accepted")
  ret = krylov_block_solve(ws, c_funloc(cb_block_A), c_null_funptr, c_null_funptr, &
                           c_loc(B), c_null_ptr, c_loc(opts))
  call check(krylov_block_is_solved(ws) == 1,         "warm-started block solve converged")
  call check(krylov_block_niter(ws) < niter_cold,     "warm start reduces iterations")
  ret = krylov_block_workspace_free(ws)

  ! -------------------------------------------------------------------------
  ! Error codes.
  ! -------------------------------------------------------------------------
  write(*,'(A)') "block error codes ..."
  ret = krylov_block_workspace_create(99_c_int, int(N,c_int), int(N,c_int), &
                                      int(P,c_int), KRYLOV_FLOAT64, KRYLOV_CPU, c_null_ptr, ws)
  call check(ret == -2, "unknown block solver returns -2")
  ret = krylov_block_workspace_create(KRYLOV_BLOCK_GMRES, int(N,c_int), int(N,c_int), &
                                      int(P,c_int), KRYLOV_FLOAT64, KRYLOV_CPU, c_null_ptr, ws)
  call check(krylov_block_workspace_free(ws) == 0, "first block free returns 0")
  call check(krylov_block_workspace_free(ws) == 1, "second block free returns 1")

  ! -------------------------------------------------------------------------
  write(*,'(/,I0,A,I0,A)') n_pass, " checks passed, ", n_fail, " failed"
  if (n_fail > 0) stop 1

contains

  ! Y = A * X for an n x p block, A = tridiag(-1, 8, -1).
  subroutine apply_A(Xin, Yout, pp)
    integer, intent(in)         :: pp
    real(c_double), intent(in)  :: Xin(:,:)
    real(c_double), intent(out) :: Yout(:,:)
    integer :: ii, jj, nn
    nn = size(Xin, 1)
    do jj = 1, pp
      do ii = 1, nn
        Yout(ii,jj) = 8.0_c_double * Xin(ii,jj)
        if (ii > 1)  Yout(ii,jj) = Yout(ii,jj) - Xin(ii-1,jj)
        if (ii < nn) Yout(ii,jj) = Yout(ii,jj) - Xin(ii+1,jj)
      end do
    end do
  end subroutine apply_A

  ! Block matvec callback:  Y = A * X
  subroutine cb_block_A(x_ptr, y_ptr, p_in, userdata) bind(c)
    type(c_ptr),    value :: x_ptr, y_ptr, userdata
    integer(c_int), value :: p_in
    real(c_double), pointer :: xx(:,:), yy(:,:)
    call c_f_pointer(x_ptr, xx, [n_glob, int(p_in)])
    call c_f_pointer(y_ptr, yy, [n_glob, int(p_in)])
    call apply_A(xx, yy, int(p_in))
  end subroutine cb_block_A

  ! Block preconditioner callback:  Y = M^-1 X  with M = 8*I
  subroutine cb_block_M(x_ptr, y_ptr, p_in, userdata) bind(c)
    type(c_ptr),    value :: x_ptr, y_ptr, userdata
    integer(c_int), value :: p_in
    real(c_double), pointer :: xx(:,:), yy(:,:)
    call c_f_pointer(x_ptr, xx, [n_glob, int(p_in)])
    call c_f_pointer(y_ptr, yy, [n_glob, int(p_in)])
    yy = xx / 8.0_c_double
  end subroutine cb_block_M

  subroutine check(cond, msg)
    logical,      intent(in) :: cond
    character(*), intent(in) :: msg
    if (cond) then
      n_pass = n_pass + 1
    else
      n_fail = n_fail + 1
      write(*,'(A,A)') "  FAIL  ", msg
    end if
  end subroutine check

end program test_block
