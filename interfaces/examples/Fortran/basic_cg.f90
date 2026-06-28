! basic_cg.f90 — solve a 5x5 SPD system with CG.
!
! A = tridiag(-1, 2, -1),  b = [1, 0, 0, 0, 1]^T
!
! Compile:
!   gfortran -o basic_cg interfaces/examples/Fortran/basic_cg.f90 \
!       interfaces/build/lib/libkrylov.so

program basic_cg
  use iso_c_binding
  implicit none
  include '../../include/krylov.f90'

  integer, parameter :: n = 5

  ! Problem data — passed to the callback as userdata
  real(c_double), target :: diag(n), off(n-1)

  real(c_double), target  :: b(n), x(n)
  type(c_ptr)             :: ws
  integer(c_int)          :: ret

  ! -------------------------------------------------------------------------
  ! Build A = tridiag(-1, 2, -1)
  ! -------------------------------------------------------------------------
  diag = 2.0_c_double
  off  = -1.0_c_double

  ! -------------------------------------------------------------------------
  ! Right-hand side
  ! -------------------------------------------------------------------------
  b = 0.0_c_double
  b(1) = 1.0_c_double
  b(n) = 1.0_c_double

  ! -------------------------------------------------------------------------
  ! Create workspace
  ! -------------------------------------------------------------------------
  ret = krylov_workspace_create(KRYLOV_CG, n, n,            &
                                KRYLOV_FLOAT64, KRYLOV_CPU, &
                                c_null_ptr,                 &  ! workspace options (defaults)
                                ws)
  if (ret /= 0) then
    write(*,*) "krylov_workspace_create failed:", ret
    stop 1
  end if

  ! -------------------------------------------------------------------------
  ! Solve
  ! -------------------------------------------------------------------------
  block
    type(KrylovOptions), target :: opts
    opts = krylov_default_options()
    opts%atol = 1.0d-10
    opts%rtol = 1.0d-10

    ret = krylov_solve(ws,                  &
                       c_funloc(matvec_A),  &  ! y = A*x
                       c_null_funptr,       &  ! y = A'*x  (CG doesn't need it)
                       c_null_funptr,       &  ! no left preconditioner
                       c_null_funptr,       &  ! no right preconditioner
                       c_loc(b),            &  ! right-hand side b (size m)
                       c_null_ptr,          &  ! c = NULL  (CG only needs one RHS)
                       c_loc(diag),         &  ! userdata: diagonal array
                       c_loc(opts))            ! solver options
  end block
  if (ret /= 0) then
    write(*,*) "krylov_solve failed:", ret
    ret = krylov_workspace_free(ws)
    stop 1
  end if

  ! -------------------------------------------------------------------------
  ! Retrieve solution
  ! -------------------------------------------------------------------------
  ret = krylov_get_x(ws, c_loc(x), int(n, c_int))
  if (ret /= 0) then
    write(*,*) "krylov_get_x failed:", ret
    ret = krylov_workspace_free(ws)
    stop 1
  end if

  ! -------------------------------------------------------------------------
  ! Print results
  ! -------------------------------------------------------------------------
  write(*,'(A,L1,A,I0,A,ES10.3,A)') &
    "Solved: ", (krylov_is_solved(ws) == 1), &
    "   niter: ", krylov_niter(ws),          &
    "   time: ", krylov_elapsed_time(ws), " s"
  write(*,'(A)', advance='no') "x = ["
  write(*,'(5F6.2)', advance='no') x
  write(*,'(A)') " ]"

  ! -------------------------------------------------------------------------
  ! Free workspace
  ! -------------------------------------------------------------------------
  ret = krylov_workspace_free(ws)

contains

  ! -------------------------------------------------------------------------
  ! Matvec callback:  y = A*x   (A = tridiag(-1, 2, -1))
  ! -------------------------------------------------------------------------
  subroutine matvec_A(x_ptr, y_ptr, userdata) bind(c)
    use iso_c_binding
    type(c_ptr), value :: x_ptr, y_ptr, userdata

    real(c_double), pointer :: xv(:), yv(:), dg(:)
    integer :: i

    call c_f_pointer(x_ptr,    xv, [n])
    call c_f_pointer(y_ptr,    yv, [n])
    call c_f_pointer(userdata, dg, [n])   ! diag array

    do i = 1, n
      yv(i) = dg(i) * xv(i)
      if (i > 1) yv(i) = yv(i) - xv(i-1)
      if (i < n) yv(i) = yv(i) - xv(i+1)
    end do
  end subroutine matvec_A

end program basic_cg
