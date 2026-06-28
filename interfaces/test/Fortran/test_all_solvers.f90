! test_all_solvers.f90 — tests all solvers accessible via the Fortran interface.
!
! Three problem families (Float64 only; the C layer is tested for all precisions
! in test_libkrylov.jl):
!   SPD    : A = tridiag(-1, 2, -1),  b = A * ones,  x_true = ones
!   NONSYM : A = tridiag(-1, n, -1),  b = A * ones,  x_true = ones
!   LS     : A = tridiag(-1, n, -1) rectangular m x n,  b = A * ones
!
! Compile (after building libkrylov):
!   gfortran -O2 -o test_all_solvers interfaces/test/Fortran/test_all_solvers.f90 \
!       interfaces/build/lib/libkrylov.so
!
! Exit code: 0 if all tests pass, 1 otherwise.

program test_all_solvers
  use iso_c_binding
  implicit none
  include '../../include/krylov.f90'

  integer, parameter :: N = 20
  integer, parameter :: M = 30

  ! -------------------------------------------------------------------------
  ! Problem matrices (flat row-major)
  ! -------------------------------------------------------------------------
  real(c_double), target :: spd_A(N*N), spd_At(N*N), spd_b(N), spd_c(N)
  real(c_double), target :: spd_qd_b(N), spd_qd_c(N)   ! quasi-definite RHS for tricg/trimr
  real(c_double), target :: nsym_A(N*N), nsym_At(N*N), nsym_b(N), nsym_c(N)
  real(c_double), target :: gpmr_b(N), gpmr_c(N)       ! gpmr RHS: [I A; B I][x;y]=[b;c]
  real(c_double), target :: ls_A(M*N), ls_At(N*M), ls_b(M)

  integer :: n_pass, n_fail, i

  ! Callback state — set before each test, read by cb_matvec_A / cb_matvec_At
  real(c_double), pointer :: cur_A(:)
  real(c_double), pointer :: cur_At(:)
  integer :: cur_m, cur_n

  ! -------------------------------------------------------------------------
  nullify(cur_A, cur_At)
  cur_m = 0; cur_n = 0

  ! Build matrices and right-hand sides
  ! -------------------------------------------------------------------------
  call make_tridiag(spd_A, N, N, 2.0_c_double, -1.0_c_double)
  call do_transpose(spd_A, spd_At, N, N)
  call matvec(spd_A,  N, N, [(1.0_c_double, i=1,N)], spd_b)
  call matvec(spd_At, N, N, [(1.0_c_double, i=1,N)], spd_c)  ! = spd_b (A symmetric)
  ! Quasi-definite RHS for tricg/trimr: x_true = y_true = ones
  ! System [I  A; A' -I][x; y] = [ones + A*ones; A'*ones - ones]
  spd_qd_b = 1.0_c_double + spd_b   ! ones + A*ones
  spd_qd_c = spd_b - 1.0_c_double   ! A*ones - ones  (A symmetric so A'*ones = A*ones)

  call make_tridiag(nsym_A, N, N, real(N, c_double), -1.0_c_double)
  call do_transpose(nsym_A, nsym_At, N, N)
  call matvec(nsym_A,  N, N, [(1.0_c_double, i=1,N)], nsym_b)
  call matvec(nsym_At, N, N, [(1.0_c_double, i=1,N)], nsym_c)  ! c = A^T * ones
  ! GPMR RHS: [I A; B I][x;y]=[b;c] with B=A^T, x_true=y_true=ones
  ! b = ones + A*ones,  c = A^T*ones + ones
  gpmr_b = 1.0_c_double + nsym_b
  gpmr_c = nsym_c + 1.0_c_double

  call make_tridiag_rect(ls_A, M, N, real(N, c_double), -1.0_c_double)
  call do_transpose(ls_A, ls_At, M, N)
  call matvec(ls_A, M, N, [(1.0_c_double, i=1,N)], ls_b)

  ! -------------------------------------------------------------------------
  ! Run tests
  ! -------------------------------------------------------------------------
  n_pass = 0
  n_fail = 0

  ! --- symmetric / Hermitian ---
  !                                                  need_At  has_y  y_size  need_c  c_vec
  call run(KRYLOV_CG,         "cg        ", .false., .false., 0, .false., spd_c,  spd_A,  spd_At,  spd_b,  N, N)
  call run(KRYLOV_CR,         "cr        ", .false., .false., 0, .false., spd_c,  spd_A,  spd_At,  spd_b,  N, N)
  call run(KRYLOV_SYMMLQ,     "symmlq    ", .false., .false., 0, .false., spd_c,  spd_A,  spd_At,  spd_b,  N, N)
  call run(KRYLOV_MINRES,     "minres    ", .false., .false., 0, .false., spd_c,  spd_A,  spd_At,  spd_b,  N, N)
  call run(KRYLOV_MINRES_QLP, "minres_qlp", .false., .false., 0, .false., spd_c,  spd_A,  spd_At,  spd_b,  N, N)
  call run(KRYLOV_CAR,        "car       ", .false., .false., 0, .false., spd_c,  spd_A,  spd_At,  spd_b,  N, N)
  call run(KRYLOV_MINARES,    "minares   ", .false., .false., 0, .false., spd_c,  spd_A,  spd_At,  spd_b,  N, N)
  call run(KRYLOV_TRICG,      "tricg     ", .true.,  .true.,  N, .true.,  spd_qd_c, spd_A, spd_At, spd_qd_b, N, N)
  call run(KRYLOV_TRIMR,      "trimr     ", .true.,  .true.,  N, .true.,  spd_qd_c, spd_A, spd_At, spd_qd_b, N, N)
  ! --- non-symmetric square ---
  call run(KRYLOV_BICGSTAB,   "bicgstab  ", .false., .false., 0, .false., nsym_c, nsym_A, nsym_At, nsym_b, N, N)
  call run(KRYLOV_CGS,        "cgs       ", .false., .false., 0, .false., nsym_c, nsym_A, nsym_At, nsym_b, N, N)
  call run(KRYLOV_BILQ,       "bilq      ", .true.,  .false., 0, .false., nsym_c, nsym_A, nsym_At, nsym_b, N, N)
  call run(KRYLOV_QMR,        "qmr       ", .true.,  .false., 0, .false., nsym_c, nsym_A, nsym_At, nsym_b, N, N)
  call run(KRYLOV_DIOM,       "diom      ", .false., .false., 0, .false., nsym_c, nsym_A, nsym_At, nsym_b, N, N)
  call run(KRYLOV_DQGMRES,    "dqgmres   ", .false., .false., 0, .false., nsym_c, nsym_A, nsym_At, nsym_b, N, N)
  call run(KRYLOV_FOM,        "fom       ", .false., .false., 0, .false., nsym_c, nsym_A, nsym_At, nsym_b, N, N)
  call run(KRYLOV_GMRES,      "gmres     ", .false., .false., 0, .false., nsym_c, nsym_A, nsym_At, nsym_b, N, N)
  call run(KRYLOV_FGMRES,     "fgmres    ", .false., .false., 0, .false., nsym_c, nsym_A, nsym_At, nsym_b, N, N)
  call run(KRYLOV_BILQR,      "bilqr     ", .true.,  .true.,  N, .true.,  nsym_c, nsym_A, nsym_At, nsym_b, N, N)
  call run(KRYLOV_TRILQR,     "trilqr    ", .true.,  .true.,  N, .true.,  nsym_c, nsym_A, nsym_At, nsym_b, N, N)
  call run(KRYLOV_USYMLQ,     "usymlq    ", .true.,  .false., 0, .true.,  nsym_c, nsym_A, nsym_At, nsym_b, N, N)
  call run(KRYLOV_USYMQR,     "usymqr    ", .true.,  .false., 0, .true.,  nsym_c, nsym_A, nsym_At, nsym_b, N, N)
  call run(KRYLOV_USYMLQR,    "usymlqr   ", .true.,  .true.,  N, .true.,  nsym_c, nsym_A, nsym_At, nsym_b, N, N)
  ! gpmr: matvec_At slot carries B (n×m)
  call run(KRYLOV_GPMR,       "gpmr      ", .true.,  .true.,  N, .true.,  gpmr_c, nsym_A, nsym_At, gpmr_b, N, N)
  ! --- least-squares / least-norm ---
  call run(KRYLOV_LSLQ,       "lslq      ", .true.,  .false., 0, .false., spd_c,  ls_A,   ls_At,   ls_b,   M, N)
  call run(KRYLOV_LSQR,       "lsqr      ", .true.,  .false., 0, .false., spd_c,  ls_A,   ls_At,   ls_b,   M, N)
  call run(KRYLOV_LSMR,       "lsmr      ", .true.,  .false., 0, .false., spd_c,  ls_A,   ls_At,   ls_b,   M, N)
  call run(KRYLOV_CGLS,       "cgls      ", .true.,  .false., 0, .false., spd_c,  ls_A,   ls_At,   ls_b,   M, N)
  call run(KRYLOV_CRLS,       "crls      ", .true.,  .false., 0, .false., spd_c,  ls_A,   ls_At,   ls_b,   M, N)
  call run(KRYLOV_CGNE,       "cgne      ", .true.,  .false., 0, .false., spd_c,  ls_A,   ls_At,   ls_b,   M, N)
  call run(KRYLOV_CRMR,       "crmr      ", .true.,  .false., 0, .false., spd_c,  ls_A,   ls_At,   ls_b,   M, N)
  call run(KRYLOV_CRAIG,      "craig     ", .true.,  .false., 0, .false., spd_c,  ls_A,   ls_At,   ls_b,   M, N)
  call run(KRYLOV_CRAIGMR,    "craigmr   ", .true.,  .false., 0, .false., spd_c,  ls_A,   ls_At,   ls_b,   M, N)
  call run(KRYLOV_LNLQ,       "lnlq      ", .true.,  .false., 0, .false., spd_c,  ls_A,   ls_At,   ls_b,   M, N)

  write(*,'(/,I0,A,I0,A)') n_pass, "/", n_pass + n_fail, " passed"
  if (n_fail > 0) stop 1

contains

  ! ---------------------------------------------------------------------------
  ! Run one solver test.
  !   solver   : KrylovSolverType constant
  !   name     : label for output
  !   need_At  : pass cb_At (otherwise c_null_funptr)
  !   has_y    : call krylov_get_y and check finiteness
  !   y_size   : size of dual vector y (ignored when has_y=.false.)
  !   need_c   : pass second RHS c (size n), for two-RHS solvers
  !   c_vec    : second RHS array of length n (only used when need_c=.true.)
  !   A_flat   : row-major operator matrix (m × n)
  !   At_flat  : row-major adjoint matrix  (n × m)
  !   b_vec    : right-hand side of length m
  !   m, n     : dimensions
  ! ---------------------------------------------------------------------------
  subroutine run(solver, name, need_At, has_y, y_size, need_c, c_vec, A_flat, At_flat, b_vec, m, n)
    integer(c_int),  intent(in) :: solver
    character(*),    intent(in) :: name
    logical,         intent(in) :: need_At, has_y, need_c
    integer,         intent(in) :: y_size, m, n
    real(c_double),  intent(in), target :: c_vec(*), A_flat(*), At_flat(*), b_vec(*)

    type(c_ptr)    :: ws
    integer(c_int) :: ret
    real(c_double), allocatable, target :: x(:), y(:)
    real(c_double) :: err
    integer        :: i

    write(*,'(A,A)', advance='no') name, " ... "

    ! Expose pointers to the callbacks
    call c_f_pointer(c_loc(A_flat),  cur_A,  [m*n])
    call c_f_pointer(c_loc(At_flat), cur_At, [n*m])
    cur_m = m
    cur_n = n

    ret = krylov_workspace_create(solver, int(m, c_int), int(n, c_int), &
                                  KRYLOV_FLOAT64, KRYLOV_CPU, c_null_ptr, ws)
    if (ret /= 0) then
      write(*,'(A,I0)') "FAIL  workspace_create returned ", ret
      n_fail = n_fail + 1; return
    end if

    block
      type(KrylovOptions), target :: opts
      opts = krylov_default_options()
      opts%atol = 1.0d-8
      opts%rtol = 1.0d-8

      ret = krylov_solve(ws,                                    &
                         c_funloc(cb_matvec_A),               &
                         merge(c_funloc(cb_matvec_At),        &
                               c_null_funptr, need_At),        &
                         c_null_funptr,                       &  ! no left preconditioner
                         c_null_funptr,                       &  ! no right preconditioner
                         c_loc(b_vec),                        &
                         merge(c_loc(c_vec), c_null_ptr, need_c), &
                         c_null_ptr,                          &
                         c_loc(opts))
    end block
    if (ret /= 0) then
      write(*,'(A,I0)') "FAIL  krylov_solve returned ", ret
      ret = krylov_workspace_free(ws)
      n_fail = n_fail + 1; return
    end if

    if (krylov_is_solved(ws) /= 1) then
      write(*,'(A,I0,A)') "FAIL  did not converge (niter=", krylov_niter(ws), ")"
      ret = krylov_workspace_free(ws)
      n_fail = n_fail + 1; return
    end if

    ! Check x ≈ ones
    allocate(x(n))
    ret = krylov_get_x(ws, c_loc(x), int(n, c_int))
    err = 0.0_c_double
    do i = 1, n
      err = err + (x(i) - 1.0_c_double)**2
    end do
    err = sqrt(err / real(n, c_double))
    deallocate(x)

    if (err > 1.0d-6) then
      write(*,'(A,ES10.3)') "FAIL  ||x - x_true||/sqrt(n) = ", err
      ret = krylov_workspace_free(ws)
      n_fail = n_fail + 1; return
    end if

    ! Check y is finite when expected
    if (has_y) then
      allocate(y(y_size))
      ret = krylov_get_y(ws, c_loc(y), int(y_size, c_int))
      if (ret /= 0) then
        write(*,'(A,I0)') "FAIL  krylov_get_y returned ", ret
        deallocate(y)
        ret = krylov_workspace_free(ws)
        n_fail = n_fail + 1; return
      end if
      if (any(.not. ieee_is_finite(y))) then
        write(*,*) "FAIL  y contains non-finite values"
        deallocate(y)
        ret = krylov_workspace_free(ws)
        n_fail = n_fail + 1; return
      end if
      deallocate(y)
    end if

    ret = krylov_workspace_free(ws)
    write(*,*) "PASS"
    n_pass = n_pass + 1
  end subroutine run

  subroutine cb_matvec_A(x_ptr, y_ptr, userdata) bind(c)
    use iso_c_binding
    type(c_ptr), value :: x_ptr, y_ptr, userdata
    real(c_double), pointer :: x(:), y(:)
    call c_f_pointer(x_ptr, x, [cur_n])
    call c_f_pointer(y_ptr, y, [cur_m])
    call matvec(cur_A, cur_m, cur_n, x, y)
  end subroutine cb_matvec_A

  subroutine cb_matvec_At(x_ptr, y_ptr, userdata) bind(c)
    use iso_c_binding
    type(c_ptr), value :: x_ptr, y_ptr, userdata
    real(c_double), pointer :: x(:), y(:)
    call c_f_pointer(x_ptr, x, [cur_m])
    call c_f_pointer(y_ptr, y, [cur_n])
    call matvec(cur_At, cur_n, cur_m, x, y)
  end subroutine cb_matvec_At

  ! ---------------------------------------------------------------------------
  ! Dense matrix-vector product: y = A(m x n, row-major) * x(n)
  ! ---------------------------------------------------------------------------
  subroutine matvec(A, m, n, x, y)
    real(c_double), intent(in)  :: A(*), x(n)
    real(c_double), intent(out) :: y(m)
    integer,        intent(in)  :: m, n
    integer :: i, j
    do i = 1, m
      y(i) = 0.0_c_double
      do j = 1, n
        y(i) = y(i) + A((i-1)*n + j) * x(j)
      end do
    end do
  end subroutine matvec

  ! Build an n×n tridiagonal into a flat row-major array (1-indexed Fortran)
  subroutine make_tridiag(A, rows, cols, diag_val, off_val)
    real(c_double), intent(out) :: A(rows*cols)
    integer,        intent(in)  :: rows, cols
    real(c_double), intent(in)  :: diag_val, off_val
    integer :: i
    A = 0.0_c_double
    do i = 1, rows
      A((i-1)*cols + i) = diag_val
      if (i > 1)    A((i-1)*cols + (i-1)) = off_val
      if (i < cols) A((i-1)*cols + (i+1)) = off_val
    end do
  end subroutine make_tridiag

  ! Build an m×n rectangular tridiagonal
  subroutine make_tridiag_rect(A, m, n, diag_val, off_val)
    real(c_double), intent(out) :: A(m*n)
    integer,        intent(in)  :: m, n
    real(c_double), intent(in)  :: diag_val, off_val
    integer :: i
    A = 0.0_c_double
    do i = 1, m
      if (i <= n) A((i-1)*n + i) = diag_val
      if (i > 1 .and. i-1 <= n) A((i-1)*n + (i-1)) = off_val
      if (i < n)                 A((i-1)*n + (i+1)) = off_val
    end do
  end subroutine make_tridiag_rect

  ! Transpose m×n → n×m (both row-major)
  subroutine do_transpose(A, At, m, n)
    real(c_double), intent(in)  :: A(m*n)
    real(c_double), intent(out) :: At(n*m)
    integer,        intent(in)  :: m, n
    integer :: i, j
    do i = 1, m
      do j = 1, n
        At((j-1)*m + i) = A((i-1)*n + j)
      end do
    end do
  end subroutine do_transpose

  ! IEEE finiteness helper (avoids ieee_arithmetic dependency on old compilers)
  elemental function ieee_is_finite(x) result(r)
    real(c_double), intent(in) :: x
    logical :: r
    r = .not. (x /= x) .and. abs(x) < huge(x)
  end function ieee_is_finite

end program test_all_solvers
