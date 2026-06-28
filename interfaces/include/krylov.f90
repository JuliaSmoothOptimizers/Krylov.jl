! krylov.f90 — Fortran interface to Krylov.jl
!
! Usage:
!   Add  use iso_c_binding  and  include 'krylov.f90'  AFTER  implicit none
!   in your program or subroutine.
!
! Example:
!
!   program my_prog
!     use iso_c_binding
!     implicit none
!     include 'krylov.f90'    ! ← here, after implicit none
!     ...
!   end program
!
! Callbacks must match the krylov_matvec abstract interface (or krylov_block_matvec
! for the block solvers) and be passed via c_funloc(my_sub).  Pass c_null_funptr
! for unused callbacks.

  ! -------------------------------------------------------------------------
  ! Enumerators  (must match krylov.h)
  ! -------------------------------------------------------------------------

  ! KrylovDataType
  integer(c_int), parameter :: KRYLOV_FLOAT32   = 0
  integer(c_int), parameter :: KRYLOV_FLOAT64   = 1
  integer(c_int), parameter :: KRYLOV_COMPLEX32 = 2
  integer(c_int), parameter :: KRYLOV_COMPLEX64 = 3

  ! KrylovDeviceType
  integer(c_int), parameter :: KRYLOV_CPU = 0

  ! KrylovSolverType
  integer(c_int), parameter :: KRYLOV_CG         =  0
  integer(c_int), parameter :: KRYLOV_CR         =  1
  integer(c_int), parameter :: KRYLOV_SYMMLQ     =  2
  integer(c_int), parameter :: KRYLOV_MINRES     =  3
  integer(c_int), parameter :: KRYLOV_MINRES_QLP =  4
  integer(c_int), parameter :: KRYLOV_DIOM       =  5
  integer(c_int), parameter :: KRYLOV_DQGMRES    =  6
  integer(c_int), parameter :: KRYLOV_FOM        =  7
  integer(c_int), parameter :: KRYLOV_GMRES      =  8
  integer(c_int), parameter :: KRYLOV_FGMRES     =  9
  integer(c_int), parameter :: KRYLOV_BICGSTAB   = 10
  integer(c_int), parameter :: KRYLOV_CGS        = 11
  integer(c_int), parameter :: KRYLOV_BILQ       = 12
  integer(c_int), parameter :: KRYLOV_QMR        = 13
  integer(c_int), parameter :: KRYLOV_USYMLQ     = 14
  integer(c_int), parameter :: KRYLOV_USYMQR     = 15
  integer(c_int), parameter :: KRYLOV_TRICG      = 16
  integer(c_int), parameter :: KRYLOV_TRIMR      = 17
  integer(c_int), parameter :: KRYLOV_TRILQR     = 18
  integer(c_int), parameter :: KRYLOV_BILQR      = 19
  integer(c_int), parameter :: KRYLOV_LSLQ       = 20
  integer(c_int), parameter :: KRYLOV_LSQR       = 21
  integer(c_int), parameter :: KRYLOV_LSMR       = 22
  integer(c_int), parameter :: KRYLOV_USYMLQR    = 23
  integer(c_int), parameter :: KRYLOV_CGLS       = 24
  integer(c_int), parameter :: KRYLOV_CRLS       = 25
  integer(c_int), parameter :: KRYLOV_CGNE       = 26
  integer(c_int), parameter :: KRYLOV_CRMR       = 27
  integer(c_int), parameter :: KRYLOV_CRAIG      = 28
  integer(c_int), parameter :: KRYLOV_CRAIGMR    = 29
  integer(c_int), parameter :: KRYLOV_LNLQ       = 30
  integer(c_int), parameter :: KRYLOV_GPMR       = 31
  integer(c_int), parameter :: KRYLOV_CAR        = 32
  integer(c_int), parameter :: KRYLOV_MINARES    = 33

  ! KrylovBlockSolverType  (block_gmres / block_minres)
  integer(c_int), parameter :: KRYLOV_BLOCK_GMRES  = 0
  integer(c_int), parameter :: KRYLOV_BLOCK_MINRES = 1

  ! -------------------------------------------------------------------------
  ! Option types  (must match the structs in krylov.h)
  !
  ! Two separate structs, mirroring when each option is consumed:
  !   * KrylovWorkspaceOptions — construction-time (krylov_workspace_create)
  !   * KrylovOptions          — solve-time        (krylov_solve)
  !
  ! Initialise with krylov_default_workspace_options() / krylov_default_options()
  ! and override only the fields you need.  Sentinel 0 (ints) or NaN (doubles)
  ! means "use the solver default".  Pass the struct via c_loc(opts).
  ! -------------------------------------------------------------------------

  type, bind(c) :: KrylovWorkspaceOptions
    integer(c_int) :: memory   ! 0 → 20 (GMRES / FGMRES / FOM / DIOM / DQGMRES / GPMR)
    integer(c_int) :: window   ! 0 → 5  (MINRES / SYMMLQ / LSQR / LSMR / LSLQ)
  end type KrylovWorkspaceOptions

  type, bind(c) :: KrylovOptions
    real(c_double) :: atol     ! NaN → sqrt(eps(T)) per precision
    real(c_double) :: rtol     ! NaN → sqrt(eps(T)) per precision
    integer(c_int) :: itmax    ! 0   → solver default
    integer(c_int) :: verbose  ! 0   = silent
    real(c_double) :: lambda   ! 0.0 = no regularisation (LSQR / LSMR / CGLS / …)
    real(c_double) :: tau      ! NaN → solver default (TriCG / TriMR : 1.0)
    real(c_double) :: nu       ! NaN → solver default (TriCG / TriMR : -1.0)
  end type KrylovOptions

  ! -------------------------------------------------------------------------
  ! Callback interface
  !
  ! Declare your matvec subroutine with this exact signature, then pass it
  ! as c_funloc(my_matvec).
  !
  ! Example:
  !
  !   subroutine my_matvec(x_ptr, y_ptr, userdata) bind(c)
  !     use iso_c_binding
  !     type(c_ptr), value :: x_ptr, y_ptr, userdata
  !     real(c_double), pointer :: x(:), y(:)
  !     call c_f_pointer(x_ptr, x, [n])
  !     call c_f_pointer(y_ptr, y, [n])
  !     y = matmul(A, x)
  !   end subroutine
  ! -------------------------------------------------------------------------

  abstract interface
    subroutine krylov_matvec(x_ptr, y_ptr, userdata) bind(c)
      use iso_c_binding
      type(c_ptr), value :: x_ptr    ! read-only input vector
      type(c_ptr), value :: y_ptr    ! output vector
      type(c_ptr), value :: userdata ! opaque user context
    end subroutine krylov_matvec
  end interface

  ! Block matvec: computes Y = A*X (or Y = M\X) for a block of p columns.
  ! X is n×p, Y is m×p, both column-major.  Pass via c_funloc(my_block_matvec).
  abstract interface
    subroutine krylov_block_matvec(x_ptr, y_ptr, p, userdata) bind(c)
      use iso_c_binding
      type(c_ptr),    value :: x_ptr    ! read-only input block (n×p)
      type(c_ptr),    value :: y_ptr    ! output block (m×p)
      integer(c_int), value :: p        ! block size (number of columns)
      type(c_ptr),    value :: userdata ! opaque user context
    end subroutine krylov_block_matvec
  end interface

  ! -------------------------------------------------------------------------
  ! C function interfaces
  ! -------------------------------------------------------------------------

  interface

    ! -----------------------------------------------------------------------
    ! krylov_default_workspace_options / krylov_default_options
    !
    ! Return an option struct with every field set to its "use default"
    ! sentinel.  Always start from these before overriding fields.
    ! -----------------------------------------------------------------------
    function krylov_default_workspace_options() &
        bind(c, name='krylov_default_workspace_options') result(wopts)
      import :: KrylovWorkspaceOptions
      type(KrylovWorkspaceOptions) :: wopts
    end function krylov_default_workspace_options

    function krylov_default_options() &
        bind(c, name='krylov_default_options') result(opts)
      import :: KrylovOptions
      type(KrylovOptions) :: opts
    end function krylov_default_options

    ! -----------------------------------------------------------------------
    ! krylov_workspace_create
    !
    ! Creates a workspace for the given solver.
    !
    !   solver  : KrylovSolverType constant (e.g. KRYLOV_CG, KRYLOV_GMRES)
    !   m, n    : operator dimensions
    !   dtype   : KrylovDataType constant (KRYLOV_FLOAT32 / KRYLOV_FLOAT64 / ...)
    !   device  : KrylovDeviceType constant (KRYLOV_CPU)
    !   wopts   : c_loc(KrylovWorkspaceOptions)  or  c_null_ptr for defaults
    !   ws      : receives the opaque workspace handle
    !
    ! Returns 0 on success, -1 on error, -2 on an unknown (solver, dtype) pair.
    ! -----------------------------------------------------------------------
    function krylov_workspace_create(solver, m, n, dtype, device, wopts, ws) &
        bind(c, name='krylov_workspace_create') result(ret)
      use iso_c_binding
      integer(c_int), value       :: solver, m, n, dtype, device
      type(c_ptr),    value       :: wopts   ! c_loc(opts) or c_null_ptr
      type(c_ptr),    intent(out) :: ws
      integer(c_int)              :: ret
    end function krylov_workspace_create

    ! -----------------------------------------------------------------------
    ! krylov_solve
    !
    !   ws         : workspace handle
    !   matvec_A   : callback  y = A*x          (required)
    !   matvec_At  : callback  y = A'*x         (c_null_funptr for CG/GMRES/...)
    !   matvec_M   : preconditioner; must compute y = M\x (solve M y = x)
    !                (c_null_funptr = no preconditioner)
    !   b          : first right-hand side pointer (c_loc of your array, size m)
    !   c          : second right-hand side pointer (c_loc of your array, size n)
    !                  c_null_ptr for solvers that only need one RHS
    !   userdata   : forwarded to every callback (c_loc or c_null_ptr)
    !   opts       : c_loc(KrylovOptions)  or  c_null_ptr for all defaults
    !
    ! Returns 0 on success, nonzero on error.
    ! -----------------------------------------------------------------------
    function krylov_solve(ws, matvec_A, matvec_At, matvec_M, &
                          b, c, userdata, opts) &
        bind(c, name='krylov_solve') result(ret)
      use iso_c_binding
      type(c_ptr),    value :: ws
      type(c_funptr), value :: matvec_A    ! y = A*x
      type(c_funptr), value :: matvec_At   ! y = A'*x  or  c_null_funptr
      type(c_funptr), value :: matvec_M    ! y = M\x   or  c_null_funptr
      type(c_ptr),    value :: b           ! c_loc(b_array), size m
      type(c_ptr),    value :: c           ! c_loc(c_array), size n  or  c_null_ptr
      type(c_ptr),    value :: userdata    ! c_loc(data)  or  c_null_ptr
      type(c_ptr),    value :: opts        ! c_loc(KrylovOptions) or c_null_ptr
      integer(c_int)        :: ret
    end function krylov_solve

    ! -----------------------------------------------------------------------
    ! krylov_get_x
    !
    ! Copies the primal solution into the buffer pointed to by x.
    ! Pass c_loc(x_array).
    !
    ! Returns 0 on success, nonzero on error.
    ! -----------------------------------------------------------------------
    function krylov_get_x(ws, x, n) &
        bind(c, name='krylov_get_x') result(ret)
      use iso_c_binding
      type(c_ptr),    value :: ws
      type(c_ptr),    value :: x   ! c_loc(x_array)
      integer(c_int), value :: n
      integer(c_int)        :: ret
    end function krylov_get_x

    ! -----------------------------------------------------------------------
    ! krylov_get_y
    !
    ! Copies the dual solution y (for solvers with two outputs: TriCG, GPMR, ...).
    ! Returns -2 if the solver has only one solution.
    ! -----------------------------------------------------------------------
    function krylov_get_y(ws, y, m) &
        bind(c, name='krylov_get_y') result(ret)
      use iso_c_binding
      type(c_ptr),    value :: ws
      type(c_ptr),    value :: y   ! c_loc(y_array)
      integer(c_int), value :: m
      integer(c_int)        :: ret
    end function krylov_get_y

    ! -----------------------------------------------------------------------
    ! krylov_is_solved
    !
    ! Returns 1 if converged, 0 if not, -1 on error.
    ! -----------------------------------------------------------------------
    function krylov_is_solved(ws) &
        bind(c, name='krylov_is_solved') result(ret)
      use iso_c_binding
      type(c_ptr),   value :: ws
      integer(c_int)       :: ret
    end function krylov_is_solved

    ! -----------------------------------------------------------------------
    ! krylov_niter
    !
    ! Returns the number of iterations performed, or -1 on error.
    ! -----------------------------------------------------------------------
    function krylov_niter(ws) &
        bind(c, name='krylov_niter') result(ret)
      use iso_c_binding
      type(c_ptr),   value :: ws
      integer(c_int)       :: ret
    end function krylov_niter

    ! -----------------------------------------------------------------------
    ! krylov_elapsed_time
    !
    ! Returns elapsed solve time in seconds, or -1.0 on error.
    ! -----------------------------------------------------------------------
    function krylov_elapsed_time(ws) &
        bind(c, name='krylov_elapsed_time') result(ret)
      use iso_c_binding
      type(c_ptr),    value :: ws
      real(c_double)        :: ret
    end function krylov_elapsed_time

    ! -----------------------------------------------------------------------
    ! krylov_warm_start
    !
    ! Sets the initial guess for the next krylov_solve call.
    ! Pass c_loc(x0_array).
    !
    ! Returns 0 on success, nonzero on error.
    ! -----------------------------------------------------------------------
    function krylov_warm_start(ws, x0, n) &
        bind(c, name='krylov_warm_start') result(ret)
      use iso_c_binding
      type(c_ptr),    value :: ws
      type(c_ptr),    value :: x0   ! c_loc(x0_array)
      integer(c_int), value :: n
      integer(c_int)        :: ret
    end function krylov_warm_start

    ! -----------------------------------------------------------------------
    ! krylov_workspace_free
    !
    ! Releases the workspace.  The handle must not be used after this call.
    ! Returns 0 on success, 1 if handle was not found.
    ! -----------------------------------------------------------------------
    function krylov_workspace_free(ws) &
        bind(c, name='krylov_workspace_free') result(ret)
      use iso_c_binding
      type(c_ptr),   value :: ws
      integer(c_int)       :: ret
    end function krylov_workspace_free

    ! -----------------------------------------------------------------------
    ! Block Krylov interface (block_gmres / block_minres)
    !
    ! The right-hand side is an m×p block B and the solution an n×p block X,
    ! both column-major.  Pass blocks via c_loc(your_2d_array).
    ! -----------------------------------------------------------------------

    ! krylov_block_workspace_create
    !   solver : KRYLOV_BLOCK_GMRES / KRYLOV_BLOCK_MINRES
    !   m, n   : operator dimensions ; p : block size (#columns)
    !   wopts  : c_loc(KrylovWorkspaceOptions) or c_null_ptr (memory: block_gmres)
    function krylov_block_workspace_create(solver, m, n, p, dtype, device, wopts, ws) &
        bind(c, name='krylov_block_workspace_create') result(ret)
      use iso_c_binding
      integer(c_int), value       :: solver, m, n, p, dtype, device
      type(c_ptr),    value       :: wopts
      type(c_ptr),    intent(out) :: ws
      integer(c_int)              :: ret
    end function krylov_block_workspace_create

    ! krylov_block_solve
    !   matvec_A : block matvec  Y = A*X        (required)
    !   matvec_M : preconditioner; must compute Y = M\X (solve M Y = X)
    !              (c_null_funptr = no preconditioner)
    !   B        : c_loc(B), m×p column-major
    !   opts     : c_loc(KrylovOptions) or c_null_ptr
    function krylov_block_solve(ws, matvec_A, matvec_M, b, userdata, opts) &
        bind(c, name='krylov_block_solve') result(ret)
      use iso_c_binding
      type(c_ptr),    value :: ws
      type(c_funptr), value :: matvec_A
      type(c_funptr), value :: matvec_M
      type(c_ptr),    value :: b
      type(c_ptr),    value :: userdata
      type(c_ptr),    value :: opts
      integer(c_int)        :: ret
    end function krylov_block_solve

    ! krylov_block_get_X — copies the n×p solution block into c_loc(X)
    function krylov_block_get_X(ws, x, n, p) &
        bind(c, name='krylov_block_get_X') result(ret)
      use iso_c_binding
      type(c_ptr),    value :: ws
      type(c_ptr),    value :: x
      integer(c_int), value :: n, p
      integer(c_int)        :: ret
    end function krylov_block_get_X

    ! krylov_block_is_solved — 1 if converged, 0 if not, -1 on error
    function krylov_block_is_solved(ws) &
        bind(c, name='krylov_block_is_solved') result(ret)
      use iso_c_binding
      type(c_ptr),   value :: ws
      integer(c_int)       :: ret
    end function krylov_block_is_solved

    ! krylov_block_niter — number of iterations performed, or -1 on error
    function krylov_block_niter(ws) &
        bind(c, name='krylov_block_niter') result(ret)
      use iso_c_binding
      type(c_ptr),   value :: ws
      integer(c_int)       :: ret
    end function krylov_block_niter

    ! krylov_block_elapsed_time — block solve time in seconds, or -1.0 on error
    function krylov_block_elapsed_time(ws) &
        bind(c, name='krylov_block_elapsed_time') result(ret)
      use iso_c_binding
      type(c_ptr),    value :: ws
      real(c_double)        :: ret
    end function krylov_block_elapsed_time

    ! krylov_block_warm_start — initial guess (n×p block) for the next solve
    function krylov_block_warm_start(ws, x0, n, p) &
        bind(c, name='krylov_block_warm_start') result(ret)
      use iso_c_binding
      type(c_ptr),    value :: ws
      type(c_ptr),    value :: x0
      integer(c_int), value :: n, p
      integer(c_int)        :: ret
    end function krylov_block_warm_start

    ! krylov_block_workspace_free — release the block workspace.
    ! Returns 0 on success, 1 if the handle was not found.
    function krylov_block_workspace_free(ws) &
        bind(c, name='krylov_block_workspace_free') result(ret)
      use iso_c_binding
      type(c_ptr),   value :: ws
      integer(c_int)       :: ret
    end function krylov_block_workspace_free

  end interface
