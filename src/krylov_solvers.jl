export KrylovSolver, MinresSolver, CgSolver, CrSolver, SymmlqSolver, CgLanczosSolver,
CgLanczosShiftSolver, MinresQlpSolver, DqgmresSolver, DiomSolver, UsymlqSolver,
UsymqrSolver, TricgSolver, TrimrSolver, TrilqrSolver, CgsSolver, BicgstabSolver,
BilqSolver, QmrSolver, BilqrSolver, CglsSolver, CrlsSolver, CgneSolver, CrmrSolver,
LslqSolver, LsqrSolver, LsmrSolver, LnlqSolver, CraigSolver, CraigmrSolver,
GmresSolver, FomSolver, GpmrSolver

export solve!, solution, nsolution, statistics, issolved, issolved_primal, issolved_dual

"Abstract type for using Krylov solvers in-place"
abstract type KrylovSolver{T,FC,S} end

"""
Type for storing the vectors required by the in-place version of MINRES.

The outer constructors

    solver = MinresSolver(n, m, S; window :: Int=5)
    solver = MinresSolver(A, b; window :: Int=5)

may be used in order to create these vectors.
"""
mutable struct MinresSolver{T,FC,S} <: KrylovSolver{T,FC,S}
  Δx      :: S
  x       :: S
  r1      :: S
  r2      :: S
  w1      :: S
  w2      :: S
  y       :: S
  v       :: S
  err_vec :: Vector{T}
  stats   :: SimpleStats{T}

  function MinresSolver(n, m, S; window :: Int=5)
    FC = eltype(S)
    T  = real(FC)
    Δx = S(undef, 0)
    x  = S(undef, n)
    r1 = S(undef, n)
    r2 = S(undef, n)
    w1 = S(undef, n)
    w2 = S(undef, n)
    y  = S(undef, n)
    v  = S(undef, 0)
    err_vec = zeros(T, window)
    stats = SimpleStats(false, false, T[], T[], T[], "unknown")
    solver = new{T,FC,S}(Δx, x, r1, r2, w1, w2, y, v, err_vec, stats)
    return solver
  end

  function MinresSolver(A, b; window :: Int=5)
    n, m = size(A)
    S = ktypeof(b)
    MinresSolver(n, m, S, window=window)
  end
end

"""
Type for storing the vectors required by the in-place version of CG.

The outer constructors

    solver = CgSolver(n, m, S)
    solver = CgSolver(A, b)

may be used in order to create these vectors.
"""
mutable struct CgSolver{T,FC,S} <: KrylovSolver{T,FC,S}
  Δx    :: S
  x     :: S
  r     :: S
  p     :: S
  Ap    :: S
  z     :: S
  stats :: SimpleStats{T}

  function CgSolver(n, m, S)
    FC = eltype(S)
    T  = real(FC)
    Δx = S(undef, 0)
    x  = S(undef, n)
    r  = S(undef, n)
    p  = S(undef, n)
    Ap = S(undef, n)
    z  = S(undef, 0)
    stats = SimpleStats(false, false, T[], T[], T[], "unknown")
    solver = new{T,FC,S}(Δx, x, r, p, Ap, z, stats)
    return solver
  end

  function CgSolver(A, b)
    n, m = size(A)
    S = ktypeof(b)
    CgSolver(n, m, S)
  end
end

"""
Type for storing the vectors required by the in-place version of CR.

The outer constructors

    solver = CrSolver(n, m, S)
    solver = CrSolver(A, b)

may be used in order to create these vectors.
"""
mutable struct CrSolver{T,FC,S} <: KrylovSolver{T,FC,S}
  x     :: S
  r     :: S
  p     :: S
  q     :: S
  Ar    :: S
  Mq    :: S
  stats :: SimpleStats{T}

  function CrSolver(n, m, S)
    FC = eltype(S)
    T  = real(FC)
    x  = S(undef, n)
    r  = S(undef, n)
    p  = S(undef, n)
    q  = S(undef, n)
    Ar = S(undef, n)
    Mq = S(undef, 0)
    stats = SimpleStats(false, false, T[], T[], T[], "unknown")
    solver = new{T,FC,S}(x, r, p, q, Ar, Mq, stats)
    return solver
  end

  function CrSolver(A, b)
    n, m = size(A)
    S = ktypeof(b)
    CrSolver(n, m, S)
  end
end

"""
Type for storing the vectors required by the in-place version of SYMMLQ.

The outer constructors

    solver = SymmlqSolver(n, m, S)
    solver = SymmlqSolver(A, b)

may be used in order to create these vectors.
"""
mutable struct SymmlqSolver{T,FC,S} <: KrylovSolver{T,FC,S}
  Δx      :: S
  x       :: S
  Mvold   :: S
  Mv      :: S
  Mv_next :: S
  w̅       :: S
  v       :: S
  clist   :: Vector{T}
  zlist   :: Vector{T}
  sprod   :: Vector{T}
  stats   :: SymmlqStats{T}

  function SymmlqSolver(n, m, S; window :: Int=5)
    FC      = eltype(S)
    T       = real(FC)
    Δx      = S(undef, 0)
    x       = S(undef, n)
    Mvold   = S(undef, n)
    Mv      = S(undef, n)
    Mv_next = S(undef, n)
    w̅       = S(undef, n)
    v       = S(undef, 0)
    clist   = zeros(T, window)
    zlist   = zeros(T, window)
    sprod   = ones(T, window)
    stats = SymmlqStats(false, T[], Union{T, Missing}[], T[], Union{T, Missing}[], T(NaN), T(NaN), "unknown")
    solver = new{T,FC,S}(Δx, x, Mvold, Mv, Mv_next, w̅, v, clist, zlist, sprod, stats)
    return solver
  end

  function SymmlqSolver(A, b; window :: Int=5)
    n, m = size(A)
    S = ktypeof(b)
    SymmlqSolver(n, m, S, window=window)
  end
end

"""
Type for storing the vectors required by the in-place version of CG-LANCZOS.

The outer constructors

    solver = CgLanczosSolver(n, m, S)
    solver = CgLanczosSolver(A, b)

may be used in order to create these vectors.
"""
mutable struct CgLanczosSolver{T,FC,S} <: KrylovSolver{T,FC,S}
  x       :: S
  Mv      :: S
  Mv_prev :: S
  p       :: S
  Mv_next :: S
  v       :: S
  stats   :: LanczosStats{T}

  function CgLanczosSolver(n, m, S)
    FC      = eltype(S)
    T       = real(FC)
    x       = S(undef, n)
    Mv      = S(undef, n)
    Mv_prev = S(undef, n)
    p       = S(undef, n)
    Mv_next = S(undef, n)
    v       = S(undef, 0)
    stats = LanczosStats(false, T[], false, T(NaN), T(NaN), "unknown")
    solver = new{T,FC,S}(x, Mv, Mv_prev, p, Mv_next, v, stats)
    return solver
  end

  function CgLanczosSolver(A, b)
    n, m = size(A)
    S = ktypeof(b)
    CgLanczosSolver(n, m, S)
  end
end

"""
Type for storing the vectors required by the in-place version of CG-LANCZOS with shifts.

The outer constructors

    solver = CgLanczosShiftSolver(n, m, nshifts, S)
    solver = CgLanczosShiftSolver(A, b, nshifts)

may be used in order to create these vectors.
"""
mutable struct CgLanczosShiftSolver{T,FC,S} <: KrylovSolver{T,FC,S}
  Mv         :: S
  Mv_prev    :: S
  Mv_next    :: S
  v          :: S
  x          :: Vector{S}
  p          :: Vector{S}
  σ          :: Vector{T}
  δhat       :: Vector{T}
  ω          :: Vector{T}
  γ          :: Vector{T}
  rNorms     :: Vector{T}
  converged  :: BitVector
  not_cv     :: BitVector
  stats      :: LanczosShiftStats{T}

  function CgLanczosShiftSolver(n, m, nshifts, S)
    FC         = eltype(S)
    T          = real(FC)
    Mv         = S(undef, n)
    Mv_prev    = S(undef, n)
    Mv_next    = S(undef, n)
    v          = S(undef, 0)
    x          = [S(undef, n) for i = 1 : nshifts]
    p          = [S(undef, n) for i = 1 : nshifts]
    σ          = Vector{T}(undef, nshifts)
    δhat       = Vector{T}(undef, nshifts)
    ω          = Vector{T}(undef, nshifts)
    γ          = Vector{T}(undef, nshifts)
    rNorms     = Vector{T}(undef, nshifts)
    indefinite = BitVector(undef, nshifts)
    converged  = BitVector(undef, nshifts)
    not_cv     = BitVector(undef, nshifts)
    stats = LanczosShiftStats(false, [T[] for i = 1 : nshifts], indefinite, T(NaN), T(NaN), "unknown")
    solver = new{T,FC,S}(Mv, Mv_prev, Mv_next, v, x, p, σ, δhat, ω, γ, rNorms, converged, not_cv, stats)
    return solver
  end

  function CgLanczosShiftSolver(A, b, nshifts)
    n, m = size(A)
    S = ktypeof(b)
    CgLanczosShiftSolver(n, m, nshifts, S)
  end
end

"""
Type for storing the vectors required by the in-place version of MINRES-QLP.

The outer constructors

    solver = MinresQlpSolver(n, m, S)
    solver = MinresQlpSolver(A, b)

may be used in order to create these vectors.
"""
mutable struct MinresQlpSolver{T,FC,S} <: KrylovSolver{T,FC,S}
  Δx      :: S
  wₖ₋₁    :: S
  wₖ      :: S
  M⁻¹vₖ₋₁ :: S
  M⁻¹vₖ   :: S
  x       :: S
  p       :: S
  vₖ      :: S
  stats   :: SimpleStats{T}

  function MinresQlpSolver(n, m, S)
    FC      = eltype(S)
    T       = real(FC)
    Δx      = S(undef, 0)
    wₖ₋₁    = S(undef, n)
    wₖ      = S(undef, n)
    M⁻¹vₖ₋₁ = S(undef, n)
    M⁻¹vₖ   = S(undef, n)
    x       = S(undef, n)
    p       = S(undef, n)
    vₖ      = S(undef, 0)
    stats = SimpleStats(false, false, T[], T[], T[], "unknown")
    solver = new{T,FC,S}(Δx, wₖ₋₁, wₖ, M⁻¹vₖ₋₁, M⁻¹vₖ, x, p, vₖ, stats)
    return solver
  end

  function MinresQlpSolver(A, b)
    n, m = size(A)
    S = ktypeof(b)
    MinresQlpSolver(n, m, S)
  end
end

"""
Type for storing the vectors required by the in-place version of DQGMRES.

The outer constructors

    solver = DqgmresSolver(n, m, memory, S)
    solver = DqgmresSolver(A, b, memory)

may be used in order to create these vectors.
"""
mutable struct DqgmresSolver{T,FC,S} <: KrylovSolver{T,FC,S}
  Δx    :: S
  x     :: S
  t     :: S
  z     :: S
  w     :: S
  P     :: Vector{S}
  V     :: Vector{S}
  c     :: Vector{T}
  s     :: Vector{T}
  H     :: Vector{T}
  stats :: SimpleStats{T}

  function DqgmresSolver(n, m, memory, S)
    FC = eltype(S)
    T  = real(FC)
    Δx = S(undef, 0)
    x  = S(undef, n)
    t  = S(undef, n)
    z  = S(undef, 0)
    w  = S(undef, 0)
    P  = [S(undef, n) for i = 1 : memory]
    V  = [S(undef, n) for i = 1 : memory]
    c  = Vector{T}(undef, memory)
    s  = Vector{T}(undef, memory)
    H  = Vector{T}(undef, memory+2)
    stats = SimpleStats(false, false, T[], T[], T[], "unknown")
    solver = new{T,FC,S}(Δx, x, t, z, w, P, V, c, s, H, stats)
    return solver
  end

  function DqgmresSolver(A, b, memory)
    n, m = size(A)
    S = ktypeof(b)
    DqgmresSolver(n, m, memory, S)
  end
end

"""
Type for storing the vectors required by the in-place version of DIOM.

The outer constructors

    solver = DiomSolver(n, m, memory, S)
    solver = DiomSolver(A, b, memory)

may be used in order to create these vectors.
"""
mutable struct DiomSolver{T,FC,S} <: KrylovSolver{T,FC,S}
  Δx    :: S
  x     :: S
  t     :: S
  z     :: S
  w     :: S
  P     :: Vector{S}
  V     :: Vector{S}
  L     :: Vector{T}
  H     :: Vector{T}
  stats :: SimpleStats{T}

  function DiomSolver(n, m, memory, S)
    FC  = eltype(S)
    T   = real(FC)
    Δx = S(undef, 0)
    x  = S(undef, n)
    t  = S(undef, n)
    z  = S(undef, 0)
    w  = S(undef, 0)
    P  = [S(undef, n) for i = 1 : memory]
    V  = [S(undef, n) for i = 1 : memory]
    L  = Vector{T}(undef, memory)
    H  = Vector{T}(undef, memory+2)
    stats = SimpleStats(false, false, T[], T[], T[], "unknown")
    solver = new{T,FC,S}(Δx, x, t, z, w, P, V, L, H, stats)
    return solver
  end

  function DiomSolver(A, b, memory)
    n, m = size(A)
    S = ktypeof(b)
    DiomSolver(n, m, memory, S)
  end
end

"""
Type for storing the vectors required by the in-place version of USYMLQ.

The outer constructors

    solver = UsymlqSolver(n, m, S)
    solver = UsymlqSolver(A, b)

may be used in order to create these vectors.
"""
mutable struct UsymlqSolver{T,FC,S} <: KrylovSolver{T,FC,S}
  uₖ₋₁  :: S
  uₖ    :: S
  p     :: S
  x     :: S
  d̅     :: S
  vₖ₋₁  :: S
  vₖ    :: S
  q     :: S
  stats :: SimpleStats{T}

  function UsymlqSolver(n, m, S)
    FC   = eltype(S)
    T    = real(FC)
    uₖ₋₁ = S(undef, m)
    uₖ   = S(undef, m)
    p    = S(undef, m)
    x    = S(undef, m)
    d̅    = S(undef, m)
    vₖ₋₁ = S(undef, n)
    vₖ   = S(undef, n)
    q    = S(undef, n)
    stats = SimpleStats(false, false, T[], T[], T[], "unknown")
    solver = new{T,FC,S}(uₖ₋₁, uₖ, p, x, d̅, vₖ₋₁, vₖ, q, stats)
    return solver
  end

  function UsymlqSolver(A, b)
    n, m = size(A)
    S = ktypeof(b)
    UsymlqSolver(n, m, S)
  end
end

"""
Type for storing the vectors required by the in-place version of USYMQR.

The outer constructors

    solver = UsymqrSolver(n, m, S)
    solver = UsymqrSolver(A, b)

may be used in order to create these vectors.
"""
mutable struct UsymqrSolver{T,FC,S} <: KrylovSolver{T,FC,S}
  vₖ₋₁  :: S
  vₖ    :: S
  q     :: S
  x     :: S
  wₖ₋₂  :: S
  wₖ₋₁  :: S
  uₖ₋₁  :: S
  uₖ    :: S
  p     :: S
  stats :: SimpleStats{T}

  function UsymqrSolver(n, m, S)
    FC   = eltype(S)
    T    = real(FC)
    vₖ₋₁ = S(undef, n)
    vₖ   = S(undef, n)
    q    = S(undef, n)
    x    = S(undef, m)
    wₖ₋₂ = S(undef, m)
    wₖ₋₁ = S(undef, m)
    uₖ₋₁ = S(undef, m)
    uₖ   = S(undef, m)
    p    = S(undef, m)
    stats = SimpleStats(false, false, T[], T[], T[], "unknown")
    solver = new{T,FC,S}(vₖ₋₁, vₖ, q, x, wₖ₋₂, wₖ₋₁, uₖ₋₁, uₖ, p, stats)
    return solver
  end

  function UsymqrSolver(A, b)
    n, m = size(A)
    S = ktypeof(b)
    UsymqrSolver(n, m, S)
  end
end

"""
Type for storing the vectors required by the in-place version of TRICG.

The outer constructors

    solver = TricgSolver(n, m, S)
    solver = TricgSolver(A, b)

may be used in order to create these vectors.
"""
mutable struct TricgSolver{T,FC,S} <: KrylovSolver{T,FC,S}
  y       :: S
  N⁻¹uₖ₋₁ :: S
  N⁻¹uₖ   :: S
  p       :: S
  gy₂ₖ₋₁  :: S
  gy₂ₖ    :: S
  x       :: S
  M⁻¹vₖ₋₁ :: S
  M⁻¹vₖ   :: S
  q       :: S
  gx₂ₖ₋₁  :: S
  gx₂ₖ    :: S
  Δx      :: S
  Δy      :: S
  uₖ      :: S
  vₖ      :: S
  stats   :: SimpleStats{T}

  function TricgSolver(n, m, S)
    FC      = eltype(S)
    T       = real(FC)
    y       = S(undef, m)
    N⁻¹uₖ₋₁ = S(undef, m)
    N⁻¹uₖ   = S(undef, m)
    p       = S(undef, m)
    gy₂ₖ₋₁  = S(undef, m)
    gy₂ₖ    = S(undef, m)
    x       = S(undef, n)
    M⁻¹vₖ₋₁ = S(undef, n)
    M⁻¹vₖ   = S(undef, n)
    q       = S(undef, n)
    gx₂ₖ₋₁  = S(undef, n)
    gx₂ₖ    = S(undef, n)
    Δx      = S(undef, 0)
    Δy      = S(undef, 0)
    uₖ      = S(undef, 0)
    vₖ      = S(undef, 0)
    stats = SimpleStats(false, false, T[], T[], T[], "unknown")
    solver = new{T,FC,S}(y, N⁻¹uₖ₋₁, N⁻¹uₖ, p, gy₂ₖ₋₁, gy₂ₖ, x, M⁻¹vₖ₋₁, M⁻¹vₖ, q, gx₂ₖ₋₁, gx₂ₖ, Δx, Δy, uₖ, vₖ, stats)
    return solver
  end

  function TricgSolver(A, b)
    n, m = size(A)
    S = ktypeof(b)
    TricgSolver(n, m, S)
  end
end

"""
Type for storing the vectors required by the in-place version of TRIMR.

The outer constructors

    solver = TrimrSolver(n, m, S)
    solver = TrimrSolver(A, b)

may be used in order to create these vectors.
"""
mutable struct TrimrSolver{T,FC,S} <: KrylovSolver{T,FC,S}
  y       :: S
  N⁻¹uₖ₋₁ :: S
  N⁻¹uₖ   :: S
  p       :: S
  gy₂ₖ₋₃  :: S
  gy₂ₖ₋₂  :: S
  gy₂ₖ₋₁  :: S
  gy₂ₖ    :: S
  x       :: S
  M⁻¹vₖ₋₁ :: S
  M⁻¹vₖ   :: S
  q       :: S
  gx₂ₖ₋₃  :: S
  gx₂ₖ₋₂  :: S
  gx₂ₖ₋₁  :: S
  gx₂ₖ    :: S
  Δx      :: S
  Δy      :: S
  uₖ      :: S
  vₖ      :: S
  stats   :: SimpleStats{T}

  function TrimrSolver(n, m, S)
    FC      = eltype(S)
    T       = real(FC)
    y       = S(undef, m)
    N⁻¹uₖ₋₁ = S(undef, m)
    N⁻¹uₖ   = S(undef, m)
    p       = S(undef, m)
    gy₂ₖ₋₃  = S(undef, m)
    gy₂ₖ₋₂  = S(undef, m)
    gy₂ₖ₋₁  = S(undef, m)
    gy₂ₖ    = S(undef, m)
    x       = S(undef, n)
    M⁻¹vₖ₋₁ = S(undef, n)
    M⁻¹vₖ   = S(undef, n)
    q       = S(undef, n)
    gx₂ₖ₋₃  = S(undef, n)
    gx₂ₖ₋₂  = S(undef, n)
    gx₂ₖ₋₁  = S(undef, n)
    gx₂ₖ    = S(undef, n)
    Δx      = S(undef, 0)
    Δy      = S(undef, 0)
    uₖ      = S(undef, 0)
    vₖ      = S(undef, 0)
    stats = SimpleStats(false, false, T[], T[], T[], "unknown")
    solver = new{T,FC,S}(y, N⁻¹uₖ₋₁, N⁻¹uₖ, p, gy₂ₖ₋₃, gy₂ₖ₋₂, gy₂ₖ₋₁, gy₂ₖ, x, M⁻¹vₖ₋₁, M⁻¹vₖ, q, gx₂ₖ₋₃, gx₂ₖ₋₂, gx₂ₖ₋₁, gx₂ₖ, Δx, Δy, uₖ, vₖ, stats)
    return solver
  end

  function TrimrSolver(A, b)
    n, m = size(A)
    S = ktypeof(b)
    TrimrSolver(n, m, S)
  end
end

"""
Type for storing the vectors required by the in-place version of TRILQR.

The outer constructors

    solver = TrilqrSolver(n, m, S)
    solver = TrilqrSolver(A, b)

may be used in order to create these vectors.
"""
mutable struct TrilqrSolver{T,FC,S} <: KrylovSolver{T,FC,S}
  uₖ₋₁  :: S
  uₖ    :: S
  p     :: S
  d̅     :: S
  x     :: S
  vₖ₋₁  :: S
  vₖ    :: S
  q     :: S
  y     :: S
  wₖ₋₃  :: S
  wₖ₋₂  :: S
  stats :: AdjointStats{T}

  function TrilqrSolver(n, m, S)
    FC   = eltype(S)
    T    = real(FC)
    uₖ₋₁ = S(undef, m)
    uₖ   = S(undef, m)
    p    = S(undef, m)
    d̅    = S(undef, m)
    x    = S(undef, m)
    vₖ₋₁ = S(undef, n)
    vₖ   = S(undef, n)
    q    = S(undef, n)
    y    = S(undef, n)
    wₖ₋₃ = S(undef, n)
    wₖ₋₂ = S(undef, n)
    stats = AdjointStats(false, false, T[], T[], "unknown")
    solver = new{T,FC,S}(uₖ₋₁, uₖ, p, d̅, x, vₖ₋₁, vₖ, q, y, wₖ₋₃, wₖ₋₂, stats)
    return solver
  end

  function TrilqrSolver(A, b)
    n, m = size(A)
    S = ktypeof(b)
    TrilqrSolver(n, m, S)
  end
end

"""
Type for storing the vectors required by the in-place version of CGS.

The outer constructorss

    solver = CgsSolver(n, m, S)
    solver = CgsSolver(A, b)

may be used in order to create these vectors.
"""
mutable struct CgsSolver{T,FC,S} <: KrylovSolver{T,FC,S}
  x     :: S
  r     :: S
  u     :: S
  p     :: S
  q     :: S
  ts    :: S
  yz    :: S
  vw    :: S
  stats :: SimpleStats{T}

  function CgsSolver(n, m, S)
    FC = eltype(S)
    T  = real(FC)
    x  = S(undef, n)
    r  = S(undef, n)
    u  = S(undef, n)
    p  = S(undef, n)
    q  = S(undef, n)
    ts = S(undef, n)
    yz = S(undef, 0)
    vw = S(undef, 0)
    stats = SimpleStats(false, false, T[], T[], T[], "unknown")
    solver = new{T,FC,S}(x, r, u, p, q, ts, yz, vw, stats)
    return solver
  end

  function CgsSolver(A, b)
    n, m = size(A)
    S = ktypeof(b)
    CgsSolver(n, m, S)
  end
end

"""
Type for storing the vectors required by the in-place version of BICGSTAB.

The outer constructors

    solver = BicgstabSolver(n, m, S)
    solver = BicgstabSolver(A, b)

may be used in order to create these vectors.
"""
mutable struct BicgstabSolver{T,FC,S} <: KrylovSolver{T,FC,S}
  x     :: S
  r     :: S
  p     :: S
  v     :: S
  s     :: S
  qd    :: S
  yz    :: S
  t     :: S
  stats :: SimpleStats{T}

  function BicgstabSolver(n, m, S)
    FC = eltype(S)
    T  = real(FC)
    x  = S(undef, n)
    r  = S(undef, n)
    p  = S(undef, n)
    v  = S(undef, n)
    s  = S(undef, n)
    qd = S(undef, n)
    yz = S(undef, 0)
    t  = S(undef, 0)
    stats = SimpleStats(false, false, T[], T[], T[], "unknown")
    solver = new{T,FC,S}(x, r, p, v, s, qd, yz, t, stats)
    return solver
  end

  function BicgstabSolver(A, b)
    n, m = size(A)
    S = ktypeof(b)
    BicgstabSolver(n, m, S)
  end
end

"""
Type for storing the vectors required by the in-place version of BILQ.

The outer constructors

    solver = BilqSolver(n, m, S)
    solver = BilqSolver(A, b)

may be used in order to create these vectors.
"""
mutable struct BilqSolver{T,FC,S} <: KrylovSolver{T,FC,S}
  uₖ₋₁  :: S
  uₖ    :: S
  q     :: S
  vₖ₋₁  :: S
  vₖ    :: S
  p     :: S
  x     :: S
  d̅     :: S
  stats :: SimpleStats{T}

  function BilqSolver(n, m, S)
    FC   = eltype(S)
    T    = real(FC)
    uₖ₋₁ = S(undef, n)
    uₖ   = S(undef, n)
    q    = S(undef, n)
    vₖ₋₁ = S(undef, n)
    vₖ   = S(undef, n)
    p    = S(undef, n)
    x    = S(undef, n)
    d̅    = S(undef, n)
    stats = SimpleStats(false, false, T[], T[], T[], "unknown")
    solver = new{T,FC,S}(uₖ₋₁, uₖ, q, vₖ₋₁, vₖ, p, x, d̅, stats)
    return solver
  end

  function BilqSolver(A, b)
    n, m = size(A)
    S = ktypeof(b)
    BilqSolver(n, m, S)
  end
end

"""
Type for storing the vectors required by the in-place version of QMR.

The outer constructors

    solver = QmrSolver(n, m, S)
    solver = QmrSolver(A, b)

may be used in order to create these vectors.
"""
mutable struct QmrSolver{T,FC,S} <: KrylovSolver{T,FC,S}
  uₖ₋₁  :: S
  uₖ    :: S
  q     :: S
  vₖ₋₁  :: S
  vₖ    :: S
  p     :: S
  x     :: S
  wₖ₋₂  :: S
  wₖ₋₁  :: S
  stats :: SimpleStats{T}

  function QmrSolver(n, m, S)
    FC   = eltype(S)
    T    = real(FC)
    uₖ₋₁ = S(undef, n)
    uₖ   = S(undef, n)
    q    = S(undef, n)
    vₖ₋₁ = S(undef, n)
    vₖ   = S(undef, n)
    p    = S(undef, n)
    x    = S(undef, n)
    wₖ₋₂ = S(undef, n)
    wₖ₋₁ = S(undef, n)
    stats = SimpleStats(false, false, T[], T[], T[], "unknown")
    solver = new{T,FC,S}(uₖ₋₁, uₖ, q, vₖ₋₁, vₖ, p, x, wₖ₋₂, wₖ₋₁, stats)
    return solver
  end

  function QmrSolver(A, b)
    n, m = size(A)
    S = ktypeof(b)
    QmrSolver(n, m, S)
  end
end

"""
Type for storing the vectors required by the in-place version of BILQR.

The outer constructors

    solver = BilqrSolver(n, m, S)
    solver = BilqrSolver(A, b)

may be used in order to create these vectors.
"""
mutable struct BilqrSolver{T,FC,S} <: KrylovSolver{T,FC,S}
  uₖ₋₁  :: S
  uₖ    :: S
  q     :: S
  vₖ₋₁  :: S
  vₖ    :: S
  p     :: S
  x     :: S
  y     :: S
  d̅     :: S
  wₖ₋₃  :: S
  wₖ₋₂  :: S
  stats :: AdjointStats{T}

  function BilqrSolver(n, m, S)
    FC   = eltype(S)
    T    = real(FC)
    uₖ₋₁ = S(undef, n)
    uₖ   = S(undef, n)
    q    = S(undef, n)
    vₖ₋₁ = S(undef, n)
    vₖ   = S(undef, n)
    p    = S(undef, n)
    x    = S(undef, n)
    y    = S(undef, n)
    d̅    = S(undef, n)
    wₖ₋₃ = S(undef, n)
    wₖ₋₂ = S(undef, n)
    stats = AdjointStats(false, false, T[], T[], "unknown")
    solver = new{T,FC,S}(uₖ₋₁, uₖ, q, vₖ₋₁, vₖ, p, x, y, d̅, wₖ₋₃, wₖ₋₂, stats)
    return solver
  end

  function BilqrSolver(A, b)
    n, m = size(A)
    S = ktypeof(b)
    BilqrSolver(n, m, S)
  end
end

"""
Type for storing the vectors required by the in-place version of CGLS.

The outer constructors

    solver = CglsSolver(n, m, S)
    solver = CglsSolver(A, b)

may be used in order to create these vectors.
"""
mutable struct CglsSolver{T,FC,S} <: KrylovSolver{T,FC,S}
  x     :: S
  p     :: S
  s     :: S
  r     :: S
  q     :: S
  Mr    :: S
  stats :: SimpleStats{T}

  function CglsSolver(n, m, S)
    FC = eltype(S)
    T  = real(FC)
    x  = S(undef, m)
    p  = S(undef, m)
    s  = S(undef, m)
    r  = S(undef, n)
    q  = S(undef, n)
    Mr = S(undef, 0)
    stats = SimpleStats(false, false, T[], T[], T[], "unknown")
    solver = new{T,FC,S}(x, p, s, r, q, Mr, stats)
    return solver
  end

  function CglsSolver(A, b)
    n, m = size(A)
    S = ktypeof(b)
    CglsSolver(n, m, S)
  end
end

"""
Type for storing the vectors required by the in-place version of CRLS.

The outer constructors

    solver = CrlsSolver(n, m, S)
    solver = CrlsSolver(A, b)

may be used in order to create these vectors.
"""
mutable struct CrlsSolver{T,FC,S} <: KrylovSolver{T,FC,S}
  x     :: S
  p     :: S
  Ar    :: S
  q     :: S
  r     :: S
  Ap    :: S
  s     :: S
  Ms    :: S
  stats :: SimpleStats{T}

  function CrlsSolver(n, m, S)
    FC = eltype(S)
    T  = real(FC)
    x  = S(undef, m)
    p  = S(undef, m)
    Ar = S(undef, m)
    q  = S(undef, m)
    r  = S(undef, n)
    Ap = S(undef, n)
    s  = S(undef, n)
    Ms = S(undef, 0)
    stats = SimpleStats(false, false, T[], T[], T[], "unknown")
    solver = new{T,FC,S}(x, p, Ar, q, r, Ap, s, Ms, stats)
    return solver
  end

  function CrlsSolver(A, b)
    n, m = size(A)
    S = ktypeof(b)
    CrlsSolver(n, m, S)
  end
end

"""
Type for storing the vectors required by the in-place version of CGNE.

The outer constructors

    solver = CgneSolver(n, m, S)
    solver = CgneSolver(A, b)

may be used in order to create these vectors.
"""
mutable struct CgneSolver{T,FC,S} <: KrylovSolver{T,FC,S}
  x     :: S
  p     :: S
  Aᵀz   :: S
  r     :: S
  q     :: S
  s     :: S
  z     :: S
  stats :: SimpleStats{T}

  function CgneSolver(n, m, S)
    FC  = eltype(S)
    T   = real(FC)
    x   = S(undef, m)
    p   = S(undef, m)
    Aᵀz = S(undef, m)
    r   = S(undef, n)
    q   = S(undef, n)
    s   = S(undef, 0)
    z   = S(undef, 0)
    stats = SimpleStats(false, false, T[], T[], T[], "unknown")
    solver = new{T,FC,S}(x, p, Aᵀz, r, q, s, z, stats)
    return solver
  end

  function CgneSolver(A, b)
    n, m = size(A)
    S = ktypeof(b)
    CgneSolver(n, m, S)
  end
end

"""
Type for storing the vectors required by the in-place version of CRMR.

The outer constructors

    solver = CrmrSolver(n, m, S)
    solver = CrmrSolver(A, b)

may be used in order to create these vectors.
"""
mutable struct CrmrSolver{T,FC,S} <: KrylovSolver{T,FC,S}
  x     :: S
  p     :: S
  Aᵀr   :: S
  r     :: S
  q     :: S
  Mq    :: S
  s     :: S
  stats :: SimpleStats{T}

  function CrmrSolver(n, m, S)
    FC  = eltype(S)
    T   = real(FC)
    x   = S(undef, m)
    p   = S(undef, m)
    Aᵀr = S(undef, m)
    r   = S(undef, n)
    q   = S(undef, n)
    Mq  = S(undef, 0)
    s   = S(undef, 0)
    stats = SimpleStats(false, false, T[], T[], T[], "unknown")
    solver = new{T,FC,S}(x, p, Aᵀr, r, q, Mq, s, stats)
    return solver
  end

  function CrmrSolver(A, b)
    n, m = size(A)
    S = ktypeof(b)
    CrmrSolver(n, m, S)
  end
end

"""
Type for storing the vectors required by the in-place version of LSLQ.

The outer constructors

    solver = LslqSolver(n, m, S)
    solver = LslqSolver(A, b)

may be used in order to create these vectors.
"""
mutable struct LslqSolver{T,FC,S} <: KrylovSolver{T,FC,S}
  x       :: S
  Nv      :: S
  Aᵀu     :: S
  w̄       :: S
  Mu      :: S
  Av      :: S
  u       :: S
  v       :: S
  err_vec :: Vector{T}
  stats   :: LSLQStats{T}

  function LslqSolver(n, m, S; window :: Int=5)
    FC  = eltype(S)
    T   = real(FC)
    x   = S(undef, m)
    Nv  = S(undef, m)
    Aᵀu = S(undef, m)
    w̄   = S(undef, m)
    Mu  = S(undef, n)
    Av  = S(undef, n)
    u   = S(undef, 0)
    v   = S(undef, 0)
    err_vec = zeros(T, window)
    stats = LSLQStats(false, false, T[], T[], T[], false, T[], T[], "unknown")
    solver = new{T,FC,S}(x, Nv, Aᵀu, w̄, Mu, Av, u, v, err_vec, stats)
    return solver
  end

  function LslqSolver(A, b; window :: Int=5)
    n, m = size(A)
    S = ktypeof(b)
    LslqSolver(n, m, S, window=window)
  end
end

"""
Type for storing the vectors required by the in-place version of LSQR.

The outer constructors

    solver = LsqrSolver(n, m, S)
    solver = LsqrSolver(A, b)

may be used in order to create these vectors.
"""
mutable struct LsqrSolver{T,FC,S} <: KrylovSolver{T,FC,S}
  x       :: S
  Nv      :: S
  Aᵀu     :: S
  w       :: S
  Mu      :: S
  Av      :: S
  u       :: S
  v       :: S
  err_vec :: Vector{T}
  stats   :: SimpleStats{T}

  function LsqrSolver(n, m, S; window :: Int=5)
    FC  = eltype(S)
    T   = real(FC)
    x   = S(undef, m)
    Nv  = S(undef, m)
    Aᵀu = S(undef, m)
    w   = S(undef, m)
    Mu  = S(undef, n)
    Av  = S(undef, n)
    u   = S(undef, 0)
    v   = S(undef, 0)
    err_vec = zeros(T, window)
    stats = SimpleStats(false, false, T[], T[], T[], "unknown")
    solver = new{T,FC,S}(x, Nv, Aᵀu, w, Mu, Av, u, v, err_vec, stats)
    return solver
  end

  function LsqrSolver(A, b; window :: Int=5)
    n, m = size(A)
    S = ktypeof(b)
    LsqrSolver(n, m, S, window=window)
  end
end

"""
Type for storing the vectors required by the in-place version of LSMR.

The outer constructors

    solver = LsmrSolver(n, m, S)
    solver = LsmrSolver(A, b)

may be used in order to create these vectors.
"""
mutable struct LsmrSolver{T,FC,S} <: KrylovSolver{T,FC,S}
  x       :: S
  Nv      :: S
  Aᵀu     :: S
  h       :: S
  hbar    :: S
  Mu      :: S
  Av      :: S
  u       :: S
  v       :: S
  err_vec :: Vector{T}
  stats   :: SimpleStats{T}

  function LsmrSolver(n, m, S; window :: Int=5)
    FC   = eltype(S)
    T    = real(FC)
    x    = S(undef, m)
    Nv   = S(undef, m)
    Aᵀu  = S(undef, m)
    h    = S(undef, m)
    hbar = S(undef, m)
    Mu   = S(undef, n)
    Av   = S(undef, n)
    u    = S(undef, 0)
    v    = S(undef, 0)
    err_vec = zeros(T, window)
    stats = SimpleStats(false, false, T[], T[], T[], "unknown")
    solver = new{T,FC,S}(x, Nv, Aᵀu, h, hbar, Mu, Av, u, v, err_vec, stats)
    return solver
  end

  function LsmrSolver(A, b; window :: Int=5)
    n, m = size(A)
    S = ktypeof(b)
    LsmrSolver(n, m, S, window=window)
  end
end

"""
Type for storing the vectors required by the in-place version of LNLQ.

The outer constructors

    solver = LnlqSolver(n, m, S)
    solver = LnlqSolver(A, b)

may be used in order to create these vectors.
"""
mutable struct LnlqSolver{T,FC,S} <: KrylovSolver{T,FC,S}
  x     :: S
  Nv    :: S
  Aᵀu   :: S
  y     :: S
  w̄     :: S
  Mu    :: S
  Av    :: S
  u     :: S
  v     :: S
  q     :: S
  stats :: LNLQStats{T}

  function LnlqSolver(n, m, S)
    FC  = eltype(S)
    T   = real(FC)
    x   = S(undef, m)
    Nv  = S(undef, m)
    Aᵀu = S(undef, m)
    y   = S(undef, n)
    w̄   = S(undef, n)
    Mu  = S(undef, n)
    Av  = S(undef, n)
    u   = S(undef, 0)
    v   = S(undef, 0)
    q   = S(undef, 0)
    stats = LNLQStats(false, T[], false, T[], T[], "unknown")
    solver = new{T,FC,S}(x, Nv, Aᵀu, y, w̄, Mu, Av, u, v, q, stats)
    return solver
  end

  function LnlqSolver(A, b)
    n, m = size(A)
    S = ktypeof(b)
    LnlqSolver(n, m, S)
  end
end

"""
Type for storing the vectors required by the in-place version of CRAIG.

The outer constructors

    solver = CraigSolver(n, m, S)
    solver = CraigSolver(A, b)

may be used in order to create these vectors.
"""
mutable struct CraigSolver{T,FC,S} <: KrylovSolver{T,FC,S}
  x     :: S
  Nv    :: S
  Aᵀu   :: S
  y     :: S
  w     :: S
  Mu    :: S
  Av    :: S
  u     :: S
  v     :: S
  w2    :: S
  stats :: SimpleStats{T}

  function CraigSolver(n, m, S)
    FC  = eltype(S)
    T   = real(FC)
    x   = S(undef, m)
    Nv  = S(undef, m)
    Aᵀu = S(undef, m)
    y   = S(undef, n)
    w   = S(undef, n)
    Mu  = S(undef, n)
    Av  = S(undef, n)
    u   = S(undef, 0)
    v   = S(undef, 0)
    w2  = S(undef, 0)
    stats = SimpleStats(false, false, T[], T[], T[], "unknown")
    solver = new{T,FC,S}(x, Nv, Aᵀu, y, w, Mu, Av, u, v, w2, stats)
    return solver
  end

  function CraigSolver(A, b)
    n, m = size(A)
    S = ktypeof(b)
    CraigSolver(n, m, S)
  end
end

"""
Type for storing the vectors required by the in-place version of CRAIGMR.

The outer constructors

    solver = CraigmrSolver(n, m, S)
    solver = CraigmrSolver(A, b)

may be used in order to create these vectors.
"""
mutable struct CraigmrSolver{T,FC,S} <: KrylovSolver{T,FC,S}
  x     :: S
  Nv    :: S
  Aᵀu   :: S
  d     :: S
  y     :: S
  Mu    :: S
  w     :: S
  wbar  :: S
  Av    :: S
  u     :: S
  v     :: S
  q     :: S
  stats :: SimpleStats{T}

  function CraigmrSolver(n, m, S)
    FC   = eltype(S)
    T    = real(FC)
    x    = S(undef, m)
    Nv   = S(undef, m)
    Aᵀu  = S(undef, m)
    d    = S(undef, m)
    y    = S(undef, n)
    Mu   = S(undef, n)
    w    = S(undef, n)
    wbar = S(undef, n)
    Av   = S(undef, n)
    u    = S(undef, 0)
    v    = S(undef, 0)
    q    = S(undef, 0)
    stats = SimpleStats(false, false, T[], T[], T[], "unknown")
    solver = new{T,FC,S}(x, Nv, Aᵀu, d, y, Mu, w, wbar, Av, u, v, q, stats)
    return solver
  end

  function CraigmrSolver(A, b)
    n, m = size(A)
    S = ktypeof(b)
    CraigmrSolver(n, m, S)
  end
end

"""
Type for storing the vectors required by the in-place version of GMRES.

The outer constructors

    solver = GmresSolver(n, m, memory, S)
    solver = GmresSolver(A, b, memory)

may be used in order to create these vectors.
"""
mutable struct GmresSolver{T,FC,S} <: KrylovSolver{T,FC,S}
  Δx    :: S
  x     :: S
  w     :: S
  p     :: S
  q     :: S
  V     :: Vector{S}
  c     :: Vector{T}
  s     :: Vector{T}
  z     :: Vector{T}
  R     :: Vector{T}
  stats :: SimpleStats{T}

  function GmresSolver(n, m, memory, S)
    FC = eltype(S)
    T  = real(FC)
    Δx = S(undef, 0)
    x  = S(undef, n)
    w  = S(undef, n)
    p  = S(undef, 0)
    q  = S(undef, 0)
    V  = [S(undef, n) for i = 1 : memory]
    c  = Vector{T}(undef, memory)
    s  = Vector{T}(undef, memory)
    z  = Vector{T}(undef, memory)
    R  = Vector{T}(undef, div(memory * (memory+1), 2))
    stats = SimpleStats(false, false, T[], T[], T[], "unknown")
    solver = new{T,FC,S}(Δx, x, w, p, q, V, c, s, z, R, stats)
    return solver
  end

  function GmresSolver(A, b, memory)
    n, m = size(A)
    S = ktypeof(b)
    GmresSolver(n, m, memory, S)
  end
end

"""
Type for storing the vectors required by the in-place version of FOM.

The outer constructors

    solver = FomSolver(n, m, memory, S)
    solver = FomSolver(A, b, memory)

may be used in order to create these vectors.
"""
mutable struct FomSolver{T,FC,S} <: KrylovSolver{T,FC,S}
  Δx    :: S
  x     :: S
  w     :: S
  p     :: S
  q     :: S
  V     :: Vector{S}
  l     :: Vector{T}
  z     :: Vector{T}
  U     :: Vector{T}
  stats :: SimpleStats{T}

  function FomSolver(n, m, memory, S)
    FC = eltype(S)
    T  = real(FC)
    Δx = S(undef, 0)
    x  = S(undef, n)
    w  = S(undef, n)
    p  = S(undef, 0)
    q  = S(undef, 0)
    V  = [S(undef, n) for i = 1 : memory]
    l  = Vector{T}(undef, memory)
    z  = Vector{T}(undef, memory)
    U  = Vector{T}(undef, div(memory * (memory+1), 2))
    stats = SimpleStats(false, false, T[], T[], T[], "unknown")
    solver = new{T,FC,S}(Δx, x, w, p, q, V, l, z, U, stats)
    return solver
  end

  function FomSolver(A, b, memory)
    n, m = size(A)
    S = ktypeof(b)
    FomSolver(n, m, memory, S)
  end
end

"""
Type for storing the vectors required by the in-place version of GPMR.

The outer constructors

    solver = GpmrSolver(n, m, memory, S)
    solver = GpmrSolver(A, b, memory)

may be used in order to create these vectors.
"""
mutable struct GpmrSolver{T,FC,S} <: KrylovSolver{T,FC,S}
  wA    :: S
  wB    :: S
  dA    :: S
  dB    :: S
  Δx    :: S
  Δy    :: S
  x     :: S
  y     :: S
  q     :: S
  p     :: S
  V     :: Vector{S}
  U     :: Vector{S}
  gs    :: Vector{T}
  gc    :: Vector{T}
  zt    :: Vector{T}
  R     :: Vector{T}
  stats :: SimpleStats{T}

  function GpmrSolver(n, m, memory, S)
    FC = eltype(S)
    T  = real(FC)
    wA = S(undef, 0)
    wB = S(undef, 0)
    dA = S(undef, n)
    dB = S(undef, m)
    Δx = S(undef, 0)
    Δy = S(undef, 0)
    x  = S(undef, n)
    y  = S(undef, m)
    q  = S(undef, 0)
    p  = S(undef, 0)
    V  = [S(undef, n) for i = 1 : memory]
    U  = [S(undef, m) for i = 1 : memory]
    gs = Vector{T}(undef, 4 * memory)
    gc = Vector{T}(undef, 4 * memory)
    zt = Vector{T}(undef, 2 * memory)
    R  = Vector{T}(undef, memory * (2memory + 1))
    stats = SimpleStats(false, false, T[], T[], T[], "unknown")
    solver = new{T,FC,S}(wA, wB, dA, dB, Δx, Δy, x, y, q, p, V, U, gs, gc, zt, R, stats)
    return solver
  end

  function GpmrSolver(A, b, memory)
    n, m = size(A)
    S = ktypeof(b)
    GpmrSolver(n, m, memory, S)
  end
end

"""
    solve!(solver, args...; kwargs...)

Use the in-place Krylov method associated to `solver`.
"""
function solve! end

"""
    solution(solver)

Return the solution(s) stored in the `solver`.
Optionally you can specify which solution you want to recover,
`solution(solver, 1)` returns `x` and `solution(solver, 2)` returns `y`.
"""
function solution end

"""
    nsolution(solver)

Return the number of outputs of `solution(solver)`.
"""
function nsolution end

"""
    statistics(solver)

Return the statistics stored in the `solver`.
"""
function statistics end

"""
    issolved(solver)

Return a boolean that determines whether the Krylov method associated to `solver` succeeded.
"""
function issolved end

for (KS, fun, nsol) in [
  (LsmrSolver          , :lsmr!      , 1)
  (CgsSolver           , :cgs!       , 1)
  (UsymlqSolver        , :usymlq!    , 1)
  (LnlqSolver          , :lnlq!      , 2)
  (BicgstabSolver      , :bicgstab!  , 1)
  (CrlsSolver          , :crls!      , 1)
  (LsqrSolver          , :lsqr!      , 1)
  (MinresSolver        , :minres!    , 1)
  (CgneSolver          , :cgne!      , 1)
  (DqgmresSolver       , :dqgmres!   , 1)
  (SymmlqSolver        , :symmlq!    , 1)
  (TrimrSolver         , :trimr!     , 2)
  (UsymqrSolver        , :usymqr!    , 1)
  (BilqrSolver         , :bilqr!     , 2)
  (CrSolver            , :cr!        , 1)
  (CraigmrSolver       , :craigmr!   , 2)
  (TricgSolver         , :tricg!     , 2)
  (CraigSolver         , :craig!     , 2)
  (DiomSolver          , :diom!      , 1)
  (LslqSolver          , :lslq!      , 1)
  (TrilqrSolver        , :trilqr!    , 2)
  (CrmrSolver          , :crmr!      , 1)
  (CgSolver            , :cg!        , 1)
  (CgLanczosShiftSolver, :cg_lanczos!, 1)
  (CglsSolver          , :cgls!      , 1)
  (CgLanczosSolver     , :cg_lanczos!, 1)
  (BilqSolver          , :bilq!      , 1)
  (MinresQlpSolver     , :minres_qlp!, 1)
  (QmrSolver           , :qmr!       , 1)
  (GmresSolver         , :gmres!     , 1)
  (FomSolver           , :fom!       , 1)
  (GpmrSolver          , :gpmr!      , 2)
]
  @eval begin
    @inline solve!(solver :: $KS, args...; kwargs...) = $(fun)(solver, args...; kwargs...)
    @inline statistics(solver :: $KS) = solver.stats
    @inline nsolution(solver :: $KS) = $nsol
    ($nsol == 1) && @inline solution(solver :: $KS) = solver.x
    ($nsol == 2) && @inline solution(solver :: $KS) = solver.x, solver.y
    ($nsol == 1) && @inline solution(solver :: $KS, p :: Integer) = (p == 1) ? solution(solver) : error("solution(solver) has only one output.")
    ($nsol == 2) && @inline solution(solver :: $KS, p :: Integer) = (1 ≤ p ≤ 2) ? solution(solver)[p] : error("solution(solver) has only two outputs.")
    if $KS ∈ (BilqrSolver, TrilqrSolver)
      @inline issolved_primal(solver :: $KS) = solver.stats.solved_primal
      @inline issolved_dual(solver :: $KS) = solver.stats.solved_dual
      @inline issolved(solver :: $KS) = issolved_primal(solver) && issolved_dual(solver)
    else
      @inline issolved(solver :: $KS) = solver.stats.solved
    end
  end
end

function show(io :: IO, solver :: KrylovSolver)
  workspace = typeof(solver)
  name_solver = workspace.name.wrapper
  precision = workspace.parameters[2]
  architecture = workspace.parameters[3] <: Vector ? "CPU" : "GPU"
  @printf(io, "┌%s┬%s┬%s┐\n", "─"^20, "─"^26, "─"^18)
  @printf(io, "│%20s│%26s│%18s│\n", name_solver, "Precision: $precision", "Architecture: $architecture")
  @printf(io, "├%s┼%s┼%s┤\n", "─"^20, "─"^26, "─"^18)
  @printf(io, "│%20s│%26s│%18s│\n", "Attribute", "Type", "Size")
  @printf(io, "├%s┼%s┼%s┤\n", "─"^20, "─"^26, "─"^18)
  for i=1:fieldcount(typeof(solver))-1 # show stats seperately
    type_i = fieldtype(typeof(solver), i)
    name_i = fieldname(typeof(solver), i)
    len = if type_i <: AbstractVector
      field_i = getfield(solver, name_i)
      ni = length(field_i)
      if eltype(type_i) <: AbstractVector
        "$(ni) x $(length(field_i[1]))"
      else
        length(field_i)
      end
    else
      0
    end
    if name_i in [:w̅, :w̄, :d̅]
      @printf(io, "│%21s│%26s│%18s│\n", string(name_i), type_i, len)
    else
      @printf(io, "│%20s│%26s│%18s│\n", string(name_i), type_i, len)
    end
  end
  @printf(io, "└%s┴%s┴%s┘\n","─"^20,"─"^26,"─"^18)
  print(io, solver.stats)
end
