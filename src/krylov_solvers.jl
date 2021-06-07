export KrylovSolver, MinresSolver, CgSolver, CrSolver, SymmlqSolver, CgLanczosSolver,
CgLanczosShiftSolver, MinresQlpSolver, DqgmresSolver, DiomSolver, UsymlqSolver,
UsymqrSolver, TricgSolver, TrimrSolver, TrilqrSolver, CgsSolver, BicgstabSolver,
BilqSolver, QmrSolver, BilqrSolver, CglsSolver, CrlsSolver, CgneSolver, CrmrSolver,
LslqSolver, LsqrSolver, LsmrSolver, LnlqSolver, CraigSolver, CraigmrSolver

"Abstract type for using Krylov solvers in-place"
abstract type KrylovSolver{T,S} end

"""
Type for storing the vectors required by the in-place version of MINRES.

The outer constructors

    solver = MinresSolver(n, m, S; window :: Int=5)
    solver = MinresSolver(A, b; window :: Int=5)

may be used in order to create these vectors.
"""
mutable struct MinresSolver{T,S} <: KrylovSolver{T,S}
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
    T  = eltype(S)
    x  = S(undef, n)
    r1 = S(undef, n)
    r2 = S(undef, n)
    w1 = S(undef, n)
    w2 = S(undef, n)
    y  = S(undef, n)
    v  = S(undef, 0)
    err_vec = zeros(T, window)
    stats = SimpleStats(false, true, T[], T[], "unknown")
    solver = new{T,S}(x, r1, r2, w1, w2, y, v, err_vec, stats)
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
mutable struct CgSolver{T,S} <: KrylovSolver{T,S}
  x  :: S
  r  :: S
  p  :: S
  Ap :: S
  z  :: S

  function CgSolver(n, m, S)
    T  = eltype(S)
    x  = S(undef, n)
    r  = S(undef, n)
    p  = S(undef, n)
    Ap = S(undef, n)
    z  = S(undef, 0)
    solver = new{T,S}(x, r, p, Ap, z)
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
mutable struct CrSolver{T,S} <: KrylovSolver{T,S}
  x  :: S
  r  :: S
  p  :: S
  q  :: S
  Ar :: S
  Mq :: S

  function CrSolver(n, m, S)
    T  = eltype(S)
    x  = S(undef, n)
    r  = S(undef, n)
    p  = S(undef, n)
    q  = S(undef, n)
    Ar = S(undef, n)
    Mq = S(undef, 0)
    solver = new{T,S}(x, r, p, q, Ar, Mq)
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
mutable struct SymmlqSolver{T,S} <: KrylovSolver{T,S}
  x       :: S
  Mvold   :: S
  Mv      :: S
  Mv_next :: S
  w̅       :: S
  v       :: S

  function SymmlqSolver(n, m, S)
    T       = eltype(S)
    x       = S(undef, n)
    Mvold   = S(undef, n)
    Mv      = S(undef, n)
    Mv_next = S(undef, n)
    w̅       = S(undef, n)
    v       = S(undef, 0)
    solver = new{T,S}(x, Mvold, Mv, Mv_next, w̅, v)
    return solver
  end

  function SymmlqSolver(A, b)
    n, m = size(A)
    S = ktypeof(b)
    SymmlqSolver(n, m, S)
  end
end

"""
Type for storing the vectors required by the in-place version of CG-LANCZOS.

The outer constructors

    solver = CgLanczosSolver(n, m, S)
    solver = CgLanczosSolver(A, b)

may be used in order to create these vectors.
"""
mutable struct CgLanczosSolver{T,S} <: KrylovSolver{T,S}
  x       :: S
  Mv      :: S
  Mv_prev :: S
  p       :: S
  Mv_next :: S
  v       :: S

  function CgLanczosSolver(n, m, S)
    T       = eltype(S)
    x       = S(undef, n)
    Mv      = S(undef, n)
    Mv_prev = S(undef, n)
    p       = S(undef, n)
    Mv_next = S(undef, n)
    v       = S(undef, 0)
    solver = new{T,S}(x, Mv, Mv_prev, p, Mv_next, v)
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
mutable struct CgLanczosShiftSolver{T,S} <: KrylovSolver{T,S}
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
  indefinite :: BitArray
  converged  :: BitArray
  not_cv     :: BitArray

  function CgLanczosShiftSolver(n, m, nshifts, S)
    T          = eltype(S)
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
    indefinite = BitArray(undef, nshifts)
    converged  = BitArray(undef, nshifts)
    not_cv     = BitArray(undef, nshifts)
    solver = new{T,S}(Mv, Mv_prev, Mv_next, v, x, p, σ, δhat, ω, γ, rNorms, indefinite, converged, not_cv)
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
mutable struct MinresQlpSolver{T,S} <: KrylovSolver{T,S}
  wₖ₋₁    :: S
  wₖ      :: S
  M⁻¹vₖ₋₁ :: S
  M⁻¹vₖ   :: S
  x       :: S
  p       :: S
  vₖ      :: S

  function MinresQlpSolver(n, m, S)
    T       = eltype(S)
    wₖ₋₁    = S(undef, n)
    wₖ      = S(undef, n)
    M⁻¹vₖ₋₁ = S(undef, n)
    M⁻¹vₖ   = S(undef, n)
    x       = S(undef, n)
    p       = S(undef, n)
    vₖ      = S(undef, 0)
    solver = new{T,S}(wₖ₋₁, wₖ, M⁻¹vₖ₋₁, M⁻¹vₖ, x, p, vₖ)
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
mutable struct DqgmresSolver{T,S} <: KrylovSolver{T,S}
  x :: S
  t :: S
  z :: S
  w :: S
  P :: Vector{S}
  V :: Vector{S}
  c :: Vector{T}
  s :: Vector{T}
  H :: Vector{T}

  function DqgmresSolver(n, m, memory, S)
    T = eltype(S)
    x = S(undef, n)
    t = S(undef, n)
    z = S(undef, 0)
    w = S(undef, 0)
    P = [S(undef, n) for i = 1 : memory]
    V = [S(undef, n) for i = 1 : memory]
    c = Vector{T}(undef, memory)
    s = Vector{T}(undef, memory)
    H = Vector{T}(undef, memory+2)
    solver = new{T,S}(x, t, z, w, P, V, c, s, H)
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
mutable struct DiomSolver{T,S} <: KrylovSolver{T,S}
  x     :: S
  x_old :: S
  t     :: S
  z     :: S
  w     :: S
  P     :: Vector{S}
  V     :: Vector{S}
  L     :: Vector{T}
  H     :: Vector{T}
  p     :: BitArray

  function DiomSolver(n, m, memory, S)
    T     = eltype(S)
    x     = S(undef, n)
    x_old = S(undef, n)
    t     = S(undef, n)
    z     = S(undef, 0)
    w     = S(undef, 0)
    P     = [S(undef, n) for i = 1 : memory]
    V     = [S(undef, n) for i = 1 : memory]
    L     = Vector{T}(undef, memory)
    H     = Vector{T}(undef, memory+2)
    p     = BitArray(undef, memory)
    solver = new{T,S}(x, x_old, t, z, w, P, V, L, H, p)
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
mutable struct UsymlqSolver{T,S} <: KrylovSolver{T,S}
  uₖ₋₁ :: S
  uₖ   :: S
  p    :: S
  x    :: S
  d̅    :: S
  vₖ₋₁ :: S
  vₖ   :: S
  q    :: S

  function UsymlqSolver(n, m, S)
    T    = eltype(S)
    uₖ₋₁ = S(undef, m)
    uₖ   = S(undef, m)
    p    = S(undef, m)
    x    = S(undef, m)
    d̅    = S(undef, m)
    vₖ₋₁ = S(undef, n)
    vₖ   = S(undef, n)
    q    = S(undef, n)
    solver = new{T,S}(uₖ₋₁, uₖ, p, x, d̅, vₖ₋₁, vₖ, q)
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
mutable struct UsymqrSolver{T,S} <: KrylovSolver{T,S}
  vₖ₋₁ :: S
  vₖ   :: S
  q    :: S
  x    :: S
  wₖ₋₂ :: S
  wₖ₋₁ :: S
  uₖ₋₁ :: S
  uₖ   :: S
  p    :: S

  function UsymqrSolver(n, m, S)
    T    = eltype(S)
    vₖ₋₁ = S(undef, n)
    vₖ   = S(undef, n)
    q    = S(undef, n)
    x    = S(undef, m)
    wₖ₋₂ = S(undef, m)
    wₖ₋₁ = S(undef, m)
    uₖ₋₁ = S(undef, m)
    uₖ   = S(undef, m)
    p    = S(undef, m)
    solver = new{T,S}(vₖ₋₁, vₖ, q, x, wₖ₋₂, wₖ₋₁, uₖ₋₁, uₖ, p)
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
mutable struct TricgSolver{T,S} <: KrylovSolver{T,S}
  yₖ      :: S
  N⁻¹uₖ₋₁ :: S
  N⁻¹uₖ   :: S
  p       :: S
  gy₂ₖ₋₁  :: S
  gy₂ₖ    :: S
  xₖ      :: S
  M⁻¹vₖ₋₁ :: S
  M⁻¹vₖ   :: S
  q       :: S
  gx₂ₖ₋₁  :: S
  gx₂ₖ    :: S
  uₖ      :: S
  vₖ      :: S

  function TricgSolver(n, m, S)
    T       = eltype(S)
    yₖ      = S(undef, m)
    N⁻¹uₖ₋₁ = S(undef, m)
    N⁻¹uₖ   = S(undef, m)
    p       = S(undef, m)
    gy₂ₖ₋₁  = S(undef, m)
    gy₂ₖ    = S(undef, m)
    xₖ      = S(undef, n)
    M⁻¹vₖ₋₁ = S(undef, n)
    M⁻¹vₖ   = S(undef, n)
    q       = S(undef, n)
    gx₂ₖ₋₁  = S(undef, n)
    gx₂ₖ    = S(undef, n)
    uₖ      = S(undef, 0)
    vₖ      = S(undef, 0)
    solver  = new{T,S}(yₖ, N⁻¹uₖ₋₁, N⁻¹uₖ, p, gy₂ₖ₋₁, gy₂ₖ, xₖ, M⁻¹vₖ₋₁, M⁻¹vₖ, q, gx₂ₖ₋₁, gx₂ₖ, uₖ, vₖ)
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
mutable struct TrimrSolver{T,S} <: KrylovSolver{T,S}
  yₖ      :: S
  N⁻¹uₖ₋₁ :: S
  N⁻¹uₖ   :: S
  p       :: S
  gy₂ₖ₋₃  :: S
  gy₂ₖ₋₂  :: S
  gy₂ₖ₋₁  :: S
  gy₂ₖ    :: S
  xₖ      :: S
  M⁻¹vₖ₋₁ :: S
  M⁻¹vₖ   :: S
  q       :: S
  gx₂ₖ₋₃  :: S
  gx₂ₖ₋₂  :: S
  gx₂ₖ₋₁  :: S
  gx₂ₖ    :: S
  uₖ      :: S
  vₖ      :: S

  function TrimrSolver(n, m, S)
    T       = eltype(S)
    yₖ      = S(undef, m)
    N⁻¹uₖ₋₁ = S(undef, m)
    N⁻¹uₖ   = S(undef, m)
    p       = S(undef, m)
    gy₂ₖ₋₃  = S(undef, m)
    gy₂ₖ₋₂  = S(undef, m)
    gy₂ₖ₋₁  = S(undef, m)
    gy₂ₖ    = S(undef, m)
    xₖ      = S(undef, n)
    M⁻¹vₖ₋₁ = S(undef, n)
    M⁻¹vₖ   = S(undef, n)
    q       = S(undef, n)
    gx₂ₖ₋₃  = S(undef, n)
    gx₂ₖ₋₂  = S(undef, n)
    gx₂ₖ₋₁  = S(undef, n)
    gx₂ₖ    = S(undef, n)
    uₖ      = S(undef, 0)
    vₖ      = S(undef, 0)
    solver = new{T,S}(yₖ, N⁻¹uₖ₋₁, N⁻¹uₖ, p, gy₂ₖ₋₃, gy₂ₖ₋₂, gy₂ₖ₋₁, gy₂ₖ, xₖ, M⁻¹vₖ₋₁, M⁻¹vₖ, q, gx₂ₖ₋₃, gx₂ₖ₋₂, gx₂ₖ₋₁, gx₂ₖ, uₖ, vₖ)
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
mutable struct TrilqrSolver{T,S} <: KrylovSolver{T,S}
  uₖ₋₁ :: S
  uₖ   :: S
  p    :: S
  d̅    :: S
  x    :: S
  vₖ₋₁ :: S
  vₖ   :: S
  q    :: S
  t    :: S
  wₖ₋₃ :: S
  wₖ₋₂ :: S

  function TrilqrSolver(n, m, S)
    T    = eltype(S)
    uₖ₋₁ = S(undef, m)
    uₖ   = S(undef, m)
    p    = S(undef, m)
    d̅    = S(undef, m)
    x    = S(undef, m)
    vₖ₋₁ = S(undef, n)
    vₖ   = S(undef, n)
    q    = S(undef, n)
    t    = S(undef, n)
    wₖ₋₃ = S(undef, n)
    wₖ₋₂ = S(undef, n)
    solver = new{T,S}(uₖ₋₁, uₖ, p, d̅, x, vₖ₋₁, vₖ, q, t, wₖ₋₃, wₖ₋₂)
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
mutable struct CgsSolver{T,S} <: KrylovSolver{T,S}
  x  :: S
  r  :: S
  u  :: S
  p  :: S
  q  :: S
  ts :: S
  yz :: S
  vw :: S

  function CgsSolver(n, m, S)
    T  = eltype(S)
    x  = S(undef, n)
    r  = S(undef, n)
    u  = S(undef, n)
    p  = S(undef, n)
    q  = S(undef, n)
    ts = S(undef, n)
    yz = S(undef, 0)
    vw = S(undef, 0)
    solver = new{T,S}(x, r, u, p, q, ts, yz, vw)
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
mutable struct BicgstabSolver{T,S} <: KrylovSolver{T,S}
  x  :: S
  r  :: S
  p  :: S
  v  :: S
  s  :: S
  qd :: S
  yz :: S
  t  :: S

  function BicgstabSolver(n, m, S)
    T  = eltype(S)
    x  = S(undef, n)
    r  = S(undef, n)
    p  = S(undef, n)
    v  = S(undef, n)
    s  = S(undef, n)
    qd = S(undef, n)
    yz = S(undef, 0)
    t  = S(undef, 0)
    solver = new{T,S}(x, r, p, v, s, qd, yz, t)
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
mutable struct BilqSolver{T,S} <: KrylovSolver{T,S}
  uₖ₋₁ :: S
  uₖ   :: S
  q    :: S
  vₖ₋₁ :: S
  vₖ   :: S
  p    :: S
  x    :: S
  d̅    :: S

  function BilqSolver(n, m, S)
    T    = eltype(S)
    uₖ₋₁ = S(undef, n)
    uₖ   = S(undef, n)
    q    = S(undef, n)
    vₖ₋₁ = S(undef, n)
    vₖ   = S(undef, n)
    p    = S(undef, n)
    x    = S(undef, n)
    d̅    = S(undef, n)
    solver = new{T,S}(uₖ₋₁, uₖ, q, vₖ₋₁, vₖ, p, x, d̅)
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
mutable struct QmrSolver{T,S} <: KrylovSolver{T,S}
  uₖ₋₁ :: S
  uₖ   :: S
  q    :: S
  vₖ₋₁ :: S
  vₖ   :: S
  p    :: S
  x    :: S
  wₖ₋₂ :: S
  wₖ₋₁ :: S

  function QmrSolver(n, m, S)
    T    = eltype(S)
    uₖ₋₁ = S(undef, n)
    uₖ   = S(undef, n)
    q    = S(undef, n)
    vₖ₋₁ = S(undef, n)
    vₖ   = S(undef, n)
    p    = S(undef, n)
    x    = S(undef, n)
    wₖ₋₂ = S(undef, n)
    wₖ₋₁ = S(undef, n)
    solver = new{T,S}(uₖ₋₁, uₖ, q, vₖ₋₁, vₖ, p, x, wₖ₋₂, wₖ₋₁)
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
mutable struct BilqrSolver{T,S} <: KrylovSolver{T,S}
  uₖ₋₁ :: S
  uₖ   :: S
  q    :: S
  vₖ₋₁ :: S
  vₖ   :: S
  p    :: S
  x    :: S
  t    :: S
  d̅    :: S
  wₖ₋₃ :: S
  wₖ₋₂ :: S

  function BilqrSolver(n, m, S)
    T    = eltype(S)
    uₖ₋₁ = S(undef, n)
    uₖ   = S(undef, n)
    q    = S(undef, n)
    vₖ₋₁ = S(undef, n)
    vₖ   = S(undef, n)
    p    = S(undef, n)
    x    = S(undef, n)
    t    = S(undef, n)
    d̅    = S(undef, n)
    wₖ₋₃ = S(undef, n)
    wₖ₋₂ = S(undef, n)
    solver = new{T,S}(uₖ₋₁, uₖ, q, vₖ₋₁, vₖ, p, x, t, d̅, wₖ₋₃, wₖ₋₂)
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
mutable struct CglsSolver{T,S} <: KrylovSolver{T,S}
  x  :: S
  p  :: S
  s  :: S
  r  :: S
  q  :: S
  Mr :: S

  function CglsSolver(n, m, S)
    T  = eltype(S)
    x  = S(undef, m)
    p  = S(undef, m)
    s  = S(undef, m)
    r  = S(undef, n)
    q  = S(undef, n)
    Mr = S(undef, 0)
    solver = new{T,S}(x, p, s, r, q, Mr)
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
mutable struct CrlsSolver{T,S} <: KrylovSolver{T,S}
  x  :: S
  p  :: S
  Ar :: S
  q  :: S
  r  :: S
  Ap :: S
  s  :: S
  Ms :: S

  function CrlsSolver(n, m, S)
    T  = eltype(S)
    x  = S(undef, m)
    p  = S(undef, m)
    Ar = S(undef, m)
    q  = S(undef, m)
    r  = S(undef, n)
    Ap = S(undef, n)
    s  = S(undef, n)
    Ms = S(undef, 0)
    solver = new{T,S}(x, p, Ar, q, r, Ap, s, Ms)
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
mutable struct CgneSolver{T,S} <: KrylovSolver{T,S}
  x   :: S
  p   :: S
  Aᵀz :: S
  r   :: S
  q   :: S
  s   :: S
  z   :: S

  function CgneSolver(n, m, S)
    T   = eltype(S)
    x   = S(undef, m)
    p   = S(undef, m)
    Aᵀz = S(undef, m)
    r   = S(undef, n)
    q   = S(undef, n)
    s   = S(undef, 0)
    z   = S(undef, 0)
    solver = new{T,S}(x, p, Aᵀz, r, q, s, z)
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
mutable struct CrmrSolver{T,S} <: KrylovSolver{T,S}
  x   :: S
  p   :: S
  Aᵀr :: S
  r   :: S
  q   :: S
  Mq  :: S
  s   :: S

  function CrmrSolver(n, m, S)
    T = eltype(S)
    x   = S(undef, m)
    p   = S(undef, m)
    Aᵀr = S(undef, m)
    r   = S(undef, n)
    q   = S(undef, n)
    Mq  = S(undef, 0)
    s   = S(undef, 0)
    solver = new{T,S}(x, p, Aᵀr, r, q, Mq, s)
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
mutable struct LslqSolver{T,S} <: KrylovSolver{T,S}
  x_lq :: S
  Nv   :: S
  Aᵀu  :: S
  w̄    :: S
  Mu   :: S
  Av   :: S
  u    :: S
  v    :: S

  function LslqSolver(n, m, S)
    T    = eltype(S)
    x_lq = S(undef, m)
    Nv   = S(undef, m)
    Aᵀu  = S(undef, m)
    w̄    = S(undef, m)
    Mu   = S(undef, n)
    Av   = S(undef, n)
    u    = S(undef, 0)
    v    = S(undef, 0)
    solver = new{T,S}(x_lq, Nv, Aᵀu, w̄, Mu, Av, u, v)
    return solver
  end

  function LslqSolver(A, b)
    n, m = size(A)
    S = ktypeof(b)
    LslqSolver(n, m, S)
  end
end

"""
Type for storing the vectors required by the in-place version of LSQR.

The outer constructors

    solver = LsqrSolver(n, m, S)
    solver = LsqrSolver(A, b)

may be used in order to create these vectors.
"""
mutable struct LsqrSolver{T,S} <: KrylovSolver{T,S}
  x   :: S
  Nv  :: S
  Aᵀu :: S
  w   :: S
  Mu  :: S
  Av  :: S
  u   :: S
  v   :: S

  function LsqrSolver(n, m, S)
    T   = eltype(S)
    x   = S(undef, m)
    Nv  = S(undef, m)
    Aᵀu = S(undef, m)
    w   = S(undef, m)
    Mu  = S(undef, n)
    Av  = S(undef, n)
    u   = S(undef, 0)
    v   = S(undef, 0)
    solver = new{T,S}(x, Nv, Aᵀu, w, Mu, Av, u, v)
    return solver
  end

  function LsqrSolver(A, b)
    n, m = size(A)
    S = ktypeof(b)
    LsqrSolver(n, m, S)
  end
end

"""
Type for storing the vectors required by the in-place version of LSMR.

The outer constructors

    solver = LsmrSolver(n, m, S)
    solver = LsmrSolver(A, b)

may be used in order to create these vectors.
"""
mutable struct LsmrSolver{T,S} <: KrylovSolver{T,S}
  x    :: S
  Nv   :: S
  Aᵀu  :: S
  h    :: S
  hbar :: S
  Mu   :: S
  Av   :: S
  u    :: S
  v    :: S

  function LsmrSolver(n, m, S)
    T    = eltype(S)
    x    = S(undef, m)
    Nv   = S(undef, m)
    Aᵀu  = S(undef, m)
    h    = S(undef, m)
    hbar = S(undef, m)
    Mu   = S(undef, n)
    Av   = S(undef, n)
    u    = S(undef, 0)
    v    = S(undef, 0)
    solver = new{T,S}(x, Nv, Aᵀu, h, hbar, Mu, Av, u, v)
    return solver
  end

  function LsmrSolver(A, b)
    n, m = size(A)
    S = ktypeof(b)
    LsmrSolver(n, m, S)
  end
end

"""
Type for storing the vectors required by the in-place version of LNLQ.

The outer constructors

    solver = LnlqSolver(n, m, S)
    solver = LnlqSolver(A, b)

may be used in order to create these vectors.
"""
mutable struct LnlqSolver{T,S} <: KrylovSolver{T,S}
  x   :: S
  Nv  :: S
  Aᵀu :: S
  y   :: S
  w̄   :: S
  Mu  :: S
  Av  :: S
  u   :: S
  v   :: S
  q   :: S

  function LnlqSolver(n, m, S)
    T  = eltype(S)
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
    solver = new{T,S}(x, Nv, Aᵀu, y, w̄, Mu, Av, u, v, q)
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
mutable struct CraigSolver{T,S} <: KrylovSolver{T,S}
  x   :: S
  Nv  :: S
  Aᵀu :: S
  y   :: S
  w   :: S
  Mu  :: S
  Av  :: S
  u   :: S
  v   :: S
  w2  :: S

  function CraigSolver(n, m, S)
    T   = eltype(S)
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
    solver = new{T,S}(x, Nv, Aᵀu, y, w, Mu, Av, u, v, w2)
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
mutable struct CraigmrSolver{T,S} <: KrylovSolver{T,S}
  x    :: S
  Nv   :: S
  Aᵀu  :: S
  y    :: S
  Mu   :: S
  w    :: S
  wbar :: S
  Av   :: S
  u    :: S
  v    :: S

  function CraigmrSolver(n, m, S)
    T    = eltype(S)
    x    = S(undef, m)
    Nv   = S(undef, m)
    Aᵀu  = S(undef, m)
    y    = S(undef, n)
    Mu   = S(undef, n)
    w    = S(undef, n)
    wbar = S(undef, n)
    Av   = S(undef, n)
    u    = S(undef, 0)
    v    = S(undef, 0)
    solver = new{T,S}(x, Nv, Aᵀu, y, Mu, w, wbar, Av, u, v)
    return solver
  end

  function CraigmrSolver(A, b)
    n, m = size(A)
    S = ktypeof(b)
    CraigmrSolver(n, m, S)
  end
end
