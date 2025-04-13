export BlockKrylovWorkspace, BlockMinresWorkspace, BlockGmresWorkspace

"Abstract type for using block Krylov solvers in-place."
abstract type BlockKrylovWorkspace{T,FC,SV,SM} end

"""
Type for storing the vectors required by the in-place version of BLOCK-MINRES.

The outer constructors

    workspace = BlockMinresWorkspace(m, n, p, SV, SM)
    workspace = BlockMinresWorkspace(A, B)

may be used in order to create these vectors.
"""
mutable struct BlockMinresWorkspace{T,FC,SV,SM} <: BlockKrylovWorkspace{T,FC,SV,SM}
  m          :: Int
  n          :: Int
  p          :: Int
  ΔX         :: SM
  X          :: SM
  P          :: SM
  Q          :: SM
  C          :: SM
  D          :: SM
  Φ          :: SM
  Vₖ₋₁       :: SM
  Vₖ         :: SM
  wₖ₋₂       :: SM
  wₖ₋₁       :: SM
  Hₖ₋₂       :: SM
  Hₖ₋₁       :: SM
  τₖ₋₂       :: SV
  τₖ₋₁       :: SV
  warm_start :: Bool
  stats      :: SimpleStats{T}
end

function BlockMinresWorkspace(m::Integer, n::Integer, p::Integer, SV::Type, SM::Type)
  FC   = eltype(SV)
  T    = real(FC)
  ΔX   = SM(undef, 0, 0)
  X    = SM(undef, n, p)
  P    = SM(undef, 0, 0)
  Q    = SM(undef, n, p)
  C    = SM(undef, p, p)
  D    = SM(undef, 2p, p)
  Φ    = SM(undef, p, p)
  Vₖ₋₁ = SM(undef, n, p)
  Vₖ   = SM(undef, n, p)
  wₖ₋₂ = SM(undef, n, p)
  wₖ₋₁ = SM(undef, n, p)
  Hₖ₋₂ = SM(undef, 2p, p)
  Hₖ₋₁ = SM(undef, 2p, p)
  τₖ₋₂ = SV(undef, p)
  τₖ₋₁ = SV(undef, p)
  SV = isconcretetype(SV) ? SV : typeof(τₖ₋₁)
  SM = isconcretetype(SM) ? SM : typeof(X)
  stats = SimpleStats(0, false, false, false, T[], T[], T[], 0.0, "unknown")
  workspace = BlockMinresWorkspace{T,FC,SV,SM}(m, n, p, ΔX, X, P, Q, C, D, Φ, Vₖ₋₁, Vₖ, wₖ₋₂, wₖ₋₁, Hₖ₋₂, Hₖ₋₁, τₖ₋₂, τₖ₋₁, false, stats)
  return workspace
end

function BlockMinresWorkspace(A, B)
  m, n = size(A)
  s, p = size(B)
  SM = typeof(B)
  SV = matrix_to_vector(SM)
  BlockMinresWorkspace(m, n, p, SV, SM)
end

"""
Type for storing the vectors required by the in-place version of BLOCK-GMRES.

The outer constructors

    workspace = BlockGmresWorkspace(m, n, p, SV, SM; memory = 5)
    workspace = BlockGmresWorkspace(A, B; memory = 5)

may be used in order to create these vectors.
`memory` is set to `div(n,p)` if the value given is larger than `div(n,p)`.
"""
mutable struct BlockGmresWorkspace{T,FC,SV,SM} <: BlockKrylovWorkspace{T,FC,SV,SM}
  m          :: Int
  n          :: Int
  p          :: Int
  ΔX         :: SM
  X          :: SM
  W          :: SM
  P          :: SM
  Q          :: SM
  C          :: SM
  D          :: SM
  V          :: Vector{SM}
  Z          :: Vector{SM}
  R          :: Vector{SM}
  H          :: Vector{SM}
  τ          :: Vector{SV}
  warm_start :: Bool
  stats      :: SimpleStats{T}
end

function BlockGmresWorkspace(m::Integer, n::Integer, p::Integer, SV::Type, SM::Type; memory::Integer = 5)
  memory = min(div(n,p), memory)
  FC = eltype(SV)
  T  = real(FC)
  ΔX = SM(undef, 0, 0)
  X  = SM(undef, n, p)
  W  = SM(undef, n, p)
  P  = SM(undef, 0, 0)
  Q  = SM(undef, 0, 0)
  C  = SM(undef, p, p)
  D  = SM(undef, 2p, p)
  V  = SM[SM(undef, n, p) for i = 1 : memory]
  Z  = SM[SM(undef, p, p) for i = 1 : memory]
  R  = SM[SM(undef, p, p) for i = 1 : div(memory * (memory+1), 2)]
  H  = SM[SM(undef, 2p, p) for i = 1 : memory]
  τ  = SV[SV(undef, p) for i = 1 : memory]
  SV = isconcretetype(SV) ? SV : typeof(τ)
  SM = isconcretetype(SM) ? SM : typeof(X)
  stats = SimpleStats(0, false, false, false, T[], T[], T[], 0.0, "unknown")
  workspace = BlockGmresWorkspace{T,FC,SV,SM}(m, n, p, ΔX, X, W, P, Q, C, D, V, Z, R, H, τ, false, stats)
  return workspace
end

function BlockGmresWorkspace(A, B; memory::Integer = 5)
  m, n = size(A)
  s, p = size(B)
  SM = typeof(B)
  SV = matrix_to_vector(SM)
  BlockGmresWorkspace(m, n, p, SV, SM; memory)
end

for (KS, fun, nsol, nA, nAt, warm_start) in [
  (:BlockMinresWorkspace, :block_minres!, 1, 1, 0, true)
  (:BlockGmresWorkspace , :block_gmres! , 1, 1, 0, true)
]
  @eval begin
    size(workspace :: $KS) = workspace.m, workspace.n
    nrhs(workspace :: $KS) = workspace.p
    statistics(workspace :: $KS) = workspace.stats
    niterations(workspace :: $KS) = workspace.stats.niter
    Aprod(workspace :: $KS) = $nA * workspace.stats.niter
    Atprod(workspace :: $KS) = $nAt * workspace.stats.niter
    nsolution(workspace :: $KS) = $nsol
    if $nsol == 1
      solution(workspace :: $KS) = workspace.X
      solution(workspace :: $KS, p :: Integer) = (p == 1) ? solution(workspace) : error("solution(workspace) has only one output.")
      results(workspace :: $KS) = (workspace.X, workspace.stats)
    end
    issolved(workspace :: $KS) = workspace.stats.solved
    if $warm_start
      function warm_start!(workspace :: $KS, X0)
        n2, p2 = size(X0)
        SM = typeof(workspace.X)
        (workspace.n == n2 && workspace.p == p2) || error("X0 should have size ($n, $p)")
        allocate_if(true, workspace, :ΔX, SM, workspace.n, workspace.p)
        copyto!(workspace.ΔX, X0)
        workspace.warm_start = true
        return workspace
      end
    end
  end
end

function ksizeof(attribute)
  if isa(attribute, Vector{<:AbstractVector}) && !isempty(attribute)
    # A vector of vectors is a vector of pointers in Julia.
    # All vectors inside a vector have the same size in Krylov.jl
    size_attribute = sizeof(attribute) + length(attribute) * ksizeof(attribute[1])
  else
    size_attribute = sizeof(attribute)
  end
  return size_attribute
end

function sizeof(stats_workspace :: Union{KrylovStats, KrylovWorkspace, BlockKrylovWorkspace})
  type = typeof(stats_solver)
  nfields = fieldcount(type)
  storage = 0
  for i = 1:nfields
    field_i = getfield(stats_workspace, i)
    size_i = ksizeof(field_i)
    storage += size_i
  end
  return storage
end

"""
    show(io, workspace; show_stats=true)

Statistics of `workspace` are displayed if `show_stats` is set to true.
"""
function show(io :: IO, workspace :: Union{KrylovWorkspace{T,FC,S}, BlockKrylovWorkspace{T,FC,S}}; show_stats :: Bool=true) where {T <: AbstractFloat, FC <: FloatOrComplex{T}, S <: AbstractVector{FC}}
  workspace = typeof(workspace)
  name_workspace = string(workspace.name.name)
  name_stats = string(typeof(workspace.stats).name.name)
  nbytes = sizeof(workspace)
  storage = format_bytes(nbytes)
  architecture = S <: Vector ? "CPU" : "GPU"
  l1 = max(length(name_solver), length(string(FC)) + 11)  # length("Precision: ") = 11
  nchar = workspace <: Union{CgLanczosShiftWorkspace, FomWorkspace, DiomWorkspace, DqgmresWorkspace, GmresWorkspace, FgmresWorkspace, GpmrWorkspace, BlockGmresWorkspace} ? 8 : 0  # length("Vector{}") = 8
  l2 = max(ndigits(workspace.m) + 7, length(architecture) + 14, length(string(S)) + nchar)  # length("nrows: ") = 7 and length("Architecture: ") = 14
  l2 = max(l2, length(name_stats) + 2 + length(string(T)))  # length("{}") = 2
  l3 = max(ndigits(workspace.n) + 7, length(storage) + 9)  # length("Storage: ") = 9 and length("cols: ") = 7
  format = Printf.Format("│%$(l1)s│%$(l2)s│%$(l3)s│\n")
  format2 = Printf.Format("│%$(l1+1)s│%$(l2)s│%$(l3)s│\n")
  @printf(io, "┌%s┬%s┬%s┐\n", "─"^l1, "─"^l2, "─"^l3)
  Printf.format(io, format, "$(name_solver)", "nrows: $(workspace.m)", "ncols: $(workspace.n)")
  @printf(io, "├%s┼%s┼%s┤\n", "─"^l1, "─"^l2, "─"^l3)
  Printf.format(io, format, "Precision: $FC", "Architecture: $architecture","Storage: $storage")
  @printf(io, "├%s┼%s┼%s┤\n", "─"^l1, "─"^l2, "─"^l3)
  Printf.format(io, format, "Attribute", "Type", "Size")
  @printf(io, "├%s┼%s┼%s┤\n", "─"^l1, "─"^l2, "─"^l3)
  for i=1:fieldcount(workspace)
    name_i = fieldname(workspace, i)
    type_i = fieldtype(workspace, i)
    field_i = getfield(workspace, name_i)
    size_i = ksizeof(field_i)
    (size_i ≠ 0) && Printf.format(io, format, string(name_i), type_i, format_bytes(size_i))
  end
  @printf(io, "└%s┴%s┴%s┘\n","─"^l1,"─"^l2,"─"^l3)
  if show_stats
    @printf(io, "\n")
    show(io, workspace.stats)
  end
  return nothing
end
