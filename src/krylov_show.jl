import Base.show, Base.sizeof, Base.format_bytes

function ksizeof(attribute)
  if isa(attribute, Vector{<:AbstractArray}) && !isempty(attribute)
    # A vector of arrays is a vector of pointers in Julia.
    # All arrays inside a vector have the same size in Krylov.jl
    size_attribute = sizeof(attribute) + length(attribute) * ksizeof(attribute[1])
  else
    size_attribute = sizeof(attribute)
  end
  return size_attribute
end

function sizeof(stats_workspace :: Union{KrylovStats, KrylovWorkspace, BlockKrylovWorkspace})
  type = typeof(stats_workspace)
  nfields = fieldcount(type)
  storage = 0
  for i = 1:nfields
    field_i = getfield(stats_workspace, i)
    size_i = ksizeof(field_i)
    storage += size_i
  end
  return storage
end

special_fields = Dict(
  :residualscg => "residuals (cg)",
  :errorscg => "errors (cg)",
  :Anorm => "‖A‖F",
  :Acond => "κ₂(A)",
  :err_ubnds_lq => "error bound LQ",
  :err_ubnds_cg => "error bound CG",
)

function show(io :: IO, stats :: KrylovStats)
  kst = typeof(stats)
  s = string(kst.name.name) * "\n"
  nfield = fieldcount(kst)
  for i = 1 : nfield
    field = fieldname(kst, i)
    field_name = if field ∈ keys(special_fields)
      special_fields[field]
    else
      replace(string(field), "_" => " ")
    end
    s *=  " " * field_name * ":"
    statfield = getfield(stats, field)
    if isa(statfield, AbstractVector) && eltype(statfield) <: Union{Missing, AbstractFloat}
      s *= @sprintf " %s\n" vec2str(statfield)
    elseif field_name == "allocation timer" || field_name == "timer"
      (statfield < 1e-3) && (s *= @sprintf " %.2fμs\n" 1e6*statfield)
      (1e-3 ≤ statfield < 1.00) && (s *= @sprintf " %.2fms\n" 1e3*statfield)
      (statfield ≥ 1.00) && (s *= @sprintf " %.2fs\n" statfield)
    else
      s *= @sprintf " %s\n" statfield
    end
  end
  print(io, s)
end

"""
    show(io, workspace; show_stats=true)

Statistics of `workspace` are displayed if `show_stats` is set to true.
"""
function show(io :: IO, workspace :: _KrylovWorkspace{T,FC,Sm,Sn}; show_stats :: Bool=true) where {T <: AbstractFloat, FC <: FloatOrComplex{T}, Sm <: AbstractVector{FC}, Sn <: AbstractVector{FC}}
  type_workspace = typeof(workspace)
  name_workspace = string(type_workspace.name.name)
  name_stats = string(typeof(workspace.stats).name.name)
  nbytes = sizeof(workspace)
  storage = format_bytes(nbytes)
  lS = max(string(Sm) |> length, string(Sn) |> length)
  lT = string(T) |> length
  lFC = string(FC) |> length
  if Sm <: Vector && Sn <: Vector
    architecture = "CPU"
  elseif Sm.name.name in (:CuArray, :ROCArray, :oneArray, :MtlArray) && Sn.name.name in (:CuArray, :ROCArray, :oneArray, :MtlArray)
    architecture = "GPU"
  else
    architecture = "CPU / GPU"
  end
  l1 = max(length(name_workspace), lFC + 11)  # length("Precision: ") = 11
  nchar = type_workspace <: Union{CgLanczosShiftWorkspace, CglsLanczosShiftWorkspace, FomWorkspace, DiomWorkspace, DqgmresWorkspace, GmresWorkspace, FgmresWorkspace, GpmrWorkspace, BlockGmresWorkspace} ? 8 : 0  # length("Vector{}") = 8
  l2 = max(ndigits(workspace.m) + 7, length(architecture) + 14, lS + nchar)  # length("nrows: ") = 7 and length("Architecture: ") = 14
  l2 = max(l2, length(name_stats) + 2 + lT)  # length("{}") = 2
  l3 = max(ndigits(workspace.n) + 7, length(storage) + 9)  # length("Storage: ") = 9 and length("cols: ") = 7
  format = Printf.Format("│%$(l1)s│%$(l2)s│%$(l3)s│\n")
  format2 = Printf.Format("│%$(l1+1)s│%$(l2)s│%$(l3)s│\n")
  @printf(io, "┌%s┬%s┬%s┐\n", "─"^l1, "─"^l2, "─"^l3)
  Printf.format(io, format, "$(name_workspace)", "nrows: $(workspace.m)", "ncols: $(workspace.n)")
  @printf(io, "├%s┼%s┼%s┤\n", "─"^l1, "─"^l2, "─"^l3)
  Printf.format(io, format, "Precision: $FC", "Architecture: $architecture","Storage: $storage")
  @printf(io, "├%s┼%s┼%s┤\n", "─"^l1, "─"^l2, "─"^l3)
  Printf.format(io, format, "Attribute", "Type", "Size")
  @printf(io, "├%s┼%s┼%s┤\n", "─"^l1, "─"^l2, "─"^l3)
  for i=1:fieldcount(type_workspace)
    name_i = fieldname(type_workspace, i)
    type_i = fieldtype(type_workspace, i)
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

function show(io :: IO, workspace :: BlockKrylovWorkspace{T,FC,SV,SM}; show_stats :: Bool=true) where {T <: AbstractFloat, FC <: FloatOrComplex{T}, SV <: AbstractVector{FC}, SM <: AbstractMatrix{FC}}
  type_workspace = typeof(workspace)
  name_workspace = string(type_workspace.name.name)
  name_stats = string(typeof(workspace.stats).name.name)
  nbytes = sizeof(workspace)
  storage = format_bytes(nbytes)
  lS = max(string(SV) |> length, string(SM) |> length)
  lT = string(T) |> length
  lFC = string(FC) |> length
  if SV <: Vector && SM <: Matrix
    architecture = "CPU"
  elseif SV.name.name in (:CuArray, :ROCArray, :oneArray, :MtlArray) && SM.name.name in (:CuArray, :ROCArray, :oneArray, :MtlArray)
    architecture = "GPU"
  else
    architecture = "CPU / GPU"
  end
  l1 = max(length(name_workspace), length(string(FC)) + 11)  # length("Precision: ") = 11
  nchar = type_workspace <: BlockGmresWorkspace ? 8 : 0  # length("Vector{}") = 8
  l2 = max(ndigits(workspace.m) + 7, length(architecture) + 14, lS + nchar)  # length("nrows: ") = 7 and length("Architecture: ") = 14
  l2 = max(l2, length(name_stats) + 2 + lT)  # length("{}") = 2
  l3 = max(ndigits(workspace.n) + 7, length(storage) + 9)  # length("Storage: ") = 9 and length("cols: ") = 7
  format = Printf.Format("│%$(l1)s│%$(l2)s│%$(l3)s│\n")
  format2 = Printf.Format("│%$(l1+1)s│%$(l2)s│%$(l3)s│\n")
  @printf(io, "┌%s┬%s┬%s┐\n", "─"^l1, "─"^l2, "─"^l3)
  Printf.format(io, format, "$(name_workspace)", "nrows: $(workspace.m)", "ncols: $(workspace.n)")
  @printf(io, "├%s┼%s┼%s┤\n", "─"^l1, "─"^l2, "─"^l3)
  Printf.format(io, format, "Precision: $FC", "Architecture: $architecture","Storage: $storage")
  @printf(io, "├%s┼%s┼%s┤\n", "─"^l1, "─"^l2, "─"^l3)
  Printf.format(io, format, "Attribute", "Type", "Size")
  @printf(io, "├%s┼%s┼%s┤\n", "─"^l1, "─"^l2, "─"^l3)
  for i=1:fieldcount(type_workspace)
    name_i = fieldname(type_workspace, i)
    type_i = fieldtype(type_workspace, i)
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
