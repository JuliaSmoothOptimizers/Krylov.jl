# Strip SparseArrays (and therefore SuiteSparse) from the libkrylov build.
#
# The C/Fortran interface only exposes the Krylov solvers, never the processes
# (hermitian_lanczos, golub_kahan, ...), so SparseArrays is dead weight here.
# This script edits the working tree in place before the bundle is built:
#
#   * Project.toml      : drop the SparseArrays dep and compat entry
#   * src/Krylov.jl     : drop `using SparseArrays`, disable the *_processes.jl
#   * src/krylov_utils.jl : remove the two SparseVector ktypeof methods
#
# Every edit is anchored on a stable string/signature rather than on line
# numbers or surrounding context, so it survives code moving around. If an
# anchor is ever missing the script errors out loudly instead of silently
# leaving SparseArrays in the build.

const ROOT = normpath(joinpath(@__DIR__, "..", ".."))

"Apply `f` to the file, asserting that it actually changed the content."
function edit(f, relpath)
    path = joinpath(ROOT, relpath)
    old = read(path, String)
    new = f(old)
    new == old && error("trim_sparsearrays: no change in $relpath — an anchor is missing, refusing to build with SparseArrays still in.")
    write(path, new)
    println("  patched $relpath")
end

println("Trimming SparseArrays from the build...")

# --- Project.toml: remove the dep uuid line and the compat line ---------------
edit("Project.toml") do s
    replace(s, r"^SparseArrays = .*\n"m => "")
end

# --- src/Krylov.jl: drop the using and disable the two process files ----------
edit("src/Krylov.jl") do s
    s = replace(s, "using LinearAlgebra, SparseArrays, Printf" =>
                   "using LinearAlgebra, Printf")
    s = replace(s, "include(\"krylov_processes.jl\")" =>
                   "# include(\"krylov_processes.jl\")        # disabled: pulls SparseArrays -> SuiteSparse")
    s = replace(s, "include(\"block_krylov_processes.jl\")" =>
                   "# include(\"block_krylov_processes.jl\")  # disabled: pulls SparseArrays -> SuiteSparse")
    return s
end

# --- src/krylov_utils.jl: remove the two SparseVector ktypeof methods ---------
# Anchored on the method signatures; `.*?\r?\nend\r?\n` matches the (brace-free)
# body up to the first closing `end`, regardless of where the methods sit in the
# file. `\r?` keeps it working on Windows checkouts that use CRLF line endings.
edit("src/krylov_utils.jl") do s
    s = replace(s, r"function ktypeof\(v::S\) where S <: SparseVector.*?\r?\nend\r?\n"s => "")
    s = replace(s, r"function ktypeof\(v::S\) where S <: AbstractSparseVector.*?\r?\nend\r?\n"s => "")
    return s
end

println("Done.")
println()
println("NOTE: this edited the working tree in place (Project.toml, src/Krylov.jl,")
println("      src/krylov_utils.jl). In CI the checkout is throwaway. For a LOCAL")
println("      build, restore your sources once the bundle is built with:")
println("          git restore Project.toml src/Krylov.jl src/krylov_utils.jl")
