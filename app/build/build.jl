using PackageCompiler

target_dir = get(ENV, "OUTDIR", "$(@__DIR__)/../CKrylov")
target_dir = replace(target_dir, "\\"=>"/")  # Change Windows paths to use "/"

println("Creating library in $target_dir")
PackageCompiler.create_library(".", target_dir;
                               lib_name="krylov",
                               precompile_execution_file=["$(@__DIR__)/generate_precompile.jl"],
                               precompile_statements_file=["$(@__DIR__)/additional_precompile.jl"],
                               incremental=false,
                               filter_stdlibs=true,
                               header_files = ["$(@__DIR__)/krylov.h"],
                               force=true
                              )
