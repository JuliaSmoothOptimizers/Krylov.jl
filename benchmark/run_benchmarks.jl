using Pkg
bmark_dir = @__DIR__
repo_name = string(split(ARGS[1], ".")[1])
bmarkname = lowercase(repo_name)

# if we are running these benchmarks from the git repository
# we want to develop the package instead of using the release
if isdir(joinpath(bmark_dir, "..", ".git"))
    Pkg.develop(PackageSpec(url = joinpath(bmark_dir, "..")))
end

using DataFrames
using GitHub
using JLD2
using JSON
using PkgBenchmark
using Plots

using SolverBenchmark

# NB: benchmarkpkg will run benchmarks/benchmarks.jl by default
commit = benchmarkpkg(repo_name)  # current state of repository
main = benchmarkpkg(repo_name, "main")
judgement = judge(commit, main)

commit_stats = bmark_results_to_dataframes(commit)
main_stats = bmark_results_to_dataframes(main)
judgement_stats = judgement_results_to_dataframes(judgement)

export_markdown("judgement_$(bmarkname).md", judgement)
export_markdown("main.md", main)
export_markdown("$(bmarkname).md", commit)

function profile_solvers_from_pkgbmark(stats::Dict{Symbol,DataFrame})
    # guard against zero gctimes
    costs = [
        df -> df[!, :time],
        df -> df[!, :memory],
        df -> df[!, :gctime] .+ 1,
        df -> df[!, :allocations],
    ]
    profile_solvers(stats, costs, ["time", "memory", "gctime+1", "allocations"])
end

# extract stats for each benchmark to plot profiles
# files_dict will be part of json_dict below
files_dict = Dict{String,Any}()
file_num = 1
for k ∈ keys(judgement_stats)
    global file_num
    k_stats = Dict{Symbol,DataFrame}(:commit => commit_stats[k], :main => main_stats[k])
    save_stats(k_stats, "$(bmarkname)_vs_main_$(k).jld2", force = true)

    k_profile = profile_solvers_from_pkgbmark(k_stats)
    savefig("profiles_commit_vs_main_$(k).svg")
    # read contents of svg file to add to gist
    k_svgfile = open("profiles_commit_vs_main_$(k).svg", "r") do fd
        readlines(fd)
    end
    # file_num makes sure svg files appear before md files (added below)
    files_dict["$(file_num)_$(k).svg"] = Dict{String,Any}("content" => join(k_svgfile))
    file_num += 1
end

for mdfile ∈ [:judgement, :main, :commit]
    global file_num
    files_dict["$(file_num)_$(mdfile).md"] =
        Dict{String,Any}("content" => "$(sprint(export_markdown, eval(mdfile)))")
    file_num += 1
end

jldopen("$(bmarkname)_vs_main_judgement.jld2", "w") do file
    file["jstats"] = judgement_stats
end

# json description of gist
json_dict = Dict{String,Any}(
    "description" => "$(repo_name) repository benchmark",
    "public" => true,
    "files" => files_dict,
)

open("gist.json", "w") do f
    JSON.print(f, json_dict)
end
