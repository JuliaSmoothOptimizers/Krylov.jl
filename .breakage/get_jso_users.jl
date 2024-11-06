import GitHub, PkgDeps  # both export users()

length(ARGS) >= 1 || error("specify at least one JSO package as argument")

jso_repos, _ = GitHub.repos("JuliaSmoothOptimizers")
jso_names = [splitext(x.name)[1] for x ∈ jso_repos]

name = splitext(ARGS[1])[1]
name ∈ jso_names || error("argument should be one of ", jso_names)

dependents = String[]
try
  global dependents = filter(x -> x ∈ jso_names, PkgDeps.users(name))
catch e
  # package not registered; don't insert into dependents
end

println(dependents)
