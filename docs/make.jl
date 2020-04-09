using Documenter, Krylov

makedocs(
  modules = [Krylov],
  doctest = true,
  linkcheck = true,
  strict = true,
  format = Documenter.HTML(assets = ["assets/style.css"], prettyurls = get(ENV, "CI", nothing) == "true"),
  sitename = "Krylov.jl",
  pages = ["Home" => "index.md",
           "API" => "api.md",
           "Solvers" => "solvers.md",
           "Reference" => "reference.md",
          ]
)

deploydocs(repo = "github.com/JuliaSmoothOptimizers/Krylov.jl.git")
