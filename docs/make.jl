using Documenter, Krylov

makedocs(
  modules = [Krylov],
  doctest = true,
  linkcheck = true,
  strict = false,
  format = Documenter.HTML(assets = ["assets/style.css"], ansicolor=true, prettyurls = get(ENV, "CI", nothing) == "true"),
  sitename = "Krylov.jl",
  pages = ["Home" => "index.md",
           "API" => "api.md",
           "Krylov methods" => ["Symmetric positive definite linear systems" => "solvers/spd.md",
                                "Symmetric indefinite linear systems" => "solvers/sid.md",
                                "Unsymmetric linear systems" => "solvers/unsymmetric.md",
                                "Least-norm problems" => "solvers/ln.md",
                                "Least-squares problems" => "solvers/ls.md",
                                "Adjoint systems" => "solvers/as.md",
                                "Saddle-point and symmetric quasi-definite systems" => "solvers/sp_sqd.md"],
           "In-place methods" => "inplace.md",
           "GPU" => "gpu.md",
           "Factorization-free operators" => "factorization-free.md",
           "Tips and tricks" => "tips_tricks.md",
           "Tutorial" => "examples.md",
           "Reference" => "reference.md"
          ]
)

deploydocs(repo = "github.com/JuliaSmoothOptimizers/Krylov.jl.git", devbranch = "main")
