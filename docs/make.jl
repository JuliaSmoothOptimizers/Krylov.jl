using Documenter, Krylov

makedocs(
  modules = [Krylov],
  doctest = true,
  linkcheck = true,
  strict = true,
  format = Documenter.HTML(assets = ["assets/style.css"],
                           ansicolor=true,
                           prettyurls = get(ENV, "CI", nothing) == "true",
                           collapselevel = 1),
  sitename = "Krylov.jl",
  pages = ["Home" => "index.md",
           "API" => "api.md",
           "Krylov methods" => ["Symmetric positive definite linear systems" => "solvers/spd.md",
                                "Symmetric indefinite linear systems" => "solvers/sid.md",
                                "Unsymmetric linear systems" => "solvers/unsymmetric.md",
                                "Minimum-norm problems" => "solvers/ln.md",
                                "Least-squares problems" => "solvers/ls.md",
                                "Adjoint systems" => "solvers/as.md",
                                "Saddle-point and symmetric quasi-definite systems" => "solvers/sp_sqd.md",
                                "Generalized saddle-point and unsymmetric partitioned systems" => "solvers/gsp.md"],
           "In-place methods" => "inplace.md",
           "Preconditioners" => "preconditioners.md",
           "GPU support" => "gpu.md",
           "Warm start" => "warm_start.md",
           "Factorization-free operators" => "factorization-free.md",
           "Callbacks" => "callbacks.md",
           "Performance tips" => "tips.md",
           "Tutorial" => "examples.md",
           "Reference" => "reference.md"
          ]
)

deploydocs(
  repo = "github.com/JuliaSmoothOptimizers/Krylov.jl.git",
  push_preview = true,
  devbranch = "main",
)
