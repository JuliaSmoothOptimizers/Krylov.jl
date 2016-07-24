const home = "https://github.com/JuliaSmoothOptimizers"
const deps = Dict{AbstractString, AbstractString}("LinearOperators" => "master")

function dep_installed(dep)
  try
    Pkg.installed(dep)  # throws an error instead of returning false
    return true
  catch
    return false
  end
end

function dep_install(dep)
  dep_installed(dep) || Pkg.clone("$home/$dep.jl.git")
  Pkg.checkout(dep, deps[dep])
  Pkg.build(dep)
end

for dep in keys(deps)
  dep_install(dep)
end

