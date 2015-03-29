module Krylov

using LinearOperators
import Base.show

# Documentation.
using Docile
@docstrings

@doc "Abstract type for statistics returned by a solver" ->
abstract KrylovStats;

include("krylov_utils.jl")
include("cg.jl")
include("cg_lanczos.jl")
include("cgls.jl")
include("crls.jl")

end
