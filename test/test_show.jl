@testset "show" begin
  solver = Krylov.SimpleStats(true, true, Float64[], Float64[], "t")
  io = IOBuffer()
  show(io, solver)
  showed = String(take!(io))
  storage_type = typeof(solver)
  expected = """Simple stats
  solved: true
  inconsistent: true
  residuals: []
  Aresiduals: []
  status: t"""
  @test strip.(split(chomp(showed), "\n")) == strip.(split(chomp(expected), "\n"))

  solver = Krylov.LanczosStats(true, Float64[], Bool[], NaN, NaN,"t")
  io = IOBuffer()
  show(io, solver)
  showed = String(take!(io))
  storage_type = typeof(solver)
  expected = """Lanczos stats
  solved: true
  residuals: []
  flagged: Bool[]
  ‖A‖F: NaN
  κ₂(A): NaN
  status: t"""
  @test strip.(split(chomp(showed), "\n")) == strip.(split(chomp(expected), "\n"))

  solver = Krylov.SymmlqStats(
    true,
    Float64[],
    Union{Float64,Missing}[],
    Float64[],
    Union{Float64,Missing}[1., missing],
    NaN,
    NaN,
    "t",
  )
  io = IOBuffer()
  show(io, solver)
  showed = String(take!(io))
  storage_type = typeof(solver)
  expected = """Symmlq stats
  solved: true
  residuals: []
  residuals (cg): []
  errors: []
  errors (cg): [ 1.0e+00  ✗✗✗✗ ]
  ‖A‖F: NaN
  κ₂(A): NaN
  status: t"""
  @test strip.(split(chomp(showed), "\n")) == strip.(split(chomp(expected), "\n"))

  solver = Krylov.AdjointStats(true, true, Float64[], Float64[],"t")
  io = IOBuffer()
  show(io, solver)
  showed = String(take!(io))
  storage_type = typeof(solver)
  expected = """Adjoint stats
  solved primal: true
  solved dual: true
  residuals primal: []
  residuals dual: []
  status: t"""
  @test strip.(split(chomp(showed), "\n")) == strip.(split(chomp(expected), "\n"))

end