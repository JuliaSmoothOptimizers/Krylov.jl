@testset "stats" begin
  stats = Krylov.SimpleStats(true, true, Float64[], Float64[], "t")
  io = IOBuffer()
  show(io, stats)
  showed = String(take!(io))
  storage_type = typeof(stats)
  expected = """Simple stats
  solved: true
  inconsistent: true
  residuals: []
  Aresiduals: []
  status: t"""
  @test strip.(split(chomp(showed), "\n")) == strip.(split(chomp(expected), "\n"))
  Krylov.reset!(stats)

  stats = Krylov.LanczosStats(true, Float64[], Bool[], NaN, NaN,"t")
  io = IOBuffer()
  show(io, stats)
  showed = String(take!(io))
  storage_type = typeof(stats)
  expected = """Lanczos stats
  solved: true
  residuals: []
  flagged: Bool[]
  ‖A‖F: NaN
  κ₂(A): NaN
  status: t"""
  @test strip.(split(chomp(showed), "\n")) == strip.(split(chomp(expected), "\n"))
  Krylov.reset!(stats)

  stats = Krylov.SymmlqStats(
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
  show(io, stats)
  showed = String(take!(io))
  storage_type = typeof(stats)
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
  Krylov.reset!(stats)

  stats = Krylov.AdjointStats(true, true, Float64[], Float64[],"t")
  io = IOBuffer()
  show(io, stats)
  showed = String(take!(io))
  storage_type = typeof(stats)
  expected = """Adjoint stats
  solved primal: true
  solved dual: true
  residuals primal: []
  residuals dual: []
  status: t"""
  @test strip.(split(chomp(showed), "\n")) == strip.(split(chomp(expected), "\n"))
  Krylov.reset!(stats)

  stats = Krylov.LNLQStats(true, Float64[], false, Float64[], Float64[], "t")
  io = IOBuffer()
  show(io, stats)
  showed = String(take!(io))
  storage_type = typeof(stats)
  expected = """LNLQ stats
  solved: true
  residuals: []
  error with bnd: false
  error bnd x: []
  error bnd y: []
  status: t"""
  @test strip.(split(chomp(showed), "\n")) == strip.(split(chomp(expected), "\n"))
  Krylov.reset!(stats)

  stats = Krylov.LSLQStats(true, false, Float64[], Float64[], Float64[], false, Float64[], Float64[], "t")
  io = IOBuffer()
  show(io, stats)
  showed = String(take!(io))
  storage_type = typeof(stats)
  expected = """LSLQ stats
  solved: true
  inconsistent: false
  residuals: []
  Aresiduals: []
  err lbnds: []
  error with bnd: false
  error bound LQ: []
  error bound CG: []
  status: t"""
  @test strip.(split(chomp(showed), "\n")) == strip.(split(chomp(expected), "\n"))
  Krylov.reset!(stats)

end
