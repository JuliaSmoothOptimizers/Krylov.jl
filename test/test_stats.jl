@testset "stats" begin
  stats = Krylov.SimpleStats(true, true, Float64[1.0], Float64[2.0], "t")
  io = IOBuffer()
  show(io, stats)
  showed = String(take!(io))
  storage_type = typeof(stats)
  expected = """Simple stats
  solved: true
  inconsistent: true
  residuals: [ 1.0e+00 ]
  Aresiduals: [ 2.0e+00 ]
  status: t"""
  @test strip.(split(chomp(showed), "\n")) == strip.(split(chomp(expected), "\n"))
  Krylov.reset!(stats)
  check_reset(stats)
  @test (VERSION < v"1.5") || (@allocated Krylov.reset!(stats)) == 0

  stats = Krylov.LanczosStats(true, Float64[3.0], true, NaN, NaN, "t")
  io = IOBuffer()
  show(io, stats)
  showed = String(take!(io))
  storage_type = typeof(stats)
  expected = """Lanczos stats
  solved: true
  residuals: [ 3.0e+00 ]
  indefinite: true
  ‖A‖F: NaN
  κ₂(A): NaN
  status: t"""
  @test strip.(split(chomp(showed), "\n")) == strip.(split(chomp(expected), "\n"))
  Krylov.reset!(stats)
  check_reset(stats)
  @test (VERSION < v"1.5") || (@allocated Krylov.reset!(stats)) == 0

  stats = Krylov.LanczosShiftStats(true, [Float64[0.9, 0.5], Float64[0.6, 0.4, 0.1]], BitVector([false, true]), NaN, NaN, "t")
  io = IOBuffer()
  show(io, stats)
  showed = String(take!(io))
  storage_type = typeof(stats)
  expected = """LanczosShift stats
  solved: true
  residuals: [[0.9, 0.5], [0.6, 0.4, 0.1]]
  indefinite: Bool[0, 1]
  ‖A‖F: NaN
  κ₂(A): NaN
  status: t"""
  @test (VERSION < v"1.5") || strip.(split(chomp(showed), "\n")) == strip.(split(chomp(expected), "\n"))
  Krylov.reset!(stats)
  @test (VERSION < v"1.5") || (@allocated Krylov.reset!(stats)) == 0

  stats = Krylov.SymmlqStats(true, Float64[4.0], Union{Float64,Missing}[5.0, missing], Float64[6.0], Union{Float64,Missing}[7.0, missing], NaN, NaN, "t")
  io = IOBuffer()
  show(io, stats)
  showed = String(take!(io))
  storage_type = typeof(stats)
  expected = """Symmlq stats
  solved: true
  residuals: [ 4.0e+00 ]
  residuals (cg): [ 5.0e+00  ✗✗✗✗ ]
  errors: [ 6.0e+00 ]
  errors (cg): [ 7.0e+00  ✗✗✗✗ ]
  ‖A‖F: NaN
  κ₂(A): NaN
  status: t"""
  @test strip.(split(chomp(showed), "\n")) == strip.(split(chomp(expected), "\n"))
  Krylov.reset!(stats)
  check_reset(stats)
  @test (VERSION < v"1.5") || (@allocated Krylov.reset!(stats)) == 0

  stats = Krylov.AdjointStats(true, true, Float64[8.0], Float64[9.0], "t")
  io = IOBuffer()
  show(io, stats)
  showed = String(take!(io))
  storage_type = typeof(stats)
  expected = """Adjoint stats
  solved primal: true
  solved dual: true
  residuals primal: [ 8.0e+00 ]
  residuals dual: [ 9.0e+00 ]
  status: t"""
  @test strip.(split(chomp(showed), "\n")) == strip.(split(chomp(expected), "\n"))
  Krylov.reset!(stats)
  check_reset(stats)
  @test (VERSION < v"1.5") || (@allocated Krylov.reset!(stats)) == 0

  stats = Krylov.LNLQStats(true, Float64[10.0], false, Float64[11.0], Float64[12.0], "t")
  io = IOBuffer()
  show(io, stats)
  showed = String(take!(io))
  storage_type = typeof(stats)
  expected = """LNLQ stats
  solved: true
  residuals: [ 1.0e+01 ]
  error with bnd: false
  error bnd x: [ 1.1e+01 ]
  error bnd y: [ 1.2e+01 ]
  status: t"""
  @test strip.(split(chomp(showed), "\n")) == strip.(split(chomp(expected), "\n"))
  Krylov.reset!(stats)
  check_reset(stats)
  @test (VERSION < v"1.5") || (@allocated Krylov.reset!(stats)) == 0

  stats = Krylov.LSLQStats(true, false, Float64[13.0], Float64[14.0], Float64[15.0], false, Float64[16.0], Float64[17.0], "t")
  io = IOBuffer()
  show(io, stats)
  showed = String(take!(io))
  storage_type = typeof(stats)
  expected = """LSLQ stats
  solved: true
  inconsistent: false
  residuals: [ 1.3e+01 ]
  Aresiduals: [ 1.4e+01 ]
  err lbnds: [ 1.5e+01 ]
  error with bnd: false
  error bound LQ: [ 1.6e+01 ]
  error bound CG: [ 1.7e+01 ]
  status: t"""
  @test strip.(split(chomp(showed), "\n")) == strip.(split(chomp(expected), "\n"))
  Krylov.reset!(stats)
  check_reset(stats)
  @test (VERSION < v"1.5") || (@allocated Krylov.reset!(stats)) == 0
end
