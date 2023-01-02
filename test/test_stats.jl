@testset "stats" begin
  stats = Krylov.SimpleStats(0, true, true, Float64[1.0], Float64[2.0], Float64[], "t")
  io = IOBuffer()
  show(io, stats)
  showed = String(take!(io))
  storage_type = typeof(stats)
  expected = """SimpleStats
  niter: 0
  solved: true
  inconsistent: true
  residuals: [ 1.0e+00 ]
  Aresiduals: [ 2.0e+00 ]
  κ₂(A): []
  status: t"""
  @test strip.(split(chomp(showed), "\n")) == strip.(split(chomp(expected), "\n"))
  Krylov.reset!(stats)
  check_reset(stats)
  nbytes_allocated = @allocated Krylov.reset!(stats)
  @test nbytes_allocated == 0

  stats = Krylov.LsmrStats(0, true, true, Float64[1.0], Float64[2.0], Float64(3.0), Float64(4.0), Float64(5.0), Float64(6.0), Float64(7.0), "t")
  io = IOBuffer()
  show(io, stats)
  showed = String(take!(io))
  storage_type = typeof(stats)
  expected = """LsmrStats
  niter: 0
  solved: true
  inconsistent: true
  residuals: [ 1.0e+00 ]
  Aresiduals: [ 2.0e+00 ]
  residual: 3.0
  Aresidual: 4.0
  κ₂(A): 5.0
  ‖A‖F: 6.0
  xNorm: 7.0
  status: t"""
  @test strip.(split(chomp(showed), "\n")) == strip.(split(chomp(expected), "\n"))
  Krylov.reset!(stats)
  check_reset(stats)
  nbytes_allocated = @allocated Krylov.reset!(stats)
  @test nbytes_allocated == 0

  stats = Krylov.LanczosStats(0, true, Float64[3.0], true, NaN, NaN, "t")
  io = IOBuffer()
  show(io, stats)
  showed = String(take!(io))
  storage_type = typeof(stats)
  expected = """LanczosStats
  niter: 0
  solved: true
  residuals: [ 3.0e+00 ]
  indefinite: true
  ‖A‖F: NaN
  κ₂(A): NaN
  status: t"""
  @test strip.(split(chomp(showed), "\n")) == strip.(split(chomp(expected), "\n"))
  Krylov.reset!(stats)
  check_reset(stats)
  nbytes_allocated = @allocated Krylov.reset!(stats)
  @test nbytes_allocated == 0

  stats = Krylov.LanczosShiftStats(0, true, [Float64[0.9, 0.5], Float64[0.6, 0.4, 0.1]], BitVector([false, true]), NaN, NaN, "t")
  io = IOBuffer()
  show(io, stats)
  showed = String(take!(io))
  storage_type = typeof(stats)
  expected = """LanczosShiftStats
  niter: 0
  solved: true
  residuals: [[0.9, 0.5], [0.6, 0.4, 0.1]]
  indefinite: Bool[0, 1]
  ‖A‖F: NaN
  κ₂(A): NaN
  status: t"""
  @test strip.(split(chomp(showed), "\n")) == strip.(split(chomp(expected), "\n"))
  Krylov.reset!(stats)
  nbytes_allocated = @allocated Krylov.reset!(stats)
  @test nbytes_allocated == 0

  stats = Krylov.SymmlqStats(0, true, Float64[4.0], Union{Float64,Missing}[5.0, missing], Float64[6.0], Union{Float64,Missing}[7.0, missing], NaN, NaN, "t")
  io = IOBuffer()
  show(io, stats)
  showed = String(take!(io))
  storage_type = typeof(stats)
  expected = """SymmlqStats
  niter: 0
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
  nbytes_allocated = @allocated Krylov.reset!(stats)
  @test nbytes_allocated == 0

  stats = Krylov.AdjointStats(0, true, true, Float64[8.0], Float64[9.0], "t")
  io = IOBuffer()
  show(io, stats)
  showed = String(take!(io))
  storage_type = typeof(stats)
  expected = """AdjointStats
  niter: 0
  solved primal: true
  solved dual: true
  residuals primal: [ 8.0e+00 ]
  residuals dual: [ 9.0e+00 ]
  status: t"""
  @test strip.(split(chomp(showed), "\n")) == strip.(split(chomp(expected), "\n"))
  Krylov.reset!(stats)
  check_reset(stats)
  nbytes_allocated = @allocated Krylov.reset!(stats)
  @test nbytes_allocated == 0

  stats = Krylov.LNLQStats(0, true, Float64[10.0], false, Float64[11.0], Float64[12.0], "t")
  io = IOBuffer()
  show(io, stats)
  showed = String(take!(io))
  storage_type = typeof(stats)
  expected = """LNLQStats
  niter: 0
  solved: true
  residuals: [ 1.0e+01 ]
  error with bnd: false
  error bnd x: [ 1.1e+01 ]
  error bnd y: [ 1.2e+01 ]
  status: t"""
  @test strip.(split(chomp(showed), "\n")) == strip.(split(chomp(expected), "\n"))
  Krylov.reset!(stats)
  check_reset(stats)
  nbytes_allocated = @allocated Krylov.reset!(stats)
  @test nbytes_allocated == 0

  stats = Krylov.LSLQStats(0, true, false, Float64[13.0], Float64[14.0], Float64[15.0], false, Float64[16.0], Float64[17.0], "t")
  io = IOBuffer()
  show(io, stats)
  showed = String(take!(io))
  storage_type = typeof(stats)
  expected = """LSLQStats
  niter: 0
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
  nbytes_allocated = @allocated Krylov.reset!(stats)
  @test nbytes_allocated == 0
end
