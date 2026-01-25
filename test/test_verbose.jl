function test_verbose(FC)
  A   = FC.(get_div_grad(4, 4, 4))  # Dimension m x n
  m,n = size(A)
  k   = div(n, 2)
  Au  = A[1:k,:]          # Dimension k x n
  Ao  = A[:,1:k]          # Dimension m x k
  b   = Ao * ones(FC, k)  # Dimension m
  c   = Au * ones(FC, n)  # Dimension k
  mem = 10

  T = real(FC)
  shifts  = T[1; 2; 3; 4; 5]
  nshifts = 5

  for method in (:cg, :cgls, :usymqr, :cgne, :cgs, :crmr, :cg_lanczos, :dqgmres, :diom, :cr, :gpmr,
                 :lslq, :lsqr, :lsmr, :lnlq, :craig, :bicgstab, :craigmr, :crls, :symmlq, :minres,
                 :bilq, :minres_qlp, :qmr, :usymlq, :tricg, :trimr, :trilqr, :bilqr, :gmres, :fom,
                 :car, :minares, :fgmres, :usymlqr, :cg_lanczos_shift, :cgls_lanczos_shift)

    @testset "$method" begin
      io = IOBuffer()
      if method in (:trilqr, :bilqr)
        krylov_solve(Val{method}(), A, b, b, verbose=1, iostream=io)
      elseif method in (:tricg, :trimr, :usymlqr)
        krylov_solve(Val{method}(), Au, c, b, verbose=1, iostream=io)
      elseif method in (:lnlq, :craig, :craigmr, :cgne, :crmr)
        krylov_solve(Val{method}(), Au, c, verbose=1, iostream=io)
      elseif method in (:lslq, :lsqr, :lsmr, :cgls, :crls)
        krylov_solve(Val{method}(), Ao, b, verbose=1, iostream=io)
      elseif method == :usymlq
        krylov_solve(Val{method}(), Au, c, b, verbose=1, iostream=io)
      elseif method == :usymqr
        krylov_solve(Val{method}(), Ao, b, c, verbose=1, iostream=io)
      elseif method == :gpmr
        krylov_solve(Val{method}(), Ao, Au, b, c, verbose=1, iostream=io)
      elseif method in (:cg_lanczos_shift, :cgls_lanczos_shift)
        krylov_solve(Val{method}(), A, b, shifts, verbose=1, iostream=io)
      else
        krylov_solve(Val{method}(), A, b, verbose=1, iostream=io)
      end

      showed = String(take!(io))
      str = split(showed, '\n', keepempty=false)
      nrows = length(str)
      first_row = method in (:bilqr, :trilqr) ? 3 : 2
      last_row = method == :cg ? nrows-1 : nrows
      str = str[first_row:last_row]
      len_header = length(str[1])
      @test mapreduce(x -> length(x) == len_header, &, str)
    end
  end
end

@testset "verbose" begin
  for FC in (Float64, ComplexF64)
    @testset "Data Type: $FC" begin
      test_verbose(FC)
    end
  end
end
