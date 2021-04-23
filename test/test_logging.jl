function test_logging()
  L   = get_div_grad(4, 4, 4)
  n   = size(L, 1)
  m   = div(n, 2)
  A   = PreallocatedLinearOperator(L) # Dimension n x n
  Au  = PreallocatedLinearOperator(L[1:m,:]) # Dimension m x n
  Ao  = PreallocatedLinearOperator(L[:,1:m]) # Dimension n x m
  b   = Ao * ones(m) # Dimension n
  c   = Au * ones(n) # Dimension m
  mem = 10

  shifts  = [1.0; 2.0; 3.0; 4.0; 5.0]
  nshifts = 5

  symmlq(A, b, verbose=1)
  cg(A, b, verbose=1)
  minres(A, b, verbose=1)
  cg_lanczos(A, b, verbose=1)
  cg_lanczos_shift_seq(A, b, shifts, verbose=1)
  dqgmres(A, b, verbose=1)
  cr(A, b, verbose=1)
  crmr(Au, c, verbose=1)
  cgs(A, b, verbose=1)
  bicgstab(A, b, verbose=1)
  craigmr(Au, c, verbose=1)
  cgne(Au, c, verbose=1)
  lnlq(Au, c, verbose=1)
  craig(Au, c, verbose=1)
  lslq(Ao, b, verbose=1)
  cgls(Ao, b, verbose=1)
  lsqr(Ao, b, verbose=1)
  crls(Ao, b, verbose=1)
  lsmr(Ao, b, verbose=1)
  usymqr(Ao, b, c, verbose=1)
  trilqr(A, b, b, verbose=1)
  bilq(A, b, verbose=1)
  bilqr(A, b, b, verbose=1)
  minres_qlp(A, b, verbose=1)
  qmr(A, b, verbose=1)
  usymlq(Au, c, b, verbose=1)
  tricg(Au, c, b, verbose=1)
  trimr(Au, c, b, verbose=1)
end

@testset "logging" begin
  test_logging()
end
