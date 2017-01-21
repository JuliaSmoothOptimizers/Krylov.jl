# Display an array in the form
# [ -3.0e-01 -5.1e-01  1.9e-01 ... -2.3e-01 -4.4e-01  2.4e-01 ]
# with ndisp/2 elements on each side.
function vec2str(x :: Array{Float64,1}; ndisp :: Int=7)
  n = length(x);
  if n <= ndisp
    ndisp = n;
    nside = n;
  else
    nside = max(1, div(ndisp - 1, 2));
  end
  s = "[ ";
  i = 1;
  while i <= nside
    s *= @sprintf("%8.1e ", x[i]);
    i += 1;
  end
  if i <= div(n, 2)
    s *= "... ";
  end;
  i = max(i, n - nside + 1);
  while i <= n
    s *= @sprintf("%8.1e ", x[i]);
    i += 1;
  end
  s *= "]";
  return s;
end
