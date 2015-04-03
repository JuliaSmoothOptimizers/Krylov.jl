# This macro evaluates its arguments before calling @printf.
print_formatted(fmt, args...) = @eval @printf($fmt, $(args...))


## The following function directly interfaces libc's printf.
function process_types(args...)
  # Process types of arguments for libc's printf family.

  # Grab everybody's type.
  types = [typeof(arg) for arg in args]

  # Perform the following conversions:
  # string -> Ptr{Uint8} (for %s)
  # arrays -> Ptr{Void}  (for %p)
  types = map(t -> (t == ASCIIString ? Ptr{Uint8} : t), types)
  types = map(t -> (issubtype(t, Base.Array) ? Ptr{Void} : t), types)
  #types = map(t -> (issubtype(t, Function) ? Ptr{Void} : t), types)
  types = append!([Ptr{Uint8}], types)

  # ccall absolutely wants a tuple as 3rd argument.
  # A variable that evaluates to a tuple just won't do.
  typexpr = Expr(:tuple, types...)
  return typexpr
end


function c_printf(fmt :: String, args...)
  typexpr = process_types(args...)
  # Ignore printf's return value (an int).
  @eval ccall((:printf, "libc"), Void, $(typexpr), $(fmt), $(args...))
end


# Display an array in the form
# [ -3.0e-01 -5.1e-01  1.9e-01 ... -2.3e-01 -4.4e-01  2.4e-01 ]
# with ndisp/2 elements on each side.
function vec2str(x :: Array{Float64,1}; ndisp=7)
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


# Parallel reduce.
preduce(func, darray) = reduce(func,
                               map(fetch,
                                   { (@spawnat p reduce(func, localpart(darray))) for p = procs(darray) } ));
