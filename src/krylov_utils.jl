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


# Parallel reduce.
preduce(func, darray) = reduce(func,
                               map(fetch,
                                   { (@spawnat p reduce(func, localpart(darray))) for p = procs(darray) } ));
