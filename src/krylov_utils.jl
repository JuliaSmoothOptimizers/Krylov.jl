# This macro evaluates its arguments before calling @printf.
print_formatted(fmt, args...) = @eval @printf($fmt, $(args...))

# Parallel reduce.
preduce(func, darray) = reduce(func,
                               map(fetch,
                                   { (@spawnat p reduce(func, localpart(darray))) for p = procs(darray) } ));
