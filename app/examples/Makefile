LIB_DIR := ../CKrylov/lib
INCLUDE_DIR := ../CKrylov/include
WLARGS := -Wl,-rpath,"$(LIB_DIR)" -Wl,-rpath,"$(LIB_DIR)/julia"

CFLAGS+=-O2 -fPIE -I$(INCLUDE_DIR)
LDFLAGS+=-lm -L$(LIB_DIR) -ljulia $(WLARGS)

main: main.o
	gcc -o $@ $< $(LDFLAGS) -lkrylov

main.o: main.c
	gcc $< -c -o $@ $(CFLAGS)

.PHONY: clean
clean:
	rm *~ *.o main
