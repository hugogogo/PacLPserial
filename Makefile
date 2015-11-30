#
# To build with a different compiler / on a different platform, use
#     make PLATFORM=xxx
#
# where xxx is
#     icc = Intel compilers
#     gcc = GNU compilers
#     clang = Clang compiler (OS X default)
#
# Or create a Makefile.in.xxx of your own!
#

PLATFORM=icc
include Makefile.in.$(PLATFORM)

.PHONY: exe clean reclean

# === Executables

exe: PacLP.x

PacLP.x: PacLP.o mt19937p.o
	$(CC) $(CFLAGS) $(OPTFLAGS) $(LDFLAGS) $^ -o $@

PacLP.o: PacLP.c
	$(CC) -c $(CFLAGS) $(OPTFLAGS) $(LDFLAGS) $<

%.o: %.c
	$(CC) -c $(CFLAGS) $(LDFLAGS) $<

# === Cleanup and tarball

clean:
	rm -f *.o PacLP.x

reclean:
	rm -f lps.o*

