# Compiler
FC=mpif90
FFLAGS=-g -check all

# Directories
PWD    = $(shell pwd)
OBJDIR = $(PWD)/obj
SRCDIR = $(PWD)/src
LIBDIR = $(PWD)/UTILS/lib
IDIR   = $(PWD)/UTILS/include

## Set ADIOS_DIR here or before doing make
override ADIOS_DIR:=/home/jas11/bin
override ADIOS_INC:=` ${ADIOS_DIR}/bin/adios_config -c -f`
override ADIOS_FLIB:=`${ADIOS_DIR}/bin/adios_config -l -f`

# Libraries
LIBS = -lsacio -lsac

# Files and folders
TARGET = generate_asdf
_OBJ = asdf_vars.o asdf_subs.o generate_asdf.o

# Make all
all: $(OBJDIR) $(TARGET)

OBJ = $(patsubst %,$(OBJDIR)/%,$(_OBJ))

$(OBJDIR):
	mkdir -p $(OBJDIR)

$(OBJDIR)/%.o: $(SRCDIR)/%.f90
	$(FC) -c -o $@ $< $(FFLAGS) -module $(OBJDIR) $(ADIOS_INC)

$(TARGET) : $(OBJ)
	$(FC) -o $@ $^ $(FFLAGS) -I$(OBJDIR) -L$(LIBDIR) $(LIBS) -I$(IDIR) $(ADIOS_FLIB)

.PHONY: clean
clean:
	rm -f $(OBJDIR)/*.f90 $(OBJDIR)/*.o $(OBJDIR)/*.mod core.*
