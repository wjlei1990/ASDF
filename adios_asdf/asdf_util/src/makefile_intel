#Directories
PWD    = $(shell pwd)
OBJDIR = ../obj
SRCDIR = $(PWD)
INCDIR = ../include
LIBDIR = ../lib

CFLAGS=-g -O3 -warn all

############################
#compiler option
FC = ifort
CC = icc
MPIFC = mpif90
MPICC = mpicc

_OBJ = asdf_helpers_definitions.o asdf_helpers_writers.o \
			 asdf_helpers.o asdf_data.o asdf_manager.o \
			 asdf_read_subs.o asdf_write_subs.o

OBJ = $(patsubst %, ${OBJDIR}/%, $(_OBJ))

## set ADIOS_DIR here or before doing make
#override ADIOS_DIR:=/home/lei/bin/adios-1.5.0
#override ADIOS_INC:=` ${ADIOS_DIR}/bin/adios_config -c -f`
#override ADIOS_FLIB:=`${ADIOS_DIR}/bin/adios_config -l -f`

ADIOS_INC=$(shell adios_config -cf)
ADIOS_FLIB=$(shell adios_config -lf)

##########################################################
PROG = TEST
default: ${PROG}

$(OBJDIR)/%.o: $(SRCDIR)/%.f90
	  $(MPIFC) ${CFLAGS} -c -o $@ $< -module $(INCDIR) $(ADIOS_INC)

$(OBJDIR)/%.o: $(SRCDIR)/%.f
	  $(MPIFC) ${CFLAGS} -c -o $@ $< -module $(INCDIR) $(ADIOS_INC)
.F.o:
	$(OBJDIR)/%.o: $(SRCDIR)/%.F
	  $(MPIFC) ${CFLAGS} -c -o $@ $< -module $(INCDIR) $(ADIOS_INC)

AR_LIB:
	ar cr $(LIBDIR)/libasdf.a $(OBJDIR)/*.o

${PROG}: $(OBJ) AR_LIB

.PHONY:clean print_var cleanall

print_var:
	@echo $(OBJ)
	@echo $(SRCDIR)

clean:
	rm -f $(OBJDIR)/* $(INCDIR)/* $(LIBDIR)/*

cleanall:
	rm -f  iasp91.*
	cd ${TAULIBDUR} ; make -f make_gfortran clean

