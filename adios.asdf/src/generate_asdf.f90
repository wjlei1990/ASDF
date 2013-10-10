!> \file generate_asdf.f90
!! \Generates adios bp files from a QuakeML file, a folder
!! \of waveform files and folder of StationXML files
!! \author JAS

program generate_asdf

  use adios_write_mod
  use asdf_vars
  use mpi
  implicit none

  integer :: comm, rank, ierr, nproc, adios_err  
  integer(kind=8) :: adios_handle, adios_group
  integer(kind=8) :: adios_groupsize, adios_totalsize

  call MPI_Init (ierr)
  call MPI_Comm_dup (MPI_COMM_WORLD, comm, ierr)
  call MPI_Comm_rank (comm, rank, ierr)
  call MPI_Comm_size (comm, nproc, ierr)

  call adios_init_noxml (comm)
  call adios_allocate_buffer (500, adios_err)

  call adios_declare_group (adios_group, "EVENTS", "", 1, adios_err)
  call adios_select_method (adios_group, "MPI", "", "", adios_err)
  
  adios_groupsize = 0
  call define_asdf_variables (adios_group, adios_groupsize)

  call adios_open (adios_handle, "EVENTS", ASDF_FILE, "w", comm, adios_err)
  call adios_group_size (adios_handle, adios_groupsize, adios_totalsize, adios_err)
  call write_asdf_variables (adios_handle, comm)
  call adios_close (adios_handle, adios_err)

  call MPI_Barrier (comm, ierr)
  call adios_finalize (rank, adios_err)
  call MPI_Finalize (ierr)

end program
