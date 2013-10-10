module asdf_vars

  ! -------------------------------------------------------------
  ! directories
  character(len=100) :: DATA_DIR = "/tigress/jas11/SDF/adios.asdf/DATA"
  ! -------------------------------------------------------------
  ! files
  character(len=100) :: ASDF_FILE = "asdf.bp"
  ! -------------------------------------------------------------
  ! global variables
  integer            :: NDATAMAX = 300000
  integer            :: nwaveforms, npoints, nreceivers
  integer            :: QuakeML_size
  real               :: B, DELTA
  integer,allocatable:: StationXML_size(:)

end module asdf_vars
