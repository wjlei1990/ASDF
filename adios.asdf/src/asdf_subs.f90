!> Defines asdf variables for earthquake test data set
!! \param adios_group adios group
!! \param my_group_size Stores the adios group size

subroutine define_asdf_variables (adios_group, adios_groupsize)

  use adios_write_mod
  use asdf_vars
  implicit none

  integer :: adios_err, stat, nerr, irec
  integer(kind=8) :: varid
  integer(kind=8),intent(inout) :: adios_group, adios_groupsize

  real, dimension(NDATAMAX)      :: displseries

  character(len=6)             :: npts_string
  character(len=100)            :: waveform, station
  character(len=200)           :: command, station_path, waveform_path

  ! Define the quakeml file

  open(10, file=trim(DATA_DIR)//"/quake.xml", status='old')
  inquire(unit=10, size=QuakeML_size)
  close(10)

  print *, "Size of quake.xml: ", QuakeML_size

  ! Define a string (adios_type=9) to store the quakeml file

  call adios_define_var (adios_group, "QuakeML", "", 9, "", "", "", varid)
  adios_groupsize = QuakeML_size

  ! Get the number of waveforms

  command = "ls -1 DATA/SAC | wc -l > NUMWAVEFORMS"
  call system(command)
  open (3, file='NUMWAVEFORMS', status='old')
  read (3, *, iostat=stat) nwaveforms
  print *, nwaveforms , " waveforms found."
  close(3)

  ! Define the waveform traces

  command = "ls -1 DATA/SAC > SAC_WAVEFORMS"
  call system(command)

  open(10, file="SAC_WAVEFORMS", status='old')
  do
    read (10,'(A)', iostat=stat) waveform
    if (stat /= 0) exit
    waveform_path = trim(trim(DATA_DIR)//"/SAC/"//waveform)
    call rsac1(trim(adjustl(waveform_path)), displseries, npoints, B, DELTA, NDATAMAX, nerr)
    write( npts_string, '(I6)' )  npoints
    call adios_define_var (adios_group, trim(waveform), "", 5, trim(npts_string), &
           "", "0", varid)
    adios_groupsize = adios_groupsize + 8*npoints
  enddo
  close(10)

  command = "ls -l DATA/StationXML | wc -l > NUMSTATIONS"
  call system(command)
  open(4, file="NUMSTATIONS", status='old')
  read(4, *, iostat=stat) nreceivers
  print *, nreceivers, " stations found."
  close(4)

  allocate(StationXML_size(nreceivers))

  command = "ls DATA/StationXML > STATION_FILES"
  call system(command)

  irec = 1
  open(5, file="STATION_FILES", status='old')
  do
    read (5,*, iostat=stat) station
    if (stat /= 0) exit
    station_path = trim(trim(DATA_DIR)//"/StationXML/"//station)
    open(6, file=station_path, status='old')
    inquire(unit=6, size=StationXML_size(irec))
    close(6)
    call adios_define_var (adios_group, trim(station), "", 9, "", "", "", varid)
    adios_groupsize = adios_groupsize + StationXML_size(irec)
    irec = irec + 1
  enddo
  close(5)
  
  print *, "Adios groupsize (bytes) is : ", adios_groupsize
  
end subroutine define_asdf_variables

!> Writes QuakeML, StationXML, and sac waveform data to an asdf data file
!! \param adios_handle The file will be saved as file_name.
!! \param my_adios_group Name of the group
!! \param sac_type Distinguishes between observed and synthetic data

subroutine write_asdf_variables (adios_handle)

  use adios_write_mod
  use asdf_vars
  implicit none

  integer                      :: adios_err, i, nerr, stat, irec, length
  integer(kind=8),intent(in)   :: adios_handle

  character(len=100)           :: waveform, station
  character(len=200)           :: command, station_path, waveform_path
  character(len=QuakeML_size)  :: QuakeML
  character(len=500000)        :: StationXML
  character(len=:), allocatable :: StationXML_string
  
  real, dimension(NDATAMAX)    :: displseries

  ! Read in QuakeML

  open(10, file=trim(DATA_DIR)//"/quake.xml", status='old', &
         recl=QuakeML_size, form='unformatted', access='direct')
  read (10, rec=1) QuakeML
  close(10)
  
  call adios_write(adios_handle, "QuakeML", trim(QuakeML), adios_err)

  ! Read in Waveforms

  displseries(1:NDATAMAX) = 0.0
  open(15, file="SAC_WAVEFORMS", status='old')
  do 
    read (15,'(A)', iostat=stat) waveform
    if (stat /= 0)  exit
    waveform_path = trim(trim(DATA_DIR)//"/SAC/"//waveform)
    call rsac1(trim(adjustl(waveform_path)), displseries, npoints, B, DELTA, NDATAMAX, nerr)
    call adios_write(adios_handle, trim(waveform), displseries, adios_err)
  enddo
  close (15)

  ! Read in StationXML files

  open(20, file="STATION_FILES", status='old')
  irec = 1
  do
    read (20,*, iostat=stat) station
    if (stat /= 0) exit
    length = StationXML_size(irec)
    allocate(character(length) :: StationXML_string)
    station_path = trim(trim(DATA_DIR)//"/StationXML/"//station)
    open(25, file=station_path, status='old', &
        recl=StationXML_size(irec), form='unformatted', access='direct')
    read(25, rec=1) StationXML_string
    StationXML = StationXML_String
    close(25)
    call adios_write(adios_handle, trim(station), trim(StationXML), adios_err)
    irec = irec + 1
    deallocate(StationXML_string)
  enddo
  close(20)

end subroutine write_asdf_variables
