!==============================================================================
!> Tools to setup and cleanup ADIOS
!------------------------------------------------------------------------------
module asdf_manager_mod

  use asdf_data

contains

!==============================================================================
!> Initialize ADIOS and setup the xml output file
subroutine asdf_setup()

  use adios_write_mod, only: adios_init
  use mpi

  implicit none

  integer :: comm, rank
  integer :: adios_err, ierr
  integer :: ASDF_BUFFER_SIZE_IN_MB = 500

  call MPI_COMM_RANK(MPI_COMM_WORLD, rank, ierr)
  call MPI_COMM_DUP(MPI_COMM_WORLD, comm, ierr)

  call adios_init_noxml (comm, adios_err);
  !sizeMB = 200 ! TODO 200MB is surely not the right size for the adios buffer
  !call adios_allocate_buffer (sizeMB , adios_err)
  call adios_allocate_buffer (ASDF_BUFFER_SIZE_IN_MB, adios_err)
end subroutine asdf_setup

!==============================================================================
!> Finalize ADIOS. Must be called once everything is written down.
subroutine asdf_cleanup()

  use adios_write_mod, only: adios_finalize
  use mpi

  implicit none
  integer :: myrank
  integer :: adios_err, ierr

  call MPI_COMM_RANK(MPI_COMM_WORLD, myrank, ierr)
  call MPI_Barrier(MPI_COMM_WORLD, ierr)
  !call MPI_Comm_rank(MPI_COMM_WORLD, myrank, ierr)
  call adios_finalize (myrank, adios_err)

end subroutine asdf_cleanup

subroutine init_asdf_data(my_asdf,nrecords,resp_flag)
!init the asdf data structure
!receiver_name, network, component and receiver_id are not allocated

  !use asdf_data
  type(asdf_event) :: my_asdf
  integer :: nrecords
  integer :: len_temp
  logical,intent(in) :: resp_flag

  my_asdf%nrecords=nrecords

  !if (resp_flag) then
  !  my_asdf%STORE_RESPONSE = .TRUE.
  !else
  !  my_asdf%STORE_RESPONSE = .FALSE.
  !endif
  my_asdf%STORE_RESPONSE = resp_flag

  my_asdf%event = ""
  !print *,"Number of Records:", my_asdf%nrecords
  !>allocate array variables
  allocate (my_asdf%npoints(my_asdf%nrecords))
  allocate (my_asdf%gmt_year(my_asdf%nrecords))
  allocate (my_asdf%gmt_hour(my_asdf%nrecords))
  allocate (my_asdf%gmt_day(my_asdf%nrecords))
  allocate (my_asdf%gmt_min(my_asdf%nrecords))
  allocate (my_asdf%gmt_sec(my_asdf%nrecords))
  allocate (my_asdf%gmt_msec(my_asdf%nrecords))

  allocate (my_asdf%event_lat(my_asdf%nrecords))
  allocate (my_asdf%event_lo(my_asdf%nrecords))
  allocate (my_asdf%event_dpt(my_asdf%nrecords))

  allocate (my_asdf%receiver_lat(my_asdf%nrecords))
  allocate (my_asdf%receiver_lo(my_asdf%nrecords))
  allocate (my_asdf%receiver_el(my_asdf%nrecords))
  allocate (my_asdf%receiver_dpt(my_asdf%nrecords))
  allocate (my_asdf%begin_value(my_asdf%nrecords))
  allocate (my_asdf%end_value(my_asdf%nrecords))
  allocate (my_asdf%cmp_azimuth(my_asdf%nrecords))
  allocate (my_asdf%cmp_incident_ang(my_asdf%nrecords))
  allocate (my_asdf%sample_rate(my_asdf%nrecords))
  allocate (my_asdf%scale_factor(my_asdf%nrecords))
  allocate (my_asdf%ev_to_sta_AZ(my_asdf%nrecords))
  allocate (my_asdf%sta_to_ev_AZ(my_asdf%nrecords))
  allocate (my_asdf%great_circle_arc(my_asdf%nrecords))
  allocate (my_asdf%dist(my_asdf%nrecords))
  allocate (my_asdf%P_pick(my_asdf%nrecords))
  allocate (my_asdf%S_pick(my_asdf%nrecords))
  !>the kernel part: allocate the record
  allocate (my_asdf%records(my_asdf%nrecords))
  allocate (my_asdf%responses(my_asdf%nrecords))
  allocate (my_asdf%receiver_name_array(my_asdf%nrecords))
  allocate (my_asdf%network_array(my_asdf%nrecords))
  allocate (my_asdf%component_array(my_asdf%nrecords))
  allocate (my_asdf%receiver_id_array(my_asdf%nrecords))
  len_temp=6*nrecords
  my_asdf%min_period=0.0
  my_asdf%max_period=0.0

end subroutine init_asdf_data

subroutine deallocate_asdf(asdf_container, ierr)

  integer :: ierr
  type(asdf_event),intent(inout) :: asdf_container
  integer :: i

  deallocate (asdf_container%gmt_hour, STAT=ierr)
  if (ierr /= 0) stop "Failed to deallocate "
  deallocate (asdf_container%gmt_day, STAT=ierr)
  if (ierr /= 0) stop "Failed to deallocate "
  deallocate (asdf_container%gmt_min, STAT=ierr)
  if (ierr /= 0) stop "Failed to deallocate "
  deallocate (asdf_container%gmt_sec, STAT=ierr)
  if (ierr /= 0) stop "Failed to deallocate "
  deallocate (asdf_container%gmt_msec, STAT=ierr)
  if (ierr /= 0) stop "Failed to deallocate "
  deallocate (asdf_container%event_lat, STAT=ierr)
  if (ierr /= 0) stop "Failed to deallocate "
  deallocate (asdf_container%event_lo, STAT=ierr)
  if (ierr /= 0) stop "Failed to deallocate "
  deallocate (asdf_container%event_dpt, STAT=ierr)
  if (ierr /= 0) stop "Failed to deallocate "
  deallocate (asdf_container%receiver_lat, STAT=ierr)
  if (ierr /= 0) stop "Failed to deallocate "
  deallocate (asdf_container%receiver_lo, STAT=ierr)
  if (ierr /= 0) stop "Failed to deallocate "
  deallocate (asdf_container%receiver_el, STAT=ierr)
  if (ierr /= 0) stop "Failed to deallocate "
  deallocate (asdf_container%receiver_dpt, STAT=ierr)
  if (ierr /= 0) stop "Failed to deallocate "
  deallocate (asdf_container%begin_value, STAT=ierr)
  if (ierr /= 0) stop "Failed to deallocate "
  deallocate (asdf_container%end_value, STAT=ierr)
  if (ierr /= 0) stop "Failed to deallocate "
  deallocate (asdf_container%cmp_azimuth, STAT=ierr)
  if (ierr /= 0) stop "Failed to deallocate "
  deallocate (asdf_container%cmp_incident_ang, STAT=ierr)
  if (ierr /= 0) stop "Failed to deallocate "
  deallocate (asdf_container%sample_rate, STAT=ierr)
  if (ierr /= 0) stop "Failed to deallocate "
  deallocate (asdf_container%scale_factor, STAT=ierr)
  if (ierr /= 0) stop "Failed to deallocate "
  deallocate (asdf_container%ev_to_sta_AZ, STAT=ierr)
  if (ierr /= 0) stop "Failed to deallocate "
  deallocate (asdf_container%sta_to_ev_AZ, STAT=ierr)
  if (ierr /= 0) stop "Failed to deallocate "
  deallocate (asdf_container%great_circle_arc, STAT=ierr)
  if (ierr /= 0) stop "Failed to deallocate "
  deallocate (asdf_container%dist, STAT=ierr)
  if (ierr /= 0) stop "Failed to deallocate "
  deallocate (asdf_container%P_pick, STAT=ierr)
  if (ierr /= 0) stop "Failed to deallocate "
  deallocate (asdf_container%S_pick, STAT=ierr)
  if (ierr /= 0) stop "Failed to deallocate "
  do i = 1, asdf_container%nrecords
    deallocate(asdf_container%records(i)%record, STAT=ierr)
    if (ierr /= 0) stop "Failed to deallocate "
  enddo
  deallocate (asdf_container%receiver_name_array, STAT=ierr)
  if (ierr /= 0) stop "Failed to deallocate "
  deallocate (asdf_container%network_array, STAT=ierr)
  if (ierr /= 0) stop "Failed to deallocate "
  deallocate (asdf_container%component_array, STAT=ierr)
  if (ierr /= 0) stop "Failed to deallocate "
  deallocate (asdf_container%receiver_id_array, STAT=ierr)
  if (ierr /= 0) stop "Failed to deallocate "

end subroutine deallocate_asdf

end module asdf_manager_mod
