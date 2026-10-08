!--------------------------------------------------------------------------------------------------!
!   CP2K: A general program to perform molecular dynamics simulations                              !
!   Copyright 2000-2026 CP2K developers group <https://cp2k.org>                                   !
!                                                                                                  !
!   SPDX-License-Identifier: GPL-2.0-or-later                                                      !
!--------------------------------------------------------------------------------------------------!
!> \brief Interface to the GPU plane-wave stencil of the native SKALA atom-composite route
!>        (NATIVE_GRID_GPU_STENCIL; kernels in xc/skala_native_grid_gpu.cu). The host makes the
!>        stencils, so the GPU uses the nodes and weights of the CPU path. Without CUDA every
!>        routine aborts.
MODULE qs_native_grid_gpu
   USE ISO_C_BINDING,                   ONLY: C_DOUBLE,&
                                              C_INT
   USE kinds,                           ONLY: dp
#include "./base/base_uses.f90"

   IMPLICIT NONE

   PRIVATE

   CHARACTER(len=*), PARAMETER, PRIVATE :: moduleN = 'qs_native_grid_gpu'

   PUBLIC :: native_grid_gpu_adjoint, native_grid_gpu_interpolate, native_grid_gpu_release, &
             native_grid_gpu_set_grid, native_grid_gpu_set_rows, native_grid_gpu_upload_field

CONTAINS

! **************************************************************************************************
!> \brief Set the plane-wave grid and the number of spin channels on the device.
!> \param npts grid points per direction
!> \param wrap whether the stencil indices wrap around the grid in each direction
!> \param nchannels spin channels (1 or 2)
! **************************************************************************************************
   SUBROUTINE native_grid_gpu_set_grid(npts, wrap, nchannels)
      INTEGER, DIMENSION(3), INTENT(IN)                  :: npts
      LOGICAL, DIMENSION(3), INTENT(IN)                  :: wrap
      INTEGER, INTENT(IN)                                :: nchannels

#if defined(__OFFLOAD_CUDA)
      INTERFACE
         SUBROUTINE set_grid_c(npts, wrap, nchannels) &
            BIND(C, name="skala_native_grid_gpu_set_grid")
            IMPORT :: C_INT
            INTEGER(KIND=C_INT), DIMENSION(3), INTENT(IN)     :: npts, wrap
            INTEGER(KIND=C_INT), VALUE                        :: nchannels
         END SUBROUTINE set_grid_c
      END INTERFACE
      INTEGER(KIND=C_INT), DIMENSION(3)                  :: npts_c, wrap_c

      npts_c = INT(npts, C_INT)
      wrap_c = MERGE(1_C_INT, 0_C_INT, wrap)
      CALL set_grid_c(npts_c, wrap_c, INT(nchannels, C_INT))
#else
      MARK_USED(npts)
      MARK_USED(wrap)
      MARK_USED(nchannels)
      CPABORT("NATIVE_GRID_GPU_STENCIL requires a CP2K build with CUDA.")
#endif
   END SUBROUTINE native_grid_gpu_set_grid

! **************************************************************************************************
!> \brief Upload one plane-wave field of one spin channel.
!> \param channel spin channel, 1 or 2
!> \param component 1 density, 2-4 gradient x, y, z, 5 kinetic-energy density
!> \param field the field on the whole grid
! **************************************************************************************************
   SUBROUTINE native_grid_gpu_upload_field(channel, component, field)
      INTEGER, INTENT(IN)                                :: channel, component
      REAL(KIND=dp), CONTIGUOUS, DIMENSION(:, :, :), &
         INTENT(IN)                                      :: field

#if defined(__OFFLOAD_CUDA)
      INTERFACE
         SUBROUTINE upload_field_c(channel, component, field) &
            BIND(C, name="skala_native_grid_gpu_upload_field")
            IMPORT :: C_DOUBLE, C_INT
            INTEGER(KIND=C_INT), VALUE                        :: channel, component
            REAL(KIND=C_DOUBLE), DIMENSION(*), INTENT(IN)     :: field
         END SUBROUTINE upload_field_c
      END INTERFACE

      CALL upload_field_c(INT(channel - 1, C_INT), INT(component - 1, C_INT), field)
#else
      MARK_USED(channel)
      MARK_USED(component)
      MARK_USED(field)
      CPABORT("NATIVE_GRID_GPU_STENCIL requires a CP2K build with CUDA.")
#endif
   END SUBROUTINE native_grid_gpu_upload_field

! **************************************************************************************************
!> \brief Upload the stencils of all rows, in the order in which values are exchanged.
!> \param first_node zero-based grid index of the first stencil node per direction and row
!> \param weights Lagrange weights per node, direction and row
!> \param active whether the stencil of a row is active
! **************************************************************************************************
   SUBROUTINE native_grid_gpu_set_rows(first_node, weights, active)
      INTEGER, DIMENSION(:, :), INTENT(IN)               :: first_node
      REAL(KIND=dp), DIMENSION(:, :, :), INTENT(IN)      :: weights
      LOGICAL, DIMENSION(:), INTENT(IN)                  :: active

#if defined(__OFFLOAD_CUDA)
      INTERFACE
         SUBROUTINE set_rows_c(nrows, first_node, weights, active) &
            BIND(C, name="skala_native_grid_gpu_set_rows")
            IMPORT :: C_DOUBLE, C_INT
            INTEGER(KIND=C_INT), VALUE                        :: nrows
            INTEGER(KIND=C_INT), DIMENSION(*), INTENT(IN)     :: first_node
            REAL(KIND=C_DOUBLE), DIMENSION(*), INTENT(IN)     :: weights
            INTEGER(KIND=C_INT), DIMENSION(*), INTENT(IN)     :: active
         END SUBROUTINE set_rows_c
      END INTERFACE
      INTEGER(KIND=C_INT), ALLOCATABLE, DIMENSION(:)     :: active_c
      INTEGER(KIND=C_INT), ALLOCATABLE, DIMENSION(:, :)  :: first_node_c

      CPASSERT(SIZE(first_node, 1) == 3)
      CPASSERT(SIZE(weights, 2) == 3)
      CPASSERT(SIZE(first_node, 2) == SIZE(active) .AND. SIZE(weights, 3) == SIZE(active))
      ALLOCATE (first_node_c(3, SIZE(active)), active_c(SIZE(active)))
      first_node_c = INT(first_node, C_INT)
      active_c = MERGE(1_C_INT, 0_C_INT, active)
      CALL set_rows_c(INT(SIZE(active), C_INT), first_node_c, weights, active_c)
      DEALLOCATE (first_node_c, active_c)
#else
      MARK_USED(first_node)
      MARK_USED(weights)
      MARK_USED(active)
      CPABORT("NATIVE_GRID_GPU_STENCIL requires a CP2K build with CUDA.")
#endif
   END SUBROUTINE native_grid_gpu_set_rows

! **************************************************************************************************
!> \brief Interpolate the uploaded fields at all rows.
!> \param values density, gradient x, y, z and kinetic-energy density per channel and row
! **************************************************************************************************
   SUBROUTINE native_grid_gpu_interpolate(values)
      REAL(KIND=dp), CONTIGUOUS, DIMENSION(:, :, :), &
         INTENT(OUT)                                     :: values

#if defined(__OFFLOAD_CUDA)
      INTERFACE
         SUBROUTINE interpolate_c(values) BIND(C, name="skala_native_grid_gpu_interpolate")
            IMPORT :: C_DOUBLE
            REAL(KIND=C_DOUBLE), DIMENSION(*), INTENT(OUT)    :: values
         END SUBROUTINE interpolate_c
      END INTERFACE

      CPASSERT(SIZE(values, 1) == 5)
      CALL interpolate_c(values)
#else
      values = 0.0_dp
      CPABORT("NATIVE_GRID_GPU_STENCIL requires a CP2K build with CUDA.")
#endif
   END SUBROUTINE native_grid_gpu_interpolate

! **************************************************************************************************
!> \brief Scatter the adjoint values of all rows to the plane-wave grid.
!> \param row_values adjoint values in the layout of native_grid_gpu_interpolate
!> \param adjoint the adjoint fields: grid point, component (as the uploaded fields), channel
! **************************************************************************************************
   SUBROUTINE native_grid_gpu_adjoint(row_values, adjoint)
      REAL(KIND=dp), CONTIGUOUS, DIMENSION(:, :, :), &
         INTENT(IN)                                      :: row_values
      REAL(KIND=dp), CONTIGUOUS, DIMENSION(:, :, :), &
         INTENT(OUT)                                     :: adjoint

#if defined(__OFFLOAD_CUDA)
      INTERFACE
         SUBROUTINE adjoint_c(row_values, adjoint) BIND(C, name="skala_native_grid_gpu_adjoint")
            IMPORT :: C_DOUBLE
            REAL(KIND=C_DOUBLE), DIMENSION(*), INTENT(IN)     :: row_values
            REAL(KIND=C_DOUBLE), DIMENSION(*), INTENT(OUT)    :: adjoint
         END SUBROUTINE adjoint_c
      END INTERFACE

      CPASSERT(SIZE(row_values, 1) == 5)
      CPASSERT(SIZE(adjoint, 2) == 5)
      CALL adjoint_c(row_values, adjoint)
#else
      MARK_USED(row_values)
      adjoint = 0.0_dp
      CPABORT("NATIVE_GRID_GPU_STENCIL requires a CP2K build with CUDA.")
#endif
   END SUBROUTINE native_grid_gpu_adjoint

! **************************************************************************************************
!> \brief Free the device memory of the GPU stencil.
! **************************************************************************************************
   SUBROUTINE native_grid_gpu_release()
#if defined(__OFFLOAD_CUDA)
      INTERFACE
         SUBROUTINE release_c() BIND(C, name="skala_native_grid_gpu_release")
         END SUBROUTINE release_c
      END INTERFACE

      CALL release_c()
#endif
   END SUBROUTINE native_grid_gpu_release

END MODULE qs_native_grid_gpu
