!--------------------------------------------------------------------------------------------------!
!   CP2K: A general program to perform molecular dynamics simulations                              !
!   Copyright 2000-2026 CP2K developers group <https://cp2k.org>                                   !
!                                                                                                  !
!   SPDX-License-Identifier: GPL-2.0-or-later                                                      !
!--------------------------------------------------------------------------------------------------!
!> \brief One-centre fields of the native SKALA atom-composite route on the GPU
!>        (NATIVE_GRID_GPU_FIELDS; kernels and cuBLAS calls in xc/skala_fields_gpu.cu):
!>        the angular assembly of calc_rho_angular for all radial shells of an atom at once, and
!>        the cross-atom interpolation of the projected fields (forward and adjoint) from tables
!>        that the host computes once per candidate list. The quantities are those of the CPU path;
!>        only the order of the sums differs. Without CUDA every routine aborts.
MODULE qs_native_fields_gpu
   USE ISO_C_BINDING,                   ONLY: C_DOUBLE,&
                                              C_INT
   USE kinds,                           ONLY: dp
   USE qs_grid_atom,                    ONLY: grid_atom_type
   USE qs_harmonics_atom,               ONLY: harmonics_atom_type
   USE qs_rho_atom_types,               ONLY: rho_atom_coeff
#include "./base/base_uses.f90"

   IMPLICIT NONE

   PRIVATE

   CHARACTER(len=*), PARAMETER, PRIVATE :: moduleN = 'qs_native_fields_gpu'

   PUBLIC :: fields_gpu_angular, fields_gpu_cross_adjoint, fields_gpu_cross_forward, &
             fields_gpu_release, fields_gpu_set_cross

#if defined(__OFFLOAD_CUDA)
   INTERFACE
      SUBROUTINE set_kind_c(kind, na, niso, slm, a) BIND(C, name="skala_fields_gpu_set_kind")
         IMPORT :: C_DOUBLE, C_INT
         INTEGER(KIND=C_INT), VALUE                        :: kind, na, niso
         REAL(KIND=C_DOUBLE), DIMENSION(*), INTENT(IN)     :: slm, a
      END SUBROUTINE set_kind_c
      SUBROUTINE angular_c(kind, nr, radial, rho_h, rho_s, drho_h, drho_s) &
         BIND(C, name="skala_fields_gpu_angular")
         IMPORT :: C_DOUBLE, C_INT
         INTEGER(KIND=C_INT), VALUE                        :: kind, nr
         REAL(KIND=C_DOUBLE), DIMENSION(*), INTENT(IN)     :: radial
         REAL(KIND=C_DOUBLE), DIMENSION(*), INTENT(OUT)    :: rho_h, rho_s, drho_h, drho_s
      END SUBROUTINE angular_c
      SUBROUTINE set_cross_c(source, nharm, nr, ncand, ngroups, group_first, cand_group, nnode, &
                             node, radius, weight, harm, nentries, shell_first, entry_cand, &
                             entry_weight) BIND(C, name="skala_fields_gpu_set_cross")
         IMPORT :: C_DOUBLE, C_INT
         INTEGER(KIND=C_INT), VALUE                        :: source, nharm, nr, ncand, ngroups
         INTEGER(KIND=C_INT), DIMENSION(*), INTENT(IN)     :: group_first, cand_group, nnode, node
         REAL(KIND=C_DOUBLE), DIMENSION(*), INTENT(IN)     :: radius, weight, harm
         INTEGER(KIND=C_INT), VALUE                        :: nentries
         INTEGER(KIND=C_INT), DIMENSION(*), INTENT(IN)     :: shell_first, entry_cand
         REAL(KIND=C_DOUBLE), DIMENSION(*), INTENT(IN)     :: entry_weight
      END SUBROUTINE set_cross_c
      SUBROUTINE cross_forward_c(source, nspins, cutoff, projection, group_fields) &
         BIND(C, name="skala_fields_gpu_cross_forward")
         IMPORT :: C_DOUBLE, C_INT
         INTEGER(KIND=C_INT), VALUE                        :: source, nspins
         REAL(KIND=C_DOUBLE), VALUE                        :: cutoff
         REAL(KIND=C_DOUBLE), DIMENSION(*), INTENT(IN)     :: projection
         REAL(KIND=C_DOUBLE), DIMENSION(*), INTENT(OUT)    :: group_fields
      END SUBROUTINE cross_forward_c
      SUBROUTINE cross_adjoint_c(source, nspins, cutoff, group_adjoint, adjoint) &
         BIND(C, name="skala_fields_gpu_cross_adjoint")
         IMPORT :: C_DOUBLE, C_INT
         INTEGER(KIND=C_INT), VALUE                        :: source, nspins
         REAL(KIND=C_DOUBLE), VALUE                        :: cutoff
         REAL(KIND=C_DOUBLE), DIMENSION(*), INTENT(IN)     :: group_adjoint
         REAL(KIND=C_DOUBLE), DIMENSION(*), INTENT(OUT)    :: adjoint
      END SUBROUTINE cross_adjoint_c
      SUBROUTINE release_c() BIND(C, name="skala_fields_gpu_release")
      END SUBROUTINE release_c
   END INTERFACE
#endif

CONTAINS

! **************************************************************************************************
!> \brief GPU counterpart of calling calc_rho_angular (with gradients) for every radial shell of an
!>        atom: rho = sum_iso r(ir, iso) slm(ia, iso) and
!>        grad_d rho = sum_iso (dr(ir, iso) a(d, ia) + r_d(ir, iso)) slm(ia, iso), hard and soft,
!>        with |grad rho| in the fourth component. The fields are overwritten (calc_rho_angular
!>        adds to fields that its callers zero).
!> \param grid_atom atom grid of the kind
!> \param harmonics spherical harmonics of the kind
!> \param nspins ...
!> \param kind_index index of the kind, which identifies the harmonics stored on the device
!> \param r_h radial density coefficients, hard (nr, max_iso_not0)
!> \param r_s soft
!> \param dr_h their radial derivatives, hard
!> \param dr_s soft
!> \param r_h_d their Cartesian-gradient coefficients (direction, spin), hard
!> \param r_s_d soft
!> \param rho_h hard density (angular point, radial point, spin)
!> \param rho_s soft density
!> \param drho_h hard gradient and its norm (4, angular point, radial point, spin)
!> \param drho_s soft gradient and its norm
! **************************************************************************************************
   SUBROUTINE fields_gpu_angular(grid_atom, harmonics, nspins, kind_index, r_h, r_s, dr_h, dr_s, &
                                 r_h_d, r_s_d, rho_h, rho_s, drho_h, drho_s)
      TYPE(grid_atom_type), POINTER                      :: grid_atom
      TYPE(harmonics_atom_type), POINTER                 :: harmonics
      INTEGER, INTENT(IN)                                :: nspins, kind_index
      TYPE(rho_atom_coeff), DIMENSION(:), POINTER        :: r_h, r_s, dr_h, dr_s
      TYPE(rho_atom_coeff), DIMENSION(:, :), POINTER     :: r_h_d, r_s_d
      REAL(dp), DIMENSION(:, :, :), POINTER              :: rho_h, rho_s
      REAL(dp), DIMENSION(:, :, :, :), POINTER           :: drho_h, drho_s

#if defined(__OFFLOAD_CUDA)
      INTEGER                                            :: handle, idir, ispin, na, niso, nr
      REAL(dp), ALLOCATABLE, DIMENSION(:, :)             :: directions, slm
      REAL(dp), ALLOCATABLE, DIMENSION(:, :, :)          :: radial

      CALL timeset("skala_fields_gpu_angular", handle)
      na = grid_atom%ng_sphere
      nr = grid_atom%nr
      niso = harmonics%max_iso_not0
      CPASSERT(SIZE(rho_h, 1) == na .AND. SIZE(rho_h, 2) == nr)
      CPASSERT(SIZE(drho_h, 1) == 4)
      CPASSERT(SIZE(drho_h, 2) == na .AND. SIZE(drho_h, 3) == nr)
      ALLOCATE (slm(na, niso), directions(3, na), radial(nr, niso, 10))
      slm(:, :) = harmonics%slm(1:na, 1:niso)
      directions(:, :) = harmonics%a(1:3, 1:na)
      CALL set_kind_c(INT(kind_index, C_INT), INT(na, C_INT), INT(niso, C_INT), slm, directions)
      DO ispin = 1, nspins
         radial(:, :, 1) = r_h(ispin)%r_coef(1:nr, 1:niso)
         radial(:, :, 2) = r_s(ispin)%r_coef(1:nr, 1:niso)
         radial(:, :, 3) = dr_h(ispin)%r_coef(1:nr, 1:niso)
         radial(:, :, 4) = dr_s(ispin)%r_coef(1:nr, 1:niso)
         DO idir = 1, 3
            radial(:, :, 4 + idir) = r_h_d(idir, ispin)%r_coef(1:nr, 1:niso)
            radial(:, :, 7 + idir) = r_s_d(idir, ispin)%r_coef(1:nr, 1:niso)
         END DO
         CALL angular_c(INT(kind_index, C_INT), INT(nr, C_INT), radial, rho_h(:, :, ispin), &
                        rho_s(:, :, ispin), drho_h(:, :, :, ispin), drho_s(:, :, :, ispin))
      END DO
      DEALLOCATE (slm, directions, radial)
      CALL timestop(handle)
#else
      MARK_USED(grid_atom)
      MARK_USED(harmonics)
      MARK_USED(nspins)
      MARK_USED(kind_index)
      MARK_USED(r_h)
      MARK_USED(r_s)
      MARK_USED(dr_h)
      MARK_USED(dr_s)
      MARK_USED(r_h_d)
      MARK_USED(r_s_d)
      MARK_USED(rho_h)
      MARK_USED(rho_s)
      MARK_USED(drho_h)
      MARK_USED(drho_s)
      CPABORT("NATIVE_GRID_GPU_FIELDS requires a CP2K build with CUDA.")
#endif
   END SUBROUTINE fields_gpu_angular

! **************************************************************************************************
!> \brief Store the cross-atom interpolation table of one source atom on the device, replacing an
!>        earlier one. Indices are 1-based here and passed 0-based.
!> \param source_atom ...
!> \param nharm harmonics of the source grid (max_s_harm)
!> \param nr radial shells of the source grid
!> \param group_first first candidate of every row group, and one past the last
!> \param cand_group group of every candidate
!> \param nnode radial nodes of every candidate (0 outside the grid)
!> \param node radial shells of the nodes (4, candidate)
!> \param radius distance of every candidate from its source image (bohr)
!> \param weight radial weights of the nodes (4, candidate)
!> \param harm harmonics in the direction of every candidate (nharm, candidate)
!> \param shell_first first entry of every radial shell, and one past the last
!> \param entry_cand candidate of every shell entry
!> \param entry_weight radial weight of every shell entry
! **************************************************************************************************
   SUBROUTINE fields_gpu_set_cross(source_atom, nharm, nr, group_first, cand_group, nnode, node, &
                                   radius, weight, harm, shell_first, entry_cand, entry_weight)
      INTEGER, INTENT(IN)                                :: source_atom, nharm, nr
      INTEGER, DIMENSION(:), INTENT(IN)                  :: group_first, cand_group, nnode
      INTEGER, DIMENSION(:, :), INTENT(IN)               :: node
      REAL(dp), DIMENSION(:), INTENT(IN)                 :: radius
      REAL(dp), DIMENSION(:, :), INTENT(IN)              :: weight, harm
      INTEGER, DIMENSION(:), INTENT(IN)                  :: shell_first, entry_cand
      REAL(dp), DIMENSION(:), INTENT(IN)                 :: entry_weight

#if defined(__OFFLOAD_CUDA)
      INTEGER                                            :: handle

      CALL timeset("skala_fields_gpu_table", handle)
      CPASSERT(SIZE(shell_first) == nr + 1)
      CPASSERT(SIZE(harm, 1) == nharm)
      CALL set_cross_c(INT(source_atom, C_INT), INT(nharm, C_INT), INT(nr, C_INT), &
                       INT(SIZE(cand_group), C_INT), INT(SIZE(group_first) - 1, C_INT), &
                       INT(group_first - 1, C_INT), INT(cand_group - 1, C_INT), &
                       INT(nnode, C_INT), INT(node - 1, C_INT), radius, weight, harm, &
                       INT(SIZE(entry_cand), C_INT), INT(shell_first - 1, C_INT), &
                       INT(entry_cand - 1, C_INT), entry_weight)
      CALL timestop(handle)
#else
      MARK_USED(source_atom)
      MARK_USED(nharm)
      MARK_USED(nr)
      MARK_USED(group_first)
      MARK_USED(cand_group)
      MARK_USED(nnode)
      MARK_USED(node)
      MARK_USED(radius)
      MARK_USED(weight)
      MARK_USED(harm)
      MARK_USED(shell_first)
      MARK_USED(entry_cand)
      MARK_USED(entry_weight)
      CPABORT("NATIVE_GRID_GPU_FIELDS requires a CP2K build with CUDA.")
#endif
   END SUBROUTINE fields_gpu_set_cross

! **************************************************************************************************
!> \brief Forward cross-atom interpolation of one source atom on the device.
!> \param source_atom ...
!> \param nspins ...
!> \param cutoff current cross cutoff of the source atom (bohr)
!> \param projection projections of the source fields (max_s_harm, nr, 5, nspins)
!> \param group_fields interpolated fields summed per row group (5, nspins, group)
! **************************************************************************************************
   SUBROUTINE fields_gpu_cross_forward(source_atom, nspins, cutoff, projection, group_fields)
      INTEGER, INTENT(IN)                                :: source_atom, nspins
      REAL(dp), INTENT(IN)                               :: cutoff
      REAL(dp), DIMENSION(:, :, :, :), INTENT(IN)        :: projection
      REAL(dp), DIMENSION(:, :, :), INTENT(OUT)          :: group_fields

#if defined(__OFFLOAD_CUDA)
      INTEGER                                            :: handle

      CALL timeset("skala_fields_gpu_forward", handle)
      CALL cross_forward_c(INT(source_atom, C_INT), INT(nspins, C_INT), REAL(cutoff, C_DOUBLE), &
                           projection, group_fields)
      CALL timestop(handle)
#else
      MARK_USED(source_atom)
      MARK_USED(nspins)
      MARK_USED(cutoff)
      MARK_USED(projection)
      group_fields = 0.0_dp
      CPABORT("NATIVE_GRID_GPU_FIELDS requires a CP2K build with CUDA.")
#endif
   END SUBROUTINE fields_gpu_cross_forward

! **************************************************************************************************
!> \brief Adjoint cross-atom interpolation of one source atom on the device.
!> \param source_atom ...
!> \param nspins ...
!> \param cutoff current cross cutoff of the source atom (bohr)
!> \param group_adjoint model derivatives of every row group (5, nspins, group)
!> \param adjoint adjoint projections (max_s_harm, nr, 5, nspins)
! **************************************************************************************************
   SUBROUTINE fields_gpu_cross_adjoint(source_atom, nspins, cutoff, group_adjoint, adjoint)
      INTEGER, INTENT(IN)                                :: source_atom, nspins
      REAL(dp), INTENT(IN)                               :: cutoff
      REAL(dp), DIMENSION(:, :, :), INTENT(IN)           :: group_adjoint
      REAL(dp), DIMENSION(:, :, :, :), INTENT(OUT)       :: adjoint

#if defined(__OFFLOAD_CUDA)
      INTEGER                                            :: handle

      CALL timeset("skala_fields_gpu_adjoint", handle)
      CALL cross_adjoint_c(INT(source_atom, C_INT), INT(nspins, C_INT), REAL(cutoff, C_DOUBLE), &
                           group_adjoint, adjoint)
      CALL timestop(handle)
#else
      MARK_USED(source_atom)
      MARK_USED(nspins)
      MARK_USED(cutoff)
      MARK_USED(group_adjoint)
      adjoint = 0.0_dp
      CPABORT("NATIVE_GRID_GPU_FIELDS requires a CP2K build with CUDA.")
#endif
   END SUBROUTINE fields_gpu_cross_adjoint

! **************************************************************************************************
!> \brief Free the device memory of NATIVE_GRID_GPU_FIELDS; does nothing if it was not used or
!>        CP2K was built without CUDA.
! **************************************************************************************************
   SUBROUTINE fields_gpu_release()

#if defined(__OFFLOAD_CUDA)
      CALL release_c()
#endif
   END SUBROUTINE fields_gpu_release

END MODULE qs_native_fields_gpu
