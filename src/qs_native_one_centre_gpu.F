!--------------------------------------------------------------------------------------------------!
!   CP2K: A general program to perform molecular dynamics simulations                              !
!   Copyright 2000-2026 CP2K developers group <https://cp2k.org>                                   !
!                                                                                                  !
!   SPDX-License-Identifier: GPL-2.0-or-later                                                      !
!--------------------------------------------------------------------------------------------------!
!> \brief One-centre matrices of the native SKALA atom-composite route on the GPU
!>        (NATIVE_GRID_GPU_ONE_CENTRE; kernels and cuBLAS calls in xc/skala_one_centre_gpu.cu):
!>        GPU counterparts of calc_tau_atom, dgaVtaudgb and gaVxcgb_GC of qs_vxc_atom_utils for
!>        the call sites of the atom-composite route. They compute the same quantities with the
!>        sums in another order: whole-atom matrix products instead of shell-by-shell ones, and in
!>        gaVxcgb_GC the angular projection of each radial integral once instead of inside the
!>        Clebsch-Gordan loop. Without CUDA every routine aborts.
MODULE qs_native_one_centre_gpu
   USE ISO_C_BINDING,                   ONLY: C_DOUBLE,&
                                              C_INT
   USE basis_set_types,                 ONLY: get_gto_basis_set,&
                                              gto_basis_set_type
   USE kinds,                           ONLY: dp
   USE orbital_pointers,                ONLY: indso,&
                                              nsoset
   USE qs_grid_atom,                    ONLY: grid_atom_type
   USE qs_harmonics_atom,               ONLY: get_none0_cg_list,&
                                              harmonics_atom_type
   USE qs_rho_atom_types,               ONLY: rho_atom_coeff,&
                                              rho_atom_type
   USE qs_vxc_atom_utils,               ONLY: tau_basis_cache_type
#include "./base/base_uses.f90"

   IMPLICIT NONE

   PRIVATE

   CHARACTER(len=*), PARAMETER, PRIVATE :: moduleN = 'qs_native_one_centre_gpu'

   PUBLIC :: calc_tau_atom_gpu, dgaVtaudgb_gpu, gaVxcgb_GC_gpu, one_centre_gpu_has_gradients, &
             one_centre_gpu_release

#if defined(__OFFLOAD_CUDA)
   INTERFACE
      FUNCTION set_gradients_c(kind, ngrid, nbas, grad) RESULT(uploaded) &
         BIND(C, name="skala_one_centre_gpu_set_gradients")
         IMPORT :: C_DOUBLE, C_INT
         INTEGER(KIND=C_INT), VALUE                        :: kind, ngrid, nbas
         REAL(KIND=C_DOUBLE), DIMENSION(*), INTENT(IN)     :: grad
         INTEGER(KIND=C_INT)                               :: uploaded
      END FUNCTION set_gradients_c
      FUNCTION has_gradients_c(kind, ngrid, nbas) RESULT(present) &
         BIND(C, name="skala_one_centre_gpu_has_gradients")
         IMPORT :: C_INT
         INTEGER(KIND=C_INT), VALUE                        :: kind, ngrid, nbas
         INTEGER(KIND=C_INT)                               :: present
      END FUNCTION has_gradients_c
      SUBROUTINE tau_c(kind, matrix, tau) BIND(C, name="skala_one_centre_gpu_tau")
         IMPORT :: C_DOUBLE, C_INT
         INTEGER(KIND=C_INT), VALUE                        :: kind
         REAL(KIND=C_DOUBLE), DIMENSION(*), INTENT(IN)     :: matrix
         REAL(KIND=C_DOUBLE), DIMENSION(*), INTENT(OUT)    :: tau
      END SUBROUTINE tau_c
      SUBROUTINE vtau_c(kind, v_tau, matrix) BIND(C, name="skala_one_centre_gpu_vtau")
         IMPORT :: C_DOUBLE, C_INT
         INTEGER(KIND=C_INT), VALUE                        :: kind
         REAL(KIND=C_DOUBLE), DIMENSION(*), INTENT(IN)     :: v_tau
         REAL(KIND=C_DOUBLE), DIMENSION(*), INTENT(OUT)    :: matrix
      END SUBROUTINE vtau_c
      SUBROUTINE projections_c(na, nr, ncol, niso, vxc, vxc_grad, vxg, radial, radial_d, &
                               radial_u, slm, projections) &
         BIND(C, name="skala_one_centre_gpu_projections")
         IMPORT :: C_DOUBLE, C_INT
         INTEGER(KIND=C_INT), VALUE                        :: na, nr, ncol, niso
         REAL(KIND=C_DOUBLE), DIMENSION(*), INTENT(IN)     :: vxc, vxc_grad, vxg, radial, &
                                                              radial_d, radial_u, slm
         REAL(KIND=C_DOUBLE), DIMENSION(*), INTENT(OUT)    :: projections
      END SUBROUTINE projections_c
      SUBROUTINE release_c() BIND(C, name="skala_one_centre_gpu_release")
      END SUBROUTINE release_c
   END INTERFACE
#endif

CONTAINS

! **************************************************************************************************
!> \brief Make the basis gradients of a kind available on the device (uploaded at the first call
!>        for the kind and kept until one_centre_gpu_release).
!> \param tau_cache basis gradients of the kind on its atomic grid
!> \param kind_index index of the kind, which identifies the stored gradients
! **************************************************************************************************
   SUBROUTINE set_gradients(tau_cache, kind_index)
      TYPE(tau_basis_cache_type), INTENT(IN)             :: tau_cache
      INTEGER, INTENT(IN)                                :: kind_index

#if defined(__OFFLOAD_CUDA)
      INTEGER                                            :: handle
      INTEGER(KIND=C_INT)                                :: uploaded

      CALL timeset("skala_1c_gpu_gradients", handle)
      IF (ALLOCATED(tau_cache%grad)) THEN
         CPASSERT(SIZE(tau_cache%grad, 1) == tau_cache%na*tau_cache%nr)
         CPASSERT(SIZE(tau_cache%grad, 2) == tau_cache%nsatbas)
         uploaded = set_gradients_c(INT(kind_index, C_INT), INT(tau_cache%na*tau_cache%nr, C_INT), &
                                    INT(tau_cache%nsatbas, C_INT), tau_cache%grad)
         MARK_USED(uploaded)
      ELSE
         ! A cache made without gradients: they must already be on the device.
         CPASSERT(one_centre_gpu_has_gradients(tau_cache, kind_index))
      END IF
      CALL timestop(handle)
#else
      MARK_USED(tau_cache)
      MARK_USED(kind_index)
      CPABORT("NATIVE_GRID_GPU_ONE_CENTRE requires a CP2K build with CUDA.")
#endif
   END SUBROUTINE set_gradients

! **************************************************************************************************
!> \brief Whether the basis gradients of a kind, with the sizes of tau_cache, are on the device.
!> \param tau_cache sizes of the kind's grid and basis (gradients need not be present)
!> \param kind_index ...
!> \return ...
! **************************************************************************************************
   FUNCTION one_centre_gpu_has_gradients(tau_cache, kind_index) RESULT(present)
      TYPE(tau_basis_cache_type), INTENT(IN)             :: tau_cache
      INTEGER, INTENT(IN)                                :: kind_index
      LOGICAL                                            :: present

#if defined(__OFFLOAD_CUDA)
      present = has_gradients_c(INT(kind_index, C_INT), INT(tau_cache%na*tau_cache%nr, C_INT), &
                                INT(tau_cache%nsatbas, C_INT)) == 1
#else
      MARK_USED(tau_cache)
      MARK_USED(kind_index)
      present = .FALSE.
#endif
   END FUNCTION one_centre_gpu_has_gradients

! **************************************************************************************************
!> \brief GPU counterpart of calc_tau_atom: tau = 1/2 sum_ij P_ij grad(phi_i) . grad(phi_j) of the
!>        hard and soft one-centre densities, on all grid points of the atom at once.
!> \param tau_h hard kinetic-energy density (angular point, radial point, spin)
!> \param tau_s soft kinetic-energy density
!> \param rho_atom one-centre density matrices of the atom
!> \param tau_cache basis gradients of the atom's kind
!> \param nspins ...
!> \param kind_index index of the atom's kind
! **************************************************************************************************
   SUBROUTINE calc_tau_atom_gpu(tau_h, tau_s, rho_atom, tau_cache, nspins, kind_index)
      REAL(dp), CONTIGUOUS, DIMENSION(:, :, :), &
         INTENT(INOUT)                                   :: tau_h, tau_s
      TYPE(rho_atom_type), POINTER                       :: rho_atom
      TYPE(tau_basis_cache_type), INTENT(IN)             :: tau_cache
      INTEGER, INTENT(IN)                                :: nspins, kind_index

#if defined(__OFFLOAD_CUDA)
      INTEGER                                            :: handle, ispin, nbas
      REAL(dp), ALLOCATABLE, DIMENSION(:, :)             :: matrix

      CALL set_gradients(tau_cache, kind_index)
      CALL timeset("skala_1c_gpu_tau", handle)
      nbas = tau_cache%nsatbas
      CPASSERT(SIZE(tau_h, 1)*SIZE(tau_h, 2) == tau_cache%na*tau_cache%nr)
      ALLOCATE (matrix(nbas, nbas))
      DO ispin = 1, nspins
         ! The density matrices are nbas x nbas, as calc_tau_atom assumes in its DGEMM.
         CPASSERT(SIZE(rho_atom%cpc_h(ispin)%r_coef, 1) == nbas)
         CPASSERT(SIZE(rho_atom%cpc_s(ispin)%r_coef, 1) == nbas)
         matrix(:, :) = rho_atom%cpc_h(ispin)%r_coef(1:nbas, 1:nbas)
         CALL tau_c(INT(kind_index, C_INT), matrix, tau_h(:, :, ispin))
         matrix(:, :) = rho_atom%cpc_s(ispin)%r_coef(1:nbas, 1:nbas)
         CALL tau_c(INT(kind_index, C_INT), matrix, tau_s(:, :, ispin))
      END DO
      DEALLOCATE (matrix)
      CALL timestop(handle)
#else
      MARK_USED(tau_h)
      MARK_USED(tau_s)
      MARK_USED(rho_atom)
      MARK_USED(tau_cache)
      MARK_USED(nspins)
      MARK_USED(kind_index)
      CPABORT("NATIVE_GRID_GPU_ONE_CENTRE requires a CP2K build with CUDA.")
#endif
   END SUBROUTINE calc_tau_atom_gpu

! **************************************************************************************************
!> \brief GPU counterpart of dgaVtaudgb: adds 1/2 sum_d (d_d phi)^T diag(v_tau) d_d phi of the hard
!>        and soft potentials to the one-centre Kohn-Sham matrices.
!> \param vtau_h hard v_tau (angular point, radial point, spin)
!> \param vtau_s soft v_tau
!> \param int_hh hard one-centre matrices
!> \param int_ss soft one-centre matrices
!> \param tau_cache basis gradients of the atom's kind
!> \param nspins ...
!> \param kind_index index of the atom's kind
! **************************************************************************************************
   SUBROUTINE dgaVtaudgb_gpu(vtau_h, vtau_s, int_hh, int_ss, tau_cache, nspins, kind_index)
      REAL(dp), DIMENSION(:, :, :), POINTER              :: vtau_h, vtau_s
      TYPE(rho_atom_coeff), DIMENSION(:), POINTER        :: int_hh, int_ss
      TYPE(tau_basis_cache_type), INTENT(IN)             :: tau_cache
      INTEGER, INTENT(IN)                                :: nspins, kind_index

#if defined(__OFFLOAD_CUDA)
      INTEGER                                            :: handle, ibas, iold, ispin, jbas, jold, &
                                                            max_old_basis, nbas
      REAL(dp), ALLOCATABLE, DIMENSION(:, :)             :: int_h, int_s, v_tau

      CPASSERT(ASSOCIATED(tau_cache%n2oindex))
      CALL set_gradients(tau_cache, kind_index)
      CALL timeset("skala_1c_gpu_vtau", handle)
      nbas = tau_cache%nsatbas
      max_old_basis = MAXVAL(tau_cache%n2oindex)
      ALLOCATE (int_h(nbas, nbas), int_s(nbas, nbas), v_tau(tau_cache%na, tau_cache%nr))
      DO ispin = 1, nspins
         CPASSERT(SIZE(int_hh(ispin)%r_coef, 1) >= max_old_basis)
         CPASSERT(SIZE(int_hh(ispin)%r_coef, 2) >= max_old_basis)
         CPASSERT(SIZE(int_ss(ispin)%r_coef, 1) >= max_old_basis)
         CPASSERT(SIZE(int_ss(ispin)%r_coef, 2) >= max_old_basis)
         v_tau(:, :) = vtau_h(1:tau_cache%na, 1:tau_cache%nr, ispin)
         CALL vtau_c(INT(kind_index, C_INT), v_tau, int_h)
         v_tau(:, :) = vtau_s(1:tau_cache%na, 1:tau_cache%nr, ispin)
         CALL vtau_c(INT(kind_index, C_INT), v_tau, int_s)
         DO jbas = 1, nbas
            DO ibas = 1, nbas
               jold = tau_cache%n2oindex(jbas)
               iold = tau_cache%n2oindex(ibas)
               int_hh(ispin)%r_coef(iold, jold) = int_hh(ispin)%r_coef(iold, jold) + &
                                                  int_h(ibas, jbas)
               int_ss(ispin)%r_coef(iold, jold) = int_ss(ispin)%r_coef(iold, jold) + &
                                                  int_s(ibas, jbas)
            END DO
         END DO
      END DO
      DEALLOCATE (int_h, int_s, v_tau)
      CALL timestop(handle)
#else
      MARK_USED(vtau_h)
      MARK_USED(vtau_s)
      MARK_USED(int_hh)
      MARK_USED(int_ss)
      MARK_USED(tau_cache)
      MARK_USED(nspins)
      MARK_USED(kind_index)
      CPABORT("NATIVE_GRID_GPU_ONE_CENTRE requires a CP2K build with CUDA.")
#endif
   END SUBROUTINE dgaVtaudgb_gpu

! **************************************************************************************************
!> \brief GPU counterpart of gaVxcgb_GC: the hard and soft one-centre XC matrices of a GGA-type
!>        potential (vxc and its gradient part vxg). The radial integrals of all primitive pairs
!>        and their projections on the real spherical harmonics are computed on the GPU; the host
!>        assembles the matrices from the projections with the Clebsch-Gordan coefficients. Pairs,
!>        columns and the assembly follow gaVxcgb_GC.
!> \param vxc_h ...
!> \param vxc_s ...
!> \param vxg_h ...
!> \param vxg_s ...
!> \param int_hh ...
!> \param int_ss ...
!> \param grid_atom ...
!> \param basis_1c ...
!> \param harmonics ...
!> \param nspins ...
!> \param col_count number of radial columns (pairs times their l values), for diagnostics
! **************************************************************************************************
   SUBROUTINE gaVxcgb_GC_gpu(vxc_h, vxc_s, vxg_h, vxg_s, int_hh, int_ss, &
                             grid_atom, basis_1c, harmonics, nspins, col_count)

      REAL(dp), DIMENSION(:, :, :), POINTER              :: vxc_h, vxc_s
      REAL(dp), DIMENSION(:, :, :, :), POINTER           :: vxg_h, vxg_s
      TYPE(rho_atom_coeff), DIMENSION(:), POINTER        :: int_hh, int_ss
      TYPE(grid_atom_type), POINTER                      :: grid_atom
      TYPE(gto_basis_set_type), POINTER                  :: basis_1c
      TYPE(harmonics_atom_type), POINTER                 :: harmonics
      INTEGER, INTENT(IN)                                :: nspins
      INTEGER, INTENT(OUT)                               :: col_count

#if defined(__OFFLOAD_CUDA)
      INTEGER :: col, dmax_iso_not0_local, handle, ia, ic, icg, idir, ipair, ipgf1, ipgf2, ir, &
         iset1, iset2, iso, iso1, iso2, ispin, l, lmax12, lmax_expansion, lmin12, m1, m2, &
         max_iso_not0, max_iso_not0_local, max_s_harm, maxl, maxso, n1, n2, na, ngau1, ngau2, &
         nngau1, npair, nr, nset, size1, sub_handle
      INTEGER, ALLOCATABLE, DIMENSION(:)                 :: cg_n_list, dcg_n_list, pair_col
      INTEGER, ALLOCATABLE, DIMENSION(:, :, :)           :: cg_list, dcg_list
      INTEGER, DIMENSION(:), POINTER                     :: lmax, lmin, npgf
      REAL(dp), ALLOCATABLE, DIMENSION(:)                :: g1, g2
      REAL(dp), ALLOCATABLE, DIMENSION(:, :)             :: dgg, gg, matso_h, matso_s, radial, &
                                                            radial_d, radial_u, slm, vxc_grad, &
                                                            vxc_plain
      REAL(dp), ALLOCATABLE, DIMENSION(:, :, :)          :: proj_h, proj_s, vxg_comp
      REAL(dp), DIMENSION(:, :), POINTER                 :: zet
      REAL(dp), DIMENSION(:, :, :), POINTER              :: my_CG
      REAL(dp), DIMENSION(:, :, :, :), POINTER           :: my_CG_dxyz

      CALL timeset("gaVxcgb_GC_gpu", handle)

      NULLIFY (lmin, lmax, npgf, zet, my_CG, my_CG_dxyz)

      CALL get_gto_basis_set(gto_basis_set=basis_1c, lmax=lmax, lmin=lmin, &
                             maxso=maxso, maxl=maxl, npgf=npgf, &
                             nset=nset, zet=zet)

      nr = grid_atom%nr
      na = grid_atom%ng_sphere
      my_CG => harmonics%my_CG
      my_CG_dxyz => harmonics%my_CG_dxyz
      max_iso_not0 = harmonics%max_iso_not0
      lmax_expansion = indso(1, max_iso_not0)
      max_s_harm = harmonics%max_s_harm

      ALLOCATE (g1(nr), g2(nr), gg(nr, 0:2*maxl), dgg(nr, 0:2*maxl))
      ALLOCATE (cg_list(2, nsoset(maxl)**2, max_s_harm), cg_n_list(max_s_harm), &
                dcg_list(2, nsoset(maxl)**2, max_s_harm), dcg_n_list(max_s_harm))
      ALLOCATE (matso_h(nsoset(maxl), nsoset(maxl)), &
                matso_s(nsoset(maxl), nsoset(maxl)))

      ! Radial functions of all primitive pairs as columns, as in gaVxcgb_GC.
      npair = 0
      DO iset1 = 1, nset
         DO iset2 = 1, nset
            npair = npair + npgf(iset1)*npgf(iset2)
         END DO
      END DO
      ALLOCATE (pair_col(npair))
      col_count = 0
      ipair = 0
      DO iset1 = 1, nset
         DO iset2 = 1, nset
            DO ipgf1 = 1, npgf(iset1)
               DO ipgf2 = 1, npgf(iset2)
                  ipair = ipair + 1
                  pair_col(ipair) = 0
                  lmin12 = lmin(iset1) + lmin(iset2)
                  IF (lmin12 > lmax_expansion) CYCLE
                  lmax12 = MIN(lmax(iset1) + lmax(iset2), lmax_expansion)
                  pair_col(ipair) = col_count + 1
                  col_count = col_count + lmax12 - lmin12 + 1
               END DO
            END DO
         END DO
      END DO
      ALLOCATE (radial(nr, col_count), radial_d(nr, col_count), radial_u(nr, col_count))
      ipair = 0
      DO iset1 = 1, nset
         DO iset2 = 1, nset
            DO ipgf1 = 1, npgf(iset1)
               g1(1:nr) = EXP(-zet(ipgf1, iset1)*grid_atom%rad2(1:nr))
               DO ipgf2 = 1, npgf(iset2)
                  ipair = ipair + 1
                  IF (pair_col(ipair) == 0) CYCLE
                  g2(1:nr) = EXP(-zet(ipgf2, iset2)*grid_atom%rad2(1:nr))
                  lmin12 = lmin(iset1) + lmin(iset2)
                  lmax12 = MIN(lmax(iset1) + lmax(iset2), lmax_expansion)
                  gg = 0.0_dp
                  dgg = 0.0_dp
                  IF (lmin12 == 0) THEN
                     gg(1:nr, lmin12) = g1(1:nr)*g2(1:nr)
                  ELSE
                     gg(1:nr, lmin12) = grid_atom%rad2l(1:nr, lmin12)*g1(1:nr)*g2(1:nr)
                  END IF
                  DO l = lmin12 + 1, lmax12
                     gg(1:nr, l) = grid_atom%rad(1:nr)*gg(:, l - 1)
                     dgg(1:nr, l - 1) = dgg(1:nr, l - 1) - 2.0_dp*(zet(ipgf1, iset1) + &
                                                                   zet(ipgf2, iset2))*gg(1:nr, l)
                  END DO
                  dgg(1:nr, lmax12) = dgg(1:nr, lmax12) - 2.0_dp*(zet(ipgf1, iset1) + &
                                                                  zet(ipgf2, iset2))*grid_atom%rad(1:nr)* &
                                      gg(1:nr, lmax12)
                  DO l = lmin12, lmax12
                     col = pair_col(ipair) + l - lmin12
                     radial(:, col) = gg(:, l)
                     radial_d(:, col) = dgg(:, l)
                     radial_u(:, col) = gg(:, l)*grid_atom%oorad2l(:, 1)
                  END DO
               END DO
            END DO
         END DO
      END DO

      ! The harmonics up to max_s_harm: the gradient loop below runs to dmax_iso_not0_local,
      ! which max_iso_not0 does not bound.
      ALLOCATE (slm(na, max_s_harm))
      slm(:, :) = harmonics%slm(1:na, 1:max_s_harm)
      ALLOCATE (vxc_plain(na, nr), vxc_grad(na, nr), vxg_comp(na, nr, 3), &
                proj_h(max_s_harm, col_count, 4), proj_s(max_s_harm, col_count, 4))

      DO ispin = 1, nspins
         ! Projections on the GPU: proj(iso, col, 1) = sum_ia slm(ia, iso) xc(ia, col) with
         ! xc = vxc radial + (vxg . a) radial_d, and proj(iso, col, 1 + d) the same for
         ! xg_d = vxg_d radial_u.
         CALL timeset("skala_1c_gpu_projections", sub_handle)
         DO ir = 1, nr
            DO ia = 1, na
               vxc_plain(ia, ir) = vxc_h(ia, ir, ispin)
               vxc_grad(ia, ir) = vxg_h(1, ia, ir, ispin)*harmonics%a(1, ia) + &
                                  vxg_h(2, ia, ir, ispin)*harmonics%a(2, ia) + &
                                  vxg_h(3, ia, ir, ispin)*harmonics%a(3, ia)
               DO idir = 1, 3
                  vxg_comp(ia, ir, idir) = vxg_h(idir, ia, ir, ispin)
               END DO
            END DO
         END DO
         CALL projections_c(INT(na, C_INT), INT(nr, C_INT), INT(col_count, C_INT), &
                            INT(max_s_harm, C_INT), vxc_plain, vxc_grad, vxg_comp, radial, &
                            radial_d, radial_u, slm, proj_h)
         DO ir = 1, nr
            DO ia = 1, na
               vxc_plain(ia, ir) = vxc_s(ia, ir, ispin)
               vxc_grad(ia, ir) = vxg_s(1, ia, ir, ispin)*harmonics%a(1, ia) + &
                                  vxg_s(2, ia, ir, ispin)*harmonics%a(2, ia) + &
                                  vxg_s(3, ia, ir, ispin)*harmonics%a(3, ia)
               DO idir = 1, 3
                  vxg_comp(ia, ir, idir) = vxg_s(idir, ia, ir, ispin)
               END DO
            END DO
         END DO
         CALL projections_c(INT(na, C_INT), INT(nr, C_INT), INT(col_count, C_INT), &
                            INT(max_s_harm, C_INT), vxc_plain, vxc_grad, vxg_comp, radial, &
                            radial_d, radial_u, slm, proj_s)
         CALL timestop(sub_handle)

         CALL timeset("skala_1c_gpu_assembly", sub_handle)
         ipair = 0
         m1 = 0
         DO iset1 = 1, nset
            n1 = nsoset(lmax(iset1))
            m2 = 0
            DO iset2 = 1, nset
               CALL get_none0_cg_list(my_CG, lmin(iset1), lmax(iset1), lmin(iset2), lmax(iset2), &
                                      max_s_harm, lmax_expansion, cg_list, cg_n_list, max_iso_not0_local)
               CPASSERT(max_iso_not0_local <= max_iso_not0)
               CALL get_none0_cg_list(my_CG_dxyz, lmin(iset1), lmax(iset1), lmin(iset2), lmax(iset2), &
                                      max_s_harm, lmax_expansion, dcg_list, dcg_n_list, dmax_iso_not0_local)
               CPASSERT(dmax_iso_not0_local <= max_s_harm)

               n2 = nsoset(lmax(iset2))
               DO ipgf1 = 1, npgf(iset1)
                  ngau1 = n1*(ipgf1 - 1) + m1
                  size1 = nsoset(lmax(iset1)) - nsoset(lmin(iset1) - 1)
                  nngau1 = nsoset(lmin(iset1) - 1) + ngau1

                  DO ipgf2 = 1, npgf(iset2)
                     ngau2 = n2*(ipgf2 - 1) + m2
                     ipair = ipair + 1
                     lmin12 = lmin(iset1) + lmin(iset2)

                     IF (lmin12 <= lmax_expansion) THEN
                        ! column of l in the projections
                        col = pair_col(ipair) - lmin12

                        matso_h = 0.0_dp
                        matso_s = 0.0_dp
                        DO iso = 1, max_iso_not0_local
                           DO icg = 1, cg_n_list(iso)
                              iso1 = cg_list(1, icg, iso)
                              iso2 = cg_list(2, icg, iso)
                              l = indso(1, iso1) + indso(1, iso2)
                              CPASSERT(l <= lmax_expansion)
                              matso_h(iso1, iso2) = matso_h(iso1, iso2) + &
                                                    proj_h(iso, col + l, 1)*my_CG(iso1, iso2, iso)
                              matso_s(iso1, iso2) = matso_s(iso1, iso2) + &
                                                    proj_s(iso, col + l, 1)*my_CG(iso1, iso2, iso)
                           END DO
                        END DO

                        DO iso = 1, dmax_iso_not0_local
                           DO icg = 1, dcg_n_list(iso)
                              iso1 = dcg_list(1, icg, iso)
                              iso2 = dcg_list(2, icg, iso)
                              l = indso(1, iso1) + indso(1, iso2)
                              CPASSERT(l <= lmax_expansion)
                              matso_h(iso1, iso2) = matso_h(iso1, iso2) + &
                                                    proj_h(iso, col + l, 2)*my_CG_dxyz(1, iso1, iso2, iso) + &
                                                    proj_h(iso, col + l, 3)*my_CG_dxyz(2, iso1, iso2, iso) + &
                                                    proj_h(iso, col + l, 4)*my_CG_dxyz(3, iso1, iso2, iso)
                              matso_s(iso1, iso2) = matso_s(iso1, iso2) + &
                                                    proj_s(iso, col + l, 2)*my_CG_dxyz(1, iso1, iso2, iso) + &
                                                    proj_s(iso, col + l, 3)*my_CG_dxyz(2, iso1, iso2, iso) + &
                                                    proj_s(iso, col + l, 4)*my_CG_dxyz(3, iso1, iso2, iso)
                           END DO
                        END DO
                     END IF

                     !  Write in the global matrix
                     DO ic = nsoset(lmin(iset2) - 1) + 1, nsoset(lmax(iset2))
                        iso1 = nsoset(lmin(iset1) - 1) + 1
                        iso2 = ngau2 + ic
                        CALL daxpy(size1, 1.0_dp, matso_h(iso1, ic), 1, &
                                   int_hh(ispin)%r_coef(nngau1 + 1, iso2), 1)
                        CALL daxpy(size1, 1.0_dp, matso_s(iso1, ic), 1, &
                                   int_ss(ispin)%r_coef(nngau1 + 1, iso2), 1)
                     END DO

                  END DO ! ipfg2
               END DO ! ipfg1
               m2 = m2 + maxso
            END DO ! iset2
            m1 = m1 + maxso
         END DO ! iset1
         CALL timestop(sub_handle)
      END DO ! ispin

      DEALLOCATE (g1, g2, gg, dgg, matso_h, matso_s, radial, radial_d, radial_u, pair_col)
      DEALLOCATE (slm, vxc_plain, vxc_grad, vxg_comp, proj_h, proj_s)
      DEALLOCATE (cg_list, cg_n_list, dcg_list, dcg_n_list)

      CALL timestop(handle)
#else
      MARK_USED(vxc_h)
      MARK_USED(vxc_s)
      MARK_USED(vxg_h)
      MARK_USED(vxg_s)
      MARK_USED(int_hh)
      MARK_USED(int_ss)
      MARK_USED(grid_atom)
      MARK_USED(basis_1c)
      MARK_USED(harmonics)
      MARK_USED(nspins)
      col_count = 0
      CPABORT("NATIVE_GRID_GPU_ONE_CENTRE requires a CP2K build with CUDA.")
#endif
   END SUBROUTINE gaVxcgb_GC_gpu

! **************************************************************************************************
!> \brief Free the device memory of the GPU one-centre matrices.
! **************************************************************************************************
   SUBROUTINE one_centre_gpu_release()
#if defined(__OFFLOAD_CUDA)
      CALL release_c()
#endif
   END SUBROUTINE one_centre_gpu_release

END MODULE qs_native_one_centre_gpu
