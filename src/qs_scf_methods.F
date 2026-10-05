!--------------------------------------------------------------------------------------------------!
!   CP2K: A general program to perform molecular dynamics simulations                              !
!   Copyright 2000-2026 CP2K developers group <https://cp2k.org>                                   !
!                                                                                                  !
!   SPDX-License-Identifier: GPL-2.0-or-later                                                      !
!--------------------------------------------------------------------------------------------------!

! **************************************************************************************************
!> \brief groups fairly general SCF methods, so that modules other than qs_scf can use them too
!>        split off from qs_scf to reduce dependencies
!> \par History
!>      - Joost VandeVondele (03.2006)
!>      - combine_ks_matrices added (05.04.06,MK)
!>      - second ROKS scheme added (15.04.06,MK)
!>      - MO occupation management moved (29.08.2008,MK)
!>      - correct_mo_eigenvalues was moved from qs_mo_types;
!>        new subroutine shift_unocc_mos (03.2016, Sergey Chulkov)
! **************************************************************************************************
MODULE qs_scf_methods
   USE cp_dbcsr_api,                    ONLY: &
        dbcsr_add, dbcsr_copy, dbcsr_create, dbcsr_desymmetrize, dbcsr_get_block_p, &
        dbcsr_iterator_blocks_left, dbcsr_iterator_next_block, dbcsr_iterator_start, &
        dbcsr_iterator_stop, dbcsr_iterator_type, dbcsr_multiply, dbcsr_p_type, dbcsr_scale, &
        dbcsr_type
   USE cp_dbcsr_contrib,                ONLY: dbcsr_dot,&
                                              dbcsr_maxabs
   USE cp_dbcsr_operations,             ONLY: copy_dbcsr_to_fm,&
                                              cp_dbcsr_sm_fm_multiply
   USE cp_fm_basic_linalg,              ONLY: cp_fm_column_scale,&
                                              cp_fm_symm,&
                                              cp_fm_triangular_multiply,&
                                              cp_fm_uplo_to_full
   USE cp_fm_cholesky,                  ONLY: cp_fm_cholesky_reduce,&
                                              cp_fm_cholesky_restore
   USE cp_fm_diag,                      ONLY: choose_eigv_solver,&
                                              cp_fm_block_jacobi,&
                                              cp_fm_geeig
   USE cp_fm_struct,                    ONLY: cp_fm_struct_create,&
                                              cp_fm_struct_equivalent,&
                                              cp_fm_struct_release,&
                                              cp_fm_struct_type
   USE cp_fm_types,                     ONLY: cp_fm_create,&
                                              cp_fm_get_info,&
                                              cp_fm_release,&
                                              cp_fm_to_fm,&
                                              cp_fm_type
   USE input_constants,                 ONLY: cholesky_inverse,&
                                              cholesky_off,&
                                              cholesky_reduce,&
                                              cholesky_restore
   USE kinds,                           ONLY: dp
   USE mathlib,                         ONLY: invert_matrix
   USE message_passing,                 ONLY: mp_para_env_type
   USE parallel_gemm_api,               ONLY: parallel_gemm
   USE qs_density_mixing_types,         ONLY: dm_pulay_storage_release,&
                                              mixing_storage_type
   USE qs_mo_types,                     ONLY: get_mo_set,&
                                              mo_set_type
#include "./base/base_uses.f90"

   IMPLICIT NONE

   PRIVATE

   CHARACTER(len=*), PARAMETER, PRIVATE :: moduleN = 'qs_scf_methods'
   REAL(KIND=dp), PARAMETER    :: ratio = 0.25_dp

   PUBLIC :: combine_ks_matrices, &
             cp_sm_mix, &
             eigensolver, &
             eigensolver_generalized, &
             eigensolver_dbcsr, &
             eigensolver_symm, &
             eigensolver_simple, &
             scf_env_density_mixing

   INTERFACE combine_ks_matrices
      MODULE PROCEDURE combine_ks_matrices_1, &
         combine_ks_matrices_2
   END INTERFACE combine_ks_matrices

CONTAINS

! **************************************************************************************************
!> \brief perform (if requested) a density mixing
!> \param p_mix_new    New density matrices
!> \param mixing_store ...
!> \param rho_ao       Density environment
!> \param para_env ...
!> \param iter_delta ...
!> \param iter_count ...
!> \param diis ...
!> \param invert       Invert mixing
!> \par History
!>      02.2003 created [fawzi]
!>      08.2014 adapted for kpoints [JGH]
!> \author fawzi
! **************************************************************************************************
   SUBROUTINE scf_env_density_mixing(p_mix_new, mixing_store, rho_ao, para_env, &
                                     iter_delta, iter_count, diis, invert)
      TYPE(dbcsr_p_type), DIMENSION(:, :), POINTER       :: p_mix_new
      TYPE(mixing_storage_type), POINTER                 :: mixing_store
      TYPE(dbcsr_p_type), DIMENSION(:, :), POINTER       :: rho_ao
      TYPE(mp_para_env_type), POINTER                    :: para_env
      REAL(KIND=dp), INTENT(INOUT)                       :: iter_delta
      INTEGER, INTENT(IN)                                :: iter_count
      LOGICAL, INTENT(in), OPTIONAL                      :: diis, invert

      CHARACTER(len=*), PARAMETER :: routineN = 'scf_env_density_mixing'

      INTEGER                                            :: handle, ic, ispin
      LOGICAL                                            :: my_diis, my_invert
      REAL(KIND=dp)                                      :: my_p_mix, tmp

      CALL timeset(routineN, handle)

      my_diis = .FALSE.
      IF (PRESENT(diis)) my_diis = diis
      my_invert = .FALSE.
      IF (PRESENT(invert)) my_invert = invert
      my_p_mix = mixing_store%alpha
      IF (my_diis .OR. iter_count < mixing_store%nskip_mixing) THEN
         my_p_mix = 1.0_dp
      END IF

      IF (mixing_store%dm_pulay) THEN
         IF (SIZE(p_mix_new, 1) == 2) THEN
            CALL dm_pulay_mixing_spin(p_mix_new, mixing_store, rho_ao, iter_delta, &
                                      full_step=my_diis .OR. iter_count < mixing_store%nskip_mixing, &
                                      invert=my_invert)
         ELSE
            CALL dm_pulay_mixing(p_mix_new, mixing_store, rho_ao, iter_delta, &
                                 full_step=my_diis .OR. iter_count < mixing_store%nskip_mixing, &
                                 invert=my_invert)
         END IF
         CALL timestop(handle)
         RETURN
      END IF

      iter_delta = 0.0_dp
      CPASSERT(ASSOCIATED(p_mix_new))
      DO ic = 1, SIZE(p_mix_new, 2)
         DO ispin = 1, SIZE(p_mix_new, 1)
            IF (my_invert) THEN
               CPASSERT(my_p_mix /= 0.0_dp)
               IF (my_p_mix /= 1.0_dp) THEN
                  CALL dbcsr_add(matrix_a=p_mix_new(ispin, ic)%matrix, &
                                 alpha_scalar=1.0_dp/my_p_mix, &
                                 matrix_b=rho_ao(ispin, ic)%matrix, &
                                 beta_scalar=(my_p_mix - 1.0_dp)/my_p_mix)
               END IF
            ELSE
               CALL cp_sm_mix(m1=p_mix_new(ispin, ic)%matrix, &
                              m2=rho_ao(ispin, ic)%matrix, &
                              p_mix=my_p_mix, &
                              delta=tmp, &
                              para_env=para_env)
               iter_delta = MAX(iter_delta, tmp)
            END IF
         END DO
      END DO

      CALL timestop(handle)

   END SUBROUTINE scf_env_density_mixing

! **************************************************************************************************
!> \brief Pulay (Anderson) mixing of the real-space density matrices (MIXING%PULAY_P)
!> \param p_mix_new    on entry the output density matrices P_out of the diagonalization, on exit
!>                     the input density matrices of the next step
!> \param mixing_store holds the history: P_in and the residual of the previous step, and NBUFFER
!>                     normalized differences of both
!> \param rho_ao       input density matrices P_in of the present step
!> \param iter_delta   largest element of |P_out - P_in| over all spins and images, the
!>                     convergence measure of DIRECT_P_MIXING
!> \param full_step    take P_out unmixed (NSKIP) and restart the history
!> \param invert       recover P_out of the present step from the mixed matrices (after convergence)
!> \par Method
!>      With residuals R = P_out - P_in and the differences dR_j = R_j+1 - R_j and
!>      dP_j = P_in,j+1 - P_in,j of consecutive steps, each pair divided by |dR_j|, the next input
!>      is P_in + alpha R - sum_j g_j (dP_j + alpha dR_j), g = (<dR_j,dR_k> + w0 delta_jk)^-1 <dR_j,R>.
!>      This is the update of broyden_mixing (qs_gspace_mixing) with the constant preconditioner
!>      alpha in place of the Kerker factor, and the Frobenius inner product of the stored matrix
!>      elements of all spins and images (off-diagonal blocks of symmetric matrices twice) in place
!>      of the G-space metric. Unlike the G-space schemes it mixes one object from which the grid
!>      densities, the kinetic energy density and the GAPW one-centre terms are all derived.
!>      The first step and a full step mix linearly with alpha.
! **************************************************************************************************
   SUBROUTINE dm_pulay_mixing(p_mix_new, mixing_store, rho_ao, iter_delta, full_step, invert)
      TYPE(dbcsr_p_type), DIMENSION(:, :), POINTER       :: p_mix_new
      TYPE(mixing_storage_type), POINTER                 :: mixing_store
      TYPE(dbcsr_p_type), DIMENSION(:, :), POINTER       :: rho_ao
      REAL(KIND=dp), INTENT(OUT)                         :: iter_delta
      LOGICAL, INTENT(IN)                                :: full_step, invert

      CHARACTER(len=*), PARAMETER                        :: routineN = 'dm_pulay_mixing'

      INTEGER                                            :: handle, ib, ic, ispin, jb, kb, nb, &
                                                            nbuffer, nimg, nspin
      REAL(KIND=dp)                                      :: alpha, inv_err, norm
      REAL(KIND=dp), ALLOCATABLE, DIMENSION(:)           :: c, g
      REAL(KIND=dp), ALLOCATABLE, DIMENSION(:, :)        :: a, b

      CALL timeset(routineN, handle)

      nspin = SIZE(p_mix_new, 1)
      nimg = SIZE(p_mix_new, 2)
      nbuffer = MAX(mixing_store%nbuffer, 1)
      alpha = mixing_store%alpha
      iter_delta = 0.0_dp

      IF (invert) THEN
         ! P_out = P_in + R of the present step; after a full step p_mix_new is P_out already
         IF (mixing_store%dm_pulay_ncall > 0) THEN
            DO ic = 1, nimg
               DO ispin = 1, nspin
                  CALL dbcsr_add(p_mix_new(ispin, ic)%matrix, rho_ao(ispin, ic)%matrix, 0.0_dp, 1.0_dp)
                  CALL dbcsr_add(p_mix_new(ispin, ic)%matrix, &
                                 mixing_store%dm_pulay_res_last(ispin, ic)%matrix, 1.0_dp, 1.0_dp)
               END DO
            END DO
         END IF
         CALL timestop(handle)
         RETURN
      END IF

      ! residual R = P_out - P_in, kept in p_mix_new until the update
      DO ic = 1, nimg
         DO ispin = 1, nspin
            CALL dbcsr_add(p_mix_new(ispin, ic)%matrix, rho_ao(ispin, ic)%matrix, 1.0_dp, -1.0_dp)
            iter_delta = MAX(iter_delta, dbcsr_maxabs(p_mix_new(ispin, ic)%matrix))
         END DO
      END DO

      IF (full_step) THEN
         DO ic = 1, nimg
            DO ispin = 1, nspin
               CALL dbcsr_add(p_mix_new(ispin, ic)%matrix, rho_ao(ispin, ic)%matrix, 1.0_dp, 1.0_dp)
            END DO
         END DO
         mixing_store%dm_pulay_ncall = 0
         CALL timestop(handle)
         RETURN
      END IF

      mixing_store%dm_pulay_ncall = mixing_store%dm_pulay_ncall + 1
      IF (mixing_store%dm_pulay_ncall == 1) THEN
         ! a new history with the sparsity of the present matrices
         CALL dm_pulay_storage_release(mixing_store)
         mixing_store%dm_pulay_ncall = 1
         ALLOCATE (mixing_store%dm_pulay_p_last(nspin, nimg), mixing_store%dm_pulay_res_last(nspin, nimg))
         ALLOCATE (mixing_store%dm_pulay_dp(nbuffer, nspin, nimg), &
                   mixing_store%dm_pulay_dres(nbuffer, nspin, nimg))
         DO ic = 1, nimg
            DO ispin = 1, nspin
               ALLOCATE (mixing_store%dm_pulay_p_last(ispin, ic)%matrix, &
                         mixing_store%dm_pulay_res_last(ispin, ic)%matrix)
               CALL dbcsr_create(mixing_store%dm_pulay_p_last(ispin, ic)%matrix, &
                                 template=rho_ao(ispin, ic)%matrix, name="PULAY_P P_IN")
               CALL dbcsr_create(mixing_store%dm_pulay_res_last(ispin, ic)%matrix, &
                                 template=rho_ao(ispin, ic)%matrix, name="PULAY_P RESIDUAL")
               DO ib = 1, nbuffer
                  ALLOCATE (mixing_store%dm_pulay_dp(ib, ispin, ic)%matrix, &
                            mixing_store%dm_pulay_dres(ib, ispin, ic)%matrix)
                  CALL dbcsr_create(mixing_store%dm_pulay_dp(ib, ispin, ic)%matrix, &
                                    template=rho_ao(ispin, ic)%matrix, name="PULAY_P DP")
                  CALL dbcsr_create(mixing_store%dm_pulay_dres(ib, ispin, ic)%matrix, &
                                    template=rho_ao(ispin, ic)%matrix, name="PULAY_P DRES")
               END DO
            END DO
         END DO
         nb = 0
      ELSE
         ! new pair of differences in slot ib, normalized by |dR|
         nb = MIN(mixing_store%dm_pulay_ncall - 1, nbuffer)
         ib = MODULO(mixing_store%dm_pulay_ncall - 2, nbuffer) + 1
         norm = 0.0_dp
         DO ic = 1, nimg
            DO ispin = 1, nspin
               CALL dbcsr_copy(mixing_store%dm_pulay_dres(ib, ispin, ic)%matrix, p_mix_new(ispin, ic)%matrix)
               CALL dbcsr_add(mixing_store%dm_pulay_dres(ib, ispin, ic)%matrix, &
                              mixing_store%dm_pulay_res_last(ispin, ic)%matrix, 1.0_dp, -1.0_dp)
               CALL dbcsr_copy(mixing_store%dm_pulay_dp(ib, ispin, ic)%matrix, rho_ao(ispin, ic)%matrix)
               CALL dbcsr_add(mixing_store%dm_pulay_dp(ib, ispin, ic)%matrix, &
                              mixing_store%dm_pulay_p_last(ispin, ic)%matrix, 1.0_dp, -1.0_dp)
               norm = norm + frobenius_dot(mixing_store%dm_pulay_dres(ib, ispin, ic)%matrix, &
                                           mixing_store%dm_pulay_dres(ib, ispin, ic)%matrix)
            END DO
         END DO
         norm = SQRT(norm)
         IF (norm > 0.0_dp) THEN
            DO ic = 1, nimg
               DO ispin = 1, nspin
                  CALL dbcsr_scale(mixing_store%dm_pulay_dres(ib, ispin, ic)%matrix, 1.0_dp/norm)
                  CALL dbcsr_scale(mixing_store%dm_pulay_dp(ib, ispin, ic)%matrix, 1.0_dp/norm)
               END DO
            END DO
         ELSE
            nb = 0 ! no new information: mix linearly
         END IF
      END IF

      ! coefficients g of the history from the normal equations
      IF (nb > 0) THEN
         ALLOCATE (a(nb, nb), b(nb, nb), c(nb), g(nb))
         a = 0.0_dp
         c = 0.0_dp
         DO jb = 1, nb
            DO kb = jb, nb
               DO ic = 1, nimg
                  DO ispin = 1, nspin
                     a(jb, kb) = a(jb, kb) + &
                                 frobenius_dot(mixing_store%dm_pulay_dres(jb, ispin, ic)%matrix, &
                                               mixing_store%dm_pulay_dres(kb, ispin, ic)%matrix)
                  END DO
               END DO
               a(kb, jb) = a(jb, kb)
            END DO
            a(jb, jb) = a(jb, jb) + mixing_store%broy_w0
            DO ic = 1, nimg
               DO ispin = 1, nspin
                  c(jb) = c(jb) + frobenius_dot(mixing_store%dm_pulay_dres(jb, ispin, ic)%matrix, &
                                                p_mix_new(ispin, ic)%matrix)
               END DO
            END DO
         END DO
         CALL invert_matrix(a, b, inv_err)
         g = MATMUL(b, c)
      END IF

      ! keep P_in and R of this step, then form the next input in place of R
      DO ic = 1, nimg
         DO ispin = 1, nspin
            CALL dbcsr_copy(mixing_store%dm_pulay_p_last(ispin, ic)%matrix, rho_ao(ispin, ic)%matrix)
            CALL dbcsr_copy(mixing_store%dm_pulay_res_last(ispin, ic)%matrix, p_mix_new(ispin, ic)%matrix)
            CALL dbcsr_add(p_mix_new(ispin, ic)%matrix, rho_ao(ispin, ic)%matrix, alpha, 1.0_dp)
            DO jb = 1, nb
               CALL dbcsr_add(p_mix_new(ispin, ic)%matrix, mixing_store%dm_pulay_dp(jb, ispin, ic)%matrix, &
                              1.0_dp, -g(jb))
               CALL dbcsr_add(p_mix_new(ispin, ic)%matrix, mixing_store%dm_pulay_dres(jb, ispin, ic)%matrix, &
                              1.0_dp, -alpha*g(jb))
            END DO
         END DO
      END DO

      IF (nb > 0) DEALLOCATE (a, b, c, g)

      CALL timestop(handle)

   END SUBROUTINE dm_pulay_mixing

! **************************************************************************************************
!> \brief Pulay mixing (MIXING%PULAY_P) of spin-polarized density matrices: Pulay's method for the
!>        total density matrix P_a + P_b, linear mixing with ALPHA_MAG for the spin density matrix
!>        P_a - P_b
!> \param p_mix_new    on entry P_out of both spins, on exit the input of the next step
!> \param mixing_store history: P_in and the residual of both spins in the previous step, and
!>                     NBUFFER normalized differences of the total density matrix and its residual
!> \param rho_ao       input density matrices P_in of the present step
!> \param iter_delta   largest element of |P_out - P_in| over both spins and all images
!> \param full_step    take P_out unmixed (NSKIP) and restart the history
!> \param invert       recover P_out of the present step (after convergence)
!> \par Why the magnetization is mixed linearly
!>      Pulay's method converges to fixed points of the SCF map whether they are stable or not.
!>      Mixing both spins jointly, it took bcc Fe and fcc Ni (PBE, smearing) from a magnetic start
!>      to the non-magnetic solution, a saddle point 0.026 Ha above the ferromagnetic state for Fe.
!>      Linear mixing of the magnetization cannot settle at that point, which it leaves when the
!>      magnetic response is unstable (Stoner), while the charge, whose long-range response limits
!>      the step, keeps the acceleration. G-space mixing in CP2K and VASP likewise treat charge and
!>      magnetization as separate channels.
! **************************************************************************************************
   SUBROUTINE dm_pulay_mixing_spin(p_mix_new, mixing_store, rho_ao, iter_delta, full_step, invert)
      TYPE(dbcsr_p_type), DIMENSION(:, :), POINTER       :: p_mix_new
      TYPE(mixing_storage_type), POINTER                 :: mixing_store
      TYPE(dbcsr_p_type), DIMENSION(:, :), POINTER       :: rho_ao
      REAL(KIND=dp), INTENT(OUT)                         :: iter_delta
      LOGICAL, INTENT(IN)                                :: full_step, invert

      CHARACTER(len=*), PARAMETER                        :: routineN = 'dm_pulay_mixing_spin'

      INTEGER                                            :: handle, ib, ic, ispin, jb, kb, nb, &
                                                            nbuffer, nimg
      REAL(KIND=dp)                                      :: alpha, alpha_mag, inv_err, norm
      REAL(KIND=dp), ALLOCATABLE, DIMENSION(:)           :: c, g
      REAL(KIND=dp), ALLOCATABLE, DIMENSION(:, :)        :: a, b

      CALL timeset(routineN, handle)

      CPASSERT(SIZE(p_mix_new, 1) == 2)
      nimg = SIZE(p_mix_new, 2)
      nbuffer = MAX(mixing_store%nbuffer, 1)
      alpha = mixing_store%alpha
      alpha_mag = mixing_store%alpha_mag
      iter_delta = 0.0_dp

      IF (invert) THEN
         IF (mixing_store%dm_pulay_ncall > 0) THEN
            DO ic = 1, nimg
               DO ispin = 1, 2
                  CALL dbcsr_add(p_mix_new(ispin, ic)%matrix, rho_ao(ispin, ic)%matrix, 0.0_dp, 1.0_dp)
                  CALL dbcsr_add(p_mix_new(ispin, ic)%matrix, &
                                 mixing_store%dm_pulay_res_last(ispin, ic)%matrix, 1.0_dp, 1.0_dp)
               END DO
            END DO
         END IF
         CALL timestop(handle)
         RETURN
      END IF

      ! residuals of both spins in p_mix_new
      DO ic = 1, nimg
         DO ispin = 1, 2
            CALL dbcsr_add(p_mix_new(ispin, ic)%matrix, rho_ao(ispin, ic)%matrix, 1.0_dp, -1.0_dp)
            iter_delta = MAX(iter_delta, dbcsr_maxabs(p_mix_new(ispin, ic)%matrix))
         END DO
      END DO

      IF (full_step) THEN
         DO ic = 1, nimg
            DO ispin = 1, 2
               CALL dbcsr_add(p_mix_new(ispin, ic)%matrix, rho_ao(ispin, ic)%matrix, 1.0_dp, 1.0_dp)
            END DO
         END DO
         mixing_store%dm_pulay_ncall = 0
         CALL timestop(handle)
         RETURN
      END IF

      mixing_store%dm_pulay_ncall = mixing_store%dm_pulay_ncall + 1
      IF (mixing_store%dm_pulay_ncall == 1) THEN
         ! per spin: last input and residual; for the total density matrix: the differences
         CALL dm_pulay_storage_release(mixing_store)
         mixing_store%dm_pulay_ncall = 1
         ALLOCATE (mixing_store%dm_pulay_p_last(2, nimg), mixing_store%dm_pulay_res_last(2, nimg))
         ALLOCATE (mixing_store%dm_pulay_dp(nbuffer, 1, nimg), mixing_store%dm_pulay_dres(nbuffer, 1, nimg))
         DO ic = 1, nimg
            DO ispin = 1, 2
               ALLOCATE (mixing_store%dm_pulay_p_last(ispin, ic)%matrix, &
                         mixing_store%dm_pulay_res_last(ispin, ic)%matrix)
               CALL dbcsr_create(mixing_store%dm_pulay_p_last(ispin, ic)%matrix, &
                                 template=rho_ao(ispin, ic)%matrix, name="PULAY_P P_IN")
               CALL dbcsr_create(mixing_store%dm_pulay_res_last(ispin, ic)%matrix, &
                                 template=rho_ao(ispin, ic)%matrix, name="PULAY_P RESIDUAL")
            END DO
            DO ib = 1, nbuffer
               ALLOCATE (mixing_store%dm_pulay_dp(ib, 1, ic)%matrix, mixing_store%dm_pulay_dres(ib, 1, ic)%matrix)
               CALL dbcsr_create(mixing_store%dm_pulay_dp(ib, 1, ic)%matrix, &
                                 template=rho_ao(1, ic)%matrix, name="PULAY_P DP")
               CALL dbcsr_create(mixing_store%dm_pulay_dres(ib, 1, ic)%matrix, &
                                 template=rho_ao(1, ic)%matrix, name="PULAY_P DRES")
            END DO
         END DO
         nb = 0
      ELSE
         ! differences of the total density matrix and its residual, divided by |dR|
         nb = MIN(mixing_store%dm_pulay_ncall - 1, nbuffer)
         ib = MODULO(mixing_store%dm_pulay_ncall - 2, nbuffer) + 1
         norm = 0.0_dp
         DO ic = 1, nimg
            ASSOCIATE (dres => mixing_store%dm_pulay_dres(ib, 1, ic)%matrix, &
                       dpin => mixing_store%dm_pulay_dp(ib, 1, ic)%matrix)
               CALL dbcsr_copy(dres, p_mix_new(1, ic)%matrix)
               CALL dbcsr_add(dres, p_mix_new(2, ic)%matrix, 1.0_dp, 1.0_dp)
               CALL dbcsr_add(dres, mixing_store%dm_pulay_res_last(1, ic)%matrix, 1.0_dp, -1.0_dp)
               CALL dbcsr_add(dres, mixing_store%dm_pulay_res_last(2, ic)%matrix, 1.0_dp, -1.0_dp)
               CALL dbcsr_copy(dpin, rho_ao(1, ic)%matrix)
               CALL dbcsr_add(dpin, rho_ao(2, ic)%matrix, 1.0_dp, 1.0_dp)
               CALL dbcsr_add(dpin, mixing_store%dm_pulay_p_last(1, ic)%matrix, 1.0_dp, -1.0_dp)
               CALL dbcsr_add(dpin, mixing_store%dm_pulay_p_last(2, ic)%matrix, 1.0_dp, -1.0_dp)
               norm = norm + frobenius_dot(dres, dres)
            END ASSOCIATE
         END DO
         norm = SQRT(norm)
         IF (norm > 0.0_dp) THEN
            DO ic = 1, nimg
               CALL dbcsr_scale(mixing_store%dm_pulay_dres(ib, 1, ic)%matrix, 1.0_dp/norm)
               CALL dbcsr_scale(mixing_store%dm_pulay_dp(ib, 1, ic)%matrix, 1.0_dp/norm)
            END DO
         ELSE
            nb = 0
         END IF
      END IF

      ! keep P_in and R of both spins, then turn p_mix_new into the residuals of the total density
      ! matrix (spin 1) and of the spin density matrix (spin 2)
      DO ic = 1, nimg
         DO ispin = 1, 2
            CALL dbcsr_copy(mixing_store%dm_pulay_p_last(ispin, ic)%matrix, rho_ao(ispin, ic)%matrix)
            CALL dbcsr_copy(mixing_store%dm_pulay_res_last(ispin, ic)%matrix, p_mix_new(ispin, ic)%matrix)
         END DO
         CALL dbcsr_add(p_mix_new(1, ic)%matrix, p_mix_new(2, ic)%matrix, 1.0_dp, 1.0_dp)
         CALL dbcsr_add(p_mix_new(2, ic)%matrix, p_mix_new(1, ic)%matrix, -2.0_dp, 1.0_dp)
      END DO

      ! Pulay coefficients from the total density matrix
      IF (nb > 0) THEN
         ALLOCATE (a(nb, nb), b(nb, nb), c(nb), g(nb))
         a = 0.0_dp
         c = 0.0_dp
         DO jb = 1, nb
            DO kb = jb, nb
               DO ic = 1, nimg
                  a(jb, kb) = a(jb, kb) + frobenius_dot(mixing_store%dm_pulay_dres(jb, 1, ic)%matrix, &
                                                        mixing_store%dm_pulay_dres(kb, 1, ic)%matrix)
               END DO
               a(kb, jb) = a(jb, kb)
            END DO
            a(jb, jb) = a(jb, jb) + mixing_store%broy_w0
            DO ic = 1, nimg
               c(jb) = c(jb) + frobenius_dot(mixing_store%dm_pulay_dres(jb, 1, ic)%matrix, &
                                             p_mix_new(1, ic)%matrix)
            END DO
         END DO
         CALL invert_matrix(a, b, inv_err)
         g = MATMUL(b, c)
      END IF

      ! steps of the total (Pulay) and spin (linear) density matrices, then back to spins
      DO ic = 1, nimg
         CALL dbcsr_scale(p_mix_new(1, ic)%matrix, alpha)
         DO jb = 1, nb
            CALL dbcsr_add(p_mix_new(1, ic)%matrix, mixing_store%dm_pulay_dp(jb, 1, ic)%matrix, 1.0_dp, -g(jb))
            CALL dbcsr_add(p_mix_new(1, ic)%matrix, mixing_store%dm_pulay_dres(jb, 1, ic)%matrix, &
                           1.0_dp, -alpha*g(jb))
         END DO
         CALL dbcsr_scale(p_mix_new(2, ic)%matrix, alpha_mag)
         CALL dbcsr_add(p_mix_new(1, ic)%matrix, p_mix_new(2, ic)%matrix, 0.5_dp, 0.5_dp)
         CALL dbcsr_add(p_mix_new(2, ic)%matrix, p_mix_new(1, ic)%matrix, -1.0_dp, 1.0_dp)
         DO ispin = 1, 2
            CALL dbcsr_add(p_mix_new(ispin, ic)%matrix, rho_ao(ispin, ic)%matrix, 1.0_dp, 1.0_dp)
         END DO
      END DO

      IF (nb > 0) DEALLOCATE (a, b, c, g)

      CALL timestop(handle)

   END SUBROUTINE dm_pulay_mixing_spin

! **************************************************************************************************
!> \brief Sum of the products of the stored elements of two matrices of the same structure
!> \param matrix_a ...
!> \param matrix_b ...
!> \return ...
! **************************************************************************************************
   FUNCTION frobenius_dot(matrix_a, matrix_b) RESULT(dot)
      TYPE(dbcsr_type), INTENT(IN)                       :: matrix_a, matrix_b
      REAL(KIND=dp)                                      :: dot

      CALL dbcsr_dot(matrix_a, matrix_b, dot)

   END FUNCTION frobenius_dot

! **************************************************************************************************
!> \brief   Diagonalise the Kohn-Sham matrix to get a new set of MO eigen-
!>          vectors and MO eigenvalues. ks will be modified
!> \param matrix_ks_fm ...
!> \param mo_set ...
!> \param ortho ...
!> \param work ...
!> \param cholesky_method ...
!> \param do_level_shift activate the level shifting technique
!> \param level_shift    amount of shift applied (in a.u.)
!> \param matrix_u_fm    matrix U : S (overlap matrix) = U^T * U
!> \param use_jacobi ...
!> \date    01.05.2001
!> \author  Matthias Krack
!> \version 1.0
! **************************************************************************************************
   SUBROUTINE eigensolver(matrix_ks_fm, mo_set, ortho, work, &
                          cholesky_method, do_level_shift, &
                          level_shift, matrix_u_fm, use_jacobi)
      TYPE(cp_fm_type), INTENT(IN)                       :: matrix_ks_fm
      TYPE(mo_set_type), INTENT(IN)                      :: mo_set
      TYPE(cp_fm_type), INTENT(IN)                       :: ortho, work
      INTEGER, INTENT(inout)                             :: cholesky_method
      LOGICAL, INTENT(in)                                :: do_level_shift
      REAL(KIND=dp), INTENT(in)                          :: level_shift
      TYPE(cp_fm_type), INTENT(IN), OPTIONAL             :: matrix_u_fm
      LOGICAL, INTENT(in)                                :: use_jacobi

      CHARACTER(len=*), PARAMETER                        :: routineN = 'eigensolver'

      INTEGER                                            :: handle, homo, nao, nmo
      REAL(KIND=dp), DIMENSION(:), POINTER               :: mo_eigenvalues
      TYPE(cp_fm_type), POINTER                          :: mo_coeff

      CALL timeset(routineN, handle)

      NULLIFY (mo_coeff)
      NULLIFY (mo_eigenvalues)

      ! Diagonalise the Kohn-Sham matrix

      CALL get_mo_set(mo_set=mo_set, &
                      nao=nao, &
                      nmo=nmo, &
                      homo=homo, &
                      eigenvalues=mo_eigenvalues, &
                      mo_coeff=mo_coeff)

      SELECT CASE (cholesky_method)
      CASE (cholesky_reduce)
         CALL cp_fm_cholesky_reduce(matrix_ks_fm, ortho)

         IF (do_level_shift) THEN
            CALL shift_unocc_mos(matrix_ks_fm=matrix_ks_fm, mo_coeff=mo_coeff, homo=homo, &
                                 level_shift=level_shift, is_triangular=.TRUE., matrix_u_fm=matrix_u_fm)
         END IF

         CALL choose_eigv_solver(matrix_ks_fm, work, mo_eigenvalues)
         CALL cp_fm_cholesky_restore(work, nmo, ortho, mo_coeff, "SOLVE")
         IF (do_level_shift) THEN
            CALL correct_mo_eigenvalues(mo_eigenvalues, homo, nmo, level_shift)
         END IF

      CASE (cholesky_restore)
         CALL cp_fm_uplo_to_full(matrix_ks_fm, work)
         CALL cp_fm_cholesky_restore(matrix_ks_fm, nao, ortho, work, &
                                     "SOLVE", pos="RIGHT")
         CALL cp_fm_cholesky_restore(work, nao, ortho, matrix_ks_fm, &
                                     "SOLVE", pos="LEFT", transa="T")

         IF (do_level_shift) THEN
            CALL shift_unocc_mos(matrix_ks_fm=matrix_ks_fm, mo_coeff=mo_coeff, homo=homo, &
                                 level_shift=level_shift, is_triangular=.TRUE., matrix_u_fm=matrix_u_fm)
         END IF

         CALL choose_eigv_solver(matrix_ks_fm, work, mo_eigenvalues)
         CALL cp_fm_cholesky_restore(work, nmo, ortho, mo_coeff, "SOLVE")

         IF (do_level_shift) THEN
            CALL correct_mo_eigenvalues(mo_eigenvalues, homo, nmo, level_shift)
         END IF

      CASE (cholesky_inverse)
         CALL cp_fm_uplo_to_full(matrix_ks_fm, work)

         CALL cp_fm_triangular_multiply(ortho, matrix_ks_fm, side="R", transpose_tr=.FALSE., &
                                        invert_tr=.FALSE., uplo_tr="U", n_rows=nao, n_cols=nao, alpha=1.0_dp)
         CALL cp_fm_triangular_multiply(ortho, matrix_ks_fm, side="L", transpose_tr=.TRUE., &
                                        invert_tr=.FALSE., uplo_tr="U", n_rows=nao, n_cols=nao, alpha=1.0_dp)

         IF (do_level_shift) THEN
            CALL shift_unocc_mos(matrix_ks_fm=matrix_ks_fm, mo_coeff=mo_coeff, homo=homo, &
                                 level_shift=level_shift, is_triangular=.TRUE., matrix_u_fm=matrix_u_fm)
         END IF

         CALL choose_eigv_solver(matrix_ks_fm, work, mo_eigenvalues)
         CALL cp_fm_triangular_multiply(ortho, work, side="L", transpose_tr=.FALSE., &
                                        invert_tr=.FALSE., uplo_tr="U", n_rows=nao, n_cols=nmo, alpha=1.0_dp)
         CALL cp_fm_to_fm(work, mo_coeff, nmo, 1, 1)

         IF (do_level_shift) THEN
            CALL correct_mo_eigenvalues(mo_eigenvalues, homo, nmo, level_shift)
         END IF

      END SELECT

      IF (use_jacobi) THEN
         CALL cp_fm_to_fm(mo_coeff, ortho)
         cholesky_method = cholesky_off
      END IF

      CALL timestop(handle)

   END SUBROUTINE eigensolver

! **************************************************************************************************
!> \brief Solve the generalized eigenvalue problem
!> \param matrix_ks_fm Kohn-Sham matrix in FM format
!> \param matrix_s     Overlap matrix (DBCSR)
!> \param mo_set       Molecular orbital set
!> \param work         Work matrix (used as eigenvector buffer)
! **************************************************************************************************
   SUBROUTINE eigensolver_generalized(matrix_ks_fm, matrix_s, mo_set, work)
      TYPE(cp_fm_type), INTENT(INOUT)                    :: matrix_ks_fm
      TYPE(dbcsr_type), INTENT(IN)                       :: matrix_s
      TYPE(mo_set_type), INTENT(IN)                      :: mo_set
      TYPE(cp_fm_type), INTENT(INOUT)                    :: work

      CHARACTER(len=*), PARAMETER :: routineN = 'eigensolver_generalized'

      INTEGER                                            :: handle, nmo
      REAL(KIND=dp), DIMENSION(:), POINTER               :: mo_eigenvalues
      TYPE(cp_fm_struct_type), POINTER                   :: fm_struct
      TYPE(cp_fm_type)                                   :: s_fm
      TYPE(cp_fm_type), POINTER                          :: mo_coeff

      CALL timeset(routineN, handle)

      NULLIFY (mo_coeff)
      NULLIFY (mo_eigenvalues)

      CALL get_mo_set(mo_set=mo_set, nmo=nmo, eigenvalues=mo_eigenvalues, mo_coeff=mo_coeff)
      CALL cp_fm_get_info(matrix_ks_fm, matrix_struct=fm_struct)

      ! Convert S matrix from DBCSR to FM (required for cuSOLVERMp)
      CALL cp_fm_create(s_fm, fm_struct)
      CALL copy_dbcsr_to_fm(matrix_s, s_fm)

      ! Solve generalized eigenvalue problem - eigenvectors output to work buffer
      CALL cp_fm_geeig(matrix_ks_fm, s_fm, mo_coeff, mo_eigenvalues, work)

      CALL cp_fm_release(s_fm)

      CALL timestop(handle)

   END SUBROUTINE eigensolver_generalized

! **************************************************************************************************
!> \brief ...
!> \param matrix_ks ...
!> \param matrix_ks_fm ...
!> \param mo_set ...
!> \param ortho_dbcsr ...
!> \param ksbuf1 ...
!> \param ksbuf2 ...
! **************************************************************************************************
   SUBROUTINE eigensolver_dbcsr(matrix_ks, matrix_ks_fm, mo_set, ortho_dbcsr, ksbuf1, ksbuf2)
      TYPE(dbcsr_type), INTENT(IN)                       :: matrix_ks
      TYPE(cp_fm_type), INTENT(INOUT)                    :: matrix_ks_fm
      TYPE(mo_set_type), INTENT(IN)                      :: mo_set
      TYPE(dbcsr_type), INTENT(IN)                       :: ortho_dbcsr
      TYPE(dbcsr_type), INTENT(INOUT)                    :: ksbuf1, ksbuf2

      CHARACTER(len=*), PARAMETER                        :: routineN = 'eigensolver_dbcsr'

      INTEGER                                            :: handle, nao, nmo
      REAL(KIND=dp), DIMENSION(:), POINTER               :: mo_eigenvalues
      TYPE(cp_fm_type)                                   :: all_evecs, nmo_evecs
      TYPE(cp_fm_type), POINTER                          :: mo_coeff

      CALL timeset(routineN, handle)

      NULLIFY (mo_coeff)
      NULLIFY (mo_eigenvalues)

      CALL get_mo_set(mo_set=mo_set, &
                      nao=nao, &
                      nmo=nmo, &
                      eigenvalues=mo_eigenvalues, &
                      mo_coeff=mo_coeff)

!    Reduce KS matrix
      CALL dbcsr_desymmetrize(matrix_ks, ksbuf2)
      CALL dbcsr_multiply('N', 'N', 1.0_dp, ksbuf2, ortho_dbcsr, 0.0_dp, ksbuf1)
      CALL dbcsr_multiply('T', 'N', 1.0_dp, ortho_dbcsr, ksbuf1, 0.0_dp, ksbuf2)

!    Solve the eigenvalue problem
      CALL copy_dbcsr_to_fm(ksbuf2, matrix_ks_fm)
      CALL cp_fm_create(all_evecs, matrix_ks_fm%matrix_struct)
      CALL choose_eigv_solver(matrix_ks_fm, all_evecs, mo_eigenvalues)

      ! select first nmo eigenvectors
      CALL cp_fm_create(nmo_evecs, mo_coeff%matrix_struct)
      CALL cp_fm_to_fm(msource=all_evecs, mtarget=nmo_evecs, ncol=nmo)
      CALL cp_fm_release(all_evecs)

!    Restore the eigenvector of the general eig. problem
      CALL cp_dbcsr_sm_fm_multiply(ortho_dbcsr, nmo_evecs, mo_coeff, nmo)

      CALL cp_fm_release(nmo_evecs)
      CALL timestop(handle)

   END SUBROUTINE eigensolver_dbcsr

! **************************************************************************************************
!> \brief ...
!> \param matrix_ks_fm ...
!> \param mo_set ...
!> \param ortho ...
!> \param work ...
!> \param do_level_shift activate the level shifting technique
!> \param level_shift    amount of shift applied (in a.u.)
!> \param matrix_u_fm    matrix U : S (overlap matrix) = U^T * U
!> \param use_jacobi ...
!> \param jacobi_threshold ...
!> \param ortho_red ...
!> \param work_red ...
!> \param matrix_ks_fm_red ...
!> \param matrix_u_fm_red ...
! **************************************************************************************************
   SUBROUTINE eigensolver_symm(matrix_ks_fm, mo_set, ortho, work, do_level_shift, &
                               level_shift, matrix_u_fm, use_jacobi, jacobi_threshold, &
                               ortho_red, work_red, matrix_ks_fm_red, matrix_u_fm_red)
      TYPE(cp_fm_type), INTENT(IN)                       :: matrix_ks_fm
      TYPE(mo_set_type), INTENT(IN)                      :: mo_set
      TYPE(cp_fm_type), INTENT(IN)                       :: ortho, work
      LOGICAL, INTENT(IN)                                :: do_level_shift
      REAL(KIND=dp), INTENT(IN)                          :: level_shift
      TYPE(cp_fm_type), INTENT(IN), OPTIONAL             :: matrix_u_fm
      LOGICAL, INTENT(IN)                                :: use_jacobi
      REAL(KIND=dp), INTENT(IN)                          :: jacobi_threshold
      TYPE(cp_fm_type), INTENT(INOUT), OPTIONAL          :: ortho_red, work_red, matrix_ks_fm_red, &
                                                            matrix_u_fm_red

      CHARACTER(len=*), PARAMETER                        :: routineN = 'eigensolver_symm'

      INTEGER                                            :: handle, homo, nao, nao_red, nelectron, &
                                                            nmo
      REAL(KIND=dp), ALLOCATABLE, DIMENSION(:)           :: eigenvalues
      REAL(KIND=dp), DIMENSION(:), POINTER               :: mo_eigenvalues
      TYPE(cp_fm_type)                                   :: work_red2
      TYPE(cp_fm_type), POINTER                          :: mo_coeff

      CALL timeset(routineN, handle)

      NULLIFY (mo_coeff)
      NULLIFY (mo_eigenvalues)

      ! Diagonalise the Kohn-Sham matrix

      CALL get_mo_set(mo_set=mo_set, &
                      nao=nao, &
                      nmo=nmo, &
                      homo=homo, &
                      nelectron=nelectron, &
                      eigenvalues=mo_eigenvalues, &
                      mo_coeff=mo_coeff)

      IF (use_jacobi) THEN

         CALL cp_fm_symm("L", "U", nao, homo, 1.0_dp, matrix_ks_fm, mo_coeff, 0.0_dp, work)
         CALL parallel_gemm("T", "N", homo, nao - homo, nao, 1.0_dp, work, mo_coeff, &
                            0.0_dp, matrix_ks_fm, b_first_col=homo + 1, c_first_col=homo + 1)

         ! Block Jacobi (pseudo-diagonalization, only one sweep)
         CALL cp_fm_block_jacobi(matrix_ks_fm, mo_coeff, mo_eigenvalues, &
                                 jacobi_threshold, homo + 1)

      ELSE ! full S^(-1/2) has been computed
         IF (PRESENT(work_red) .AND. PRESENT(ortho_red) .AND. PRESENT(matrix_ks_fm_red)) THEN
            CALL cp_fm_get_info(ortho_red, ncol_global=nao_red)
            CALL cp_fm_symm("L", "U", nao, nao_red, 1.0_dp, matrix_ks_fm, ortho_red, 0.0_dp, work_red)
            CALL parallel_gemm("T", "N", nao_red, nao_red, nao, 1.0_dp, ortho_red, work_red, 0.0_dp, matrix_ks_fm_red)

            IF (do_level_shift) THEN
               CALL shift_unocc_mos(matrix_ks_fm=matrix_ks_fm_red, mo_coeff=mo_coeff, homo=homo, &
                                    level_shift=level_shift, is_triangular=.FALSE., matrix_u_fm=matrix_u_fm_red)
            END IF

            CALL cp_fm_create(work_red2, matrix_ks_fm_red%matrix_struct)
            ALLOCATE (eigenvalues(nao_red))
            CALL choose_eigv_solver(matrix_ks_fm_red, work_red2, eigenvalues)
            mo_eigenvalues(1:MIN(nao_red, nmo)) = eigenvalues(1:MIN(nao_red, nmo))
            CALL parallel_gemm("N", "N", nao, nmo, nao_red, 1.0_dp, ortho_red, work_red2, 0.0_dp, &
                               mo_coeff)
            CALL cp_fm_release(work_red2)
         ELSE
            CALL cp_fm_symm("L", "U", nao, nao, 1.0_dp, matrix_ks_fm, ortho, 0.0_dp, work)
            CALL parallel_gemm("T", "N", nao, nao, nao, 1.0_dp, ortho, work, 0.0_dp, matrix_ks_fm)
            IF (do_level_shift) THEN
               CALL shift_unocc_mos(matrix_ks_fm=matrix_ks_fm, mo_coeff=mo_coeff, homo=homo, &
                                    level_shift=level_shift, is_triangular=.FALSE., matrix_u_fm=matrix_u_fm)
            END IF
            CALL choose_eigv_solver(matrix_ks_fm, work, mo_eigenvalues)
            CALL parallel_gemm("N", "N", nao, nmo, nao, 1.0_dp, ortho, work, 0.0_dp, &
                               mo_coeff)
         END IF

         IF (do_level_shift) THEN
            CALL correct_mo_eigenvalues(mo_eigenvalues, homo, nmo, level_shift)
         END IF

      END IF

      CALL timestop(handle)

   END SUBROUTINE eigensolver_symm

! **************************************************************************************************

! **************************************************************************************************
!> \brief ...
!> \param matrix_ks ...
!> \param mo_set ...
!> \param work ...
!> \param do_level_shift activate the level shifting technique
!> \param level_shift    amount of shift applied (in a.u.)
!> \param use_jacobi ...
!> \param jacobi_threshold ...
! **************************************************************************************************
   SUBROUTINE eigensolver_simple(matrix_ks, mo_set, work, do_level_shift, &
                                 level_shift, use_jacobi, jacobi_threshold)

      TYPE(cp_fm_type), INTENT(IN)                       :: matrix_ks
      TYPE(mo_set_type), INTENT(IN)                      :: mo_set
      TYPE(cp_fm_type), INTENT(IN)                       :: work
      LOGICAL, INTENT(IN)                                :: do_level_shift
      REAL(KIND=dp), INTENT(IN)                          :: level_shift
      LOGICAL, INTENT(IN)                                :: use_jacobi
      REAL(KIND=dp), INTENT(IN)                          :: jacobi_threshold

      CHARACTER(len=*), PARAMETER :: routineN = 'eigensolver_simple'

      INTEGER                                            :: handle, homo, nao, nelectron, nmo
      REAL(KIND=dp), DIMENSION(:), POINTER               :: mo_eigenvalues
      TYPE(cp_fm_type), POINTER                          :: mo_coeff

      CALL timeset(routineN, handle)

      NULLIFY (mo_coeff)
      NULLIFY (mo_eigenvalues)

      CALL get_mo_set(mo_set=mo_set, &
                      nao=nao, &
                      nmo=nmo, &
                      homo=homo, &
                      nelectron=nelectron, &
                      eigenvalues=mo_eigenvalues, &
                      mo_coeff=mo_coeff)

      IF (do_level_shift) THEN
         ! matrix_u_fm is simply an identity matrix, so we omit it here
         CALL shift_unocc_mos(matrix_ks_fm=matrix_ks, mo_coeff=mo_coeff, homo=homo, &
                              level_shift=level_shift, is_triangular=.FALSE.)
      END IF

      IF (use_jacobi) THEN
         CALL cp_fm_symm("L", "U", nao, homo, 1.0_dp, matrix_ks, mo_coeff, 0.0_dp, work)
         CALL parallel_gemm("T", "N", homo, nao - homo, nao, 1.0_dp, work, mo_coeff, &
                            0.0_dp, matrix_ks, b_first_col=homo + 1, c_first_col=homo + 1)
         ! Block Jacobi (pseudo-diagonalization, only one sweep)
         CALL cp_fm_block_jacobi(matrix_ks, mo_coeff, mo_eigenvalues, jacobi_threshold, homo + 1)
      ELSE

         CALL choose_eigv_solver(matrix_ks, work, mo_eigenvalues)

         CALL cp_fm_to_fm(work, mo_coeff, nmo, 1, 1)

      END IF

      IF (do_level_shift) THEN
         CALL correct_mo_eigenvalues(mo_eigenvalues, homo, nmo, level_shift)
      END IF

      CALL timestop(handle)

   END SUBROUTINE eigensolver_simple

! **************************************************************************************************
!> \brief Perform a mixing of the given matrixes into the first matrix
!>      m1 = m2 + p_mix (m1-m2)
!> \param m1 first (new) matrix, is modified
!> \param m2 the second (old) matrix
!> \param p_mix how much m1 is conserved (0: none, 1: all)
!> \param delta maximum norm of m1-m2
!> \param para_env ...
!> \param m3 ...
!> \par History
!>      02.2003 rewamped [fawzi]
!> \author fawzi
!> \note
!>      if you what to store the result in m2 swap m1 and m2 an use
!>      (1-pmix) as pmix
!>      para_env should be removed (embedded in matrix)
! **************************************************************************************************
   SUBROUTINE cp_sm_mix(m1, m2, p_mix, delta, para_env, m3)

      TYPE(dbcsr_type), POINTER                          :: m1, m2
      REAL(KIND=dp), INTENT(IN)                          :: p_mix
      REAL(KIND=dp), INTENT(OUT)                         :: delta
      TYPE(mp_para_env_type), POINTER                    :: para_env
      TYPE(dbcsr_type), OPTIONAL, POINTER                :: m3

      CHARACTER(len=*), PARAMETER                        :: routineN = 'cp_sm_mix'

      INTEGER                                            :: handle, i, iblock_col, iblock_row, j
      LOGICAL                                            :: found
      REAL(KIND=dp), DIMENSION(:, :), POINTER            :: p_delta_block, p_new_block, p_old_block
      TYPE(dbcsr_iterator_type)                          :: iter

      CALL timeset(routineN, handle)
      delta = 0.0_dp

      CALL dbcsr_iterator_start(iter, m1)
      DO WHILE (dbcsr_iterator_blocks_left(iter))
         CALL dbcsr_iterator_next_block(iter, iblock_row, iblock_col, p_new_block)
         CALL dbcsr_get_block_p(matrix=m2, row=iblock_row, col=iblock_col, &
                                BLOCK=p_old_block, found=found)
         CPASSERT(ASSOCIATED(p_old_block))
         IF (PRESENT(m3)) THEN
            CALL dbcsr_get_block_p(matrix=m3, row=iblock_row, col=iblock_col, &
                                   BLOCK=p_delta_block, found=found)
            CPASSERT(ASSOCIATED(p_delta_block))

            DO j = 1, SIZE(p_new_block, 2)
               DO i = 1, SIZE(p_new_block, 1)
                  p_delta_block(i, j) = p_new_block(i, j) - p_old_block(i, j)
                  delta = MAX(delta, ABS(p_delta_block(i, j)))
               END DO
            END DO
         ELSE
            DO j = 1, SIZE(p_new_block, 2)
               DO i = 1, SIZE(p_new_block, 1)
                  p_new_block(i, j) = p_new_block(i, j) - p_old_block(i, j)
                  delta = MAX(delta, ABS(p_new_block(i, j)))
                  p_new_block(i, j) = p_old_block(i, j) + p_mix*p_new_block(i, j)
               END DO
            END DO
         END IF
      END DO
      CALL dbcsr_iterator_stop(iter)

      CALL para_env%max(delta)

      CALL timestop(handle)

   END SUBROUTINE cp_sm_mix

! **************************************************************************************************
!> \brief ...
!> \param ksa ...
!> \param ksb ...
!> \param occa ...
!> \param occb ...
!> \param roks_parameter ...
! **************************************************************************************************
   SUBROUTINE combine_ks_matrices_1(ksa, ksb, occa, occb, roks_parameter)

      ! Combine the alpha and beta Kohn-Sham matrices during a restricted open
      ! Kohn-Sham (ROKS) calculation
      ! On input ksa and ksb contain the alpha and beta Kohn-Sham matrices,
      ! respectively. occa and occb contain the corresponding MO occupation
      ! numbers. On output the combined ROKS operator matrix is returned in ksa.

      ! Literature: - C. C. J. Roothaan, Rev. Mod. Phys. 32, 179 (1960)
      !             - M. F. Guest and V. R. Saunders, Mol. Phys. 28(3), 819 (1974)

      TYPE(cp_fm_type), INTENT(IN)                       :: ksa, ksb
      REAL(KIND=dp), DIMENSION(:), INTENT(IN)            :: occa, occb
      REAL(KIND=dp), DIMENSION(0:2, 0:2, 1:2), &
         INTENT(IN)                                      :: roks_parameter

      CHARACTER(LEN=*), PARAMETER :: routineN = 'combine_ks_matrices_1'

      INTEGER                                            :: handle, i, icol_global, icol_local, &
                                                            irow_global, irow_local, j, &
                                                            ncol_local, nrow_local
      INTEGER, DIMENSION(:), POINTER                     :: col_indices, row_indices
      LOGICAL                                            :: compatible_matrices
      REAL(KIND=dp), CONTIGUOUS, DIMENSION(:, :), &
         POINTER                                         :: fa, fb
      TYPE(cp_fm_struct_type), POINTER                   :: ksa_struct, ksb_struct

! -------------------------------------------------------------------------

      CALL timeset(routineN, handle)

      CALL cp_fm_get_info(matrix=ksa, &
                          matrix_struct=ksa_struct, &
                          nrow_local=nrow_local, &
                          ncol_local=ncol_local, &
                          row_indices=row_indices, &
                          col_indices=col_indices, &
                          local_data=fa)

      CALL cp_fm_get_info(matrix=ksb, &
                          matrix_struct=ksb_struct, &
                          local_data=fb)

      compatible_matrices = cp_fm_struct_equivalent(ksa_struct, ksb_struct)
      CPASSERT(compatible_matrices)

      IF (SUM(occb) == 0.0_dp) fb = 0.0_dp

      DO icol_local = 1, ncol_local
         icol_global = col_indices(icol_local)
         j = INT(occa(icol_global)) + INT(occb(icol_global))
         DO irow_local = 1, nrow_local
            irow_global = row_indices(irow_local)
            i = INT(occa(irow_global)) + INT(occb(irow_global))
            fa(irow_local, icol_local) = &
               roks_parameter(i, j, 1)*fa(irow_local, icol_local) + &
               roks_parameter(i, j, 2)*fb(irow_local, icol_local)
         END DO
      END DO

      CALL timestop(handle)

   END SUBROUTINE combine_ks_matrices_1

! **************************************************************************************************
!> \brief ...
!> \param ksa ...
!> \param ksb ...
!> \param occa ...
!> \param occb ...
!> \param f ...
!> \param nalpha ...
!> \param nbeta ...
! **************************************************************************************************
   SUBROUTINE combine_ks_matrices_2(ksa, ksb, occa, occb, f, nalpha, nbeta)

      ! Combine the alpha and beta Kohn-Sham matrices during a restricted open
      ! Kohn-Sham (ROKS) calculation
      ! On input ksa and ksb contain the alpha and beta Kohn-Sham matrices,
      ! respectively. occa and occb contain the corresponding MO occupation
      ! numbers. On output the combined ROKS operator matrix is returned in ksa.

      ! Literature: - C. C. J. Roothaan, Rev. Mod. Phys. 32, 179 (1960)
      !             - M. Filatov and S. Shaik, Chem. Phys. Lett. 288, 689 (1998)

      TYPE(cp_fm_type), INTENT(IN)                       :: ksa, ksb
      REAL(KIND=dp), DIMENSION(:), INTENT(IN)            :: occa, occb
      REAL(KIND=dp), INTENT(IN)                          :: f
      INTEGER, INTENT(IN)                                :: nalpha, nbeta

      CHARACTER(LEN=*), PARAMETER :: routineN = 'combine_ks_matrices_2'

      INTEGER                                            :: handle, icol_global, icol_local, &
                                                            irow_global, irow_local, ncol_local, &
                                                            nrow_local
      INTEGER, DIMENSION(:), POINTER                     :: col_indices, row_indices
      LOGICAL                                            :: compatible_matrices
      REAL(KIND=dp)                                      :: beta, t1, t2, ta, tb
      REAL(KIND=dp), CONTIGUOUS, DIMENSION(:, :), &
         POINTER                                         :: fa, fb
      TYPE(cp_fm_struct_type), POINTER                   :: ksa_struct, ksb_struct

! -------------------------------------------------------------------------

      CALL timeset(routineN, handle)

      CALL cp_fm_get_info(matrix=ksa, &
                          matrix_struct=ksa_struct, &
                          nrow_local=nrow_local, &
                          ncol_local=ncol_local, &
                          row_indices=row_indices, &
                          col_indices=col_indices, &
                          local_data=fa)

      CALL cp_fm_get_info(matrix=ksb, &
                          matrix_struct=ksb_struct, &
                          local_data=fb)

      compatible_matrices = cp_fm_struct_equivalent(ksa_struct, ksb_struct)
      CPASSERT(compatible_matrices)

      beta = 1.0_dp/(1.0_dp - f)

      DO icol_local = 1, ncol_local

         icol_global = col_indices(icol_local)

         DO irow_local = 1, nrow_local

            irow_global = row_indices(irow_local)

            t1 = 0.5_dp*(fa(irow_local, icol_local) + fb(irow_local, icol_local))

            IF ((0 < irow_global) .AND. (irow_global <= nbeta)) THEN
               IF ((0 < icol_global) .AND. (icol_global <= nbeta)) THEN
                  ! closed-closed
                  fa(irow_local, icol_local) = t1
               ELSE IF ((nbeta < icol_global) .AND. (icol_global <= nalpha)) THEN
                  ! closed-open
                  ta = 0.5_dp*(f - REAL(occa(icol_global), KIND=dp))/f
                  tb = 0.5_dp*(f - REAL(occb(icol_global), KIND=dp))/f
                  t2 = ta*fa(irow_local, icol_local) + tb*fb(irow_local, icol_local)
                  fa(irow_local, icol_local) = t1 + (beta - 1.0_dp)*t2
               ELSE
                  ! closed-virtual
                  fa(irow_local, icol_local) = t1
               END IF
            ELSE IF ((nbeta < irow_global) .AND. (irow_global <= nalpha)) THEN
               IF ((0 < irow_global) .AND. (irow_global <= nbeta)) THEN
                  ! open-closed
                  ta = 0.5_dp*(f - REAL(occa(irow_global), KIND=dp))/f
                  tb = 0.5_dp*(f - REAL(occb(irow_global), KIND=dp))/f
                  t2 = ta*fa(irow_local, icol_local) + tb*fb(irow_local, icol_local)
                  fa(irow_local, icol_local) = t1 + (beta - 1.0_dp)*t2
               ELSE IF ((nbeta < icol_global) .AND. (icol_global <= nalpha)) THEN
                  ! open-open
                  ta = 0.5_dp*(f - REAL(occa(icol_global), KIND=dp))/f
                  tb = 0.5_dp*(f - REAL(occb(icol_global), KIND=dp))/f
                  t2 = ta*fa(irow_local, icol_local) + tb*fb(irow_local, icol_local)
                  IF (irow_global == icol_global) THEN
                     fa(irow_local, icol_local) = t1 - t2
                  ELSE
                     fa(irow_local, icol_local) = t1 - 0.5_dp*t2
                  END IF
               ELSE
                  ! open-virtual
                  ta = 0.5_dp*(f - REAL(occa(irow_global), KIND=dp))/f
                  tb = 0.5_dp*(f - REAL(occb(irow_global), KIND=dp))/f
                  t2 = ta*fa(irow_local, icol_local) + tb*fb(irow_local, icol_local)
                  fa(irow_local, icol_local) = t1 - t2
               END IF
            ELSE
               IF ((0 < irow_global) .AND. (irow_global < nbeta)) THEN
                  ! virtual-closed
                  fa(irow_local, icol_local) = t1
               ELSE IF ((nbeta < icol_global) .AND. (icol_global <= nalpha)) THEN
                  ! virtual-open
                  ta = 0.5_dp*(f - REAL(occa(icol_global), KIND=dp))/f
                  tb = 0.5_dp*(f - REAL(occb(icol_global), KIND=dp))/f
                  t2 = ta*fa(irow_local, icol_local) + tb*fb(irow_local, icol_local)
                  fa(irow_local, icol_local) = t1 - t2
               ELSE
                  ! virtual-virtual
                  fa(irow_local, icol_local) = t1
               END IF
            END IF

         END DO
      END DO

      CALL timestop(handle)

   END SUBROUTINE combine_ks_matrices_2

! **************************************************************************************************
!> \brief   Correct MO eigenvalues after MO level shifting.
!> \param mo_eigenvalues vector of eigenvalues
!> \param homo index of the highest occupied molecular orbital
!> \param nmo  number of molecular orbitals
!> \param level_shift amount of applied level shifting (in a.u.)
!> \date    19.04.2002
!> \par History
!>      - correct_mo_eigenvalues added (18.04.02,MK)
!>      - moved from module qs_mo_types, revised interface (03.2016, Sergey Chulkov)
!> \author  MK
!> \version 1.0
! **************************************************************************************************
   PURE SUBROUTINE correct_mo_eigenvalues(mo_eigenvalues, homo, nmo, level_shift)

      REAL(kind=dp), DIMENSION(:), INTENT(inout)         :: mo_eigenvalues
      INTEGER, INTENT(in)                                :: homo, nmo
      REAL(kind=dp), INTENT(in)                          :: level_shift

      INTEGER                                            :: imo

      DO imo = homo + 1, nmo
         mo_eigenvalues(imo) = mo_eigenvalues(imo) - level_shift
      END DO

   END SUBROUTINE correct_mo_eigenvalues

! **************************************************************************************************
!> \brief Adjust the Kohn-Sham matrix by shifting the orbital energies of all
!>        unoccupied molecular orbitals
!> \param matrix_ks_fm   transformed Kohn-Sham matrix = U^{-1,T} * KS * U^{-1}
!> \param mo_coeff       matrix of molecular orbitals (C)
!> \param homo           number of occupied molecular orbitals
!> \param level_shift    amount of shift applying (in a.u.)
!> \param is_triangular  indicates that matrix_u_fm contains an upper triangular matrix
!> \param matrix_u_fm    matrix U: S (overlap matrix) = U^T * U;
!>                       assume an identity matrix if omitted
!> \par History
!>      03.2016 created [Sergey Chulkov]
! **************************************************************************************************
   SUBROUTINE shift_unocc_mos(matrix_ks_fm, mo_coeff, homo, &
                              level_shift, is_triangular, matrix_u_fm)

      TYPE(cp_fm_type), INTENT(IN)                       :: matrix_ks_fm, mo_coeff
      INTEGER, INTENT(in)                                :: homo
      REAL(kind=dp), INTENT(in)                          :: level_shift
      LOGICAL, INTENT(in)                                :: is_triangular
      TYPE(cp_fm_type), INTENT(IN), OPTIONAL             :: matrix_u_fm

      CHARACTER(len=*), PARAMETER                        :: routineN = 'shift_unocc_mos'

      INTEGER                                            :: handle, nao, nao_red, nmo
      REAL(kind=dp), ALLOCATABLE, DIMENSION(:)           :: weights
      TYPE(cp_fm_struct_type), POINTER                   :: ao_mo_fmstruct
      TYPE(cp_fm_type)                                   :: u_mo, u_mo_scaled

      CALL timeset(routineN, handle)

      IF (PRESENT(matrix_u_fm)) THEN
         CALL cp_fm_get_info(mo_coeff, ncol_global=nmo)
         CALL cp_fm_get_info(matrix_u_fm, nrow_global=nao_red, ncol_global=nao)
      ELSE
         CALL cp_fm_get_info(mo_coeff, ncol_global=nmo, nrow_global=nao)
         nao_red = nao
      END IF

      NULLIFY (ao_mo_fmstruct)
      CALL cp_fm_struct_create(ao_mo_fmstruct, nrow_global=nao_red, ncol_global=nmo, &
                               para_env=mo_coeff%matrix_struct%para_env, context=mo_coeff%matrix_struct%context)

      CALL cp_fm_create(u_mo, ao_mo_fmstruct)
      CALL cp_fm_create(u_mo_scaled, ao_mo_fmstruct)

      CALL cp_fm_struct_release(ao_mo_fmstruct)

      ! U * C
      IF (PRESENT(matrix_u_fm)) THEN
         IF (is_triangular) THEN
            CALL cp_fm_to_fm(mo_coeff, u_mo)
            CALL cp_fm_triangular_multiply(matrix_u_fm, u_mo, side="L", transpose_tr=.FALSE., &
                                           invert_tr=.FALSE., uplo_tr="U", n_rows=nao, n_cols=nmo, alpha=1.0_dp)
         ELSE
            CALL parallel_gemm("N", "N", nao_red, nmo, nao, 1.0_dp, matrix_u_fm, mo_coeff, 0.0_dp, u_mo)
         END IF
      ELSE
         ! assume U is an identity matrix
         CALL cp_fm_to_fm(mo_coeff, u_mo)
      END IF

      CALL cp_fm_to_fm(u_mo, u_mo_scaled)

      ! up-shift all unoccupied molecular orbitals by the amount of 'level_shift'
      ! weight = diag(DELTA) = (0, ... 0, level_shift, ..., level_shift)
      !             MO index :  1 .. homo   homo+1     ...  nmo
      ALLOCATE (weights(nmo))
      weights(1:homo) = 0.0_dp
      weights(homo + 1:nmo) = level_shift
      ! DELTA * U * C
      ! DELTA is a diagonal matrix, so simply scale all the columns of (U * C) by weights(:)
      CALL cp_fm_column_scale(u_mo_scaled, weights)
      DEALLOCATE (weights)

      ! NewKS = U^{-1,T} * KS * U^{-1} + (U * C) * DELTA * (U * C)^T
      CALL parallel_gemm("N", "T", nao_red, nao_red, nmo, 1.0_dp, u_mo, u_mo_scaled, 1.0_dp, matrix_ks_fm)

      CALL cp_fm_release(u_mo_scaled)
      CALL cp_fm_release(u_mo)

      CALL timestop(handle)

   END SUBROUTINE shift_unocc_mos

END MODULE qs_scf_methods
