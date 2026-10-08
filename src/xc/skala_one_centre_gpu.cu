/*----------------------------------------------------------------------------*/
/*  CP2K: A general program to perform molecular dynamics simulations         */
/*  Copyright 2000-2026 CP2K developers group <https://cp2k.org>              */
/*                                                                            */
/*  SPDX-License-Identifier: GPL-2.0-or-later                                 */
/*----------------------------------------------------------------------------*/

/*******************************************************************************
 * \brief One-centre matrices of the native SKALA atom-composite route on the
 *        GPU (NATIVE_GRID_GPU_ONE_CENTRE), with cuBLAS:
 *        - kinetic-energy density tau = 1/2 sum_d sum_ij P_ij d_d phi_i
 *          d_d phi_j on all grid points of an atom (calc_tau_atom);
 *        - its Kohn-Sham matrix 1/2 sum_d (d_d phi)^T diag(v_tau) d_d phi
 *          (dgaVtaudgb);
 *        - the radial integrals of gaVxcgb_GC and their projections on the
 *          real spherical harmonics, from which the host assembles the
 *          matrices with the Clebsch-Gordan coefficients.
 *        The basis gradients of each kind (tau_basis_cache%grad, column-major
 *        (points, basis functions, direction)) stay on the device until
 *        released, since they depend only on the kind's grid and basis.
 *        Matrices are column-major as in Fortran.
 ******************************************************************************/

#include "../offload/offload_library.h"
#include "../offload/offload_runtime.h"

#if defined(__OFFLOAD_CUDA)

#include <cublas_v2.h>
#include <stdio.h>
#include <stdlib.h>

#define MAX_KINDS 64

#define CUBLAS_CHECK(call)                                                     \
  do {                                                                         \
    const cublasStatus_t status = (call);                                      \
    if (status != CUBLAS_STATUS_SUCCESS) {                                     \
      fprintf(stderr, "cuBLAS error %d at %s:%d\n", (int)status, __FILE__,     \
              __LINE__);                                                       \
      abort();                                                                 \
    }                                                                          \
  } while (0)

typedef struct {
  int ngrid; // grid points of the kind (radial x angular)
  int nbas;  // basis functions (tau_basis_cache%nsatbas)
  double *grad; // device, ngrid x nbas x 3
} kind_gradients;

// Device buffers that grow as needed; the handle is created at first use.
static cublasHandle_t handle = NULL;
static kind_gradients kinds[MAX_KINDS];
static double *work = NULL;  // 3 x ngrid x nbas, or the projection inputs
static size_t work_count = 0;
static double *small = NULL; // matrices, v_tau, tau and projection results
static size_t small_count = 0;

/*******************************************************************************
 * \brief Grows a device buffer to hold at least 'count' doubles.
 ******************************************************************************/
static void ensure(double **buffer, size_t *capacity, const size_t count) {
  if (*buffer != NULL && *capacity >= count) {
    return;
  }
  if (*buffer != NULL) {
    offloadFree(*buffer);
  }
  offloadMalloc((void **)buffer, count * sizeof(double));
  *capacity = count;
}

static void activate(void) {
  offload_activate_chosen_device();
  if (handle == NULL) {
    CUBLAS_CHECK(cublasCreate(&handle));
  }
}

/*******************************************************************************
 * \brief tau[g] = sum_d sum_b grad_d[g, b] * product_d[g, b], directions and
 *        basis functions in the order of calc_tau_atom.
 ******************************************************************************/
__global__ static void tau_from_products(const int ngrid, const int nbas,
                                         const double *__restrict__ grad,
                                         const double *__restrict__ product,
                                         double *__restrict__ tau) {
  const int g = blockIdx.x * blockDim.x + threadIdx.x;
  if (g >= ngrid) {
    return;
  }
  const size_t plane = (size_t)ngrid * nbas;
  double sum = 0.0;
  for (int d = 0; d < 3; d++) {
    for (int b = 0; b < nbas; b++) {
      const size_t i = d * plane + (size_t)b * ngrid + g;
      sum += grad[i] * product[i];
    }
  }
  tau[g] = sum;
}

/*******************************************************************************
 * \brief weighted[g, b, d] = v_tau[g] * grad[g, b, d].
 ******************************************************************************/
__global__ static void weight_gradients(const size_t count, const int ngrid,
                                        const double *__restrict__ v_tau,
                                        const double *__restrict__ grad,
                                        double *__restrict__ weighted) {
  const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count) {
    return;
  }
  weighted[i] = v_tau[i % ngrid] * grad[i];
}

extern "C" {

/*******************************************************************************
 * \brief Makes the basis gradients of kind 'kind' (1-based) available on the
 *        device, uploading them unless the same kind with the same sizes is
 *        already there. Returns 1 if uploaded, 0 if reused.
 ******************************************************************************/
int skala_one_centre_gpu_set_gradients(const int kind, const int ngrid,
                                       const int nbas, const double *grad) {
  activate();
  if (kind < 1 || kind > MAX_KINDS) {
    fprintf(stderr, "skala_one_centre_gpu: kind %d out of range\n", kind);
    abort();
  }
  kind_gradients *k = &kinds[kind - 1];
  if (k->grad != NULL && k->ngrid == ngrid && k->nbas == nbas) {
    return 0;
  }
  if (k->grad != NULL) {
    offloadFree(k->grad);
  }
  const size_t count = (size_t)ngrid * nbas * 3;
  offloadMalloc((void **)&k->grad, count * sizeof(double));
  offloadMemcpyHtoD(k->grad, grad, count * sizeof(double));
  k->ngrid = ngrid;
  k->nbas = nbas;
  return 1;
}

/*******************************************************************************
 * \brief tau on all grid points of kind 'kind' for the density matrix
 *        'matrix' (nbas x nbas): product_d = 1/2 grad_d matrix^T, then
 *        tau = sum_d sum_b grad_d .* product_d, as calc_tau_atom.
 ******************************************************************************/
void skala_one_centre_gpu_tau(const int kind, const double *matrix,
                              double *host_tau) {
  activate();
  const kind_gradients *k = &kinds[kind - 1];
  const int ngrid = k->ngrid, nbas = k->nbas;
  const size_t plane = (size_t)ngrid * nbas;
  ensure(&work, &work_count, 3 * plane);
  ensure(&small, &small_count, (size_t)nbas * nbas + ngrid);
  double *matrix_dev = small;
  double *tau_dev = small + (size_t)nbas * nbas;
  offloadMemcpyHtoD(matrix_dev, matrix, (size_t)nbas * nbas * sizeof(double));
  const double half = 0.5, zero = 0.0;
  for (int d = 0; d < 3; d++) {
    CUBLAS_CHECK(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, ngrid, nbas,
                             nbas, &half, k->grad + d * plane, ngrid,
                             matrix_dev, nbas, &zero, work + d * plane, ngrid));
  }
  tau_from_products<<<(ngrid + 255) / 256, 256>>>(ngrid, nbas, k->grad, work,
                                                  tau_dev);
  OFFLOAD_CHECK(offloadGetLastError());
  offloadMemcpyDtoH(host_tau, tau_dev, (size_t)ngrid * sizeof(double));
}

/*******************************************************************************
 * \brief Kohn-Sham matrix of v_tau on kind 'kind': 1/2 sum_d grad_d^T
 *        diag(v_tau) grad_d (nbas x nbas), as dgaVtaudgb before the mapping
 *        to the one-centre basis.
 ******************************************************************************/
void skala_one_centre_gpu_vtau(const int kind, const double *host_v_tau,
                               double *host_matrix) {
  activate();
  const kind_gradients *k = &kinds[kind - 1];
  const int ngrid = k->ngrid, nbas = k->nbas;
  const size_t plane = (size_t)ngrid * nbas;
  ensure(&work, &work_count, 3 * plane);
  ensure(&small, &small_count, (size_t)nbas * nbas + ngrid);
  double *matrix_dev = small;
  double *v_tau_dev = small + (size_t)nbas * nbas;
  offloadMemcpyHtoD(v_tau_dev, host_v_tau, (size_t)ngrid * sizeof(double));
  const size_t count = 3 * plane;
  weight_gradients<<<(unsigned int)((count + 255) / 256), 256>>>(
      count, ngrid, v_tau_dev, k->grad, work);
  OFFLOAD_CHECK(offloadGetLastError());
  const double half = 0.5, zero = 0.0, one = 1.0;
  for (int d = 0; d < 3; d++) {
    CUBLAS_CHECK(cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, nbas, nbas,
                             ngrid, &half, k->grad + d * plane, ngrid,
                             work + d * plane, ngrid, d == 0 ? &zero : &one,
                             matrix_dev, nbas));
  }
  offloadMemcpyDtoH(host_matrix, matrix_dev,
                    (size_t)nbas * nbas * sizeof(double));
}

/*******************************************************************************
 * \brief Radial integrals of gaVxcgb_GC and their angular projections:
 *          xc   = vxc radial + vxc_grad radial_d        (na x ncol)
 *          xg_d = vxg_d radial_u, d = 1, 2, 3           (na x ncol)
 *          P_0  = slm^T xc,  P_d = slm^T xg_d           (niso x ncol)
 *        Inputs: vxc, vxc_grad (na x nr), vxg (na x nr x 3), radial,
 *        radial_d, radial_u (nr x ncol), slm (na x niso). Output: P
 *        (niso x ncol x 4).
 ******************************************************************************/
void skala_one_centre_gpu_projections(
    const int na, const int nr, const int ncol, const int niso,
    const double *vxc, const double *vxc_grad, const double *vxg,
    const double *radial, const double *radial_d, const double *radial_u,
    const double *slm, double *host_projections) {
  activate();
  const size_t grid = (size_t)na * nr, cols = (size_t)nr * ncol;
  const size_t xcount = (size_t)na * ncol, pcount = (size_t)niso * ncol;
  // work: vxc, vxc_grad, vxg (5 grids), radial x 3, slm, xc and xg (4)
  ensure(&work, &work_count,
         5 * grid + 3 * cols + (size_t)na * niso + 4 * xcount);
  ensure(&small, &small_count, 4 * pcount);
  double *vxc_dev = work, *vxc_grad_dev = vxc_dev + grid;
  double *vxg_dev = vxc_grad_dev + grid;
  double *radial_dev = vxg_dev + 3 * grid, *radial_d_dev = radial_dev + cols;
  double *radial_u_dev = radial_d_dev + cols;
  double *slm_dev = radial_u_dev + cols;
  double *x_dev = slm_dev + (size_t)na * niso; // xc, xg_1, xg_2, xg_3
  offloadMemcpyHtoD(vxc_dev, vxc, grid * sizeof(double));
  offloadMemcpyHtoD(vxc_grad_dev, vxc_grad, grid * sizeof(double));
  offloadMemcpyHtoD(vxg_dev, vxg, 3 * grid * sizeof(double));
  offloadMemcpyHtoD(radial_dev, radial, cols * sizeof(double));
  offloadMemcpyHtoD(radial_d_dev, radial_d, cols * sizeof(double));
  offloadMemcpyHtoD(radial_u_dev, radial_u, cols * sizeof(double));
  offloadMemcpyHtoD(slm_dev, slm, (size_t)na * niso * sizeof(double));
  const double one = 1.0, zero = 0.0;
  CUBLAS_CHECK(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, na, ncol, nr,
                           &one, vxc_dev, na, radial_dev, nr, &zero, x_dev,
                           na));
  CUBLAS_CHECK(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, na, ncol, nr,
                           &one, vxc_grad_dev, na, radial_d_dev, nr, &one,
                           x_dev, na));
  for (int d = 0; d < 3; d++) {
    CUBLAS_CHECK(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, na, ncol, nr,
                             &one, vxg_dev + d * grid, na, radial_u_dev, nr,
                             &zero, x_dev + (d + 1) * xcount, na));
  }
  for (int k = 0; k < 4; k++) {
    CUBLAS_CHECK(cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, niso, ncol, na,
                             &one, slm_dev, na, x_dev + k * xcount, na, &zero,
                             small + k * pcount, niso));
  }
  offloadMemcpyDtoH(host_projections, small, 4 * pcount * sizeof(double));
}

/*******************************************************************************
 * \brief Frees the device memory and the cuBLAS handle; does nothing if the
 *        GPU one-centre matrices were not used.
 ******************************************************************************/
void skala_one_centre_gpu_release(void) {
  int used = handle != NULL || work != NULL || small != NULL;
  for (int i = 0; i < MAX_KINDS; i++) {
    used = used || kinds[i].grad != NULL;
  }
  if (!used) {
    return;
  }
  offload_activate_chosen_device();
  for (int i = 0; i < MAX_KINDS; i++) {
    if (kinds[i].grad != NULL) {
      offloadFree(kinds[i].grad);
    }
    kinds[i].grad = NULL;
    kinds[i].ngrid = 0;
    kinds[i].nbas = 0;
  }
  if (work != NULL) {
    offloadFree(work);
  }
  if (small != NULL) {
    offloadFree(small);
  }
  work = NULL;
  small = NULL;
  work_count = 0;
  small_count = 0;
  if (handle != NULL) {
    CUBLAS_CHECK(cublasDestroy(handle));
  }
  handle = NULL;
}

} // extern "C"

#endif // __OFFLOAD_CUDA
