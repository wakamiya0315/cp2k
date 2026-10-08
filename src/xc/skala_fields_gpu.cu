/*----------------------------------------------------------------------------*/
/*  CP2K: A general program to perform molecular dynamics simulations         */
/*  Copyright 2000-2026 CP2K developers group <https://cp2k.org>              */
/*                                                                            */
/*  SPDX-License-Identifier: GPL-2.0-or-later                                 */
/*----------------------------------------------------------------------------*/

/*******************************************************************************
 * \brief One-centre fields of the native SKALA atom-composite route on the GPU
 *        (NATIVE_GRID_GPU_FIELDS):
 *        - the angular assembly of calc_rho_angular for all radial shells of
 *          an atom at once: rho = slm r^T and grad_d rho = a_d slm dr^T +
 *          slm r_d^T, with |grad rho| formed per point;
 *        - the cross-atom interpolation of the projected fields of a source
 *          atom at the composite points near its images (forward), and its
 *          transpose into adjoint projections (adjoint).
 *        The interpolation tables (radial nodes and weights, harmonics in the
 *        direction of every candidate point) are computed by the host, as the
 *        CPU path computes them, and stay on the device until released. Every
 *        sum runs in a fixed order and no atomic additions are used, so the
 *        results do not change from run to run. Arrays are column-major as in
 *        Fortran; indices passed from Fortran are converted to 0-based by the
 *        caller.
 ******************************************************************************/

#include "../offload/offload_library.h"
#include "../offload/offload_runtime.h"

#if defined(__OFFLOAD_CUDA)

#include <cublas_v2.h>
#include <math.h>
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
  int na;      // angular points of the kind's grid
  int niso;    // harmonics with non-zero density coefficients (max_iso_not0)
  double *slm; // device, na x niso
  double *a;   // device, 3 x na: directions of the angular points
} kind_harmonics;

// Interpolation table of the cross-atom candidates of one source atom. The
// candidates come in groups, one group per composite row, in list order.
typedef struct {
  int nharm, nr, ncand, ngroups, nentries;
  int *group_first; // ngroups + 1: first candidate of every group
  int *cand_group;  // ncand: group of every candidate
  int *nnode;       // ncand: radial nodes (0 for points outside the grid)
  int *node;        // 4 x ncand: radial shells
  double *radius;   // ncand: distance from the source image (bohr)
  double *weight;   // 4 x ncand: radial weights
  double *harm;     // nharm x ncand: harmonics in the direction of the point
  int *shell_first; // nr + 1: first entry of every radial shell
  int *entry_cand;  // nentries: candidate of the entry, ascending per shell
  double *entry_weight; // nentries: radial weight of the entry
} cross_table;

static cublasHandle_t handle = NULL;
static kind_harmonics kinds[MAX_KINDS];
static cross_table *tables = NULL; // indexed by source atom - 1
static int ntables = 0;
static double *work = NULL; // per-call inputs and outputs
static size_t work_count = 0;

/*******************************************************************************
 * \brief Grows the work buffer to hold at least 'count' doubles.
 ******************************************************************************/
static void ensure_work(const size_t count) {
  if (work != NULL && work_count >= count) {
    return;
  }
  if (work != NULL) {
    offloadFree(work);
  }
  offloadMalloc((void **)&work, count * sizeof(double));
  work_count = count;
}

static void activate(void) {
  offload_activate_chosen_device();
  if (handle == NULL) {
    CUBLAS_CHECK(cublasCreate(&handle));
  }
}

static void *upload(const void *host, const size_t bytes) {
  void *device = NULL;
  offloadMalloc(&device, bytes > 0 ? bytes : 1);
  if (bytes > 0) {
    offloadMemcpyHtoD(device, host, bytes);
  }
  return device;
}

static void free_table(cross_table *t) {
  void *arrays[] = {t->group_first, t->cand_group,  t->nnode,
                    t->node,        t->radius,      t->weight,
                    t->harm,        t->shell_first, t->entry_cand,
                    t->entry_weight};
  for (size_t i = 0; i < sizeof(arrays) / sizeof(arrays[0]); i++) {
    if (arrays[i] != NULL) {
      offloadFree(arrays[i]);
    }
  }
  *t = (cross_table){0};
}

/*******************************************************************************
 * \brief rho = x and grad_d rho = a_d dr + r_d on every grid point, with
 *        |grad rho| formed as calc_rho_angular forms it. x, dr and the three
 *        planes of r_d are the products slm r^T, slm dr^T and slm r_d^T.
 ******************************************************************************/
__global__ static void angular_fields(const int na, const int npoints,
                                      const double *__restrict__ a,
                                      const double *__restrict__ x,
                                      const double *__restrict__ dr,
                                      const double *__restrict__ r_d,
                                      double *__restrict__ rho,
                                      double *__restrict__ drho) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= npoints) {
    return;
  }
  const int ia = i % na;
  rho[i] = x[i];
  double square = 0.0;
  for (int d = 0; d < 3; d++) {
    const double value = a[3 * ia + d] * dr[i] + r_d[(size_t)d * npoints + i];
    drho[4 * (size_t)i + d] = value;
    square += value * value;
  }
  drho[4 * (size_t)i + 3] = sqrt(square);
}

/*******************************************************************************
 * \brief Forward interpolation: for every group (composite row), field and
 *        spin, the sum over the group's candidates within the cutoff of
 *        sum_n weight_n sum_h harm_h projection(h, node_n, field, spin).
 ******************************************************************************/
__global__ static void cross_forward(
    const int ngroups, const int nfield, const int nharm, const int nr,
    const double cutoff, const int *__restrict__ group_first,
    const double *__restrict__ radius, const int *__restrict__ nnode,
    const int *__restrict__ node, const double *__restrict__ weight,
    const double *__restrict__ harm, const double *__restrict__ projection,
    double *__restrict__ group_fields) {
  const int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= ngroups * nfield) {
    return;
  }
  const int g = t / nfield, k = t % nfield; // k = field + 5 * spin
  const double *p = projection + (size_t)k * nharm * nr;
  double sum = 0.0;
  for (int c = group_first[g]; c < group_first[g + 1]; c++) {
    if (radius[c] > cutoff) {
      continue;
    }
    const double *y = harm + (size_t)c * nharm;
    double field = 0.0;
    for (int n = 0; n < nnode[c]; n++) {
      const double *shell_projection = p + (size_t)node[4 * c + n] * nharm;
      double shell = 0.0;
      for (int h = 0; h < nharm; h++) {
        shell += y[h] * shell_projection[h];
      }
      field += weight[4 * c + n] * shell;
    }
    sum += field;
  }
  group_fields[t] = sum;
}

/*******************************************************************************
 * \brief Adjoint interpolation, one block per radial shell: adjoint(h, shell,
 *        field, spin) = sum over the shell's entries within the cutoff of
 *        weight * group_adjoint(field, spin, group) * harm_h, in entry order.
 ******************************************************************************/
__global__ static void cross_adjoint(
    const int nfield, const int nharm, const int nr, const double cutoff,
    const int *__restrict__ shell_first, const int *__restrict__ entry_cand,
    const double *__restrict__ entry_weight,
    const int *__restrict__ cand_group, const double *__restrict__ radius,
    const double *__restrict__ harm, const double *__restrict__ group_adjoint,
    double *__restrict__ adjoint) {
  const int ir = blockIdx.x;
  for (int t = threadIdx.x; t < nharm * nfield; t += blockDim.x) {
    const int h = t % nharm, k = t / nharm; // k = field + 5 * spin
    double sum = 0.0;
    for (int e = shell_first[ir]; e < shell_first[ir + 1]; e++) {
      const int c = entry_cand[e];
      if (radius[c] > cutoff) {
        continue;
      }
      sum += entry_weight[e] * group_adjoint[k + nfield * cand_group[c]] *
             harm[(size_t)c * nharm + h];
    }
    adjoint[h + (size_t)nharm * (ir + (size_t)nr * k)] = sum;
  }
}

extern "C" {

/*******************************************************************************
 * \brief Makes slm (na x niso) and the directions a (3 x na) of kind 'kind'
 *        (1-based) available on the device, uploading them unless the kind is
 *        there with the same sizes.
 ******************************************************************************/
void skala_fields_gpu_set_kind(const int kind, const int na, const int niso,
                               const double *slm, const double *a) {
  activate();
  if (kind < 1 || kind > MAX_KINDS) {
    fprintf(stderr, "skala_fields_gpu: kind %d out of range\n", kind);
    abort();
  }
  kind_harmonics *k = &kinds[kind - 1];
  if (k->slm != NULL && k->na == na && k->niso == niso) {
    return;
  }
  if (k->slm != NULL) {
    offloadFree(k->slm);
    offloadFree(k->a);
  }
  k->slm = (double *)upload(slm, (size_t)na * niso * sizeof(double));
  k->a = (double *)upload(a, (size_t)3 * na * sizeof(double));
  k->na = na;
  k->niso = niso;
}

/*******************************************************************************
 * \brief Angular assembly of one atom and spin. 'radial' holds ten nr x niso
 *        blocks: r_h, r_s, dr_h, dr_s, r_h_d(1:3), r_s_d(1:3). Outputs rho_h,
 *        rho_s (na x nr) and drho_h, drho_s (4 x na x nr).
 ******************************************************************************/
void skala_fields_gpu_angular(const int kind, const int nr,
                              const double *radial, double *rho_h,
                              double *rho_s, double *drho_h, double *drho_s) {
  activate();
  const kind_harmonics *k = &kinds[kind - 1];
  const int na = k->na, niso = k->niso;
  const size_t rblock = (size_t)nr * niso, grid = (size_t)na * nr;
  // work: the ten radial blocks, their ten products, then rho and drho
  ensure_work(10 * rblock + 10 * grid + 5 * grid);
  double *radial_dev = work, *x_dev = work + 10 * rblock;
  double *rho_dev = x_dev + 10 * grid, *drho_dev = rho_dev + grid;
  offloadMemcpyHtoD(radial_dev, radial, 10 * rblock * sizeof(double));
  const double one = 1.0, zero = 0.0;
  CUBLAS_CHECK(cublasDgemmStridedBatched(
      handle, CUBLAS_OP_N, CUBLAS_OP_T, na, nr, niso, &one, k->slm, na, 0,
      radial_dev, nr, (long long)rblock, &zero, x_dev, na, (long long)grid,
      10));
  // Products in the order of 'radial': 0 r_h, 1 r_s, 2 dr_h, 3 dr_s,
  // 4-6 r_h_d, 7-9 r_s_d; hs = 0 is the hard set, hs = 1 the soft one.
  double *rho_out[2] = {rho_h, rho_s}, *drho_out[2] = {drho_h, drho_s};
  for (int hs = 0; hs < 2; hs++) {
    angular_fields<<<(unsigned int)((grid + 255) / 256), 256>>>(
        na, (int)grid, k->a, x_dev + hs * grid, x_dev + (2 + hs) * grid,
        x_dev + (4 + 3 * hs) * grid, rho_dev, drho_dev);
    OFFLOAD_CHECK(offloadGetLastError());
    offloadMemcpyDtoH(rho_out[hs], rho_dev, grid * sizeof(double));
    offloadMemcpyDtoH(drho_out[hs], drho_dev, 4 * grid * sizeof(double));
  }
}

/*******************************************************************************
 * \brief Stores the interpolation table of source atom 'source' (1-based),
 *        replacing an earlier one. Index arrays are 0-based.
 ******************************************************************************/
void skala_fields_gpu_set_cross(
    const int source, const int nharm, const int nr, const int ncand,
    const int ngroups, const int *group_first, const int *cand_group,
    const int *nnode, const int *node, const double *radius,
    const double *weight, const double *harm, const int nentries,
    const int *shell_first, const int *entry_cand,
    const double *entry_weight) {
  activate();
  if (source < 1) {
    fprintf(stderr, "skala_fields_gpu: source atom %d out of range\n", source);
    abort();
  }
  if (source > ntables) {
    tables = (cross_table *)realloc(tables, source * sizeof(cross_table));
    for (int i = ntables; i < source; i++) {
      tables[i] = (cross_table){0};
    }
    ntables = source;
  }
  cross_table *t = &tables[source - 1];
  free_table(t);
  t->nharm = nharm;
  t->nr = nr;
  t->ncand = ncand;
  t->ngroups = ngroups;
  t->nentries = nentries;
  t->group_first = (int *)upload(group_first, (ngroups + 1) * sizeof(int));
  t->cand_group = (int *)upload(cand_group, ncand * sizeof(int));
  t->nnode = (int *)upload(nnode, ncand * sizeof(int));
  t->node = (int *)upload(node, 4 * (size_t)ncand * sizeof(int));
  t->radius = (double *)upload(radius, ncand * sizeof(double));
  t->weight = (double *)upload(weight, 4 * (size_t)ncand * sizeof(double));
  t->harm =
      (double *)upload(harm, (size_t)nharm * ncand * sizeof(double));
  t->shell_first = (int *)upload(shell_first, (nr + 1) * sizeof(int));
  t->entry_cand = (int *)upload(entry_cand, nentries * sizeof(int));
  t->entry_weight =
      (double *)upload(entry_weight, nentries * sizeof(double));
}

/*******************************************************************************
 * \brief Forward interpolation of the projections (nharm x nr x 5 x nspins)
 *        of source atom 'source' within 'cutoff' (bohr); group_fields is
 *        5 x nspins x ngroups.
 ******************************************************************************/
void skala_fields_gpu_cross_forward(const int source, const int nspins,
                                    const double cutoff,
                                    const double *projection,
                                    double *group_fields) {
  activate();
  const cross_table *t = &tables[source - 1];
  const int nfield = 5 * nspins;
  const size_t pcount = (size_t)t->nharm * t->nr * nfield;
  const size_t gcount = (size_t)nfield * t->ngroups;
  ensure_work(pcount + gcount);
  offloadMemcpyHtoD(work, projection, pcount * sizeof(double));
  if (gcount > 0) {
    cross_forward<<<(unsigned int)((gcount + 127) / 128), 128>>>(
        t->ngroups, nfield, t->nharm, t->nr, cutoff, t->group_first,
        t->radius, t->nnode, t->node, t->weight, t->harm, work,
        work + pcount);
    OFFLOAD_CHECK(offloadGetLastError());
    offloadMemcpyDtoH(group_fields, work + pcount, gcount * sizeof(double));
  }
}

/*******************************************************************************
 * \brief Adjoint interpolation of source atom 'source' within 'cutoff'
 *        (bohr): from group_adjoint (5 x nspins x ngroups) to the adjoint
 *        projections (nharm x nr x 5 x nspins).
 ******************************************************************************/
void skala_fields_gpu_cross_adjoint(const int source, const int nspins,
                                    const double cutoff,
                                    const double *group_adjoint,
                                    double *adjoint) {
  activate();
  const cross_table *t = &tables[source - 1];
  const int nfield = 5 * nspins;
  const size_t gcount = (size_t)nfield * t->ngroups;
  const size_t acount = (size_t)t->nharm * t->nr * nfield;
  ensure_work(gcount + acount);
  if (gcount > 0) {
    offloadMemcpyHtoD(work, group_adjoint, gcount * sizeof(double));
  }
  int threads = t->nharm * nfield;
  threads = threads > 512 ? 512 : threads;
  threads = ((threads + 31) / 32) * 32;
  cross_adjoint<<<t->nr, threads>>>(nfield, t->nharm, t->nr, cutoff,
                                    t->shell_first, t->entry_cand,
                                    t->entry_weight, t->cand_group, t->radius,
                                    t->harm, work, work + gcount);
  OFFLOAD_CHECK(offloadGetLastError());
  offloadMemcpyDtoH(adjoint, work + gcount, acount * sizeof(double));
}

/*******************************************************************************
 * \brief Frees all device memory and the cuBLAS handle; does nothing if the
 *        GPU fields were not used.
 ******************************************************************************/
void skala_fields_gpu_release(void) {
  int used = handle != NULL || work != NULL || tables != NULL;
  for (int i = 0; i < MAX_KINDS; i++) {
    used = used || kinds[i].slm != NULL;
  }
  if (!used) {
    return;
  }
  offload_activate_chosen_device();
  for (int i = 0; i < MAX_KINDS; i++) {
    if (kinds[i].slm != NULL) {
      offloadFree(kinds[i].slm);
      offloadFree(kinds[i].a);
    }
    kinds[i] = (kind_harmonics){0};
  }
  for (int i = 0; i < ntables; i++) {
    free_table(&tables[i]);
  }
  free(tables);
  tables = NULL;
  ntables = 0;
  if (work != NULL) {
    offloadFree(work);
  }
  work = NULL;
  work_count = 0;
  if (handle != NULL) {
    CUBLAS_CHECK(cublasDestroy(handle));
  }
  handle = NULL;
}

} // extern "C"

#endif // __OFFLOAD_CUDA
