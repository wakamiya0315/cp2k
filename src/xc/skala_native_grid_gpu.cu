/*----------------------------------------------------------------------------*/
/*  CP2K: A general program to perform molecular dynamics simulations         */
/*  Copyright 2000-2026 CP2K developers group <https://cp2k.org>              */
/*                                                                            */
/*  SPDX-License-Identifier: GPL-2.0-or-later                                 */
/*----------------------------------------------------------------------------*/

/*******************************************************************************
 * \brief Plane-wave stencil of the native SKALA atom-composite route on the GPU
 *        (NATIVE_GRID_GPU_STENCIL): interpolation of the smooth density,
 *        gradient and kinetic-energy density at the atom-grid rows, and the
 *        adjoint scatter back to the plane-wave grid.
 *
 *        The stencils are made on the host by
 *        create_native_grid_interpolation_stencil (qs_vxc_atom.F) and uploaded
 *        once per geometry: for each row the zero-based grid index of its first
 *        node in each direction and the 12 x 3 Lagrange weights. The GPU uses
 *        the same nodes and weights as the CPU path; only the order of the sums
 *        differs. The adjoint adds with atomicAdd, so its sums are not
 *        reproducible from run to run in the last digits.
 *
 *        Fields on the device: field[(channel*5 + component)*npts + point] with
 *        components density, gradient x, y, z and kinetic-energy density, and
 *        point = i + n1*(j + n2*k) as in add_native_grid_fields_adjoint_tile.
 *        Row values: value[(row*nchannels + channel)*5 + component], rows in
 *        the order of their upload.
 ******************************************************************************/

#include "../offload/offload_library.h"
#include "../offload/offload_runtime.h"

#if defined(__OFFLOAD_CUDA)

#define STENCIL_NODES 12
#define FIELD_COMPONENTS 5
#define WARP 32
#define WARPS_PER_BLOCK 8

typedef struct {
  int npts[3];
  int wrap[3];
  int nchannels;
  int nrows;
  size_t field_capacity;   // doubles allocated for fields and for adjoints
  size_t row_capacity;     // rows allocated for stencils and row values
  double *fields;          // device
  double *adjoint;         // device, same layout as fields
  double *row_values;      // device
  int *first_node;         // device, [row][3]
  double *weights;         // device, [row][3][12]
  int *active;             // device, [row]
} native_grid_gpu_state;

static native_grid_gpu_state state = {{0, 0, 0}, {0, 0, 0}, 0, 0, 0, 0,
                                      NULL, NULL, NULL, NULL, NULL, NULL};

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 600
/*******************************************************************************
 * \brief atomicAdd for doubles on devices before compute capability 6.0, which
 *        lack it (the build also compiles for older targets); as in
 *        grid/gpu/grid_gpu_internal_header.h.
 ******************************************************************************/
__device__ static inline double atomicAdd(double *address, double val) {
  unsigned long long int *address_as_ull = (unsigned long long int *)address;
  unsigned long long int old = *address_as_ull, assumed;
  do {
    assumed = old;
    old = atomicCAS(address_as_ull, assumed,
                    __double_as_longlong(val + __longlong_as_double(assumed)));
  } while (assumed != old); // integer comparison, so NaN cannot hang the loop
  return __longlong_as_double(old);
}
#endif

/*******************************************************************************
 * \brief Grid index of node 'node' of a stencil whose first node is 'first'
 *        along one direction, or -1 outside a direction that does not wrap.
 ******************************************************************************/
__device__ static inline int node_index(const int first, const int node,
                                        const int npts, const int wrap) {
  int index = first + node;
  if (wrap) {
    index %= npts; // first is already in [0, npts) for wrapped directions
  } else if (index < 0 || index >= npts) {
    index = -1;
  }
  return index;
}

/*******************************************************************************
 * \brief Interpolation, one warp per row: each lane sums over its share of the
 *        12^3 nodes, and the warp adds the partial sums.
 ******************************************************************************/
__global__ static void interpolate_rows(
    const int nrows, const int nchannels, const int n1, const int n2,
    const int n3, const int wrap1, const int wrap2, const int wrap3,
    const int *__restrict__ first_node, const double *__restrict__ weights,
    const int *__restrict__ active, const double *__restrict__ fields,
    double *__restrict__ row_values) {
  const int row = blockIdx.x * WARPS_PER_BLOCK + threadIdx.x / WARP;
  const int lane = threadIdx.x % WARP;
  if (row >= nrows) {
    return;
  }
  const size_t npts = (size_t)n1 * n2 * n3;
  const int *first = first_node + 3 * row;
  const double *w = weights + 3 * STENCIL_NODES * row;
  double sum[2 * FIELD_COMPONENTS];
  for (int k = 0; k < 2 * FIELD_COMPONENTS; k++) {
    sum[k] = 0.0;
  }
  if (active[row]) {
    for (int node = lane; node < STENCIL_NODES * STENCIL_NODES * STENCIL_NODES;
         node += WARP) {
      const int ix = node % STENCIL_NODES;
      const int iy = (node / STENCIL_NODES) % STENCIL_NODES;
      const int iz = node / (STENCIL_NODES * STENCIL_NODES);
      const int gx = node_index(first[0], ix, n1, wrap1);
      const int gy = node_index(first[1], iy, n2, wrap2);
      const int gz = node_index(first[2], iz, n3, wrap3);
      if (gx < 0 || gy < 0 || gz < 0) {
        continue;
      }
      // The same product order as interpolate_native_grid_fields.
      const double coefficient = w[ix] * w[STENCIL_NODES + iy] *
                                 w[2 * STENCIL_NODES + iz];
      const size_t point = gx + (size_t)n1 * (gy + (size_t)n2 * gz);
      for (int channel = 0; channel < nchannels; channel++) {
        for (int c = 0; c < FIELD_COMPONENTS; c++) {
          sum[channel * FIELD_COMPONENTS + c] +=
              coefficient *
              fields[(channel * FIELD_COMPONENTS + c) * npts + point];
        }
      }
    }
  }
  for (int k = 0; k < nchannels * FIELD_COMPONENTS; k++) {
    for (int offset = WARP / 2; offset > 0; offset /= 2) {
      sum[k] += __shfl_down_sync(0xffffffff, sum[k], offset);
    }
  }
  if (lane == 0) {
    for (int k = 0; k < nchannels * FIELD_COMPONENTS; k++) {
      row_values[(size_t)row * nchannels * FIELD_COMPONENTS + k] = sum[k];
    }
  }
}

/*******************************************************************************
 * \brief Adjoint of the interpolation, one warp per row: every node of the
 *        row's stencil receives coefficient x the row's adjoint values.
 ******************************************************************************/
__global__ static void scatter_rows(
    const int nrows, const int nchannels, const int n1, const int n2,
    const int n3, const int wrap1, const int wrap2, const int wrap3,
    const int *__restrict__ first_node, const double *__restrict__ weights,
    const int *__restrict__ active, const double *__restrict__ row_values,
    double *adjoint) {
  const int row = blockIdx.x * WARPS_PER_BLOCK + threadIdx.x / WARP;
  const int lane = threadIdx.x % WARP;
  if (row >= nrows || !active[row]) {
    return;
  }
  const size_t npts = (size_t)n1 * n2 * n3;
  const int *first = first_node + 3 * row;
  const double *w = weights + 3 * STENCIL_NODES * row;
  double value[2 * FIELD_COMPONENTS];
  for (int k = 0; k < nchannels * FIELD_COMPONENTS; k++) {
    value[k] = row_values[(size_t)row * nchannels * FIELD_COMPONENTS + k];
  }
  for (int node = lane; node < STENCIL_NODES * STENCIL_NODES * STENCIL_NODES;
       node += WARP) {
    const int ix = node % STENCIL_NODES;
    const int iy = (node / STENCIL_NODES) % STENCIL_NODES;
    const int iz = node / (STENCIL_NODES * STENCIL_NODES);
    const int gx = node_index(first[0], ix, n1, wrap1);
    const int gy = node_index(first[1], iy, n2, wrap2);
    const int gz = node_index(first[2], iz, n3, wrap3);
    if (gx < 0 || gy < 0 || gz < 0) {
      continue;
    }
    const double coefficient =
        w[ix] * w[STENCIL_NODES + iy] * w[2 * STENCIL_NODES + iz];
    const size_t point = gx + (size_t)n1 * (gy + (size_t)n2 * gz);
    for (int k = 0; k < nchannels * FIELD_COMPONENTS; k++) {
      atomicAdd(&adjoint[(size_t)k * npts + point], coefficient * value[k]);
    }
  }
}

/*******************************************************************************
 * \brief Grows a device buffer to hold at least 'count' elements of 'size'.
 ******************************************************************************/
static void ensure_capacity(void **buffer, const size_t old_count,
                            const size_t count, const size_t size) {
  if (*buffer != NULL && old_count >= count) {
    return;
  }
  if (*buffer != NULL) {
    offloadFree(*buffer);
  }
  offloadMalloc(buffer, count * size);
}

extern "C" {

/*******************************************************************************
 * \brief Sets the plane-wave grid and the number of spin channels, and makes
 *        room for the fields and the adjoint on the device.
 ******************************************************************************/
void skala_native_grid_gpu_set_grid(const int *npts, const int *wrap,
                                    const int nchannels) {
  offload_activate_chosen_device();
  const size_t count =
      (size_t)npts[0] * npts[1] * npts[2] * nchannels * FIELD_COMPONENTS;
  ensure_capacity((void **)&state.fields, state.field_capacity, count,
                  sizeof(double));
  ensure_capacity((void **)&state.adjoint, state.field_capacity, count,
                  sizeof(double));
  if (count > state.field_capacity) {
    state.field_capacity = count;
  }
  for (int d = 0; d < 3; d++) {
    state.npts[d] = npts[d];
    state.wrap[d] = wrap[d];
  }
  state.nchannels = nchannels;
}

/*******************************************************************************
 * \brief Uploads one plane-wave field (component 0 density, 1-3 gradient,
 *        4 kinetic-energy density) of one spin channel.
 ******************************************************************************/
void skala_native_grid_gpu_upload_field(const int channel, const int component,
                                        const double *host_field) {
  offload_activate_chosen_device();
  const size_t npts = (size_t)state.npts[0] * state.npts[1] * state.npts[2];
  offloadMemcpyHtoD(
      state.fields + (channel * FIELD_COMPONENTS + component) * npts,
      host_field, npts * sizeof(double));
}

/*******************************************************************************
 * \brief Uploads the stencils of 'nrows' rows: first node (3 per row), weights
 *        (12 x 3 per row) and whether the stencil is active (1) or not (0).
 ******************************************************************************/
void skala_native_grid_gpu_set_rows(const int nrows, const int *first_node,
                                    const double *weights, const int *active) {
  offload_activate_chosen_device();
  const size_t rows = (size_t)nrows;
  ensure_capacity((void **)&state.first_node, state.row_capacity, rows,
                  3 * sizeof(int));
  ensure_capacity((void **)&state.weights, state.row_capacity, rows,
                  3 * STENCIL_NODES * sizeof(double));
  ensure_capacity((void **)&state.active, state.row_capacity, rows,
                  sizeof(int));
  ensure_capacity((void **)&state.row_values, state.row_capacity, rows,
                  2 * FIELD_COMPONENTS * sizeof(double));
  if (rows > state.row_capacity) {
    state.row_capacity = rows;
  }
  offloadMemcpyHtoD(state.first_node, first_node, rows * 3 * sizeof(int));
  offloadMemcpyHtoD(state.weights, weights,
                    rows * 3 * STENCIL_NODES * sizeof(double));
  offloadMemcpyHtoD(state.active, active, rows * sizeof(int));
  state.nrows = nrows;
}

/*******************************************************************************
 * \brief Interpolates the uploaded fields at all rows and copies the values
 *        (5 per channel and row) to 'host_values'.
 ******************************************************************************/
void skala_native_grid_gpu_interpolate(double *host_values) {
  offload_activate_chosen_device();
  if (state.nrows == 0) {
    return;
  }
  const int blocks = (state.nrows + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
  interpolate_rows<<<blocks, WARPS_PER_BLOCK * WARP>>>(
      state.nrows, state.nchannels, state.npts[0], state.npts[1],
      state.npts[2], state.wrap[0], state.wrap[1], state.wrap[2],
      state.first_node, state.weights, state.active, state.fields,
      state.row_values);
  OFFLOAD_CHECK(offloadGetLastError());
  offloadMemcpyDtoH(host_values, state.row_values,
                    (size_t)state.nrows * state.nchannels * FIELD_COMPONENTS *
                        sizeof(double));
}

/*******************************************************************************
 * \brief Scatters the adjoint values of all rows (5 per channel and row, same
 *        layout as the interpolated values) to the plane-wave grid and copies
 *        the result, laid out as the fields, to 'host_adjoint'.
 ******************************************************************************/
void skala_native_grid_gpu_adjoint(const double *host_row_values,
                                   double *host_adjoint) {
  offload_activate_chosen_device();
  const size_t count = (size_t)state.npts[0] * state.npts[1] * state.npts[2] *
                       state.nchannels * FIELD_COMPONENTS;
  offloadMemset(state.adjoint, 0, count * sizeof(double));
  if (state.nrows > 0) {
    offloadMemcpyHtoD(state.row_values, host_row_values,
                      (size_t)state.nrows * state.nchannels * FIELD_COMPONENTS *
                          sizeof(double));
    const int blocks = (state.nrows + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    scatter_rows<<<blocks, WARPS_PER_BLOCK * WARP>>>(
        state.nrows, state.nchannels, state.npts[0], state.npts[1],
        state.npts[2], state.wrap[0], state.wrap[1], state.wrap[2],
        state.first_node, state.weights, state.active, state.row_values,
        state.adjoint);
    OFFLOAD_CHECK(offloadGetLastError());
  }
  offloadMemcpyDtoH(host_adjoint, state.adjoint, count * sizeof(double));
}

/*******************************************************************************
 * \brief Frees the device memory.
 ******************************************************************************/
void skala_native_grid_gpu_release(void) {
  void *buffers[] = {state.fields,     state.adjoint, state.row_values,
                     state.first_node, state.weights, state.active};
  int allocated = 0;
  for (int i = 0; i < 6; i++) {
    allocated = allocated || buffers[i] != NULL;
  }
  if (!allocated) {
    return; // the GPU stencil was not used; do not touch the device
  }
  offload_activate_chosen_device();
  for (int i = 0; i < 6; i++) {
    if (buffers[i] != NULL) {
      offloadFree(buffers[i]);
    }
  }
  native_grid_gpu_state empty = {{0, 0, 0}, {0, 0, 0}, 0, 0, 0, 0,
                                 NULL, NULL, NULL, NULL, NULL, NULL};
  state = empty;
}

} // extern "C"

#endif // __OFFLOAD_CUDA
