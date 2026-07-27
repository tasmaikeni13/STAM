// STAM fused CUDA kernels.
//
// Four operations dominate the cost of certified landscape probing, and all four are
// memory-bound streaming reductions or broadcasts over the parameter vector:
//
//   plane_point   theta = theta0 + sum_i c_i v_i          (write a probe point)
//   project       s_i   = <g, v_i>  for all i, one pass   (read a gradient/HVP out)
//   gram_chunk    G    += D_chunk D_chunk^T               (trajectory basis, streaming)
//   pu_taylor     partition-of-unity Taylor reconstruction on the render grid
//
// The first three are what makes the per-anchor fixed overhead tau small; tau enters
// the optimal experiment design directly, so this file is part of the method, not an
// optimisation afterthought.
//
// Architecture adaptivity is handled in three places:
//   * compile time -- __CUDA_ARCH__ guards select warp primitives and staging paths;
//   * link time    -- the extension is built for exactly the architectures present;
//   * run time     -- the host wrappers read cudaDeviceProp and choose block counts,
//                     vector widths and the accumulation strategy (compensated fp32
//                     where fp64 is rate-limited, native fp64 where it is not).
//
// Every reduction is deterministic: a fixed block count with a two-stage tree, never
// atomics over floats, so repeated runs give bit-identical results.

#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <vector>

#ifndef STAM_MAX_BASIS
#define STAM_MAX_BASIS 8
#endif

namespace {

// Warp width is 32 on every NVIDIA architecture to date; the host layer reads the
// real value from cudaDeviceProp and refuses to dispatch to these kernels if it ever
// differs, so a compile-time constant is safe and lets the reductions unroll.
constexpr int kWarp = 32;

__device__ __forceinline__ float to_float(float x) { return x; }
__device__ __forceinline__ float to_float(double x) { return static_cast<float>(x); }
__device__ __forceinline__ float to_float(c10::Half x) { return static_cast<float>(x); }
__device__ __forceinline__ float to_float(c10::BFloat16 x) { return static_cast<float>(x); }

// Warp-level sum.  __shfl_down_sync exists from sm_30 and remains the fastest float
// reduction on every architecture through Blackwell -- redux.sync (sm_80+) is
// integer-only, so there is nothing newer to dispatch to for floats.  The explicit
// mask is required for correctness under independent thread scheduling (sm_70+).
__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int offset = kWarp / 2; offset > 0; offset >>= 1) {
    v += __shfl_down_sync(0xffffffffu, v, offset);
  }
  return v;
}

// Block-level sum over warp leaders.  Depth is log2(blockDim), so this stage
// contributes negligible rounding error and needs no compensation.  Callers must
// __syncthreads() between successive uses of the same shared buffer.
template <int BLOCK>
__device__ __forceinline__ float block_sum(float v, float* shared) {
  constexpr int NWARPS = BLOCK / kWarp;
  const int lane = threadIdx.x & (kWarp - 1);
  const int wid = threadIdx.x / kWarp;

  v = warp_sum(v);
  if (lane == 0) shared[wid] = v;
  __syncthreads();
  v = (threadIdx.x < NWARPS) ? shared[lane] : 0.f;
  if (wid == 0) v = warp_sum(v);
  return v;
}

// Neumaier compensated accumulation.  A grid-stride loop inside one thread can run
// 10^5..10^7 terms; in plain fp32 that loses roughly log2(n) bits.  Compensation costs
// three extra flops per element on a memory-bound kernel -- effectively free -- and is
// what makes fp32 accumulation acceptable on parts where fp64 runs at 1/32 rate.
struct Neumaier {
  float sum;
  float c;

  __device__ __forceinline__ void reset() {
    sum = 0.f;
    c = 0.f;
  }
  __device__ __forceinline__ void add(float x) {
    const float t = sum + x;
    const float lo = (fabsf(sum) >= fabsf(x)) ? ((sum - t) + x) : ((x - t) + sum);
    c += lo;
    sum = t;
  }
  __device__ __forceinline__ float value() const { return sum + c; }
};

inline int sm_count(int device) {
  int n = 0;
  cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, device);
  return n > 0 ? n : 1;
}

// Enough blocks to fill the device several times over so the grid-stride loop
// amortises launch overhead, capped so the second reduction stage stays small.
inline int reduction_blocks(int device, int64_t n, int block) {
  constexpr int kWaves = 8;
  const int64_t by_size = (n + block - 1) / block;
  const int64_t by_device = static_cast<int64_t>(sm_count(device)) * kWaves;
  return static_cast<int>(std::max<int64_t>(1, std::min(by_size, by_device)));
}

}  // namespace

// ---------------------------------------------------------------------------
// 1. plane_point:  out = theta0 + sum_{i<K} coords[i] * basis[i]
// ---------------------------------------------------------------------------

template <typename bscalar_t, typename oscalar_t, int K>
__global__ void plane_point_kernel(const oscalar_t* __restrict__ theta0,
                                   const bscalar_t* __restrict__ basis, int64_t basis_stride,
                                   const float* __restrict__ coords, oscalar_t* __restrict__ out,
                                   int64_t n) {
  float c[K];
#pragma unroll
  for (int i = 0; i < K; ++i) c[i] = coords[i];

  const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (int64_t idx = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x; idx < n;
       idx += stride) {
    float acc = to_float(theta0[idx]);
#pragma unroll
    for (int i = 0; i < K; ++i) {
      acc = fmaf(c[i], to_float(basis[i * basis_stride + idx]), acc);
    }
    out[idx] = static_cast<oscalar_t>(acc);
  }
}

// Vectorised fp32 path: 128-bit loads on every operand.
template <int K>
__global__ void plane_point_vec4_kernel(const float4* __restrict__ theta0,
                                        const float* __restrict__ basis, int64_t basis_stride,
                                        const float* __restrict__ coords, float4* __restrict__ out,
                                        int64_t n4) {
  float c[K];
#pragma unroll
  for (int i = 0; i < K; ++i) c[i] = coords[i];

  const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (int64_t idx = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x; idx < n4;
       idx += stride) {
    float4 acc = theta0[idx];
#pragma unroll
    for (int i = 0; i < K; ++i) {
      const float4 b = reinterpret_cast<const float4*>(basis + i * basis_stride)[idx];
      acc.x = fmaf(c[i], b.x, acc.x);
      acc.y = fmaf(c[i], b.y, acc.y);
      acc.z = fmaf(c[i], b.z, acc.z);
      acc.w = fmaf(c[i], b.w, acc.w);
    }
    out[idx] = acc;
  }
}

// Launch tables.  These must live at file scope: a preprocessor directive inside a
// macro argument (AT_DISPATCH_* is itself a macro) is undefined behaviour and nvcc
// silently drops it.
#define STAM_LAUNCH_VEC4(KV)                                                                  \
  case KV:                                                                                    \
    plane_point_vec4_kernel<KV><<<grid, kBlock, 0, stream>>>(                                 \
        reinterpret_cast<const float4*>(theta0.data_ptr<float>()), basis_c.data_ptr<float>(), \
        bstride, coords_f.data_ptr<float>(), reinterpret_cast<float4*>(out.data_ptr<float>()),\
        n4);                                                                                  \
    break;

#define STAM_LAUNCH_GEN(KV)                                                                   \
  case KV:                                                                                    \
    plane_point_kernel<bscalar_t, oscalar_t, KV><<<grid, kBlock, 0, stream>>>(                \
        theta0.data_ptr<oscalar_t>(), basis_c.data_ptr<bscalar_t>(), bstride,                 \
        coords_f.data_ptr<float>(), out.data_ptr<oscalar_t>(), n);                            \
    break;

#define STAM_LAUNCH_PROJ(KV, KAHAN_V)                                                         \
  case KV:                                                                                    \
    project_partial_kernel<gscalar_t, bscalar_t, KV, kBlock, KAHAN_V>                         \
        <<<grid, kBlock, 0, stream>>>(g_c.data_ptr<gscalar_t>(),                              \
                                      basis_c.data_ptr<bscalar_t>(), basis_c.stride(0),       \
                                      partial.data_ptr<float>(), n);                          \
    break;

#define STAM_PROJ_ON(KV) STAM_LAUNCH_PROJ(KV, true)
#define STAM_PROJ_OFF(KV) STAM_LAUNCH_PROJ(KV, false)

#define STAM_K_CASES(MACRO) \
  MACRO(1) MACRO(2) MACRO(3) MACRO(4) MACRO(5) MACRO(6) MACRO(7) MACRO(8)

torch::Tensor plane_point(torch::Tensor theta0, torch::Tensor basis, torch::Tensor coords,
                          c10::optional<torch::Tensor> out_opt) {
  TORCH_CHECK(theta0.is_cuda() && basis.is_cuda(), "plane_point: tensors must be CUDA");
  TORCH_CHECK(theta0.is_contiguous(), "plane_point: theta0 must be contiguous");
  TORCH_CHECK(basis.dim() == 2, "plane_point: basis must be (K, N)");
  TORCH_CHECK(basis.size(1) == theta0.numel(), "plane_point: basis/theta0 size mismatch");
  const int K = static_cast<int>(basis.size(0));
  TORCH_CHECK(K >= 1 && K <= STAM_MAX_BASIS, "plane_point: K must be in [1, ", STAM_MAX_BASIS, "]");

  const at::cuda::CUDAGuard guard(theta0.device());
  auto coords_f = coords.to(torch::kFloat32).contiguous();
  auto basis_c = basis.contiguous();
  torch::Tensor out = out_opt.has_value() ? out_opt.value() : torch::empty_like(theta0);
  TORCH_CHECK(out.is_contiguous() && out.numel() == theta0.numel(), "plane_point: bad out tensor");

  const int64_t n = theta0.numel();
  const int64_t bstride = basis_c.stride(0);
  constexpr int kBlock = 256;
  const int device = theta0.device().index();
  auto stream = at::cuda::getCurrentCUDAStream();

  const bool vec4_ok = theta0.scalar_type() == torch::kFloat32 &&
                       basis_c.scalar_type() == torch::kFloat32 && (n % 4 == 0) &&
                       (bstride % 4 == 0) &&
                       (reinterpret_cast<uintptr_t>(theta0.data_ptr()) % 16 == 0) &&
                       (reinterpret_cast<uintptr_t>(basis_c.data_ptr()) % 16 == 0) &&
                       (reinterpret_cast<uintptr_t>(out.data_ptr()) % 16 == 0);

  if (vec4_ok) {
    const int64_t n4 = n / 4;
    const int grid = static_cast<int>(
        std::min<int64_t>((n4 + kBlock - 1) / kBlock, static_cast<int64_t>(sm_count(device)) * 32));
    switch (K) {
      STAM_K_CASES(STAM_LAUNCH_VEC4)
      default:
        TORCH_CHECK(false, "plane_point: unsupported K");
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
  }

  const int grid = static_cast<int>(
      std::min<int64_t>((n + kBlock - 1) / kBlock, static_cast<int64_t>(sm_count(device)) * 32));
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kHalf, at::kBFloat16, basis_c.scalar_type(), "plane_point_basis", [&] {
        using bscalar_t = scalar_t;
        AT_DISPATCH_FLOATING_TYPES_AND2(
            at::kHalf, at::kBFloat16, theta0.scalar_type(), "plane_point_out", [&] {
              using oscalar_t = scalar_t;
              switch (K) {
                STAM_K_CASES(STAM_LAUNCH_GEN)
                default:
                  TORCH_CHECK(false, "plane_point: unsupported K");
              }
            });
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

// ---------------------------------------------------------------------------
// 2. project:  s[i] = <g, basis[i]>, all K in a single pass over g
// ---------------------------------------------------------------------------
//
// The win is bandwidth, not flops: the naive form reads g once per basis vector.
// Fusing K dots reads g once and basis once, so traffic drops from (K+1)N floats to
// N + K*N*sizeof(basis)/4.

template <typename gscalar_t, typename bscalar_t, int K, int BLOCK, bool KAHAN>
__global__ void project_partial_kernel(const gscalar_t* __restrict__ g,
                                       const bscalar_t* __restrict__ basis, int64_t basis_stride,
                                       float* __restrict__ partial, int64_t n) {
  __shared__ float shared[BLOCK / kWarp];

  Neumaier acc_k[K];
  float acc_p[K];
#pragma unroll
  for (int i = 0; i < K; ++i) {
    acc_k[i].reset();
    acc_p[i] = 0.f;
  }

  const int64_t stride = static_cast<int64_t>(BLOCK) * gridDim.x;
  for (int64_t idx = blockIdx.x * static_cast<int64_t>(BLOCK) + threadIdx.x; idx < n;
       idx += stride) {
    const float gv = to_float(g[idx]);
#pragma unroll
    for (int i = 0; i < K; ++i) {
      const float prod = gv * to_float(basis[i * basis_stride + idx]);
      if (KAHAN) {
        acc_k[i].add(prod);
      } else {
        acc_p[i] += prod;
      }
    }
  }

#pragma unroll
  for (int i = 0; i < K; ++i) {
    const float v = block_sum<BLOCK>(KAHAN ? acc_k[i].value() : acc_p[i], shared);
    if (threadIdx.x == 0) partial[i * gridDim.x + blockIdx.x] = v;
    __syncthreads();
  }
}

// Second stage: one block per basis vector, over the (few) block partials.
template <int BLOCK>
__global__ void project_finalize_kernel(const float* __restrict__ partial, float* __restrict__ out,
                                        int nblocks) {
  __shared__ float shared[BLOCK / kWarp];
  const int row = blockIdx.x;
  float v = 0.f;
  for (int i = threadIdx.x; i < nblocks; i += BLOCK) v += partial[row * nblocks + i];
  v = block_sum<BLOCK>(v, shared);
  if (threadIdx.x == 0) out[row] = v;
}

torch::Tensor project(torch::Tensor g, torch::Tensor basis, bool compensated) {
  TORCH_CHECK(g.is_cuda() && basis.is_cuda(), "project: tensors must be CUDA");
  TORCH_CHECK(basis.dim() == 2 && basis.size(1) == g.numel(), "project: basis must be (K, N)");
  const int K = static_cast<int>(basis.size(0));
  TORCH_CHECK(K >= 1 && K <= STAM_MAX_BASIS, "project: K must be in [1, ", STAM_MAX_BASIS, "]");

  const at::cuda::CUDAGuard guard(g.device());
  auto g_c = g.contiguous();
  auto basis_c = basis.contiguous();
  const int64_t n = g_c.numel();
  constexpr int kBlock = 256;
  const int device = g.device().index();
  const int grid = reduction_blocks(device, n, kBlock);
  auto stream = at::cuda::getCurrentCUDAStream();

  auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(g.device());
  auto partial = torch::empty({K, grid}, opts);
  auto out = torch::empty({K}, opts);

  AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, g_c.scalar_type(), "project_g", [&] {
    using gscalar_t = scalar_t;
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kHalf, at::kBFloat16, basis_c.scalar_type(), "project_b", [&] {
          using bscalar_t = scalar_t;
          if (compensated) {
            switch (K) {
              STAM_K_CASES(STAM_PROJ_ON)
              default:
                TORCH_CHECK(false, "project: unsupported K");
            }
          } else {
            switch (K) {
              STAM_K_CASES(STAM_PROJ_OFF)
              default:
                TORCH_CHECK(false, "project: unsupported K");
            }
          }
        });
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  project_finalize_kernel<256>
      <<<K, 256, 0, stream>>>(partial.data_ptr<float>(), out.data_ptr<float>(), grid);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

// ---------------------------------------------------------------------------
// 3. gram_chunk:  G += D D^T for a column chunk of the trajectory matrix
// ---------------------------------------------------------------------------
//
// D is (T, N) with T ~ 10^2 and N ~ 10^5..10^8: extremely skinny, so the Gram product
// is bandwidth-bound rather than flop-bound.  Two consequences:
//   * half-precision *storage* with fp32 compute halves the traffic outright, and for
//     a 5M-parameter model with 400 snapshots is the difference between 8 GB and 4 GB;
//   * compensated accumulation costs nothing and matters, because each entry sums N
//     terms of one sign.
// Column chunking lets the trajectory live in host memory and stream through the GPU.

template <typename scalar_t, int TILE, int BLOCK>
__global__ void gram_chunk_kernel(const scalar_t* __restrict__ D, int64_t row_stride, int T,
                                  int64_t n, double* __restrict__ G, int64_t ldg) {
  const int i0 = blockIdx.y * TILE;
  const int j0 = blockIdx.x * TILE;
  if (i0 >= T || j0 >= T || j0 > i0) return;  // lower block-triangle only

  __shared__ float sA[TILE][BLOCK + 1];
  __shared__ float sB[TILE][BLOCK + 1];
  __shared__ float shared[BLOCK / kWarp];

  // Fixed-extent accumulator so every index is a compile-time constant and the array
  // stays in registers instead of spilling to local memory.
  Neumaier acc[TILE][TILE];
#pragma unroll
  for (int a = 0; a < TILE; ++a)
#pragma unroll
    for (int b = 0; b < TILE; ++b) acc[a][b].reset();

  for (int64_t base = 0; base < n; base += BLOCK) {
    const int64_t idx = base + threadIdx.x;
    const bool valid = idx < n;
#pragma unroll
    for (int a = 0; a < TILE; ++a) {
      const int r = i0 + a;
      sA[a][threadIdx.x] = (valid && r < T) ? to_float(D[r * row_stride + idx]) : 0.f;
    }
#pragma unroll
    for (int b = 0; b < TILE; ++b) {
      const int r = j0 + b;
      sB[b][threadIdx.x] = (valid && r < T) ? to_float(D[r * row_stride + idx]) : 0.f;
    }
    __syncthreads();
#pragma unroll
    for (int a = 0; a < TILE; ++a) {
      const float va = sA[a][threadIdx.x];
#pragma unroll
      for (int b = 0; b < TILE; ++b) {
        acc[a][b].add(va * sB[b][threadIdx.x]);
      }
    }
    __syncthreads();
  }

#pragma unroll
  for (int a = 0; a < TILE; ++a) {
#pragma unroll
    for (int b = 0; b < TILE; ++b) {
      const float v = block_sum<BLOCK>(acc[a][b].value(), shared);
      const int r = i0 + a, c = j0 + b;
      if (threadIdx.x == 0 && r < T && c < T && c <= r) {
        // One fp64 atomic per (tile, block) pair: the count is tiny, and the outer
        // accumulation across column chunks must be wide because each term is itself
        // a large partial sum.
        atomicAdd(&G[r * ldg + c], static_cast<double>(v));
      }
      __syncthreads();
    }
  }
}

torch::Tensor gram_chunk(torch::Tensor D, c10::optional<torch::Tensor> G_opt) {
  TORCH_CHECK(D.is_cuda() && D.dim() == 2, "gram_chunk: D must be a 2-D CUDA tensor");
  const int T = static_cast<int>(D.size(0));
  const int64_t n = D.size(1);
  const at::cuda::CUDAGuard guard(D.device());

  torch::Tensor G =
      G_opt.has_value()
          ? G_opt.value()
          : torch::zeros({T, T}, torch::TensorOptions().dtype(torch::kFloat64).device(D.device()));
  TORCH_CHECK(G.scalar_type() == torch::kFloat64 && G.size(0) == T && G.size(1) == T,
              "gram_chunk: G must be (T, T) float64");

  constexpr int TILE = 4;
  constexpr int kBlock = 256;
  const int tiles = (T + TILE - 1) / TILE;
  const dim3 grid(tiles, tiles);
  auto stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, D.scalar_type(), "gram_chunk", [&] {
    gram_chunk_kernel<scalar_t, TILE, kBlock><<<grid, kBlock, 0, stream>>>(
        D.data_ptr<scalar_t>(), D.stride(0), T, n, G.data_ptr<double>(), G.stride(0));
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return G;
}

// ---------------------------------------------------------------------------
// 4. pu_taylor: partition-of-unity Taylor reconstruction and its exact gradient
// ---------------------------------------------------------------------------
//
//   R(q)   = sum_i w_i(q) Q_i(q) / W(q),  Q_i(q) = f_i + g_i.d + 0.5 d^T H_i d
//   grad R = (1/W) [ sum_i w_i grad Q_i + sum_i (Q_i - R) grad w_i ]
//
// with Wendland C^2 weights w_i(q) = phi(|q-a_i|/rho_i), phi(r) = (1-r)_+^4 (4r+1).
// The second term is what makes the rendered quiver field the *exact* gradient of the
// rendered surface, rather than an independently interpolated field that need not be
// consistent with the surface it is drawn on.
//
// One thread per query point; anchors are staged through shared memory in tiles so a
// warp reads each anchor record once for 32 queries.

__device__ __forceinline__ void wendland_c2(float r, float& w, float& dw) {
  if (r >= 1.f) {
    w = 0.f;
    dw = 0.f;
    return;
  }
  const float om = 1.f - r;
  const float om2 = om * om;
  const float om3 = om2 * om;
  w = om3 * om * (4.f * r + 1.f);
  dw = -20.f * r * om3;  // d/dr of the above
}

template <int TILE>
__global__ void pu_taylor_kernel(const float* __restrict__ anchors,  // (n, 2)
                                 const float* __restrict__ radii,    // (n,)
                                 const float* __restrict__ fval,     // (n,)
                                 const float* __restrict__ grad,     // (n, 2)
                                 const float* __restrict__ hess,     // (n, 3) xx, xy, yy
                                 const float* __restrict__ queries,  // (m, 2)
                                 float* __restrict__ out_val,        // (m,)
                                 float* __restrict__ out_grad,       // (m, 2)
                                 float* __restrict__ out_wsum,       // (m,)
                                 int n, int m, int order, float eps) {
  __shared__ float s_a[TILE][2];
  __shared__ float s_r[TILE];
  __shared__ float s_f[TILE];
  __shared__ float s_g[TILE][2];
  __shared__ float s_h[TILE][3];

  const int q = blockIdx.x * blockDim.x + threadIdx.x;
  const float qx = (q < m) ? queries[2 * q + 0] : 0.f;
  const float qy = (q < m) ? queries[2 * q + 1] : 0.f;

  float W = 0.f, WQ = 0.f;
  float WGx = 0.f, WGy = 0.f;    // sum w_i grad Q_i
  float DWx = 0.f, DWy = 0.f;    // sum grad w_i
  float DWQx = 0.f, DWQy = 0.f;  // sum Q_i grad w_i

  for (int tile = 0; tile < n; tile += TILE) {
    for (int t = threadIdx.x; t < TILE; t += blockDim.x) {
      const int a = tile + t;
      const bool ok = a < n;
      s_a[t][0] = ok ? anchors[2 * a + 0] : 0.f;
      s_a[t][1] = ok ? anchors[2 * a + 1] : 0.f;
      s_r[t] = ok ? radii[a] : 1.f;
      s_f[t] = ok ? fval[a] : 0.f;
      s_g[t][0] = ok ? grad[2 * a + 0] : 0.f;
      s_g[t][1] = ok ? grad[2 * a + 1] : 0.f;
      s_h[t][0] = ok ? hess[3 * a + 0] : 0.f;
      s_h[t][1] = ok ? hess[3 * a + 1] : 0.f;
      s_h[t][2] = ok ? hess[3 * a + 2] : 0.f;
    }
    __syncthreads();

    const int lim = min(TILE, n - tile);
    for (int k = 0; k < lim; ++k) {
      const float dx = qx - s_a[k][0];
      const float dy = qy - s_a[k][1];
      const float dist = sqrtf(dx * dx + dy * dy);
      const float rho = s_r[k];
      float w, dphi;
      wendland_c2(dist / rho, w, dphi);
      if (w <= 0.f) continue;

      float Q = s_f[k];
      float Gx = 0.f, Gy = 0.f;
      if (order >= 1) {
        Q += s_g[k][0] * dx + s_g[k][1] * dy;
        Gx += s_g[k][0];
        Gy += s_g[k][1];
      }
      if (order >= 2) {
        const float hxx = s_h[k][0], hxy = s_h[k][1], hyy = s_h[k][2];
        Q += 0.5f * (hxx * dx * dx + 2.f * hxy * dx * dy + hyy * dy * dy);
        Gx += hxx * dx + hxy * dy;
        Gy += hxy * dx + hyy * dy;
      }

      // grad w = phi'(r) d / (rho |d|); identically zero at d = 0 since phi'(0) = 0.
      const float scale = (dist > eps) ? (dphi / (rho * dist)) : 0.f;
      const float gwx = scale * dx;
      const float gwy = scale * dy;

      W += w;
      WQ += w * Q;
      WGx += w * Gx;
      WGy += w * Gy;
      DWx += gwx;
      DWy += gwy;
      DWQx += Q * gwx;
      DWQy += Q * gwy;
    }
    __syncthreads();
  }

  if (q < m) {
    const bool covered = W > eps;
    const float invW = covered ? (1.f / W) : 0.f;
    const float R = WQ * invW;
    out_val[q] = covered ? R : NAN;
    out_grad[2 * q + 0] = covered ? invW * (WGx + (DWQx - R * DWx)) : NAN;
    out_grad[2 * q + 1] = covered ? invW * (WGy + (DWQy - R * DWy)) : NAN;
    out_wsum[q] = W;
  }
}

std::vector<torch::Tensor> pu_taylor(torch::Tensor anchors, torch::Tensor radii,
                                     torch::Tensor fval, torch::Tensor grad, torch::Tensor hess,
                                     torch::Tensor queries, int64_t order) {
  TORCH_CHECK(anchors.is_cuda(), "pu_taylor: tensors must be CUDA");
  TORCH_CHECK(anchors.dim() == 2 && anchors.size(1) == 2, "pu_taylor: anchors must be (n, 2)");
  TORCH_CHECK(queries.dim() == 2 && queries.size(1) == 2, "pu_taylor: queries must be (m, 2)");
  const at::cuda::CUDAGuard guard(anchors.device());

  auto f32 = [](const torch::Tensor& t) { return t.to(torch::kFloat32).contiguous(); };
  auto a = f32(anchors), rr = f32(radii), ff = f32(fval), gg = f32(grad), hh = f32(hess),
       qq = f32(queries);

  const int n = static_cast<int>(a.size(0));
  const int m = static_cast<int>(qq.size(0));
  auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(anchors.device());
  auto out_val = torch::empty({m}, opts);
  auto out_grad = torch::empty({m, 2}, opts);
  auto out_wsum = torch::empty({m}, opts);

  constexpr int TILE = 64;
  constexpr int kBlock = 128;
  const int grid = std::max(1, (m + kBlock - 1) / kBlock);
  auto stream = at::cuda::getCurrentCUDAStream();
  pu_taylor_kernel<TILE><<<grid, kBlock, 0, stream>>>(
      a.data_ptr<float>(), rr.data_ptr<float>(), ff.data_ptr<float>(), gg.data_ptr<float>(),
      hh.data_ptr<float>(), qq.data_ptr<float>(), out_val.data_ptr<float>(),
      out_grad.data_ptr<float>(), out_wsum.data_ptr<float>(), n, m, static_cast<int>(order),
      1e-12f);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {out_val, out_grad, out_wsum};
}

// ---------------------------------------------------------------------------
// 5. Introspection: the full cudaDeviceProp, so the Python policy never guesses
// ---------------------------------------------------------------------------

py::dict device_info(int64_t index) {
  cudaDeviceProp p{};
  C10_CUDA_CHECK(cudaGetDeviceProperties(&p, static_cast<int>(index)));
  py::dict d;
  d["name"] = std::string(p.name);
  d["major"] = p.major;
  d["minor"] = p.minor;
  d["total_memory"] = static_cast<int64_t>(p.totalGlobalMem);
  d["multi_processor_count"] = p.multiProcessorCount;
  d["warp_size"] = p.warpSize;
  d["max_threads_per_block"] = p.maxThreadsPerBlock;
  d["max_threads_per_sm"] = p.maxThreadsPerMultiProcessor;
  d["shared_memory_per_block"] = static_cast<int64_t>(p.sharedMemPerBlock);
  d["shared_memory_per_sm"] = static_cast<int64_t>(p.sharedMemPerMultiprocessor);
  d["regs_per_block"] = p.regsPerBlock;
  d["l2_cache_size"] = p.l2CacheSize;
  d["memory_bus_width"] = p.memoryBusWidth;
  d["memory_clock_khz"] = p.memoryClockRate;
  d["clock_khz"] = p.clockRate;
  d["cooperative_launch"] = p.cooperativeLaunch;
  return d;
}

py::dict build_info() {
  py::dict d;
  d["cuda_runtime"] = CUDART_VERSION;
  d["nvcc_major"] = __CUDACC_VER_MAJOR__;
  d["nvcc_minor"] = __CUDACC_VER_MINOR__;
  d["max_basis"] = STAM_MAX_BASIS;
  d["warp"] = kWarp;
  return d;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "STAM fused CUDA kernels";
  m.def("plane_point", &plane_point, "theta0 + sum c_i v_i (fused)", py::arg("theta0"),
        py::arg("basis"), py::arg("coords"), py::arg("out") = c10::nullopt);
  m.def("project", &project, "fused multi-vector dot product", py::arg("g"), py::arg("basis"),
        py::arg("compensated") = true);
  m.def("gram_chunk", &gram_chunk, "G += D D^T for a column chunk", py::arg("D"),
        py::arg("G") = c10::nullopt);
  m.def("pu_taylor", &pu_taylor, "partition-of-unity Taylor reconstruction", py::arg("anchors"),
        py::arg("radii"), py::arg("fval"), py::arg("grad"), py::arg("hess"), py::arg("queries"),
        py::arg("order"));
  m.def("device_info", &device_info, "cudaDeviceProp as a dict", py::arg("index"));
  m.def("build_info", &build_info, "compile-time configuration");
}
