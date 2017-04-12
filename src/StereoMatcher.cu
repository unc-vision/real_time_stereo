#include "StereoMatcher.h"
// Author: True Price <jtprice at cs.unc.edu>
//
// BSD License
// Copyright (C) 2017  The University of North Carolina at Chapel Hill
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// * Redistributions of source code must retain the above copyright notice, this
//   list of conditions and the following disclaimer.
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the following disclaimer in the documentation
//   and/or other materials provided with the distribution.
// * Neither the name of the original author nor the names of contributors may
//   be used to endorse or promote products derived from this software without
//   specific prior written permission.
//
// NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY
// THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
// CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT
// NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
// PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
// OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
// WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
// OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
// ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include <math.h>
#include "helper_math.h"

#include "cudaArray/cudaArray2D.h"
#include "cudaArray/cudaTexture2D.h"
#include "cudaArray/cudaSurface2D.h"
//#include "cudaArray/cudaArray3D.h"
#include "cudaArray/cudaSurface3D.h"

#include "glog/logging.h"

const float INV_255 = 0.00392156863;

__constant__ StereoMatcher::Options c_options;

//------------------------------------------------------------------------------
#define CUDA_CHECK(ans) \
  { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line,
                      bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file,
            line);
    if (abort) exit(code);
  }
}
#define CUDA_CHECK_ERROR CUDA_CHECK(cudaPeekAtLastError())

// uncomment the bottom two lines to include timing information

#define CUDA_TIMER_START
#define CUDA_TIMER_STOP(var)
//#define CUDA_TIMER_START     \
//  {                          \
//    cudaEvent_t start, stop; \
//    cudaEventCreate(&start); \
//    cudaEventCreate(&stop);  \
//    cudaEventRecord(start);
//
//#define CUDA_TIMER_STOP(var)               \
//  cudaEventRecord(stop);                   \
//  cudaEventSynchronize(stop);              \
//  cudaEventElapsedTime(&var, start, stop); \
//  }

//------------------------------------------------------------------------------

// converts a BGRA color into an intensity value in the range [0, 1]
__device__ float bgra_to_intensity(uchar4 color) {
  // the multiplication divides by 255
  return (0.2126 * color.z + 0.7152 * color.y + 0.0722 * color.x) * INV_255;
}

//------------------------------------------------------------------------------

struct StereoMatcher::StereoMatcherImpl {
  cua::CudaTexture2D<float2> ur_map1, ur_map2; // undistort+rectify maps
  cua::CudaTexture2D<unsigned char> orig_image1, orig_image2;  // input GPU images
  cua::CudaArray2D<float> image1, image2;  // undistorted GPU images
  // for the window around each pixel, compute the mean and inverse L2 norm
  cua::CudaArray2D<float2> image1_mean_inv_norm, image2_mean_inv_norm;
  cua::CudaSurface3D<float> raw_shift_scores;
  cua::CudaSurface3D<float> shift_scores;
  cua::CudaSurface2D<float> depth_map;

  // shift: number of shift elements (= max_shift - min_shift)
  StereoMatcherImpl(unsigned int width, unsigned int height, unsigned int shift)
      : ur_map1(width, height),
        ur_map2(width, height),
        orig_image1(width, height, cudaFilterModeLinear, cudaAddressModeBorder,
                    cudaReadModeNormalizedFloat),
        orig_image2(width, height, cudaFilterModeLinear, cudaAddressModeBorder,
                    cudaReadModeNormalizedFloat),
        image1(width, height),
        image2(width, height),
        image1_mean_inv_norm(width, height),
        image2_mean_inv_norm(width, height),
        raw_shift_scores(width, height, shift),
        shift_scores(width, height, shift),
        depth_map(width, height) {}
};

//------------------------------------------------------------------------------

// for each input pixel, consider all possible left-image-to-right-image matches
// in the range [-max_shift-1,-min_shift] and select the one with the best NCC
// [threadIdx.x]: equal to the shift radius
// [threadIdx.y]: TODO x-position to consider, offset by the block position
__global__ void StereoMatcher_NCC_kernel(
    StereoMatcher::StereoMatcherImpl impl) {
  extern __shared__ int shmem[];

  //
  // local constants
  //

  const int dx = threadIdx.x; // 0 to (max_shift - min_shift - 1)
  const int x = blockIdx.x;
  const int y = blockIdx.y;

  const int &window_size = c_options.window_size;
  const int &min_shift = c_options.min_shift;
  const int &max_shift = c_options.max_shift;
  const int window_radius = window_size >> 1;

  //
  // shared memory
  //

  // intensities for the window in the first image
  float *im1 = (float *)shmem;

  // intensities for the window in the second image
  float *im2 = (float *)&im1[window_size];

  //
  // begin computation
  //

  const float2 im1_mean_inv_norm = impl.image1_mean_inv_norm.get(x, y);
  const float2 im2_mean_inv_norm =
      impl.image2_mean_inv_norm.get(x + dx + min_shift - max_shift + 1, y);

  // collect the windows for both images into shared memory
  im2[dx] =
      impl.image2.get(x + dx + min_shift - max_shift - window_radius + 1, y);

  if (dx < window_size) {
    im1[dx] = impl.image1.get(x + dx - window_radius, y) - im1_mean_inv_norm.x;
    im2[dx + max_shift - min_shift - 1] =
        impl.image2.get(x + dx + min_shift - window_radius, y);
  }

  __syncthreads();

  // compute NCC
  float score = 0;
  for (int i = 0; i < window_size; ++i) {
    score += im1[i] * (im2[i + dx] - im2_mean_inv_norm.x);
  }

  score *= im1_mean_inv_norm.y * im2_mean_inv_norm.y;

  impl.raw_shift_scores.set(x, y, dx, score);
}

//------------------------------------------------------------------------------

__global__ void StereoMatcher_Smoothing_kernel(
    StereoMatcher::StereoMatcherImpl impl) {
  const int x0 = blockIdx.x * blockDim.x - 5;
  const int y0 = blockIdx.y * blockDim.y - 5;
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  const int z = blockIdx.z * blockDim.z + threadIdx.z;

  const int step = blockDim.x * blockDim.y;
  const int w = (blockDim.x + 10);
  const int h = (blockDim.y + 10);

  // load the (padded) local window into shared memory
  extern __shared__ int shmem[];
  float *values = (float *)(shmem + w * h * threadIdx.z);

  for (int i = blockDim.x * threadIdx.y + threadIdx.x; i < w * h; i += step) {
    const int u = x0 + i % w;
    const int v = y0 + i / w;
    values[i] = impl.raw_shift_scores.get(u, v, z);
  }

  __syncthreads();

  float score = 0.f;
  for (int i = 0; i < 11; ++i) {
    for (int j = 0; j < 11; ++j) {
      score += values[(threadIdx.y + i) * w + threadIdx.x + j];
    }
  }

  impl.shift_scores.set(x, y, z, score);
}

/*
__global__ void StereoMatcher_Smoothing_kernel(
    StereoMatcher::StereoMatcherImpl impl) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  const int z = blockIdx.z * blockDim.z + threadIdx.z;

  float score = 0.f;
  for (int i = -5; i <= 5; ++i) {
    for (int j = -5; j <= 5; ++j) {
      score += impl.raw_shift_scores.get(x + j, y + i, z);
    }
  }

  impl.shift_scores.set(x, y, z, score);
}
*/

//------------------------------------------------------------------------------
//
// class definitions
//
//------------------------------------------------------------------------------

StereoMatcher::StereoMatcher(const StereoMatcher::Options &options)
    : options_(options),
      impl_(new StereoMatcherImpl(options.width, options.height,
                                  options.max_shift)) {
  if (options_.max_shift < options_.window_size) {
    fprintf(stderr, "max shift smaller than window size!");
    exit(1);
  }

  cudaMemcpyToSymbol(c_options, &options_, sizeof(StereoMatcher::Options));
}

//------------------------------------------------------------------------------

StereoMatcher::~StereoMatcher() {}

//------------------------------------------------------------------------------

void StereoMatcher::initUndistortRectifyMaps(const void *ur_map1,
                                             const void *ur_map2) {
  impl_->ur_map1 = (float2*)ur_map1;
  impl_->ur_map2 = (float2*)ur_map2;
}

//------------------------------------------------------------------------------

void StereoMatcher::init_frame(void *image1_cpu, void *image2_cpu) {
  auto &impl = *impl_;

  impl.orig_image1 = (unsigned char *)image1_cpu;  // copy to GPU
  impl.orig_image2 = (unsigned char *)image2_cpu;  // copy to GPU

  // undistort+rectify (using bilinear interpolation), and convert to intensity
  impl.image1.apply_op([=] __device__(const size_t x, const size_t y) {
    const float2 uv = impl.ur_map1.get(x, y);
    return impl.orig_image1.interp<float>(uv.x, uv.y);
  });

  impl.image2.apply_op([=] __device__(const size_t x, const size_t y) {
    const float2 uv = impl.ur_map2.get(x, y);
    return impl.orig_image2.interp<float>(uv.x, uv.y);
  });

  // compute the mean and std. dev. of the window centered around each pixel
  impl.image1_mean_inv_norm.apply_op([=] __device__(const size_t x,
                                                    const size_t y) {
    const int &window_size = c_options.window_size;
    const int window_radius = window_size >> 1;
    float avg = 0.f;
    float inv_norm = 0.f;

    for (int i = 0; i < c_options.window_size; ++i) {
      avg += impl.image1.get(x + i - window_radius, y);
    }

    avg /= c_options.window_size;

    for (int i = 0; i < window_size; ++i) {
      const float intensity = impl.image1.get(x + i - window_radius, y) - avg;
      inv_norm += intensity * intensity;
    }

    inv_norm = rsqrt(inv_norm);  // = 1 / sqrt(x)

    return make_float2(avg, inv_norm);
  });

  impl.image2_mean_inv_norm.apply_op([=] __device__(const size_t x,
                                                    const size_t y) {
    const int &window_size = c_options.window_size;
    const int window_radius = window_size >> 1;
    float avg = 0.f;
    float inv_norm = 0.f;

    for (int i = 0; i < c_options.window_size; ++i) {
      avg += impl.image2.get(x + i - window_radius, y);
    }

    avg /= c_options.window_size;

    for (int i = 0; i < window_size; ++i) {
      const float intensity = impl.image2.get(x + i - window_radius, y) - avg;
      inv_norm += intensity * intensity;
    }

    inv_norm = rsqrt(inv_norm);  // = 1 / sqrt(x)

    return make_float2(avg, inv_norm);
  });
}

//------------------------------------------------------------------------------

void StereoMatcher::download_image1(void *image) const {
  impl_->image1.CopyTo((float *)image);
}

void StereoMatcher::download_image2(void *image) const {
  impl_->image2.CopyTo((float *)image);
}

void StereoMatcher::download_depth(void *image) const {
  impl_->depth_map.CopyTo((float *)image);
}

//------------------------------------------------------------------------------

void StereoMatcher::match() {
  float t1 = 0.f, t2 = 0.f, t3 = 0.f;

  auto &impl = *impl_;

  //
  // raw matching
  //

  {
    const dim3 block_dim = dim3(options_.shift());
    const dim3 grid_dim = dim3(options_.width, options_.height);

    // shared memory breakdown:
    // im1: window_size elements
    // im2: max_shift + window_size - 1 elements
    const size_t shmem_size =
        (2 * options_.window_size + options_.shift() - 1) * sizeof(float);

    CUDA_TIMER_START
    StereoMatcher_NCC_kernel<<<grid_dim, block_dim, shmem_size>>>(impl);
    CUDA_TIMER_STOP(t1)
  }

  //
  // smoothing
  //

  {
    const int w = 32; // pixels per thread block
    const int h = 32; // pixels per thread block
    const int d = 1; // shift values per thread block
    const dim3 block_dim = dim3(w, h, d);
    const dim3 grid_dim =
        dim3((options_.width + w - 1) / w, (options_.height + h - 1) / h,
             (options_.shift() + d - 1) / d);

    // shared memory breakdown:
    const size_t shmem_size = (w + 10) * (h + 10) * d * sizeof(float);

    CUDA_TIMER_START
    StereoMatcher_Smoothing_kernel<<<grid_dim, block_dim, shmem_size>>>(impl);
    CUDA_TIMER_STOP(t2)
  }

  // find the best shift
  CUDA_TIMER_START
  impl.depth_map.apply_op([=] __device__(const size_t x, const size_t y) {
    float best_dx = c_options.max_shift - 1;
    float best_score = impl.shift_scores.get(
        x, y, c_options.max_shift - c_options.min_shift - 1);

    for (int dx = c_options.max_shift - 2; dx >= c_options.min_shift; --dx) {
      const float score = impl.shift_scores.get(x, y, dx - c_options.min_shift);
      if (score > best_score) {
        best_score = score;
        best_dx = dx;
      }
    }

    const float disparity = (float)(c_options.max_shift - best_dx);
    return disparity;
  });
  CUDA_TIMER_STOP(t3)
}

//------------------------------------------------------------------------------

// match calculates disparity; this converts to depth values
// TODO (True): this is currently just for visualization purposes; it can really
// be moved into match()
void StereoMatcher::calculate_depth() {
  auto &impl = *impl_;

  impl.depth_map.apply_op([=] __device__(const size_t x, const size_t y) {
    const float depth =
        c_options.focal_length * c_options.baseline / impl.depth_map.get(x, y);
    return min(max(depth, c_options.min_depth), c_options.max_depth);
  });
}
