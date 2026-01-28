/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#ifndef CUDA_RASTERIZER_AUXILIARY_H_INCLUDED
#define CUDA_RASTERIZER_AUXILIARY_H_INCLUDED

#include "config.h"
#include "stdio.h"
#include <glm/glm.hpp>

#define BLOCK_SIZE (BLOCK_X * BLOCK_Y)
#define NUM_WARPS (BLOCK_SIZE/32)
#define DGR_FIX_AA

// Spherical harmonics coefficients
__device__ const float SH_C0 = 0.28209479177387814f;
__device__ const float SH_C1 = 0.4886025119029199f;
__device__ const float SH_C2[] = {
	1.0925484305920792f,
	-1.0925484305920792f,
	0.31539156525252005f,
	-1.0925484305920792f,
	0.5462742152960396f
};
__device__ const float SH_C3[] = {
	-0.5900435899266435f,
	2.890611442640554f,
	-0.4570457994644658f,
	0.3731763325901154f,
	-0.4570457994644658f,
	1.445305721320277f,
	-0.5900435899266435f
};

// Windowing coefficients for Low-Error Reconstruction (sinc window)
// W_l = sinc(pi * l / (L + 1))  where L=3
__device__ const float SH_W1 = 0.900316316f;
__device__ const float SH_W2 = 0.636619772f;
__device__ const float SH_W3 = 0.300105438f;

__forceinline__ __device__ float ndc2Pix(float v, int S)
{
	return ((v + 1.0) * S - 1.0) * 0.5;
}

__forceinline__ __device__ void getRect(const float2 p, int max_radius, uint2& rect_min, uint2& rect_max, dim3 grid)
{
	rect_min = {
		min(grid.x, max((int)0, (int)((p.x - max_radius) / BLOCK_X))),
		min(grid.y, max((int)0, (int)((p.y - max_radius) / BLOCK_Y)))
	};
	rect_max = {
		min(grid.x, max((int)0, (int)((p.x + max_radius + BLOCK_X - 1) / BLOCK_X))),
		min(grid.y, max((int)0, (int)((p.y + max_radius + BLOCK_Y - 1) / BLOCK_Y)))
	};
}

__forceinline__ __device__ void getRect(const float2 p, int2 ext_rect, uint2& rect_min, uint2& rect_max, dim3 grid)
{
	rect_min = {
		min(grid.x, max((int)0, (int)((p.x - ext_rect.x) / BLOCK_X))),
		min(grid.y, max((int)0, (int)((p.y - ext_rect.y) / BLOCK_Y)))
	};
	rect_max = {
		min(grid.x, max((int)0, (int)((p.x + ext_rect.x + BLOCK_X - 1) / BLOCK_X))),
		min(grid.y, max((int)0, (int)((p.y + ext_rect.y + BLOCK_Y - 1) / BLOCK_Y)))
	};
}

__forceinline__ __device__ float3 transformPoint4x3(const float3& p, const float* matrix)
{
	float3 transformed = {
		matrix[0] * p.x + matrix[4] * p.y + matrix[8] * p.z + matrix[12],
		matrix[1] * p.x + matrix[5] * p.y + matrix[9] * p.z + matrix[13],
		matrix[2] * p.x + matrix[6] * p.y + matrix[10] * p.z + matrix[14],
	};
	return transformed;
}

__forceinline__ __device__ float4 transformPoint4x4(const float3& p, const float* matrix)
{
	float4 transformed = {
		matrix[0] * p.x + matrix[4] * p.y + matrix[8] * p.z + matrix[12],
		matrix[1] * p.x + matrix[5] * p.y + matrix[9] * p.z + matrix[13],
		matrix[2] * p.x + matrix[6] * p.y + matrix[10] * p.z + matrix[14],
		matrix[3] * p.x + matrix[7] * p.y + matrix[11] * p.z + matrix[15]
	};
	return transformed;
}

__forceinline__ __device__ float3 transformVec4x3(const float3& p, const float* matrix)
{
	float3 transformed = {
		matrix[0] * p.x + matrix[4] * p.y + matrix[8] * p.z,
		matrix[1] * p.x + matrix[5] * p.y + matrix[9] * p.z,
		matrix[2] * p.x + matrix[6] * p.y + matrix[10] * p.z,
	};
	return transformed;
}

__forceinline__ __device__ float3 transformVec4x3Transpose(const float3& p, const float* matrix)
{
	float3 transformed = {
		matrix[0] * p.x + matrix[1] * p.y + matrix[2] * p.z,
		matrix[4] * p.x + matrix[5] * p.y + matrix[6] * p.z,
		matrix[8] * p.x + matrix[9] * p.y + matrix[10] * p.z,
	};
	return transformed;
}

__forceinline__ __device__ float dnormvdz(float3 v, float3 dv)
{
	float sum2 = v.x * v.x + v.y * v.y + v.z * v.z;
	float invsum32 = 1.0f / sqrt(sum2 * sum2 * sum2);
	float dnormvdz = (-v.x * v.z * dv.x - v.y * v.z * dv.y + (sum2 - v.z * v.z) * dv.z) * invsum32;
	return dnormvdz;
}

__forceinline__ __device__ float3 dnormvdv(float3 v, float3 dv)
{
	float sum2 = v.x * v.x + v.y * v.y + v.z * v.z;
	float invsum32 = 1.0f / sqrt(sum2 * sum2 * sum2);

	float3 dnormvdv;
	dnormvdv.x = ((+sum2 - v.x * v.x) * dv.x - v.y * v.x * dv.y - v.z * v.x * dv.z) * invsum32;
	dnormvdv.y = (-v.x * v.y * dv.x + (sum2 - v.y * v.y) * dv.y - v.z * v.y * dv.z) * invsum32;
	dnormvdv.z = (-v.x * v.z * dv.x - v.y * v.z * dv.y + (sum2 - v.z * v.z) * dv.z) * invsum32;
	return dnormvdv;
}

__forceinline__ __device__ float4 dnormvdv(float4 v, float4 dv)
{
	float sum2 = v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
	float invsum32 = 1.0f / sqrt(sum2 * sum2 * sum2);

	float4 vdv = { v.x * dv.x, v.y * dv.y, v.z * dv.z, v.w * dv.w };
	float vdv_sum = vdv.x + vdv.y + vdv.z + vdv.w;
	float4 dnormvdv;
	dnormvdv.x = ((sum2 - v.x * v.x) * dv.x - v.x * (vdv_sum - vdv.x)) * invsum32;
	dnormvdv.y = ((sum2 - v.y * v.y) * dv.y - v.y * (vdv_sum - vdv.y)) * invsum32;
	dnormvdv.z = ((sum2 - v.z * v.z) * dv.z - v.z * (vdv_sum - vdv.z)) * invsum32;
	dnormvdv.w = ((sum2 - v.w * v.w) * dv.w - v.w * (vdv_sum - vdv.w)) * invsum32;
	return dnormvdv;
}

__forceinline__ __device__ float sigmoid(float x)
{
	return 1.0f / (1.0f + expf(-x));
}

__forceinline__ __device__ bool in_frustum(int idx,
	const float* orig_points,
	const float* viewmatrix,
	const float* projmatrix,
	bool prefiltered,
	float3& p_view)
{
	float3 p_orig = { orig_points[3 * idx], orig_points[3 * idx + 1], orig_points[3 * idx + 2] };

	// Bring points to screen space
	float4 p_hom = transformPoint4x4(p_orig, projmatrix);
	float p_w = 1.0f / (p_hom.w + 0.0000001f);
	// float3 p_proj = { p_hom.x * p_w, p_hom.y * p_w, p_hom.z * p_w };
	p_view = transformPoint4x3(p_orig, viewmatrix); // rot

	if (p_view.z <= 0.2f)// || ((p_proj.x < -1.3 || p_proj.x > 1.3 || p_proj.y < -1.3 || p_proj.y > 1.3)))
	{
		if (prefiltered)
		{
			printf("Point is filtered although prefiltered is set. This shouldn't happen!");
			__trap();
		}
		return false;
	}
	return true;
}

// --------------------------------------------------------------------------
// SH / SG Helper functions (Moved from forward.cu / backward.cu)
// --------------------------------------------------------------------------

// Evaluate SH basis functions Y_lm(dir) for a given direction.
// Stores results in the provided output array (up to degree 3, 16 coeffs).
__forceinline__ __device__ void evalSHBasis(const glm::vec3 dir, float* result)
{
	result[0] = 0.28209479177387814f; // Y_00

	float x = dir.x;
	float y = dir.y;
	float z = dir.z;

	result[1] = -0.4886025119029199f * y;
	result[2] = 0.4886025119029199f * z;
	result[3] = -0.4886025119029199f * x;

	float xx = x * x, yy = y * y, zz = z * z;
	float xy = x * y, yz = y * z, xz = x * z;

	result[4] = 1.0925484305920792f * xy;
	result[5] = -1.0925484305920792f * yz;
	result[6] = 0.31539156525252005f * (2.0f * zz - xx - yy);
	result[7] = -1.0925484305920792f * xz;
	result[8] = 0.5462742152960396f * (xx - yy);

	result[9] = -0.5900435899266435f * y * (3.0f * xx - yy);
	result[10] = 2.890611442640554f * xy * z;
	result[11] = -0.4570457994644658f * y * (4.0f * zz - xx - yy);
	result[12] = 0.3731763325901154f * z * (2.0f * zz - 3.0f * xx - 3.0f * yy);
	result[13] = -0.4570457994644658f * x * (4.0f * zz - xx - yy);
	result[14] = 1.445305721320277f * z * (xx - yy);
	result[15] = -0.5900435899266435f * x * (xx - 3.0f * yy);
}

// Evaluate gradients of SH basis functions Y_lm(dir) w.r.t direction.
// Result stored as 16 vec3s.
__forceinline__ __device__ void dEvalSHBasis(const glm::vec3 dir, glm::vec3* results)
{
	// Y00 const, grad 0
	results[0] = {0.f, 0.f, 0.f};

	// Y1m
	// Y1-1 = -C1*y, Y10 = C1*z, Y11 = -C1*x
	// grad: (0, -C1, 0), (0, 0, C1), (-C1, 0, 0)
	float C1 = 0.4886025119029199f;
	results[1] = {0.f, -C1, 0.f};
	results[2] = {0.f, 0.f, C1};
	results[3] = {-C1, 0.f, 0.f};

	float x = dir.x;
	float y = dir.y;
	float z = dir.z;

	// Band 2
	float C2_0 = 1.0925484305920792f;
	float C2_1 = 1.0925484305920792f;
	float C2_2 = 0.31539156525252005f;
	float C2_3 = 1.0925484305920792f;
	float C2_4 = 0.5462742152960396f;

	results[4] = {C2_0 * y, C2_0 * x, 0.f}; // xy
	results[5] = {0.f, -C2_1 * z, -C2_1 * y}; // yz
	results[6] = {C2_2 * -2.f * x, C2_2 * -2.f * y, C2_2 * 4.f * z}; 
	results[7] = {-C2_3 * z, 0.f, -C2_3 * x}; // xz
	results[8] = {C2_4 * 2.f * x, -C2_4 * 2.f * y, 0.f}; // xx-yy

	// Band 3
	float C3_0 = 0.5900435899266435f;
	float C3_1 = 2.890611442640554f;
	float C3_2 = 0.4570457994644658f;
	float C3_3 = 0.3731763325901154f;
	float C3_4 = 0.4570457994644658f;
	float C3_5 = 1.445305721320277f;
	float C3_6 = 0.5900435899266435f;

	float xx = x * x, yy = y * y, zz = z * z;
	float xy = x * y, yz = y * z, xz = x * z;

	results[9] = {-C3_0 * 6.f * xy, -C3_0 * 3.f * (xx - yy), 0.f};
	results[10] = {C3_1 * yz, C3_1 * xz, C3_1 * xy};
	results[11] = {2.f * C3_2 * xy, -C3_2 * (4.f * zz - xx - 3.f * yy), -8.f * C3_2 * yz};
	results[12] = {-6.f * C3_3 * xz, -6.f * C3_3 * yz, C3_3 * (6.f * zz - 3.f * xx - 3.f * yy)};
	results[13] = {-C3_4 * (4.f * zz - 3.f * xx - yy), 2.f * C3_4 * xy, -8.f * C3_4 * xz};
	results[14] = {2.f * C3_5 * xz, -2.f * C3_5 * yz, C3_5 * (xx - yy)};
	results[15] = {-C3_6 * 3.f * (xx - yy), 6.f * C3_6 * xy, 0.f};
}

// Evaluate Zonal Harmonic (ZH) coefficient g_{i,l} Approx
__forceinline__ __device__ float evalZHApprox(int l, float lambda)
{
	if (l == 0) return 1.0f - exp(-2.0f * lambda);
	return exp(-(float)(l * (l + 1)) / (2.0f * lambda));
}

__forceinline__ __device__ float evalZHApproxVal(int l, float lambda)
{
    if (l == 0) return 1.0f - exp(-2.0f * lambda);
    return exp(-(float)(l * (l + 1)) / (2.0f * lambda));
}

__forceinline__ __device__ float evalZHApproxGrad(int l, float lambda)
{
	if (l == 0) return 2.0f * exp(-2.0f * lambda);
	
	float num = (float)(l * (l + 1));
	float exponent = -num / (2.0f * lambda);
	float g = exp(exponent);
	return g * (num / (2.0f * lambda * lambda));
}

#define CHECK_CUDA(A, debug) \
A; if(debug) { \
auto ret = cudaDeviceSynchronize(); \
if (ret != cudaSuccess) { \
std::cerr << "\n[CUDA ERROR] in " << __FILE__ << "\nLine " << __LINE__ << ": " << cudaGetErrorString(ret); \
throw std::runtime_error(cudaGetErrorString(ret)); \
} \
}

// Convert Spherical Gaussian (SG) parameters to SH coefficients using Analytic Projection.
// f_lm = Sum_i [ Y_lm(xi_i) * sqrt(4pi/(2l+1)) * g_{i,l} * w_i ]
// Max supported degree 3 (16 coeffs).

// Evaluate SH sum for a given direction (Color Evaluation)
// DC is the first coefficient (Band 0).
// shs points to Band 1 coefficients (vec3 array).
// Result is linear RGB (before clamping).
__forceinline__ __device__ glm::vec3 evalSHColor(const glm::vec3 dir, const glm::vec3 dc, const glm::vec3* shs, int deg)
{
	glm::vec3 result = SH_C0 * dc;
	if (deg > 0)
	{
		float x = dir.x;
		float y = dir.y;
		float z = dir.z;
		result = result - SH_C1 * SH_W1 * y * shs[0] + SH_C1 * SH_W1 * z * shs[1] - SH_C1 * SH_W1 * x * shs[2];

		if (deg > 1)
		{
			float xx = x * x, yy = y * y, zz = z * z;
			float xy = x * y, yz = y * z, xz = x * z;
			result = result +
				SH_C2[0] * SH_W2 * xy * shs[3] +
				SH_C2[1] * SH_W2 * yz * shs[4] +
				SH_C2[2] * SH_W2 * (2.0f * zz - xx - yy) * shs[5] +
				SH_C2[3] * SH_W2 * xz * shs[6] +
				SH_C2[4] * SH_W2 * (xx - yy) * shs[7];

			if (deg > 2)
			{
				result = result +
					SH_C3[0] * SH_W3 * y * (3.0f * xx - yy) * shs[8] +
					SH_C3[1] * SH_W3 * xy * z * shs[9] +
					SH_C3[2] * SH_W3 * y * (4.0f * zz - xx - yy) * shs[10] +
					SH_C3[3] * SH_W3 * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * shs[11] +
					SH_C3[4] * SH_W3 * x * (4.0f * zz - xx - yy) * shs[12] +
					SH_C3[5] * SH_W3 * z * (xx - yy) * shs[13] +
					SH_C3[6] * SH_W3 * x * (xx - 3.0f * yy) * shs[14];
			}
		}
	}
	result += 0.5f;
	return result; 
}

// Evaluate gradients for SH color (Backward Pass)
// dL_dRGB: Gradient of loss w.r.t Color
// dir: View direction
// shs: Forward SH coefficients (Band 1, vec3*)
// dL_ddc: Gradient w.r.t DC (Output)
// dL_dshs: Gradient w.r.t SH (Band 1, vec3* Output)
// dRGBdx, dRGBdy, dRGBdz: Gradient of Color w.r.t direction (Output, accumulated)
__forceinline__ __device__ void dEvalSHColor(
	const glm::vec3 dL_dRGB, 
	const glm::vec3 dir, 
	const glm::vec3* shs, 
	int deg,
	glm::vec3& dL_ddc,
	glm::vec3* dL_dshs,
	glm::vec3& dRGBdx, glm::vec3& dRGBdy, glm::vec3& dRGBdz)
{
	// No tricks here, just high school-level calculus.
	float dRGBdsh0 = SH_C0;
	dL_ddc = dRGBdsh0 * dL_dRGB;
	
	if (deg > 0)
	{
		float x = dir.x; float y = dir.y; float z = dir.z;
		
		float dRGBdsh1 = -SH_C1 * SH_W1 * y;
		float dRGBdsh2 = SH_C1 * SH_W1 * z;
		float dRGBdsh3 = -SH_C1 * SH_W1 * x;
		dL_dshs[0] = dRGBdsh1 * dL_dRGB;
		dL_dshs[1] = dRGBdsh2 * dL_dRGB;
		dL_dshs[2] = dRGBdsh3 * dL_dRGB;

		dRGBdx = -SH_C1 * SH_W1 * shs[2]; // sh[2] is shs[2]
		dRGBdy = -SH_C1 * SH_W1 * shs[0];
		dRGBdz = SH_C1 * SH_W1 * shs[1];

		if (deg > 1)
		{
			float xx = x * x, yy = y * y, zz = z * z;
			float xy = x * y, yz = y * z, xz = x * z;

			float dRGBdsh4 = SH_C2[0] * SH_W2 * xy;
			float dRGBdsh5 = SH_C2[1] * SH_W2 * yz;
			float dRGBdsh6 = SH_C2[2] * SH_W2 * (2.f * zz - xx - yy);
			float dRGBdsh7 = SH_C2[3] * SH_W2 * xz;
			float dRGBdsh8 = SH_C2[4] * SH_W2 * (xx - yy);
			dL_dshs[3] = dRGBdsh4 * dL_dRGB;
			dL_dshs[4] = dRGBdsh5 * dL_dRGB;
			dL_dshs[5] = dRGBdsh6 * dL_dRGB;
			dL_dshs[6] = dRGBdsh7 * dL_dRGB;
			dL_dshs[7] = dRGBdsh8 * dL_dRGB;

			dRGBdx += SH_C2[0] * SH_W2 * y * shs[3] + SH_C2[2] * SH_W2 * 2.f * -x * shs[5] + SH_C2[3] * SH_W2 * z * shs[6] + SH_C2[4] * SH_W2 * 2.f * x * shs[7];
			dRGBdy += SH_C2[0] * SH_W2 * x * shs[3] + SH_C2[1] * SH_W2 * z * shs[4] + SH_C2[2] * SH_W2 * 2.f * -y * shs[5] + SH_C2[4] * SH_W2 * 2.f * -y * shs[7];
			dRGBdz += SH_C2[1] * SH_W2 * y * shs[4] + SH_C2[2] * SH_W2 * 2.f * 2.f * z * shs[5] + SH_C2[3] * SH_W2 * x * shs[6];

			if (deg > 2)
			{
				float dRGBdsh9 = SH_C3[0] * SH_W3 * y * (3.f * xx - yy);
				float dRGBdsh10 = SH_C3[1] * SH_W3 * xy * z;
				float dRGBdsh11 = SH_C3[2] * SH_W3 * y * (4.f * zz - xx - yy);
				float dRGBdsh12 = SH_C3[3] * SH_W3 * z * (2.f * zz - 3.f * xx - 3.f * yy);
				float dRGBdsh13 = SH_C3[4] * SH_W3 * x * (4.f * zz - xx - yy);
				float dRGBdsh14 = SH_C3[5] * SH_W3 * z * (xx - yy);
				float dRGBdsh15 = SH_C3[6] * SH_W3 * x * (xx - 3.f * yy);
				dL_dshs[8] = dRGBdsh9 * dL_dRGB;
				dL_dshs[9] = dRGBdsh10 * dL_dRGB;
				dL_dshs[10] = dRGBdsh11 * dL_dRGB;
				dL_dshs[11] = dRGBdsh12 * dL_dRGB;
				dL_dshs[12] = dRGBdsh13 * dL_dRGB;
				dL_dshs[13] = dRGBdsh14 * dL_dRGB;
				dL_dshs[14] = dRGBdsh15 * dL_dRGB;

				dRGBdx += (
					SH_C3[0] * SH_W3 * shs[8] * 3.f * 2.f * xy +
					SH_C3[1] * SH_W3 * shs[9] * yz +
					SH_C3[2] * SH_W3 * shs[10] * -2.f * xy +
					SH_C3[3] * SH_W3 * shs[11] * -3.f * 2.f * xz +
					SH_C3[4] * SH_W3 * shs[12] * (-3.f * xx + 4.f * zz - yy) +
					SH_C3[5] * SH_W3 * shs[13] * 2.f * xz +
					SH_C3[6] * SH_W3 * shs[14] * 3.f * (xx - yy));

				dRGBdy += (
					SH_C3[0] * SH_W3 * shs[8] * 3.f * (xx - yy) +
					SH_C3[1] * SH_W3 * shs[9] * xz +
					SH_C3[2] * SH_W3 * shs[10] * (-3.f * yy + 4.f * zz - xx) +
					SH_C3[3] * SH_W3 * shs[11] * -3.f * 2.f * yz +
					SH_C3[4] * SH_W3 * shs[12] * -2.f * xy +
					SH_C3[5] * SH_W3 * shs[13] * -2.f * yz +
					SH_C3[6] * SH_W3 * shs[14] * -3.f * 2.f * xy);

				dRGBdz += (
					SH_C3[1] * SH_W3 * shs[9] * xy +
					SH_C3[2] * SH_W3 * shs[10] * 4.f * 2.f * yz +
					SH_C3[3] * SH_W3 * shs[11] * 3.f * (2.f * zz - xx - yy) +
					SH_C3[4] * SH_W3 * shs[12] * 4.f * 2.f * xz +
					SH_C3[5] * SH_W3 * shs[13] * (xx - yy));
			}
		}
	}
}

__forceinline__ __device__ void computeSHFromSG(int idx, int max_coeffs_sg, const float* shs, float* result_sh_coeffs)
{
	// Initialize SH coeffs to 0
	for (int i = 0; i < 16 * 3; ++i) result_sh_coeffs[i] = 0.0f;

	// shs points to the start of SG params for this Gaussian
	// Layout per lobe (7 floats):
	// [0-2]: Amplitude (RGB)
	// [3-5]: Axis (XYZ)
	// [6]: Sharpness (Scalar)
	const float* params = shs + idx * max_coeffs_sg;
	
	// Determine number of lobes based on stride 7
	// If M is not multiple of 7, this implies error or extra padding, but we assume packed 7.
	int num_lobes = max_coeffs_sg / 7;

	// Temporary storage for SH basis evaluated at lobe axis
	float basis[16];

	// Constants for Normalization: sqrt(4pi / (2l+1))
	const float C_norm[4] = {
		3.544907701811032f,  // l=0: sqrt(4pi)
		2.046653415892977f,  // l=1: sqrt(4pi/3)
		1.585330919042404f,  // l=2: sqrt(4pi/5)
		1.354055224345298f   // l=3: sqrt(4pi/7)
	};

	for (int k = 0; k < num_lobes; ++k)
	{
		// Offset for current lobe
		const float* lobe_params = params + k * 7;
		
		glm::vec3 amplitude = {lobe_params[0], lobe_params[1], lobe_params[2]};
		glm::vec3 axis = {lobe_params[3], lobe_params[4], lobe_params[5]};
		float sharpness = lobe_params[6];

		// Normalize axis
		float axis_len = glm::length(axis);
		if (axis_len > 1e-6f)
			axis = axis / axis_len;
		
		// Evaluate SH basis at lobe axis: Y_lm(xi)
		evalSHBasis(axis, basis);

		// Sharpness lambda (stored as log scale in sharpness?) 
		// User code in scene_model just inits 7 dim tensor. 
		// Usually 3DGS stores "activation" applied later. 
		// Standard: exp(param) to ensure pos. 
		float lambda = exp(sharpness);

		// Band 0 (l=0)
		float g0 = evalZHApprox(0, lambda);
		float scale0 = C_norm[0] * g0;
		result_sh_coeffs[0] += amplitude.x * scale0 * basis[0];
		result_sh_coeffs[1] += amplitude.y * scale0 * basis[0];
		result_sh_coeffs[2] += amplitude.z * scale0 * basis[0];

		// Band 1 (l=1)
		float g1 = evalZHApprox(1, lambda);
		float scale1 = C_norm[1] * g1;
		for (int i = 1; i <= 3; ++i) {
			result_sh_coeffs[i * 3 + 0] += amplitude.x * scale1 * basis[i];
			result_sh_coeffs[i * 3 + 1] += amplitude.y * scale1 * basis[i];
			result_sh_coeffs[i * 3 + 2] += amplitude.z * scale1 * basis[i];
		}

		// Band 2 (l=2)
		float g2 = evalZHApprox(2, lambda);
		float scale2 = C_norm[2] * g2;
		for (int i = 4; i <= 8; ++i) {
			result_sh_coeffs[i * 3 + 0] += amplitude.x * scale2 * basis[i];
			result_sh_coeffs[i * 3 + 1] += amplitude.y * scale2 * basis[i];
			result_sh_coeffs[i * 3 + 2] += amplitude.z * scale2 * basis[i];
		}

		// Band 3 (l=3)
		float g3 = evalZHApprox(3, lambda);
		float scale3 = C_norm[3] * g3;
		for (int i = 9; i <= 15; ++i) {
			result_sh_coeffs[i * 3 + 0] += amplitude.x * scale3 * basis[i];
			result_sh_coeffs[i * 3 + 1] += amplitude.y * scale3 * basis[i];
			result_sh_coeffs[i * 3 + 2] += amplitude.z * scale3 * basis[i];
		}
	}
}

#endif
