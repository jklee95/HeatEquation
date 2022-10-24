#pragma once


static inline __device__ float cubic_spline_kernel_laplacian(const float r, const float radius)
{
	const auto q = 2.0f * fabs(r) / radius;
	//return (q < EPSILON) ? 0.0f :
	//	((q) <= 1.0f ? (powf(2.0f - q, 3) - 4.0f * powf(1.0f - q, 3)) :
	//	(q) <= 2.0f ? (powf(2.0f - q, 3)) :
	//		0.0f) / (4.0f * PI * powf(radius, 3));
	if (q > 2.0f || q < EPSILON) return 0.0f;
	else
	{
		const auto a = 4 / (PI * (q + EPSILON) * radius * radius * radius * radius * radius);
		return a * ((q > 1.0f) ? ((3 - q) * q - 2)
			: ((3 * q - 3) * q));
	}
}



__device__ void contributeFluidTemerpature(float* temperature, const int i, float3* pos, const int cellStart, const int cellEnd, const float radius, const float dt)
{
	auto j = cellStart;
	float sph_func, dT, coeff = 0.001f;
	while (j < cellEnd)
	{
		// dT/dt
		sph_func = (temperature[i] - temperature[j]) * cubic_spline_kernel(length(pos[i] - pos[j]), radius);
		dT = coeff * sph_func * dt;

		//T(i+1) = T(i) + dT
		temperature[i] += dT;

		++j;
	}
	//printf("temp %2d = %f\n", i, temperature[i]);
	return;
}



__global__ void computeTemperature_CUDA(float* massFluid, float* temperature, const int num, float3* posFluid,
	int* cellStartFluid, int3 cellSize, const float cellLength, const float radius, const float dt)
{
	const unsigned int i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (i >= num) return;
	__syncthreads();
#pragma unroll
	for (auto m = 0; m < 27; __syncthreads(), ++m)
	{
		const auto cellID = particlePos2cellIdx(
			make_int3(posFluid[i] / cellLength) + make_int3(m / 9 - 1, (m % 9) / 3 - 1, m % 3 - 1), cellSize);
		if (cellID == (cellSize.x * cellSize.y * cellSize.z)) continue;
		contributeFluidTemerpature(temperature, i, posFluid, cellStartFluid[cellID], cellStartFluid[cellID + 1], radius, dt);
	}
	return;
}

void heat(std::shared_ptr<SPHParticles>& fluids, const DArray<int>& cellStartFluid, 
	int3 cellSize, float cellLength, float radius, float dt)
{
	int num = fluids->size();

	computeTemperature_CUDA <<< ((fluids->size()) - 1) / block_size + 1, block_size >>> (fluids->getMassPtr(), fluids->getTemperaturePtr(), num,
		fluids->getPosPtr(), cellStartFluid.addr(), cellSize, cellLength, radius,dt);
		
}