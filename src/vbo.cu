// Copyright (C) 2019 Xiao Zhai
// 
// This file is part of CPP-Fluid-Particles.
// 
// CPP-Fluid-Particles is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
// 
// CPP-Fluid-Particles is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
// 
// You should have received a copy of the GNU General Public License
// along with CPP-Fluid-Particles.  If not, see <http://www.gnu.org/licenses/>.

#include <vector>
#include <memory>
#include <cuda_runtime.h>
#include "helper_math.h"
#include "DArray.h"
#include "Particles.h"
#include "SPHParticles.h"

__global__ void generate_dots_CUDA(float3* dot, float3* posColor, float3* pos, float* density, const int num)
{
	const unsigned int i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (i >= num) return;

	dot[i] = pos[i];
	float a = density[i] * 0.01;


	// purple -> black
	if (a > -500 && a <= -1000)
		posColor[i] = make_float3(0.5 - (-a - 500.0) / 500.0, 0.0, 0.5 - (-a - 500.0) / 500.0);

	// red -> purple
	else if (a > -500 && a <= 0)
		posColor[i] = make_float3(1.0 - (-a / 500.0), 0.0, -a / 1000.0);

	// ###### low temperature ######


	// ###### high temperature ######

	// red -> yellow
	else if (a > 0 && a <= 500)
		posColor[i] = make_float3(1.0, a / 500.0, 0.0);

	// yellow -> white
	else if (a > 500 && a <= 1000)
		posColor[i] = make_float3(1.0, 1.0, (a - 500.0) / 500.0);



	/*
	// pink
	if (a > -900 && a <= -800)
		posColor[i] = make_float3(1.0, 0.0, -a - 8.0);
	else if (a > -800 && a <= -700)
		posColor[i] = make_float3(0.75, 0.0, -a - 7.0);
	else if (a > -700 && a <= -600)
		posColor[i] = make_float3(0.5, 0.0, -a - 6.0);
	else if (a > -600 && a <= -500)
		posColor[i] = make_float3(0.25, 0.0, -a - 5.0);

	// sky blue
	else if (a > -500 && a <= -400)
		posColor[i] = make_float3(0.0, 1.0, -a - 4.0);
	else if (a > -400 && a <= -300)
		posColor[i] = make_float3(0.0, 0.75, -a - 3.0);
	else if (a > -300 && a <= -200)
		posColor[i] = make_float3(0.0, 0.5, -a - 2.0);
	else if (a > -200 && a <= -100)
		posColor[i] = make_float3(0.0, 0.25, -a - 1.0);

	// blue
	else if (a > -100 && a <= 0)
		posColor[i] = make_float3(0.0, 0.0, -a);

	// ###### low temperature ######


	// ###### high temperature ######

	// red
	else if (a > 0 && a <= 100)
		posColor[i] = make_float3(a, 0.0, 0.0);

	// orange, yellow
	else if (a > 100 && a <= 200)
		posColor[i] = make_float3(a - 1.0, 0.25, 0.0);
	else if (a > 200 && a <= 300)
		posColor[i] = make_float3(a - 2.0, 0.5, 0.0);
	else if (a > 300 && a <= 400)
		posColor[i] = make_float3(a - 3.0, 0.75, 0.0);
	else if (a > 400 && a <= 500)
		posColor[i] = make_float3(a - 4.0, 1.0, 0.0);

	// white
	else if (a > 500 && a <= 600)
		posColor[i] = make_float3(a - 5.0, 1.0, 0.25);
	else if (a > 600 && a <= 700)
		posColor[i] = make_float3(a - 6.0, 1.0, 0.5);
	else if (a > 700 && a <= 800)
		posColor[i] = make_float3(a - 7.0, 1.0, 0.75);
	else if (a > 800 && a <= 900)
		posColor[i] = make_float3(a - 8.0, 1.0, 1.0);
	*/



	//printf("temp = %f\n", density[i]);

	/*
	if (density[i] < 0.75f)	{
		posColor[i] = make_float3(0.34f, 0.46f, 0.7f);
	}
	else if (density[i] < 1.0f) {
		const auto w = (density[i] - 0.75f) * 4.0f;
		posColor[i] = w * make_float3(0.9f) + (1 - w) * make_float3(0.34f, 0.46f, 0.7f);
	}
	else {
		auto w = (powf(density[i], 2) - 1.0f)*4.0f;
		w = fminf(w, 1.0f);
		posColor[i] = (1-w)*make_float3(0.9f) + w*make_float3(1.0f, 0.4f, 0.7f);
	}*/
}

extern "C" void generate_dots(float3* dot, float3* color, const std::shared_ptr<SPHParticles> particles) {
	generate_dots_CUDA <<<(particles->size() - 1) / block_size + 1, block_size >>> 
												//						//
		(dot, color, particles->getPosPtr(), particles->getTemperaturePtr(), particles->size());
	cudaDeviceSynchronize(); CHECK_KERNEL();
	return;
}