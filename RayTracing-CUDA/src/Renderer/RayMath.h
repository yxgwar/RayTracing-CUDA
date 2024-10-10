#pragma once

#include "Math/Mathcu.h"
#include <glm/glm.hpp>
#include <curand_kernel.h>

namespace RayTracing
{
	//return float 0~1
	static __device__ float randomf(curandState state)
	{
		return curand_uniform(&state);
	}

	//return vec3 -1~1
	static __device__ glmcu::vec3 randomv(curandState state)
	{
		while (true)
		{
			glmcu::vec3 rv = { curand_uniform(&state), curand_uniform(&state), curand_uniform(&state) };
			float l = rv.length();
			if (l <= 1.0f && l >= 1e-6)
				return rv / l;
		}
	}

	//return int -1~1
	static __device__ int randomi(curandState state)
	{
		float r = curand_uniform(&state);
		if (r < 0.33f)
			return -1;
		else if (r < 0.66f)
			return 0;
		else return 1;
	}

	static __device__ int clamp(int a, int min, int max)
	{
		if (a < min)
			return min;
		if (a > max)
			return max;
		return a;
	}
}