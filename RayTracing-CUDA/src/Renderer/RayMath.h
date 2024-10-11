#pragma once

#include "Math/Mathcu.h"
#include <glm/glm.hpp>
#include <curand_kernel.h>

namespace RayTracing
{
	static __device__ float sqrtcuR(float x)
	{
		float xhalf = 0.5f * x;
		int i = *(int*)&x;

		if (!x) return 0;

		i = 0x5f375a86 - (i >> 1); // beautiful number
		x = *(float*)&i;
		x = x * (1.5f - xhalf * x * x); // 牛顿迭代法，提高精度
		x = x * (1.5f - xhalf * x * x); // 牛顿迭代法，提高精度
		x = x * (1.5f - xhalf * x * x); // 牛顿迭代法，提高精度

		return 1 / x;
	}

	static __device__ glmcu::vec3 normalizeR(glmcu::vec3& v)
	{
		float  x = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
		float k = 1.0f / sqrtcuR(x);
		return glmcu::vec3{ v[0] * k,v[1] * k,v[2] * k };
	}

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
			if (l < 1.0f)
				return rv;
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