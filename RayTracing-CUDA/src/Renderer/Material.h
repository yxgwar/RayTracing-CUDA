#pragma once
#include "Hittable.h"
#include "RayMath.h"
#include <curand_kernel.h>

namespace RayTracing
{
	class Material
	{
	public:
		__device__ virtual bool Scatter(Ray& ray, HitData& hitData, glmcu::vec3& color, curandState& rand) = 0;
	};
}