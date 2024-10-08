#pragma once
#include "Hittable.h"
#include "RayMath.h"
#include <curand_kernel.h>

namespace RayTracing
{
	class Material
	{
	public:
		__device__ virtual ~Material() = default;

		__device__ virtual bool Scatter(Ray& ray, HitData& hitData, glm::vec3& color, curandState rand) = 0;
	};
}