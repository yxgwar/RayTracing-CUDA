#pragma once
#include "Ray.h"
#include <memory>

namespace RayTracing
{
	class Material;
	
	struct HitData
	{
		glmcu::vec3 hitPosition;
		glmcu::vec3 normal;
		float t;
		int index;
	};

	class Hittable
	{
	public:
		__device__ virtual ~Hittable() = default;
		__device__ virtual bool IsHit(Ray& ray, HitData& hitData) = 0;
		__host__ __device__ virtual int GetIndex() = 0;
	};
}