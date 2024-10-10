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
		virtual ~Hittable() = default;
		__device__ virtual bool IsHit(Ray& ray, HitData& hitData) = 0;
	};

	class Sphere :public Hittable
	{
	public:
		Sphere(const glm::vec3& position, float radius, int index);
		~Sphere() = default;

		__device__ bool IsHit(Ray& ray, HitData& hitData) override;
	private:
		glm::vec3 m_Position{ 0.0f };
		float m_Radius = 0.5f;
		int m_Index;
	};
}