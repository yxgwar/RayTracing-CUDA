#pragma once
#include "Ray.h"
#include <memory>

namespace RayTracing
{
	class Material;
	
	struct HitData
	{
		glm::vec3 hitPosition;
		glm::vec3 normal;
		float t;
		int index;
	};

	class Hittable
	{
	public:
		virtual ~Hittable() = default;
		virtual bool IsHit(const Ray& ray, HitData& hitData) = 0;
	};

	class Sphere :public Hittable
	{
	public:
		Sphere(const glm::vec3& position, float radius, int index);
		~Sphere() = default;

		bool IsHit(const Ray& ray, HitData& hitData) override;
	private:
		glm::vec3 m_Position{ 0.0f };
		float m_Radius = 0.5f;
		int m_Index;
	};
}