#pragma once
#include "Hittable.h"
#include "Math/Mathcu.h"

namespace RayTracing
{
	static __device__ glmcu::vec3 normalizeS(glmcu::vec3& v)
	{
		float k = 1.0f / sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
		return glmcu::vec3{ v[0] * k,v[1] * k,v[2] * k };
	}

	class Sphere :public Hittable
	{
	public:
		__device__ Sphere(glmcu::vec3& position, float radius, int index)
			:m_Position(position), m_Radius(radius), m_Index(index) {};

		__device__ bool IsHit(Ray& ray, HitData& hitData, float cloest) override;
		__host__ __device__ virtual int GetIndex() override { return m_Index; }
	private:
		glmcu::vec3 m_Position{ 0.0f };
		float m_Radius = 0.5f;
		int m_Index;
	};

	__device__ bool Sphere::IsHit(Ray& ray, HitData& hitData, float cloest)
	{
		glmcu::vec3 oc = ray.origin - m_Position;

		/*
		* (x - P.x)^2 + (y - P.y)^2 + (z - P.z)^2 = r^2
		* (HitPosition - m_Position)^2 = r^2
		* HitPosition = rayOrigin + rayDirection * t
		* a = rDrD
		* b = 2rD(rO - P)
		* c = (rO - P)(rO - P) - r^2
		*/

		float a = glmcu::dot(ray.direction, ray.direction);
		float b2 = glmcu::dot(oc, ray.direction);
		float c = glmcu::dot(oc, oc) - m_Radius * m_Radius;

		float discriminant = b2 * b2 - a * c;

		if (discriminant < 0)
			return false;
		else
		{
			float t = (-b2 - sqrt(discriminant)) / a;
			if (t < 0 || t > cloest)
				return false;
			glmcu::vec3 hitPosition = ray.origin + ray.direction * t;
			glmcu::vec3 nor = hitPosition - m_Position;
			glmcu::vec3 normal = nor / m_Radius;

			hitData.hitPosition = hitPosition;
			hitData.normal = normal;
			hitData.t = t;
			hitData.index = m_Index;
			return true;
		}
	}
}