#pragma once
#include "Hittable.h"
#include <curand_kernel.h>

namespace RayTracing
{
	class Material
	{
	public:
		virtual ~Material() = default;

		__host__ __device__ virtual bool Scatter(Ray& ray, HitData& hitData, glm::vec3& color, curandState rand) = 0;
	};

	class Metal :public Material
	{
	public:
		Metal(const glm::vec3& albedo, float fuzz) :m_Albedo(albedo), m_Fuzz(fuzz) {}
		~Metal() = default;

		__host__ __device__ bool Scatter(Ray& ray, HitData& hitData, glm::vec3& color, curandState rand) override;
	private:
		glm::vec3 m_Albedo;
		float m_Fuzz;
	};

	class Lambertian :public Material
	{
	public:
		Lambertian(const glm::vec3& albedo) :m_Albedo(albedo) {}
		~Lambertian() = default;

		__host__ __device__ bool Scatter(Ray& ray, HitData& hitData, glm::vec3& color, curandState rand) override;
	private:
		glm::vec3 m_Albedo;
	};

	class Dielectric :public Material
	{
	public:
		Dielectric(float refractionIndex) :m_RefractionIndex(refractionIndex) {}
		~Dielectric() = default;

		__host__ __device__ bool Scatter(Ray& ray, HitData& hitData, glm::vec3& color, curandState rand) override;
	private:
		float m_RefractionIndex;
	};
}