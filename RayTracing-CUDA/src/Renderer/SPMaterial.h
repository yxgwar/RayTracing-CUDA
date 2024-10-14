#pragma once
#include "Material.h"

namespace RayTracing
{
	class Metal :public Material
	{
	public:
		__device__ Metal(const glmcu::vec3& albedo, float fuzz) :m_Albedo(albedo), m_Fuzz(fuzz) {}

		__device__ bool Scatter(Ray& ray, HitData& hitData, glmcu::vec3& color, curandState& rand) override;
	private:
		glmcu::vec3 m_Albedo;
		float m_Fuzz;
	};

	class Lambertian :public Material
	{
	public:
		__device__ Lambertian(const glmcu::vec3& albedo) :m_Albedo(albedo) {}

		__device__ bool Scatter(Ray& ray, HitData& hitData, glmcu::vec3& color, curandState& rand) override;
	private:
		glmcu::vec3 m_Albedo;
	};

	class Dielectric :public Material
	{
	public:
		__device__ Dielectric(float refractionIndex) :m_RefractionIndex(refractionIndex) {}

		__device__ bool Scatter(Ray& ray, HitData& hitData, glmcu::vec3& color, curandState& rand) override;
	private:
		float m_RefractionIndex;
	};

	static __device__ float reflectance(float cosine, float refraction_index) {
		// Use Schlick's approximation for reflectance.
		auto r0 = (1.0f - refraction_index) / (1.0f + refraction_index);
		r0 = r0 * r0;
		return r0 + (1.0f - r0) * std::pow((1.0f - cosine), 5);
	}

	static __device__ glmcu::vec3 refract(glmcu::vec3& direction, glmcu::vec3& normal, float refractionIndex)
	{
		//R'_垂直 = eta / eta' * (R + cos(theta) * n)
		//R'_平行 = -sqrt(1 - R'_垂直^2) * n
		glmcu::vec3 r_per = (direction - glmcu::dot(direction, normal) * normal) * refractionIndex;
		glmcu::vec3 r_par = -sqrt(1.0f - glmcu::dot(r_per, r_per)) * normal;
		return r_per + r_par;
	}

	//镜面反射
	__device__ bool Metal::Scatter(Ray& ray, HitData& hitData, glmcu::vec3& color, curandState& rand)
	{
		//OUT = IN - 2*dot(IN,N)*N
		glmcu::vec3 re = glmcu::reflect(ray.direction, hitData.normal) + randomv(rand) * m_Fuzz;
		ray.direction = glmcu::normalize(re);
		//ray.direction = glmcu::normalize(glmcu::reflect(ray.direction, hitData.normal) + randomv(rand) * m_Fuzz);
		ray.origin = hitData.hitPosition + ray.direction * 0.00001f;
		color = m_Albedo;
		return glmcu::dot(hitData.normal, ray.direction) > 0;
	}

	//漫反射
	__device__ bool Lambertian::Scatter(Ray& ray, HitData& hitData, glmcu::vec3& color, curandState& rand)
	{
		glmcu::vec3 diffuse = hitData.normal + randomv(rand);
		ray.direction = glmcu::normalize(diffuse);
		ray.origin = hitData.hitPosition + ray.direction * 0.00001f;
		color = m_Albedo;
		return true;
	}

	//折射
	__device__ bool Dielectric::Scatter(Ray& ray, HitData& hitData, glmcu::vec3& color, curandState& rand)
	{
#if 1
		float theta = glmcu::dot(ray.direction, hitData.normal);
		float refraction;
		glmcu::vec3 nn;
		if (theta < 0.0f)
		{
			//从外向内射折射率取倒数
			refraction = 1.0f / m_RefractionIndex;
			nn = hitData.normal;
		}
		else
		{
			refraction = m_RefractionIndex;
			nn = -hitData.normal;
		}
		//判断是折射还是全反射
		//m_RefractionIndex * sin(theta) > 1为全反射
		if (sqrt(1.0f - theta * theta) * refraction > 1 || reflectance(-theta, refraction) > randomf(rand))
			ray.direction = glmcu::reflect(ray.direction, nn);
		else
			ray.direction = refract(ray.direction, nn, refraction);
		ray.origin = hitData.hitPosition + ray.direction * 0.00001f;
		color = glmcu::vec3(1.0f);
		return true;
#else
		glmcu::vec3 outward_normal;
		glmcu::vec3 reflected = glmcu::reflect(ray.direction, hitData.normal);
		float ni_over_nt;
		color = glmcu::vec3(1.0f, 1.0f, 1.0f);
		glmcu::vec3 refracted;
		float reflect_prob;
		float cosine;
		if (glmcu::dot(ray.direction, hitData.normal) > 0.0f) {
			outward_normal = -hitData.normal;
			ni_over_nt = m_RefractionIndex;
			cosine = dot(ray.direction, hitData.normal) / ray.direction.length();
			cosine = sqrt(1.0f - m_RefractionIndex * m_RefractionIndex * (1 - cosine * cosine));
		}
		else {
			outward_normal = hitData.normal;
			ni_over_nt = 1.0f / m_RefractionIndex;
			cosine = -glmcu::dot(ray.direction, hitData.normal) / ray.direction.length();
		}
		if (refract(ray.direction, outward_normal, ni_over_nt, refracted))
			reflect_prob = reflectance(cosine, m_RefractionIndex);
		else
			reflect_prob = 1.0f;
		if (randomf(rand) < reflect_prob)
			ray.direction = reflected;
		else
			ray.direction = refracted;
		ray.origin = hitData.hitPosition + ray.direction * 0.00001f;
		return true;
#endif
	}
}