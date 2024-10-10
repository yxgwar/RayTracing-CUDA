#include "Material.h"
#include "RayMath.h"
#include "Core/CheckCudaError.h"
#include "Math/Mathcu.h"

namespace RayTracing
{
	static __host__ __device__ float reflectance(float cosine, float refraction_index) {
		// Use Schlick's approximation for reflectance.
		auto r0 = (1 - refraction_index) / (1 + refraction_index);
		r0 = r0 * r0;
		return r0 + (1 - r0) * std::pow((1 - cosine), 5);
	}

	__host__ __device__ static glmcu::vec3 refract(glmcu::vec3& direction, glmcu::vec3& normal, float refractionIndex)
	{
		//R'_垂直 = eta / eta' * (R + cos(theta) * n)
		//R'_平行 = -sqrt(1 - R'_垂直^2) * n
		glmcu::vec3 r_per = (direction - glmcu::dot(direction, normal) * normal) * refractionIndex;
		glmcu::vec3 r_par = -sqrt(1.0f - glmcu::dot(r_per, r_per)) * normal;
		return r_per + r_par;
	}

	//镜面反射
	__host__ __device__ bool Metal::Scatter(Ray& ray, HitData& hitData, glm::vec3& color, curandState rand)
	{
		//OUT = IN - 2*(IN*N)*N
		ray.direction = glmcu::normalize(glmcu::reflect(ray.direction, hitData.normal) + randomv(rand) * m_Fuzz);
		ray.origin = hitData.hitPosition + ray.direction * 0.00001f;
		color = m_Albedo;
		return glmcu::dot(hitData.normal, ray.direction) > 0;
	}

	//漫反射
	__host__ __device__ bool Lambertian::Scatter(Ray& ray, HitData& hitData, glm::vec3& color, curandState rand)
	{
		glmcu::vec3 diffuse = hitData.normal + randomv(rand);
		diffuse = diffuse.length() < 1e-6 ? hitData.normal : diffuse;
		ray.direction = glmcu::normalize(diffuse);
		ray.origin = hitData.hitPosition + diffuse * 0.00001f;
		color = m_Albedo;
		return true;
	}

	//折射
	__host__ __device__ bool Dielectric::Scatter(Ray& ray, HitData& hitData, glm::vec3& color, curandState rand)
	{
		float theta = glmcu::dot(ray.direction, hitData.normal);
		//从外向内射折射率取倒数
		float refraction = theta < 0.0f ? 1.0f / m_RefractionIndex : m_RefractionIndex;
		//判断是折射还是全反射
		//m_RefractionIndex * sin(theta) > 1为全反射
		if (sqrt(1.0f - theta * theta) * refraction > 1 || reflectance(-theta, refraction) > randomf(rand))
			ray.direction = glmcu::reflect(ray.direction, hitData.normal);
		else
			ray.direction = refract(ray.direction, hitData.normal, refraction);
		ray.origin = hitData.hitPosition + ray.direction * 0.00001f;
		color = glm::vec3(1.0f);
		return true;
	}
}