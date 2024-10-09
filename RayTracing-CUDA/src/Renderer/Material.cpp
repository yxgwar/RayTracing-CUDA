#include "Material.h"
#include "RayMath.h"

namespace RayTracing
{
	static float reflectance(float cosine, float refraction_index) {
		// Use Schlick's approximation for reflectance.
		auto r0 = (1 - refraction_index) / (1 + refraction_index);
		r0 = r0 * r0;
		return r0 + (1 - r0) * std::pow((1 - cosine), 5);
	}

	static glm::vec3 refract(glm::vec3& direction, glm::vec3& normal, float refractionIndex)
	{
		//R'_��ֱ = eta / eta' * (R + cos(theta) * n)
		//R'_ƽ�� = -sqrt(1 - R'_��ֱ^2) * n
		glm::vec3 r_per = refractionIndex * (direction - glm::dot(direction, normal) * normal);
		glm::vec3 r_par = -glm::sqrt(1.0f - glm::dot(r_per, r_per)) * normal;
		return r_per + r_par;
	}

	//���淴��
	bool Metal::Scatter(Ray& ray, HitData& hitData, glm::vec3& color)
	{
		//OUT = IN - 2*(IN*N)*N
		ray.direction = glm::normalize(glm::reflect(ray.direction, hitData.normal) + m_Fuzz * RayMath::RandomVec());
		ray.origin = hitData.hitPosition + ray.direction * 0.00001f;
		color = m_Albedo;
		return glm::dot(hitData.normal, ray.direction) > 0;
	}

	//������
	bool Lambertian::Scatter(Ray& ray, HitData& hitData, glm::vec3& color)
	{
		glm::vec3 diffuse = hitData.normal + RayMath::RandomVec();
		diffuse = glm::length(diffuse) < 1e-6 ? hitData.normal : diffuse;
		ray.direction = glm::normalize(diffuse);
		ray.origin = hitData.hitPosition + diffuse * 0.00001f;
		color = m_Albedo;
		return true;
	}

	//����
	bool Dielectric::Scatter(Ray& ray, HitData& hitData, glm::vec3& color)
	{
		float theta = glm::dot(ray.direction, hitData.normal);
		//����������������ȡ����
		float refraction = theta < 0.0f ? 1.0f / m_RefractionIndex : m_RefractionIndex;
		//�ж������仹��ȫ����
		//m_RefractionIndex * sin(theta) > 1Ϊȫ����
		if (sqrt(1.0f - theta * theta) * refraction > 1 || reflectance(-theta, refraction) > RayMath::Randomf())
			ray.direction = glm::reflect(ray.direction, hitData.normal);
		else
			ray.direction = refract(ray.direction, hitData.normal, refraction);
		ray.origin = hitData.hitPosition + ray.direction * 0.00001f;
		color = glm::vec3(1.0f);
		return true;
	}
}