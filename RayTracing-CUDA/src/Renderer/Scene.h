#pragma once
#include "Hittable.h"
#include "Material.h"

#include <vector>
#include <glm/glm.hpp>
#include <memory>

namespace RayTracing
{
	class Scene 
	{
	public:
		Scene() = default;
		~Scene();

		void AddObjects(Sphere* sphere) { m_Objects.push_back(sphere); }
		void AddMaterials(Material* material) { m_Material.push_back(material); }

		inline int LastMaterial() { return (int)m_Material.size() - 1; }
		inline const std::vector<Hittable*> GetObjects() const { return m_Objects; }
		inline const std::vector<Material*> GetMaterial() const { return m_Material; }
		//inline const std::vector<std::shared_ptr<Hittable>>& GetObjects() const { return m_Objects; }

		bool IsHit(Ray& ray, HitData& hitData);
		bool Scatter(Ray& ray, HitData& hitData, glm::vec3& color, curandState rand) { return m_Material[hitData.index]->Scatter(ray, hitData, color, rand); }
	private:
		std::vector<Hittable*> m_Objects;
		std::vector<Material*> m_Material;
	};
}