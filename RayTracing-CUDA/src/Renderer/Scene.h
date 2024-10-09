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

		void AddObjects(std::shared_ptr<Sphere> sphere) { m_Objects.push_back(sphere); }
		void AddMaterials(std::shared_ptr<Material> material) { m_Material.push_back(material); }

		inline int LastMaterial() { return m_Material.size() - 1; }
		//inline const std::vector<std::shared_ptr<Hittable>>& GetObjects() const { return m_Objects; }

		bool IsHit(const Ray& ray, HitData& hitData);
		bool Scatter(Ray& ray, HitData& hitData, glm::vec3& color) { return m_Material[hitData.index]->Scatter(ray, hitData, color); }
	private:
		std::vector<std::shared_ptr<Hittable>> m_Objects;
		std::vector<std::shared_ptr<Material>> m_Material;
	};
}