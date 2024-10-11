#pragma once

#include <vector>
#include <glm/glm.hpp>
#include <memory>

namespace RayTracing
{
	class Hittable;
	class Material;

	class Scene 
	{
	public:
		Scene();
		~Scene();

		void CreateWorld(int hit, int mat);

		//void AddObjects(Sphere* sphere) { m_Objects.push_back(sphere); }
		//void AddMaterials(Material* material) { m_Material.push_back(material); }

		//inline int LastMaterial() { return (int)m_Material.size() - 1; }
		inline Hittable** GetHit() const { return m_Objects; }
		inline Material** GetMat() const { return m_Materials; }
		inline int GetHitCount() const { return m_Hit; }
		inline int GetMatCount() const { return m_Mat; }

		//bool IsHit(Ray& ray, HitData& hitData);
		//bool Scatter(Ray& ray, HitData& hitData, glm::vec3& color, curandState rand) { return m_Material[hitData.index]->Scatter(ray, hitData, color, rand); }
	private:
		/*std::vector<Hittable*> m_Objects;
		std::vector<Material*> m_Material;*/
		Hittable** m_Objects;
		Material** m_Materials;
		int m_Hit;
		int m_Mat;
	};
}