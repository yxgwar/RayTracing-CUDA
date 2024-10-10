#include "Scene.h"

namespace RayTracing
{
	Scene::~Scene()
	{
		for (int i = 0; i < m_Objects.size(); i++) 
			delete m_Objects[i];
		m_Objects.clear();

		for (int i = 0; i < m_Material.size(); i++)
			delete m_Material[i];
		m_Material.clear();
	}

	

	__device__ bool Scene::IsHit(Ray& ray, HitData& hitData)
	{
		HitData temp;
		bool isHit = false;
		float mint = FLT_MAX;
		for (auto& object : m_Objects)
		{
			if (object->IsHit(ray, temp))
			{
				isHit = true;
				if (temp.t < mint)
				{
					hitData = temp;
					mint = temp.t;
				}
			}
		}
		return isHit;
	}
}