#include "Scene.h"

namespace RayTracing
{
	Scene::~Scene()
	{
		m_Objects.clear();
	}

	bool Scene::IsHit(const Ray& ray, HitData& hitData)
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