#include "Scene.h"
//#include "Sphere.h"

#include "Core/CheckCudaError.h"

namespace RayTracing
{
	Scene::Scene()
	{
		m_Hit = 0;
		m_Mat = 0;
		m_Objects = nullptr;
		m_Materials = nullptr;
	}

	Scene::~Scene()
	{
		checkCudaErrors(cudaFree(m_Objects));
		checkCudaErrors(cudaFree(m_Materials));
	}

	void Scene::CreateWorld(int hit, int mat)
	{
		m_Hit = hit;
		m_Mat = mat;
		checkCudaErrors(cudaMallocManaged(&m_Objects, hit * sizeof(Hittable*)));
		checkCudaErrors(cudaMallocManaged(&m_Materials, mat * sizeof(Material*)));
		//create_world << <1, 1 >> > (m_Objects, m_Materials);
		//checkCudaErrors(cudaGetLastError());
		//checkCudaErrors(cudaDeviceSynchronize());
	}
}