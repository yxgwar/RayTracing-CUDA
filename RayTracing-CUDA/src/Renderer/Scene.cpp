#include "Scene.h"
#include "Sphere.h"
#include "SPMaterial.h"
#include "Core/CheckCudaError.h"

namespace RayTracing
{
	__global__ void create_world(Hittable** hit, Material** mat)
	{
#if 1
		int i = 0;
		mat[i++] = new Metal(glm::vec3(0.9f), 0.5f);
		mat[i++] = new Dielectric(1.5f);
		mat[i++] = new Dielectric(1.0f / 1.5f);
		mat[i++] = new Lambertian(glm::vec3{ 0.8f, 0.2f, 0.7f });
		mat[i] = new Lambertian(glm::vec3{ 0.5f, 0.5f, 0.5f });
		i = 0;
		hit[i++] = new Sphere(glm::vec3(0.0f), 0.5f, 3);
		hit[i++] = new Sphere(glm::vec3(1.0f, 0.0f, 0.0f), 0.5f, 1);
		hit[i++] = new Sphere(glm::vec3(1.0f, 0.0f, 0.0f), 0.4f, 2);
		hit[i++] = new Sphere(glm::vec3(-1.0f, 0.0f, 0.0f), 0.5f, 0);
		hit[i] = new Sphere(glm::vec3{ 0.0f, -100.5f, 0.0f }, 100.0f, 4);
#else
		m_Scene.AddMaterials(std::make_shared<Lambertian>(glm::vec3(0.5f, 0.5f, 0.5f)));
		m_Scene.AddObjects(std::make_shared<Sphere>(glm::vec3{ 0.0f, -1000.0f, 0.0f }, 1000.0f, 0));

		for (int a = -11; a < 11; a++) {
			for (int b = -11; b < 11; b++) {
				auto choose_mat = RayMath::Randomf();
				glm::vec3 center{ a + 0.9f * RayMath::Randomf(), 0.2f, b + 0.9f * RayMath::Randomf() };

				if (glm::length(center - glm::vec3{ 4.0f, 0.2f, 0.0f }) > 0.9) {

					if (choose_mat < 0.8) {
						// diffuse
						auto albedo = RayMath::RandomVec() * RayMath::RandomVec();
						m_Scene.AddMaterials(std::make_shared<Lambertian>(albedo));
					}
					else if (choose_mat < 0.95) {
						// metal
						auto albedo = RayMath::RandomVec() * 0.5f + 0.5f;
						auto fuzz = RayMath::Randomf() * 0.5f;
						m_Scene.AddMaterials(std::make_shared<Metal>(albedo, fuzz));
					}
					else {
						// glass
						m_Scene.AddMaterials(std::make_shared<Dielectric>(1.5f));
					}
					m_Scene.AddObjects(std::make_shared<Sphere>(center, 0.2f, m_Scene.LastMaterial()));
				}
			}
		}

		m_Scene.AddMaterials(std::make_shared<Dielectric>(1.5f));
		m_Scene.AddObjects(std::make_shared<Sphere>(glm::vec3{ 0.0f, 1.0f, 0.0f }, 1.0f, m_Scene.LastMaterial()));

		m_Scene.AddMaterials(std::make_shared<Lambertian>(glm::vec3{ 0.4f, 0.2f, 0.1f }));
		m_Scene.AddObjects(std::make_shared<Sphere>(glm::vec3{ -4.0f, 1.0f, 0.0f }, 1.0f, m_Scene.LastMaterial()));

		m_Scene.AddMaterials(std::make_shared<Metal>(glm::vec3{ 0.7f, 0.6f, 0.5f }, 0.0f));
		m_Scene.AddObjects(std::make_shared<Sphere>(glm::vec3{ 4.0f, 1.0f, 0.0f }, 1.0f, m_Scene.LastMaterial()));
#endif
	}

	__global__ void free(Hittable** hit, Material** mat, int h, int m)
	{
		for (int i = 0; i < h; i++)
		{
			delete hit[i];
		}
		delete[] hit;
		for (int i = 0; i < m; i++)
		{
			delete mat[i];
		}
		delete[] mat;
	}

	Scene::~Scene()
	{
		free << <1, 1 >> > (m_Objects, m_Materials, m_Hit, m_Mat);
		checkCudaErrors(cudaFree(m_Objects));
		checkCudaErrors(cudaFree(m_Materials));
	}

	void Scene::CreateWorld(int hit, int mat)
	{
		m_Hit = hit;
		m_Mat = mat;
		checkCudaErrors(cudaMallocManaged(&m_Objects, hit * sizeof(Hittable*)));
		checkCudaErrors(cudaMallocManaged(&m_Materials, mat * sizeof(Material*)));
		create_world << <1, 1 >> > (m_Objects, m_Materials);
		checkCudaErrors(cudaGetLastError());
		checkCudaErrors(cudaDeviceSynchronize());
	}
}