#include "App.h"
#include "Log.h"
#include "Renderer/Image.h"
#include "Renderer/Render.h"
#include "Renderer/RayMath.h"
#include "Renderer/Material.h"
#include <chrono>

namespace RayTracing
{
	App::App(int width, int height, int channels)
		:m_Width(width), m_Height(height), m_Channels(channels), m_AspectRatio((float)width / height),
		m_Camera(45.0f, 0.1f, 100.0f, float(width), float(height))
	{
		RAY_INFO("Start!");
	}

	void App::Run()
	{
		RayMath::Init();
		Image image = Image(m_Width, m_Height);

#if 1
		auto metal = std::make_shared<Metal>(glm::vec3(0.9f), 0.5f);
		auto right = std::make_shared<Dielectric>(1.5f);
		auto bubble = std::make_shared<Dielectric>(1.0f / 1.5f);
		auto center = std::make_shared<Lambertian>(glm::vec3{ 0.8f, 0.2f, 0.7f });
		auto land = std::make_shared<Lambertian>(glm::vec3{ 0.5f, 0.5f, 0.5f });
		m_Scene.AddMaterials(metal);
		m_Scene.AddMaterials(right);
		m_Scene.AddMaterials(bubble);
		m_Scene.AddMaterials(center);
		m_Scene.AddMaterials(land);
		m_Scene.AddObjects(std::make_shared<Sphere>(glm::vec3(0.0f), 0.5f, 3));
		m_Scene.AddObjects(std::make_shared<Sphere>(glm::vec3(1.0f, 0.0f, 0.0f), 0.5f, 1));
		m_Scene.AddObjects(std::make_shared<Sphere>(glm::vec3(1.0f, 0.0f, 0.0f), 0.4f, 2));
		m_Scene.AddObjects(std::make_shared<Sphere>(glm::vec3(-1.0f, 0.0f, 0.0f), 0.5f, 0));
		m_Scene.AddObjects(std::make_shared<Sphere>(glm::vec3{ 0.0f, -100.5f, 0.0f }, 100.0f, 4));
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

		RAY_INFO("Start setting data!");

		auto start = std::chrono::high_resolution_clock::now();
		m_Camera.SetPosition({ 0.0f, 0.0f, 3.0f });
		Render::StartRendering(m_Scene, m_Camera, image, m_Width, m_Height, m_Channels);
		auto end = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double> duration = end - start;
		std::cout << "Time taken: " << duration.count() << " seconds" << std::endl;

		RAY_INFO("Start Generating image!");
		if (image.GenerateImage())
			RAY_INFO("Success!");
		else
			RAY_ERROR("Fail!");
	}
}