#include "App.h"
#include "Log.h"
#include "Renderer/Image.h"
#include "Renderer/Render.h"
#include <chrono>

namespace RayTracing
{
	App::App(int width, int height, int channels)
		:m_Width(width), m_Height(height), m_Channels(channels), m_AspectRatio((float)width / height)
		/*m_Camera(30.0, 0.1f, 100.0f, width, height), m_Scene()*/
	{
		RAY_INFO("Start!");
	}

	void App::Run()
	{
		Image image = Image(m_Width, m_Height);

		//m_Scene.CreateWorld(4, 4);
		m_Scene.CreateWorld(22 * 22 + 1 + 3, 22 * 22 + 1 + 3);

		RAY_INFO("Start setting data!");

		auto start = std::chrono::high_resolution_clock::now();
		//m_Camera.SetPosition({ 0.0f, 0.0f, 3.0f });
		//m_Camera.SetPosition({ 13.0f, 2.0f, 3.0f });
		auto end = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double> duration = end - start;
		std::cout << "Camera Time taken: " << duration.count() << " seconds" << std::endl;

		Render::StartRendering(m_Scene, /*m_Camera, */image, m_Width, m_Height, m_Channels);
		auto end1 = std::chrono::high_resolution_clock::now();
		duration = end1 - end;
		std::cout << "Render Time taken: " << duration.count() << " seconds" << std::endl;

		RAY_INFO("Start Generating image!");
		if (image.GenerateImage())
			RAY_INFO("Success!");
		else
			RAY_ERROR("Fail!");
	}
}