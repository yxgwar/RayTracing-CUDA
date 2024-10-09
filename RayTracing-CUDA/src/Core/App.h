#pragma once

#include "Renderer/Camera.h"
#include "Renderer/Scene.h"

namespace RayTracing
{
	class App
	{
	public:
		App(int width = 1920, int height = 1080, int channels = 4);
		~App() = default;

		void Run();
	private:
		Camera m_Camera;
		Scene m_Scene;

		int m_Width, m_Height, m_Channels;
		float m_AspectRatio;
	};
}