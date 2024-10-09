#pragma once

#include "Image.h"
#include "Camera.h"
#include "Scene.h"
#include "Ray.h"
#include <glm/glm.hpp>

namespace RayTracing
{
	class Render
	{
	public:
		static void StartRendering(Scene& scene, Camera& camera, Image& image, int width, int height, int channels);
	private:
		static glm::vec4 perPixel(Scene& scene, Ray& ray);
	};
}