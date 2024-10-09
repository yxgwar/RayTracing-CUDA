#include "Render.h"
#include "RayMath.h"
#include "Material.h"
#include <thread>

namespace RayTracing
{
	void Render::StartRendering(Scene& scene, Camera& camera, Image& image, int width, int height, int channels)
	{
		glm::vec3 rayOrigin = camera.GetOrigin();
		int samplers = 100;
#if 1
		std::size_t num_threads = std::thread::hardware_concurrency();
		std::vector<std::thread> threads;
		int total = width * height;
		int step = total / num_threads;
		for (int th = 0; th < num_threads; th++)
		{
			threads.emplace_back([&, th]()
				{
					Ray ray;
					ray.origin = rayOrigin;
					int start = th * step;
					int end = (th == num_threads - 1) ? total : (th + 1) * step;
					for (int k = start; k < end; k++) {
						int i = k % width;
						int j = k / width;
						glm::vec4 colorS(0.0f);
						for (int s = 0; s < samplers; s++)
						{
							int x = RayMath::clamp(i + RayMath::RandomI(), 0, width - 1);
							int y = RayMath::clamp(j + RayMath::RandomI(), 0, height - 1);
							ray.direction = camera.GetRayDirections()[x + y * width];
							colorS += perPixel(scene, ray);
						}

						glm::vec4 color = colorS / float(samplers);
						color = glm::clamp(color, glm::vec4(0.0f), glm::vec4(1.0f));//将颜色限制在0~255
						image.SetPixelData(color, k);
					}
				});
		}

		for (auto& thread : threads)
			thread.join();
#else
		Ray ray;
		ray.origin = rayOrigin;

		for (int j = 0; j < height; j++) {
			for (int i = 0; i < width; i++) {
				glm::vec4 colorS(0.0f);
				for (int s = 0; s < samplers; s++)
				{
					int x = RayMath::clamp(i + RayMath::RandomI(), 0, width - 1);
					int y = RayMath::clamp(j + RayMath::RandomI(), 0, height - 1);
					ray.direction = camera.GetRayDirections()[x + y * width];
					colorS += perPixel(scene, ray);
				}

				glm::vec4 color = colorS / float(samplers);
				color = glm::clamp(color, glm::vec4(0.0f), glm::vec4(1.0f));//将颜色限制在0~255
				image.SetPixelData(color, i + j * width);
			}
		}
#endif
	}

	glm::vec4 Render::perPixel(Scene& scene, Ray& ray)
	{
		HitData hitData;
		Ray traceRay(ray);

		glm::vec3 color(1.0f);
		int bounces = 10;
		for (size_t i = 0; i < bounces; i++)
		{
			if (scene.IsHit(traceRay, hitData))
			{
				glm::vec3 rColor;
				if (scene.Scatter(traceRay, hitData, rColor))
					color *= rColor;
				else
				{
					color = glm::vec3(0.0f);
					break;
				}
			}
			else
			{
				glm::vec3 dir = traceRay.direction;
				float a = (dir.y + 1.0f) * 0.5f;
				color *= (1.0f - a) * glm::vec3(1.0f) + a * glm::vec3(0.3f, 0.5f, 1.0f);
				break;
			}
		}
		return { color, 1.0f };
	}
}