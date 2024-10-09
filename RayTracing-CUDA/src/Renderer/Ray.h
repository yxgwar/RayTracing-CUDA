#pragma once

#include <glm/glm.hpp>

namespace RayTracing
{
	struct Ray
	{
		glm::vec3 origin{ 0.0f };
		glm::vec3 direction{ 1.0f };
	};
}