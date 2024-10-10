#pragma once

#include "Math/Mathcu.h"

namespace RayTracing
{
	struct Ray
	{
		glmcu::vec3 origin{ 0.0f };
		glmcu::vec3 direction{ 1.0f };
	};
}