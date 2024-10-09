#include "RayMath.h"

namespace RayTracing
{
	thread_local std::mt19937 RayMath::gen;
	std::uniform_real_distribution<> RayMath::disf;
	std::uniform_int_distribution<> RayMath::disi;

	glm::vec3 RayMath::RandomVec()
	{
		while (true)
		{
			glm::vec3 rv = { disf(gen), disf(gen), disf(gen) };
			float l = glm::length(rv);
			if (l <= 1.0f && l >= 1e-6)
				return rv / l;
		}
	}

	int RayMath::clamp(int a, int min, int max)
	{
		if (a < min)
			return min;
		if (a > max)
			return max;
		return a;
	}
}