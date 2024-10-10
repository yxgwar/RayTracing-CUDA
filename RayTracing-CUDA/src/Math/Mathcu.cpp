#include "Mathcu.h"

namespace glmcu
{
	mat4::mat4(glm::mat4& m)
	{
		for (int i = 0; i < 4; i++)
		{
			for (int j = 0; j < 4; j++)
			{
				e[i][j] = m[i][j];
			}
		}
	}
}