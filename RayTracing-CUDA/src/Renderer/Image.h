#pragma once

#include <glm/glm.hpp>

namespace RayTracing
{
	class Image {
	public:
		Image(int width, int height, int channels = 4);
		~Image();

		void SetPixelData(glm::vec4 color, int position);
		bool GenerateImage();
	private:
		int m_Width, m_Height, m_Channels;
		int size;
		unsigned char* m_Image;
	};
}