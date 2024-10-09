#include "Image.h"

#define STB_IMAGE_IMPLEMENTATION
#include <stb_img/stb_image.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_img/stb_image_write.h>

namespace RayTracing
{
	Image::Image(int width, int height, int channels)
		:m_Width(width), m_Height(height), m_Channels(channels)
	{
		size = m_Width * m_Height * m_Channels;
		m_Image = new stbi_uc[size];
	}

	Image::~Image()
	{
		stbi_image_free(m_Image);
	}

	void Image::SetPixelData(glm::vec4 color, int position)
	{
		position *= 4;
		m_Image[position++] = int(std::sqrtf(color.r) * 255.0f);
		m_Image[position++] = int(std::sqrtf(color.g) * 255.0f);
		m_Image[position++] = int(std::sqrtf(color.b) * 255.0f);
		m_Image[position] = int(std::sqrtf(color.a) * 255.0f);
	}

	bool Image::GenerateImage()
	{
		return stbi_write_png("./img/test.png", m_Width, m_Height, m_Channels, m_Image, 0);
	}
}