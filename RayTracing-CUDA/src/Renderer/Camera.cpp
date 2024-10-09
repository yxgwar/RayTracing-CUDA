#include "Camera.h"
#include "Math/Mathcu.h"
#include <Core/CheckCudaError.h>

#include <glm/gtc/matrix_transform.hpp>

namespace RayTracing
{
	__global__ void calcu(glm::vec3* rd, glmcu::mat4 inverseProjectionMatrix, glmcu::mat4 inverseViewMatrix, int viewportWidth, int viewportHeight)
	{
		int index = blockIdx.x * blockDim.x + threadIdx.x;
		int stride = blockDim.x * gridDim.x;
		int size = viewportWidth * viewportHeight;
		for (int i = index; i < size; i += stride)
		{
			int x = i % viewportWidth;
			int y = i / viewportWidth;
			glmcu::vec2 coord = { float(x) / viewportWidth, float(y) / viewportHeight };
			coord = coord * 2.0f - 1.0f;								//将坐标调整到-1~1
			coord[1] *= -1;

			//二维投影坐标 = ProjectionM * 三维空间坐标， 三维空间坐标 = (ProjectionM)^-1 * 二维投影坐标
			//改变第一个1.0f可以调整二维屏幕到摄像机的距离
			glmcu::vec4 target = inverseProjectionMatrix * glmcu::vec4(coord[0], coord[1], 1.0f, 1.0f);
			glmcu::vec3 targetP = glmcu::vec3(target) / target[3];  //齐次

			//摄像机空间坐标 = ViewM * 世界空间坐标
			//使用1扩充会得到位置坐标
			//glm::vec4 world = m_InverseViewMatrix * glm::vec4(targetP, 1.0f);
			//glm::vec3 worldP = glm::vec3(world) / world.w;
			//m_RayDirections[i + j * m_ViewportWidth] = glm::normalize(worldP - m_Position);

			//使用0扩充至四维向量可以保证变换时不受平移影响，直接得到方向向量
			glmcu::vec3 rayDirection = glmcu::normalize(glmcu::vec3(inverseViewMatrix * glmcu::vec4(glmcu::normalize(targetP), 0.0f)));
			rd[i] = { rayDirection[0], rayDirection[1], rayDirection[2] };
		}
	}

	Camera::Camera(float FOV, float nearClip, float farClip, int viewportWidth, int viewportHeight)
		:m_FOV(FOV), m_NearClip(nearClip), m_FarClip(farClip), m_ViewportWidth(viewportWidth), m_ViewportHeight(viewportHeight)
	{
		m_AspectRatio = (float)m_ViewportWidth / m_ViewportHeight;
		calculateViewMatrix();
		calculateProjectionMatrix();
		calculateRayDirections();
	}

	Camera::~Camera()
	{
		checkCudaErrors(cudaFree(m_RayDirections));
	}

	void Camera::calculateViewMatrix()
	{
		//暂时使用glm::lookat
		//m_Transform = glm::translate(glm::mat4(1.0f), m_Position);
		//m_ViewMatrix = glm::inverse(m_Transform);
		m_ViewMatrix = glm::lookAt(m_Position, m_Position + m_Direction, glm::vec3(0.0f, 1.0f, 0.0f));
		m_InverseViewMatrix = glm::inverse(m_ViewMatrix);
	}

	void Camera::calculateProjectionMatrix()
	{
		m_ProjectionMatrix = glm::perspective(glm::radians(m_FOV), m_AspectRatio, m_NearClip, m_FarClip);
		m_InverseProjectionMatrix = glm::inverse(m_ProjectionMatrix);
	}

	void Camera::calculateRayDirections()
	{
		checkCudaErrors(cudaMallocManaged(&m_RayDirections, m_ViewportWidth * m_ViewportHeight * sizeof(glm::vec3)));

		int blocksize = 256;
		int numBlocks = (m_ViewportWidth * m_ViewportHeight + blocksize - 1) / blocksize;
		calcu << <numBlocks, blocksize >> > (m_RayDirections, glmcu::mat4(m_InverseProjectionMatrix), glmcu::mat4(m_InverseViewMatrix), m_ViewportWidth, m_ViewportHeight);
		checkCudaErrors(cudaGetLastError());
		checkCudaErrors(cudaDeviceSynchronize());
	}
}