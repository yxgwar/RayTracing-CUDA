#include "Camera.h"
#include "Math/Mathcu.h"
#include <Core/CheckCudaError.h>

#include <glm/gtc/matrix_transform.hpp>

namespace RayTracing
{
	__global__ void calcu(glmcu::vec3* rd, glmcu::mat4 inverseProjectionMatrix, glmcu::mat4 inverseViewMatrix, int viewportWidth, int viewportHeight)
	{
		for (int j = 0; j < viewportHeight; j++) {
			for (int i = 0; i < viewportWidth; i++) {
				glmcu::vec2 coord = { float(i) / viewportWidth, float(j) / viewportHeight };
				coord = coord * 2.0f - 1.0f;								//�����������-1~1
				coord[1] *= -1;

				//��άͶӰ���� = ProjectionM * ��ά�ռ����꣬ ��ά�ռ����� = (ProjectionM)^-1 * ��άͶӰ����
				//�ı��һ��1.0f���Ե�����ά��Ļ��������ľ���
				glmcu::vec4 target = inverseProjectionMatrix * glmcu::vec4(coord[0], coord[1], 1.0f, 1.0f);
				glmcu::vec3 targetP = glmcu::vec3(target) / target[3];  //���

				//������ռ����� = ViewM * ����ռ�����
				//ʹ��1�����õ�λ������
				//glm::vec4 world = m_InverseViewMatrix * glm::vec4(targetP, 1.0f);
				//glm::vec3 worldP = glm::vec3(world) / world.w;
				//m_RayDirections[i + j * m_ViewportWidth] = glm::normalize(worldP - m_Position);

				//ʹ��0��������ά�������Ա�֤�任ʱ����ƽ��Ӱ�죬ֱ�ӵõ���������
				glmcu::vec3 rayDirection = glmcu::vec3(inverseViewMatrix * glmcu::vec4(glmcu::normalize(targetP), 0.0f));
				rd[i + j * viewportWidth] = glmcu::normalize(rayDirection);
			}
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

	void Camera::calculateViewMatrix()
	{
		//��ʱʹ��glm::lookat
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
		glmcu::vec3* rd;
		checkCudaErrors(cudaMallocManaged(&rd, m_ViewportWidth * m_ViewportHeight * sizeof(glmcu::vec3)));
		calcu << <1, 1 >> > (rd, glmcu::mat4(m_InverseProjectionMatrix), glmcu::mat4(m_InverseViewMatrix), m_ViewportWidth, m_ViewportHeight);
		checkCudaErrors(cudaGetLastError());
		checkCudaErrors(cudaDeviceSynchronize());

		m_RayDirections.resize(m_ViewportWidth * m_ViewportHeight);
		glmcu::vec3 rr;
		for (int i = 0; i < m_ViewportWidth * m_ViewportHeight; i++)
		{
			rr = rd[i];
			m_RayDirections[i] = glm::vec3(rr[0], rr[1], rr[2]);
		}
		checkCudaErrors(cudaFree(rd));
	}


}