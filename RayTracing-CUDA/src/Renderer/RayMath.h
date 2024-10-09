#pragma once
#include <glm/glm.hpp>
#include <random>

namespace RayTracing
{
	class RayMath
	{
	public:
		static void Init()
		{
			std::random_device rd; // ����豸
			gen = std::mt19937(rd());
			disf = std::uniform_real_distribution<>(-1.0, 1.0);// ���ȷֲ��� [-1.0, 1.0] ��Χ�ڵ�˫������
			disi = std::uniform_int_distribution<>(-1, 1); // ���ȷֲ��� [-1, 1] ��Χ�ڵ�����
		}

		//����0~1�ĸ�����
		static float Randomf() { return disf(gen) * 0.5f + 0.5f; }

		//����-1~1����ά����
		static glm::vec3 RandomVec();

		//����-1~1������
		static int RandomI() { return disi(gen); }

		static int clamp(int a, int min, int max);

	private:
		static thread_local std::mt19937 gen;
		static std::uniform_real_distribution<> disf;
		static std::uniform_int_distribution<> disi;
	};
}