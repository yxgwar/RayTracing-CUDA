#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cmath>
#include <glm/glm.hpp>

namespace glmcu
{
	class vec2
	{
	public:
		__host__ __device__ vec2() :e{ 0.0f, 0.0f } {}
		__host__ __device__ vec2(float e0, float e1) : e{ e0, e1 } { }

		//__device__ inline const vec3& operator+() const { return *this; }
		//__device__ inline vec3 operator-() const { return vec3(-e[0], -e[1], -e[2]); }
		__device__ inline float operator[](int i) const { return e[i]; }
		__device__ inline float& operator[](int i) { return e[i]; };

		float e[2];

		friend static __device__ vec2 operator*(vec2& v1, float v);
		friend static __device__ vec2 operator-(vec2& v1, float v);
	};

	class vec3
	{
	public:
		__host__ __device__ vec3() :e{ 0.0f, 0.0f, 0.0f } {}
		__host__ __device__ vec3(float e) : e{ e } {}
		__host__ __device__ vec3(float e0, float e1, float e2) : e{ e0, e1, e2 } { }
		__host__ __device__ vec3(glm::vec3 v) : e{ v.x, v.y, v.z } { }

		//__device__ inline const vec3& operator+() const { return *this; }
		//__device__ inline vec3 operator-() const { return vec3(-e[0], -e[1], -e[2]); }
		__host__ __device__ inline float operator[](int i) const { return e[i]; }
		__host__ __device__ inline float& operator[](int i) { return e[i]; };

		__device__ vec3& operator*=(const vec3& v2) { e[0] *= v2[0]; e[1] *= v2[1]; e[2] *= v2[2]; return *this;}
		__device__ vec3& operator*=(const glm::vec3& v2) { e[0] *= v2.x; e[1] *= v2.y; e[2] *= v2.z; return *this;}

		__device__ inline float length() const { return sqrt(e[0] * e[0] + e[1] * e[1] + e[2] * e[2]); }
		__device__ inline float squared_length() const { return e[0] * e[0] + e[1] * e[1] + e[2] * e[2]; }


		float e[3];

		friend static __device__ vec3 operator*(vec3& v1, vec3& v2);
		friend static __device__ vec3 operator-(vec3& v1, vec3& v2);
		friend static __device__ vec3 operator*(vec3& v1, float e);
		friend static __device__ vec3 operator*(float e, vec3& v1);
		friend static __device__ vec3 operator/(vec3& v1, float e);
		friend static __device__ vec3 operator-(float e, vec3& v1);
		friend static __device__ vec3 normalize(vec3& v);
	};

	class vec4
	{
	public:
		__host__ __device__ vec4() :e{ 0.0f, 0.0f, 0.0f, 0.0f } {}
		__host__ __device__ vec4(float e) : e{ e } {}
		__host__ __device__ vec4(vec3 v, float e) : e{ v[0], v[1], v[2], e } {}
		__host__ __device__ vec4(float e0, float e1, float e2, float e3) : e{ e0, e1, e2, e3 } { }

		//__device__ inline const vec3& operator+() const { return *this; }
		//__device__ inline vec3 operator-() const { return vec3(-e[0], -e[1], -e[2]); }
		__host__ __device__ inline float operator[](int i) const { return e[i]; }
		__host__ __device__ inline float& operator[](int i) { return e[i]; };
		__device__ inline operator vec3() { return vec3{ e[0],e[1],e[2] }; }

		__device__ vec4& operator+=(const vec4& v2) { e[0] += v2[0]; e[1] += v2[1]; e[2] += v2[2]; e[3] += v2[3];  return *this; }

		__device__ inline float length() const { return sqrt(e[0] * e[0] + e[1] * e[1] + e[2] * e[2] + e[3] * e[3]); }
		__device__ inline float squared_length() const { return e[0] * e[0] + e[1] * e[1] + e[2] * e[2] + e[3] * e[3]; }


		float e[4];

		friend static __device__ vec4 operator*(vec4& v1, vec4& v2);
		friend static __device__ vec4 operator+(vec4& v1, vec4& v2);
		friend static __device__ vec4 operator/(vec4& v1, float v);
	};

	class mat4
	{
	public:
		__host__ __device__ mat4() :e() {}
		mat4(glm::mat4& m);

		__host__ __device__ inline vec4 operator[](int i) const { return e[i]; }
		__host__ __device__ inline vec4& operator[](int i) { return e[i]; };

		vec4 e[4];

		friend static __device__ vec4 operator*(mat4& m, vec4& v);
	};

	__device__ vec3 normalize(vec3& v)
	{
		float k = 1.0f / v.length();
		return vec3{ v[0] * k,v[1] * k,v[2] * k };
	}

	__device__ static vec2 operator*(vec2& v1, float v)
	{
		return vec2(v1[0] * v, v1[1] * v);
	}

	__device__ static vec2 operator-(vec2& v1, float v)
	{
		return vec2(v1[0] - v, v1[1] - v);
	}

	__device__ static vec3 operator+(vec3& v1, vec3& v2)
	{
		return vec3(v1[0] + v2[0], v1[1] + v2[1], v1[2] + v2[2]);
	}

	__device__ static vec3 operator*(vec3& v1, vec3& v2)
	{
		return vec3(v1[0] * v2[0], v1[1] * v2[1], v1[2] * v2[2]);
	}

	__device__ static vec3 operator-(vec3& v1, vec3& v2)
	{
		return vec3(v1[0] - v2[0], v1[1] - v2[1], v1[2] - v2[2]);
	}

	__device__ static vec3 operator*(vec3& v1, float e)
	{
		return vec3(v1[0] * e, v1[1] * e, v1[2] * e);
	}

	__device__ static vec3 operator*(float e, vec3& v1)
	{
		return vec3(e * v1[0], e * v1[1], e * v1[2]);
	}

	__device__ static vec3 operator/(vec3& v1, float e)
	{
		return vec3(v1[0] / e, v1[1] / e, v1[2] / e);
	}

	__device__ static vec3 operator-(float e, vec3& v1)
	{
		return vec3(e - v1[0], e - v1[1], e - v1[2]);
	}

	__device__ static vec4 operator+(vec4& v1, vec4& v2)
	{
		return vec4(v1[0] + v2[0], v1[1] + v2[1], v1[2] + v2[2], v1[3] + v2[3]);
	}

	__device__ static vec4 operator*(vec4& v1, vec4& v2)
	{
		return vec4(v1[0] * v2[0], v1[1] * v2[1], v1[2] * v2[2], v1[3] * v2[3]);
	}

	__device__ static vec4 operator/(vec4& v1, float e)
	{
		return vec4(v1[0] / e, v1[1] / e, v1[2] / e, v1[3] / e);
	}

	__device__ static vec4 operator*(mat4& m, vec4& v)
	{
		return vec4
		{
			m[0][0] * v[0] + m[1][0] * v[1] + m[2][0] * v[2] + m[3][0] * v[3],
			m[0][1] * v[0] + m[1][1] * v[1] + m[2][1] * v[2] + m[3][1] * v[3],
			m[0][2] * v[0] + m[1][2] * v[1] + m[2][2] * v[2] + m[3][2] * v[3],
			m[0][3] * v[0] + m[1][3] * v[1] + m[2][3] * v[2] + m[3][3] * v[3]
		};
	}

	__device__ static float dot(vec3& v1, vec3& v2)
	{
		return v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2];
	}

	__device__ static vec4 clamp(vec4& a, vec4& min, vec4& max)
	{
		vec4 re;
		for (int i = 0; i < 4; i++)
		{
			if (a[i] < min[i])
				re[i] = min[i];
			else if (a[i] > max[i])
				re[i] = max[i];
			else
				re[i] = a[i];
		}
		return re;
	}

	__device__ static vec3 reflect(vec3& in, vec3& n)
	{
		return in - 2.0f * (in * n) * n;
	}
}