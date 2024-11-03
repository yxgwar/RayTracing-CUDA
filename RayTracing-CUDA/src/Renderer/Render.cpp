#include "Render.h"
#include "RayMath.h"
#include "SPMaterial.h"
#include "Sphere.h"
#include "Core/CheckCudaError.h"
#include "Camera.h"

namespace RayTracing
{
	static __device__ bool FindHit(Ray& ray, HitData& hitData, Hittable** objects, int size)
	{
		HitData temp;
		bool isHit = false;
		float mint = FLT_MAX;
		for (int i = 0; i < size; i++)
		{
			if (objects[i]->IsHit(ray, temp, mint))
			{
				isHit = true;
				hitData = temp;
				mint = temp.t;
			}
		}
		return isHit;
	}

	__device__ glmcu::vec3 perPixel(Hittable** hitcu, int sizehit, Material** matcu, int sizemat, Ray& ray, curandState rand)
	{
		HitData hitData;
		Ray traceRay(ray);

		glmcu::vec3 color(1.0f);
		int bounces = 50;
		for (int i = 0; i < bounces; i++)
		{
			if(FindHit(traceRay, hitData, hitcu, sizehit))
			{
				glmcu::vec3 rColor(0.0f);
				if (matcu[hitData.index]->Scatter(traceRay, hitData, rColor, rand))
				{
					color *= rColor;
				}
				else
				{
					return glmcu::vec3(0.0f);
				}
			}
			else
			{
				glmcu::vec3 dir = traceRay.direction;
				float a = (dir[1] + 1.0f) * 0.5f;
				color *= (1.0f - a) * glmcu::vec3(1.0f) + a * glmcu::vec3(0.3f, 0.5f, 1.0f);
				return color;
			}
		}
	}

	__global__ void render(Hittable** hitcu, int sizehit, Material** matcu, int sizemat, glm::vec3* rd, unsigned char* pix, Ray ray, int width, int height, int samplers,
		glmcu::vec3 rightD, glmcu::vec3 downD)
	{
		int i = threadIdx.x + blockIdx.x * blockDim.x;
		int j = threadIdx.y + blockIdx.y * blockDim.y;
		if ((i >= width) || (j >= height)) return;
		int k = (height - j - 1) * width + i;
		curandState rand;
		curand_init(1984 + k, 0, 0, &rand);
		glmcu::vec3 colorS(0.0f);
		for (int s = 0; s < samplers; s++)
		{
			float x = 0.0f, y = 0.0f;
			if (i != width - 1)
				x = randomf(rand);
			if (j != height - 1)
				y = randomf(rand);
			
			ray.direction = glmcu::normalize(rd[k] + x * rightD + y * downD);
			colorS += perPixel(hitcu, sizehit, matcu, sizemat, ray, rand);
		}
		glmcu::vec3 color = colorS / float(samplers);

		color = clamp(color, glmcu::vec3(0.0f), glmcu::vec3(1.0f));//将颜色限制在0~255
		pix[4 * k] = int(sqrt(color[0]) * 255.0f);
		pix[4 * k + 1] = int(sqrt(color[1]) * 255.0f);
		pix[4 * k + 2] = int(sqrt(color[2]) * 255.0f);
		pix[4 * k + 3] = 255;
	}

	static __global__ void create_world(Hittable** hit, Material** mat)
	{
#if 0
		int i = 0;
		mat[i++] = new Metal(glm::vec3(0.9f), 0.5f);
		mat[i++] = new Dielectric(1.5f);
		mat[i++] = new Dielectric(1.0f / 1.5f);
		mat[i++] = new Lambertian(glm::vec3{ 0.8f, 0.2f, 0.7f });
		mat[i] = new Lambertian(glm::vec3{ 0.5f, 0.5f, 0.5f });
		i = 0;
		hit[i++] = new Sphere(glm::vec3(0.0f), 0.5f, 3);
		hit[i++] = new Sphere(glm::vec3(1.0f, 0.0f, 0.0f), 0.5f, 1);
		hit[i++] = new Sphere(glm::vec3(1.0f, 0.0f, 0.0f), 0.4f, 2);
		hit[i++] = new Sphere(glm::vec3(-1.0f, 0.0f, 0.0f), 0.5f, 0);
		hit[i] = new Sphere(glm::vec3{ 0.0f, -100.5f, 0.0f }, 100.0f, 4);
#else
		int i = 0, j = 0;
		int k = 0;
		mat[i++] = new Lambertian(glmcu::vec3{ 0.5f, 0.5f, 0.5f });
		hit[j++] = new Sphere(glmcu::vec3{ 0.0f, -1000.0f, 0.0f }, 1000.0f, 0);

		curandState rand;
		curand_init(1984, 0, 0, &rand);
		for (int a = -11; a < 11; a++) 
		{
			for (int b = -11; b < 11; b++) 
			{
				auto choose_mat = randomf(rand);
				glmcu::vec3 center{ a + randomf(rand), 0.2f, b + randomf(rand) };
				if (choose_mat < 0.8f) {
					// diffuse
					auto albedo = randomv(rand) * randomv(rand);
					mat[i++] = new Lambertian(albedo);
				}
				else if (choose_mat < 0.95f) {
					// metal
					auto albedo = randomv(rand) * 0.5f + 0.5f;
					auto fuzz = randomf(rand) * 0.5f;
					mat[i++] = new Metal(albedo, fuzz);
				}
				else {
					// glass
					mat[i++] = new Dielectric(1.5f);
				}
				hit[j++] = new Sphere(center, 0.2f, i - 1);
			}
		}

		mat[i++] = new Dielectric(1.5f);
		hit[j++] = new Sphere(glmcu::vec3{ 0.0f, 1.0f, 0.0f }, 1.0f, i - 1);

		mat[i++] = new Lambertian(glmcu::vec3{ 0.4f, 0.2f, 0.1f });
		hit[j++] = new Sphere(glmcu::vec3{ -4.0f, 1.0f, 0.0f }, 1.0f, i - 1);

		mat[i++] = new Metal(glmcu::vec3{ 0.7f, 0.6f, 0.5f }, 0.0f);
		hit[j++] = new Sphere(glmcu::vec3{ 4.0f, 1.0f, 0.0f }, 1.0f, i - 1);
#endif
	}

	static __global__ void free(Hittable** hit, Material** mat, int h, int m)
	{
		for (int i = 0; i < h; i++)
		{
			delete hit[i];
		}
		for (int i = 0; i < m; i++)
		{
			delete mat[i];
		}
	}

	void Render::StartRendering(Scene& scene, Camera& camera, Image& image, int width, int height, int channels)
	{
		create_world << <1, 1 >> > (scene.GetHit(), scene.GetMat());
		checkCudaErrors(cudaGetLastError());
		checkCudaErrors(cudaDeviceSynchronize());
		glm::vec3 rayOrigin = camera.GetOrigin();
		int samplers = 500;
		Ray ray;
		ray.origin = rayOrigin;
		
		unsigned char* pix;
		checkCudaErrors(cudaMallocManaged(&pix, width * height * channels * sizeof(unsigned char)));

		//int blocksize = 256;
		//int numBlocks = (width * height + blocksize - 1) / blocksize;
		dim3 blocks(width / 16 + 1, height / 16 + 1);
		dim3 threads(16, 16);
		render << <blocks,threads >> > (scene.GetHit(), scene.GetHitCount(), scene.GetMat(), scene.GetMatCount(), camera.GetRayDirections(), pix, ray, width, height, samplers,
			glmcu::vec3(camera.GetRightD()), glmcu::vec3(camera.GetDownD()));
		checkCudaErrors(cudaGetLastError());
		checkCudaErrors(cudaDeviceSynchronize());

		free << <1, 1 >> > (scene.GetHit(), scene.GetMat(), scene.GetHitCount(), scene.GetMatCount());
		checkCudaErrors(cudaGetLastError());
		checkCudaErrors(cudaDeviceSynchronize());

		checkCudaErrors(cudaMemcpy(image.GeiImage(), pix, width * height * channels * sizeof(unsigned char), cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaFree(pix));
	}
}