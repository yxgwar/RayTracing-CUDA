#include "Render.h"
#include "RayMath.h"
#include "Material.h"
#include "Core/CheckCudaError.h"

namespace RayTracing
{
	static __device__ bool FindHit(Ray& ray, HitData& hitData, Hittable** objects, int size)
	{
		HitData temp;
		bool isHit = false;
		float mint = FLT_MAX;
		for (int i = 0; i < size; i++)
		{
			if (objects[i]->IsHit(ray, temp))
			{
				isHit = true;
				if (temp.t < mint)
				{
					hitData = temp;
					mint = temp.t;
				}
			}
		}
		return isHit;
	}

	__device__ glmcu::vec4 perPixel(Hittable** hitcu, int sizehit, Material** matcu, int sizemat, Ray& ray, curandState rand)
	{
		HitData hitData;
		Ray traceRay(ray);

		glmcu::vec3 color(1.0f);
		int bounces = 10;
		for (int i = 0; i < bounces; i++)
		{
			if(FindHit(traceRay, hitData, hitcu, sizehit))
			{
				glmcu::vec3 rColor(0.0f);
					//printf("%f, %f, %f\n", rColor[0], rColor[1], rColor[2]);
				if (matcu[hitData.index]->Scatter(traceRay, hitData, rColor, rand))
				{
					color *= rColor;
				}
				else
				{
					color = glmcu::vec3(0.0f);
					break;
				}
			}
			else
			{
				glmcu::vec3 dir = traceRay.direction;
				float a = (dir[1] + 1.0f) * 0.5f;
				color *= (1.0f - a) * glmcu::vec3(1.0f) + a * glmcu::vec3(0.3f, 0.5f, 1.0f);
				break;
			}
		}
		return { color, 1.0f };
	}

	__global__ void render(Hittable** hitcu, int sizehit, Material** matcu, int sizemat, glm::vec3* rd, unsigned char* pix, Ray ray, int width, int height, int samplers)
	{
		int index = blockIdx.x * blockDim.x + threadIdx.x;
		int stride = blockDim.x * gridDim.x;
		for (int k = index; k < width * height; k += stride)
		//for (int k = 0; k < width * height; k++) 
		{
			curandState rand;
			curand_init(1984 + k, 0, 0, &rand);
			int i = k % width;
			int j = k / width;
			glmcu::vec4 colorS(0.0f);
			for (int s = 0; s < samplers; s++)
			{
				int x = clamp(i + randomi(rand), 0, width - 1);
				int y = clamp(j + randomi(rand), 0, height - 1);
				ray.direction = rd[x + y * width];
				colorS += perPixel(hitcu, sizehit, matcu, sizemat, ray, rand);
			}
			glmcu::vec4 color = colorS / float(samplers);

			color = clamp(color, glmcu::vec4(0.0f), glmcu::vec4(1.0f));//将颜色限制在0~255
			pix[4 * k] = int(sqrt(color[0]) * 255.0f);
			pix[4 * k + 1] = int(sqrt(color[1]) * 255.0f);
			pix[4 * k + 2] = int(sqrt(color[2]) * 255.0f);
			pix[4 * k + 3] = int(sqrt(color[3]) * 255.0f);
		}
	}

	void Render::StartRendering(Scene& scene, Camera& camera, Image& image, int width, int height, int channels)
	{
		glm::vec3 rayOrigin = camera.GetOrigin();
		int samplers = 100;
		Ray ray;
		ray.origin = rayOrigin;
		
		unsigned char* pix;
		checkCudaErrors(cudaMallocManaged(&pix, width * height * channels * sizeof(unsigned char)));

		int blocksize = 256;
		int numBlocks = (width * height + blocksize - 1) / blocksize;
		render << <numBlocks,blocksize >> > (scene.GetHit(), scene.GetHitCount(), scene.GetMat(), scene.GetMatCount(), camera.GetRayDirections(), pix, ray, width, height, samplers);
		checkCudaErrors(cudaGetLastError());
		checkCudaErrors(cudaDeviceSynchronize());

		checkCudaErrors(cudaMemcpy(image.GeiImage(), pix, width * height * channels * sizeof(unsigned char), cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaFree(pix));
	}
}