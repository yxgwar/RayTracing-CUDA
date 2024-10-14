#ifndef CAMERAH
#define CAMERAH

#include <curand_kernel.h>
#include "Renderer/Ray.h"
#include "Math/Mathcu.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace RayTracing
{
    __device__ glmcu::vec3 random_in_unit_disk(curandState* local_rand_state) {
        glmcu::vec3 p;
        do {
            p = 2.0f * glmcu::vec3(curand_uniform(local_rand_state), curand_uniform(local_rand_state), 0) - glmcu::vec3(1, 1, 0);
        } while (glmcu::dot(p, p) >= 1.0f);
        return p;
    }

    class camera {
    public:
        __device__ camera(glmcu::vec3 lookfrom, glmcu::vec3 lookat, glmcu::vec3 vup, float vfov, float aspect, float aperture, float focus_dist) { // vfov is top to bottom in degrees
            lens_radius = aperture / 2.0f;
            float theta = vfov * ((float)M_PI) / 180.0f;
            float half_height = tan(theta / 2.0f);
            float half_width = aspect * half_height;
            origin = lookfrom;
            w = glmcu::normalize(lookfrom - lookat);
            u = glmcu::normalize(cross(vup, w));
            v = cross(w, u);
            lower_left_corner = origin - half_width * focus_dist * u - half_height * focus_dist * v - focus_dist * w;
            horizontal = 2.0f * half_width * focus_dist * u;
            vertical = 2.0f * half_height * focus_dist * v;
        }
        __device__ Ray get_ray(float s, float t, curandState* local_rand_state) {
            glmcu::vec3 rd = lens_radius * random_in_unit_disk(local_rand_state);
            glmcu::vec3 offset = u * rd[0] + v * rd[1];
            Ray r;
            r.origin = origin + offset;
            r.direction = glmcu::normalize(lower_left_corner + s * horizontal + t * vertical - origin - offset);
            return r;
        }

        glmcu::vec3 origin;
        glmcu::vec3 lower_left_corner;
        glmcu::vec3 horizontal;
        glmcu::vec3 vertical;
        glmcu::vec3 u, v, w;
        float lens_radius;
    };
}

#endif
