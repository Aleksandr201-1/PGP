#ifndef SCENE_CUH
#define SCENE_CUH

#include <vector>
#include <functional>
#include "Triangle.cuh"
#include "GPUErrorCheck.cuh"

enum Method {
    CPU,
    GPU
};

class Scene {
    private:
        //void renderGPU ();//(Vec3 pc, Vec3 pv, Triangle *triangles, uchar4 *points, int width, int height, double angle, Vec3 light_source, Vec3 light_shade, int n, int recursion_step, uchar4 *texture);
        //void smoothingGPU ();//(uchar4 *points, uchar4 *smoothing_points, int width, int height, int multiplier);
        void renderCPU ();//(Vec3 pc, Vec3 pv, Triangle *triangles, uchar4 *points, int width, int height, double angle, Vec3 light_source, Vec3 light_shade, int n, int recursion_step, uchar4 *texture);
        void smoothingCPU ();//(uchar4 *points, uchar4 *smoothing_points, int width, int height, int multiplier);
        void drawGPU ();
        void drawCPU ();
        //void floor(Vec3 a, Vec3 b, Vec3 c, Vec3 d, Vec3 color, std::vector<Triangle> &triangles);
    public:
        Scene (const std::vector<Triangle> &figures);
        ~Scene ();

        void setTexture (const std::vector<uchar4> &texture);
        //void setPoints (const std::vector<uchar4> &points, const std::vector<uchar4> &smooth);
        void setPointsSize (uint64_t width, uint64_t height);
        void setSmoothSize (uint64_t width, uint64_t height);
        void setTextureSize (uint64_t width, uint64_t height);
        void setMultiplier (uint64_t multiplier);
        void setLightSource (const Vec3 &source, const Vec3 &shade, uint64_t count, uint64_t step);
        void setCamera (const Vec3 &pc, const Vec3 &pv, double angle);

        std::vector<uchar4> getPoints () const;

        void draw (Method method);
    private:
        std::vector<Triangle> triangles;
        std::vector<uchar4> points, smooth, texture;
        uint64_t pointsWidth, pointsHeight, smoothWidth, smoothHeight, textureWidth, textureHeight;
        uint64_t count, step, multiplier;
        double angle;
        Vec3 lightSource, lightShade, pc, pv;
};

#endif