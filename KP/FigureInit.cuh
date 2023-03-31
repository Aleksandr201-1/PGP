#ifndef FIGURE_INIT_CUH
#define FIGURE_INIT_CUH

#include <vector>
#include "Triangle.cuh"

void figureInit (const std::vector<Vec3> &centers, const std::vector<double> &r, const std::vector<Vec3> &colors, std::vector<Triangle> &triangles);

void floorInit (Vec3 a, Vec3 b, Vec3 c, Vec3 d, Vec3 color, std::vector<Triangle> &triangles);

#endif