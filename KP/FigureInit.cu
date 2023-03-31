#include "FigureInit.cuh"

void figureInit (const std::vector<Vec3> &centers, const std::vector<double> &r, const std::vector<Vec3> &colors, std::vector<Triangle> &triangles) {
    
    Vec3 color = colors[0].getColor();
    
    Vec3 pointA  = createVec3(r[0] / std::sqrt(3) + centers[0][X], r[0] / std::sqrt(3) + centers[0][Y], r[0] / std::sqrt(3) + centers[0][Z]);
    Vec3 pointB  = createVec3(r[0] / std::sqrt(3) + centers[0][X], -r[0] / std::sqrt(3) + centers[0][Y], -r[0] / std::sqrt(3) + centers[0][Z]);
    Vec3 pointC  = createVec3(-r[0] / std::sqrt(3) + centers[0][X], r[0] / std::sqrt(3) + centers[0][Y], -r[0] / std::sqrt(3) + centers[0][Z]);
    Vec3 pointD  = createVec3(-r[0] / std::sqrt(3) + centers[0][X], -r[0] / std::sqrt(3) + centers[0][Y], r[0] / std::sqrt(3) + centers[0][Z]);

    triangles.push_back({pointA, pointB, pointD, color});
    triangles.push_back({pointA, pointC, pointD, color});
    triangles.push_back({pointB, pointC, pointD, color});
    triangles.push_back({pointA, pointB, pointC, color}); 

    color = colors[1].getColor();

    pointA = createVec3(-r[1] / std::sqrt(3) + centers[1][X], -r[1] / std::sqrt(3) + centers[1][Y], -r[1] / std::sqrt(3) + centers[1][Z]);
    pointB = createVec3(-r[1] / std::sqrt(3) + centers[1][X], -r[1] / std::sqrt(3) + centers[1][Y],  r[1] / std::sqrt(3) + centers[1][Z]);
    pointC = createVec3(-r[1] / std::sqrt(3) + centers[1][X],  r[1] / std::sqrt(3) + centers[1][Y], -r[1] / std::sqrt(3) + centers[1][Z]);
    pointD = createVec3(-r[1] / std::sqrt(3) + centers[1][X],  r[1] / std::sqrt(3) + centers[1][Y],  r[1] / std::sqrt(3) + centers[1][Z]);
    Vec3 pointE = createVec3( r[1] / std::sqrt(3) + centers[1][X], -r[1] / std::sqrt(3) + centers[1][Y], -r[1] / std::sqrt(3) + centers[1][Z]);
    Vec3 pointF = createVec3( r[1] / std::sqrt(3) + centers[1][X], -r[1] / std::sqrt(3) + centers[1][Y],  r[1] / std::sqrt(3) + centers[1][Z]);
    Vec3 pointG = createVec3( r[1] / std::sqrt(3) + centers[1][X],  r[1] / std::sqrt(3) + centers[1][Y], -r[1] / std::sqrt(3) + centers[1][Z]);
    Vec3 pointH = createVec3( r[1] / std::sqrt(3) + centers[1][X],  r[1] / std::sqrt(3) + centers[1][Y],  r[1] / std::sqrt(3) + centers[1][Z]);
 

    triangles.push_back({pointA, pointB, pointD, color});
    triangles.push_back({pointA, pointC, pointD, color});
    triangles.push_back({pointB, pointF, pointH, color});
    triangles.push_back({pointB, pointD, pointH, color});
    triangles.push_back({pointE, pointF, pointH, color});
    triangles.push_back({pointE, pointG, pointH, color});
    triangles.push_back({pointA, pointE, pointG, color});
    triangles.push_back({pointA, pointC, pointG, color});
    triangles.push_back({pointA, pointB, pointF, color});
    triangles.push_back({pointA, pointE, pointF, color});
    triangles.push_back({pointC, pointD, pointH, color});
    triangles.push_back({pointC, pointG, pointH, color});

    color = colors[2].getColor();
    double a = (1 + std::sqrt(5)) / 2;

    pointA = createVec3(centers[2][X], -r[2] / std::sqrt(3) + centers[2][Y], r[2] * a / std::sqrt(3) + centers[2][Z]);
    pointB = createVec3(centers[2][X], r[2] / std::sqrt(3) + centers[2][Y], r[2] * a / std::sqrt(3) + centers[2][Z]);
    pointC = createVec3(centers[2][X] - r[2] * a / std::sqrt(3), centers[2][Y], r[2] / std::sqrt(3) + centers[2][Z]);
    pointD = createVec3(centers[2][X] + r[2] * a / std::sqrt(3), centers[2][Y], r[2] / std::sqrt(3) + centers[2][Z]);
    pointE = createVec3(centers[2][X] - r[2] / std::sqrt(3), r[2] * a / std::sqrt(3) + centers[2][Y], centers[2][Z]);
    pointF = createVec3(centers[2][X] + r[2] / std::sqrt(3), r[2] * a / std::sqrt(3) + centers[2][Y], centers[2][Z]);
    pointG = createVec3(centers[2][X] + r[2] / std::sqrt(3), -r[2] * a / std::sqrt(3) + centers[2][Y], centers[2][Z]);
    pointH = createVec3(centers[2][X] - r[2] / std::sqrt(3), -r[2] * a / std::sqrt(3) + centers[2][Y], centers[2][Z]);
    Vec3 pointI = createVec3(centers[2][X] - r[2] * a / std::sqrt(3), centers[2][Y], -r[2] / std::sqrt(3) + centers[2][Z]);
    Vec3 pointJ = createVec3(centers[2][X] + r[2] * a / std::sqrt(3), centers[2][Y], -r[2] / std::sqrt(3) + centers[2][Z]);
    Vec3 pointK = createVec3(centers[2][X], -r[2] / std::sqrt(3) + centers[2][Y], -r[2] * a / std::sqrt(3) + centers[2][Z]);
    Vec3 pointL = createVec3(centers[2][X], r[2] / std::sqrt(3) + centers[2][Y], -r[2] * a / std::sqrt(3) + centers[2][Z]);

    triangles.push_back({ pointA,  pointB,  pointC, color});
    triangles.push_back({ pointB,  pointA,  pointD, color});
    triangles.push_back({ pointA,  pointC,  pointH, color});
    triangles.push_back({ pointC,  pointB,  pointE, color});
    triangles.push_back({ pointE,  pointB,  pointF, color});
    triangles.push_back({ pointG,  pointA,  pointH, color});
    triangles.push_back({ pointD,  pointA,  pointG, color});
    triangles.push_back({ pointB,  pointD,  pointF, color});
    triangles.push_back({ pointE,  pointF,  pointL, color});
    triangles.push_back({ pointG,  pointH,  pointK, color});
    triangles.push_back({ pointD,  pointG,  pointJ, color});
    triangles.push_back({ pointF,  pointD,  pointJ, color});
    triangles.push_back({ pointH,  pointC,  pointI, color});
    triangles.push_back({ pointC,  pointE,  pointI, color});
    triangles.push_back({ pointJ,  pointK,  pointL, color});
    triangles.push_back({ pointK,  pointI,  pointL, color});
    triangles.push_back({ pointF,  pointJ,  pointL, color});
    triangles.push_back({ pointJ,  pointG,  pointK, color});
    triangles.push_back({ pointH,  pointI,  pointK, color});
    triangles.push_back({ pointI,  pointE,  pointL, color});
}

void floorInit (Vec3 a, Vec3 b, Vec3 c, Vec3 d, Vec3 color, std::vector<Triangle> &triangles) {
    triangles.push_back(Triangle{a, b, c, color.getColor()});
    triangles.push_back(Triangle{c, d, a, color.getColor()});
}