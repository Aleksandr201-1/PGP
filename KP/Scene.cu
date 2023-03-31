#include "Scene.cuh"

__host__ __device__ uchar4 getTextureColor (uchar4 *texture, double x, double y, Triangle *triangles) {
    return texture[(int)((triangles[36].a[X] - x) / abs(triangles[36].a[X] - triangles[36].c[X]) * 200 + ((triangles[36].c[Y] - y) / abs(triangles[36].a[X] - triangles[36].c[X]) * 200) * 200)];
}

__host__ __device__ uchar4 ray (Vec3 pos, Vec3 dir, Triangle *triangles, Vec3 lightSource, Vec3 lightShade, int n, bool shadowsReflections, int step, uchar4 *texture, bool with_texture) {
    int minK = -1;
    double minTS;
    pos = pos + dir * 0.01;
    for (int i = 0; i < n; i++) {
        Vec3 e1 = triangles[i].b - triangles[i].a;
        Vec3 e2 = triangles[i].c - triangles[i].a;
        Vec3 p = prod(dir, e2);
        double div = dot(p, e1);
        if (fabs(div) < 1e-10) {
            continue;
        }
        Vec3 t = pos - triangles[i].a;
        double u = dot(p, t) / div;
        if (u < 0.0 || u > 1.0) {
            continue;
        }

        Vec3 q = prod(t, e1);
        double v = dot(q, dir) / div;
        if (v < 0.0 || v + u > 1.0) {
            continue;
        }

        double ts = dot(q, e2) / div;
        if (ts < 0.0) {
            continue;
        }

        if (minK == -1 || ts < minTS) {
            minK = i;
            minTS = ts;
        }

    }

    if (minK == -1) {
        return {0, 0, 0, 0};
    }

    if (shadowsReflections) {
        Vec3 tmpPos = dir * minTS + pos;
        Vec3 newDir = normalize(lightSource - tmpPos);
        for (int i = 0; i < n; i++) {
            Vec3 e1 = triangles[i].b - triangles[i].a;
            Vec3 e2 = triangles[i].c - triangles[i].a;
            Vec3 p = prod(newDir, e2);
            double div = dot(p, e1);
            if (fabs(div) < 1e-10) {
                continue;
            }
            
            Vec3 t = tmpPos - triangles[i] .a;
            double u = dot(p, t) / div;
            
            if (u < 0.0 || u > 1.0) {
                continue;
            }
            Vec3 q = prod(t, e1);
            double v = dot(q, newDir) / div;
            if (v < 0.0 || v + u > 1.0) {
                continue;
            }
            double ts = dot(q, e2) / div;
            if (ts > 0.0 && ts < Vec3Length(lightSource - tmpPos) && i != minK) {
                return {0, 0, 0, 0};
            }
    
        }
        uchar4 minColor = {0, 0, 0, 0};
        Vec3 result = triangles[minK].color;
        Vec3 reflections = triangles[minK].color;
        if ((minK == 36 || minK == 37) && with_texture) {
            result = uchar4ToVec3(getTextureColor(texture, tmpPos[X], tmpPos[Y], triangles)); 
        }
        if (step > 0) {
            Vec3 reflection_dir = reflect(dir, normalize(prod(triangles[minK].b - triangles[minK].a, triangles[minK].c - triangles[minK].a)));
            double reflection_scale = 0.5;
            double transparency_scale = 0.5;
            reflections = (reflections * (1.0 - reflection_scale) + uchar4ToVec3(ray(tmpPos, reflection_dir, triangles, lightSource, lightShade, n, true, step - 1, texture, with_texture)) * reflection_scale);
            result = (reflections * (1.0 - transparency_scale) + uchar4ToVec3(ray(tmpPos, dir, triangles, lightSource, lightShade, n, true, step - 1, texture, with_texture)) * transparency_scale);
        }

        if ((minK == 36 || minK == 37) && with_texture) {
            minColor.x += result[X] * lightShade[X];
            minColor.y += result[Y] * lightShade[Y];
            minColor.z += result[Z] * lightShade[Z];    
        } else {
            minColor.x += result[X] * lightShade[X];
            minColor.y += result[Y] * lightShade[Y];
            minColor.z += result[Z] * lightShade[Z];    
        }
        minColor.w = 0;
        return minColor;  
    } else {
        return {220, 220, 220};
    }
}

__global__ void renderGpuKernel (Vec3 pc, Vec3 pv, Triangle *triangles, uchar4 *points, int width, int height, double angle, Vec3 lightSource, Vec3 lightShade, int n, int step, uchar4 *texture) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;
    int offsetx = blockDim.x * gridDim.x;
    int offsety = blockDim.y * gridDim.y;

    double dw = 2.0 / (width - 1);
    double dh = 2.0 / (height - 1);
    double z = 1.0 / std::tan(angle * std::acos(-1.0) / 360.0);
    Vec3 bz = normalize(pv - pc);
    Vec3 bx = normalize(prod(bz, {0.0, 0.0, 1.0}));
    Vec3 by = prod(bx, bz);
    for (int i = idx; i < width; i += offsetx) {
        for (int j = idy; j < height; j += offsety) {
            Vec3 a = {-1.0 + dw * i, (-1.0 + dh * j) * height / width, z};
            Vec3 dir = normalize(multiple(bx, by, bz, a));
            points[(height - 1 - j) * width + i] = ray(pc, dir, triangles, lightSource, lightShade, n, true, step, texture, false); //меняем индексацию чтобы не получить перевернутое изображение
        }
    }
}

__global__ void smoothingGpuKernel (uchar4 *points, uchar4 *smoothPoints, int width, int height, int multiplier) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int offsetx = blockDim.x * gridDim.x;
    int offsety = blockDim.y * gridDim.y;

    int mult = multiplier * multiplier;
    for (int y = idy; y < height; y += offsety) {
        for (int x = idx; x < width; x += offsetx) {
            Vec3 mid = createVec3(0, 0, 0);
            for (int j = 0; j < multiplier; j++) {
                for (int i = 0; i < multiplier; i++) {
                    mid = mid + uchar4ToVec3(smoothPoints[i + j * width * multiplier + x * multiplier + y * width * mult]);
                }
            }
            points[x + width * y] = Vec3ToUchar4(mid / (mult));
        }
    }
}

void Scene::renderCPU () {
    double dw = 2.0 / (pointsWidth - 1);
    double dh = 2.0 / (pointsHeight - 1);
    double z = 1.0 / std::tan(angle * std::acos(-1.0) / 360.0);
    Vec3 bz = normalize(pv - pc);
    Vec3 bx = normalize(prod(bz, createVec3(0.0, 0.0, 1.0)));
    Vec3 by = prod(bx, bz);
    for (int i = 0; i < pointsWidth; i++) {
        for (int j = 0; j < pointsHeight; j++) {
            Vec3 a = createVec3(-1.0 + dw * i, (-1.0 + dh * j) * pointsHeight / pointsWidth, z);
            Vec3 dir = normalize(multiple(bx, by, bz, a));
            points[(pointsHeight - 1 - j) * pointsWidth + i] = ray(pc, dir, triangles.data(), lightSource, lightShade, triangles.size(), true, step, texture.data(), true);
        }
    }
}

void Scene::smoothingCPU () {
    int multiplier2 = multiplier * multiplier;
    for (int y = 0; y < pointsHeight; y++) {
        for (int x = 0; x < pointsWidth; x++) {
            Vec3 mid = createVec3(0, 0, 0);
            for (int j = 0; j < multiplier; j++) {
                for (int i = 0; i < multiplier; i++) {
                    mid = mid + uchar4ToVec3(smooth[i + j * pointsWidth * multiplier + x * multiplier + y * pointsWidth * multiplier2]);
                }
            }
            points[x + pointsWidth * y] = Vec3ToUchar4(mid / (multiplier2));
        }
    }
}

void Scene::drawGPU () {
    uchar4 *gpuTexture;
    gpuErrorCheck(cudaMalloc(&gpuTexture, textureWidth * textureHeight * sizeof(uchar4)));
    uchar4 *gpuPoints;
    gpuErrorCheck(cudaMalloc(&gpuPoints, pointsWidth * pointsHeight * sizeof(uchar4)));
    uchar4 *gpuSmooth;
    gpuErrorCheck(cudaMalloc(&gpuSmooth, smoothWidth * smoothHeight * sizeof(uchar4)));
    Triangle *gpuFigures;
    gpuErrorCheck(cudaMalloc(&gpuFigures, triangles.size() * sizeof(Triangle)));
    gpuErrorCheck(cudaMemcpy(gpuFigures, triangles.data(), triangles.size() * sizeof(Triangle), cudaMemcpyHostToDevice));
    gpuErrorCheck(cudaMemcpy(gpuTexture, texture.data(), textureWidth * textureHeight * sizeof(uchar4), cudaMemcpyHostToDevice));
    renderGpuKernel<<<1, 256>>>(pc, pv, gpuFigures, gpuSmooth, smoothWidth, smoothHeight, angle, lightSource, lightShade, triangles.size(), step, gpuTexture);
    gpuErrorCheck(cudaGetLastError());          
    smoothingGpuKernel<<<256, 256>>>(gpuPoints, gpuSmooth, pointsWidth, pointsHeight, multiplier);
    gpuErrorCheck(cudaGetLastError());
    points.resize(pointsWidth * pointsHeight);
    gpuErrorCheck(cudaMemcpy(&points[0], gpuPoints, pointsWidth * pointsHeight * sizeof(uchar4), cudaMemcpyDeviceToHost));           
    gpuErrorCheck(cudaFree(gpuPoints));
    gpuErrorCheck(cudaFree(gpuSmooth));
    gpuErrorCheck(cudaFree(gpuFigures));
    gpuErrorCheck(cudaFree(gpuTexture));
}

void Scene::drawCPU () {
    renderCPU();
    smoothingCPU();
}

Scene::Scene (const std::vector<Triangle> &figures) : triangles(figures) {}

Scene::~Scene () {}

void Scene::setTexture (const std::vector<uchar4> &texture) {
    this->texture = texture;
}

void Scene::setPointsSize (uint64_t width, uint64_t height) {
    this->pointsWidth = width;
    this->pointsHeight = height;
    points.resize(width * height);
}

void Scene::setSmoothSize (uint64_t width, uint64_t height) {
    this->smoothWidth = width;
    this->smoothHeight = height;
    smooth.resize(width * height);
}

void Scene::setTextureSize (uint64_t width, uint64_t height) {
    this->textureWidth = width;
    this->textureHeight = height;
}

void Scene::setMultiplier (uint64_t multiplier) {
    this->multiplier = multiplier;
}

void Scene::setLightSource (const Vec3 &source, const Vec3 &shade, uint64_t count, uint64_t step) {
    this->lightSource = source;
    this->lightShade = shade;
    this->count = count;
    this->step = step;
}

void Scene::setCamera (const Vec3 &pc, const Vec3 &pv, double angle) {
    this->pc = pc;
    this->pv = pv;
    this->angle = angle;
}

std::vector<uchar4> Scene::getPoints () const {
    return points;
}

void Scene::draw (Method method) {
    switch (method) {
        case CPU:
            drawCPU();
            break;
        case GPU:
            drawGPU();
            break;
        default:
            break;
    }
}