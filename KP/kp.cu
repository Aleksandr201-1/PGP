#include <iostream>
#include <chrono>
#include <fstream>
#include <iomanip>
#include "Scene.cuh"
#include "FigureInit.cuh"

int main(int argc, char* argv[]) {
    std::vector<std::string> args(argv, argv + argc);
    Method method;

    if (args.size() < 2) {
        args.push_back("--gpu");
    }
    std::string &command = args[1];
    if (command == "--default") {
        std::cout << "10\n ./img_%d.data\n640 480 120\n5.0 3.0 0.0 2.0 1.0 2.0 6.0 1.0 0.0 0.0 2.0 0.0 0.0 0.5 0.1 1.0 4.0 1.0 0.0 0.0\n 0.0 0.0 1.0            1.0 0.5 1.0               2.0 1.0 1.0 1.0\n -2.0 3.0 1.0            0.7 0.5 0.5               2.0 1.0 1.0 1.0\n -3.0 -3.0 1.0          0.5 0.5 1.0               2.0 1.0 1.0 1.0\n -10.0 -10.0 -1.0\n  -10.0 10.0 -1.0\n 10.0 10.0 -1.0\n 10.0 -10.0 -1.0\n floor.data\n  1.0 1.0 1.0 0.5\n 1\n 20 -20 100\n   1.0 1.0 1.0\n 3 2\n";
        return 0;
    } else if (command == "--cpu") {
        method = CPU;
    } else if (command == "--gpu") {
        method = GPU;
    }
    int framesCount, width, height, angle;
    std::string path;
    double r0c, z0c, phi0c, Arc, Azc, wrc, wzc, wphic, prc, pzc, r0n, z0n, phi0n, Arn, Azn, wrn, wzn, wphin, prn, pzn;
    std::string texturePNG;
    std::string tbc;
    Vec3 floorA, floorB, floorC, floorD;
    Vec3 lightSource, lightShade;
    std::vector<Triangle> figures;
    std::vector<uchar4> texture;
    int recursionStep;
    int multiplier;
    std::vector<Vec3> centers(3), colors(4);
    std::vector<double> r(3);
    std::cin >> framesCount;
    std::cin >> path;
    std::cin >> width >> height >> angle;
    double rc, zc, phic, rn, zn, phin;
    Vec3 pc, pv;
    int smoothing_rays;

    std::cin >> r0c >> z0c >> phi0c >> Arc >> Azc >> wrc >> wzc >> wphic >> prc >> pzc >> r0n >> z0n >> phi0n >> Arn >> Azn >> wrn >> wzn >> wphin >> prn >> pzn;

    for (uint64_t i = 0; i < centers.size(); ++i) {
        std::cin >> centers[i][X] >> centers[i][Y] >> centers[i][Z] >> colors[i][X] >> colors[i][Y] >> colors[i][Z] >> r[i] >> tbc >> tbc >> tbc;
    }
    figureInit(centers, r, colors, figures);

    std::cin >> floorA[X] >> floorA[Y] >> floorA[Z];
    std::cin >> floorB[X] >> floorB[Y] >> floorB[Z];
    std::cin >> floorC[X] >> floorC[Y] >> floorC[Z];
    std::cin >> floorD[X] >> floorD[Y] >> floorD[Z];

    std::cin >> texturePNG;

    std::ifstream file(texturePNG, std::ios::binary);

    int textureWidth, textureHeight;

    file.read(reinterpret_cast<char*>(&textureWidth), sizeof(int));
    file.read(reinterpret_cast<char*>(&textureHeight), sizeof(int));
    texture.resize(textureWidth * textureHeight);
    file.read(reinterpret_cast<char*>(&texture[0]), sizeof(uchar4) * textureWidth * textureHeight);
    file.close();

    std::cin >> colors[3][X] >> colors[3][Y] >> colors[3][Z];
    std::cin >> tbc;

    floorInit(floorA, floorB, floorC, floorD, colors[3], figures);

    std::cin >> tbc;
    std::cin >> lightSource[X] >> lightSource[Y] >> lightSource[Z];
    std::cin >> lightShade[X] >> lightShade[Y] >> lightShade[Z];

    std::cin >> recursionStep >> multiplier;

    int smoothWidth = multiplier * width;
    int smoothHeight = multiplier * height;

    //texture.resize(width * height);

    double totalTime = 0;

    std::cout << "Total Triangles: " << figures.size() << "\n";
    std::cout << "Resolution: " << width << "x" << height << "\n";
    std::cout << "Total frames: " << framesCount << "\n";
    std::cout << " ______________________________________\n";
    std::cout << "|   Frame    | Time (ms)  | Total rays |\n";
    std::cout << "|____________|____________|____________|\n";
    uint64_t tableWidth = 10;

    Scene scene(figures);
    scene.setTexture(texture);
    scene.setPointsSize(width, height);
    scene.setSmoothSize(smoothWidth, smoothHeight);
    scene.setTextureSize(textureWidth, textureHeight);
    scene.setMultiplier(multiplier);

    uint64_t pos = path.find('%', 0);

    for (int i = 0; i < framesCount; i++) {

        cudaEvent_t cudaStart, cudaStop;
        cudaEventCreate(&cudaStart);
        cudaEventCreate(&cudaStop);
        cudaEventRecord(cudaStart);
        auto start = std::chrono::steady_clock::now();
        
        double timeStep = 2.0 * std::acos(-1.0) / framesCount;
        double t = timeStep * i;

        rc = r0c + Arc * std::sin(wrc * t + prc);
        zc = z0c + Azc * std::sin(wzc * t + pzc);
        phic = phi0c + wphin * t;

        rn = r0n + Arn * std::sin(wrn * t + prn);
        zn = z0n + Azn * std::sin(wzn * t + pzn);
        phin = phi0n + wphin * t;

        pc = createVec3(rc * std::cos(phic), rc * std::sin(phic), zc);
        pv = createVec3(rn * std::cos(phin), rn * std::sin(phin), zn);

        scene.setCamera(pc, pv, angle);

        smoothing_rays = smoothWidth * smoothHeight;

        scene.draw(method);

        std::vector<uchar4> points = scene.getPoints();

        cudaEventRecord(cudaStop);
        auto end = std::chrono::steady_clock::now();
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, cudaStart, cudaStop);
        std::cout << "| " << std::setw(tableWidth) << i + 1;
        totalTime += std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        std::cout << " | " << std::setw(tableWidth) << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        std::cout << " | " << std::setw(tableWidth) << smoothing_rays << " |\n";
        std::cout << "|____________|____________|____________|\n";

        std::string filename = path.substr(0, pos) + std::to_string(i + 1) + path.substr(pos + 2, path.size() - pos);

        std::ofstream output(filename, std::ios::binary);
        output.write(reinterpret_cast<const char*>(&width), sizeof(width));
        output.write(reinterpret_cast<const char*>(&height), sizeof(height));
        output.write(reinterpret_cast<const char*>(points.data()), sizeof(uchar4) * width * height);
        output.close();
    }

    std::cout << "Total time: " << totalTime << "ms\n";
    return 0;
}
