#ifndef DIRIHLE_TASK_HPP
#define DIRIHLE_TASK_HPP

#include <iomanip>
#include <fstream>
#include <vector>
#include <cstdint>
#include <cmath>
#include <mpi.h>

class DirihleTask {
    private:
        enum Axis {
            X = 0,
            Y,
            Z,
            COUNT_OF_AXES
        };
        enum Direction {
            LEFT = 0,
            RIGHT,
            FRONT,
            BACK,
            DOWN,
            UP,
            COUNT_OF_DIRECTIONS
        };
    private:
        uint64_t index (uint64_t i, uint64_t j, uint64_t k, int *dim);
        uint64_t indexBlock (uint64_t i, uint64_t j, uint64_t k, int *block);

        int indexBlockX (int id, int *block);
        int indexBlockY (int id, int *block);
        int indexBlockZ (int id, int *block);

        void iteration1 (int ax, const std::vector<int> &b, std::vector<double> &buff, std::vector<double> &data);
        void iteration2 (int ax, const std::vector<int> &b, std::vector<double> &buff, std::vector<double> &data);
    public:
        DirihleTask (int argc, char *argv[]);
        ~DirihleTask ();

        void inputTaskInfo ();
        void YakobiSolve ();
    private:
        int id;
        MPI_Status status;
        std::vector<double> l, u, data;
        int block[3], dimension[3];
        double eps, u0;
        std::string output;
};

#endif