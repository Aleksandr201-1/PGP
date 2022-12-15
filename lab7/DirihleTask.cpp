#include "DirihleTask.hpp"

uint64_t DirihleTask::index (uint64_t i, uint64_t j, uint64_t k, int *dim) {
    return (k + 1) * (dim[Y] + 2) * (dim[X] + 2) + (j + 1) * (dim[X] + 2) + i + 1;
}

uint64_t DirihleTask::indexBlock (uint64_t i, uint64_t j, uint64_t k, int *block) {
    return k * block[Y] * block[X] + j * block[X] + i;
}

int DirihleTask::indexBlockX (int id, int *block) {
    return id % block[X];
}

int DirihleTask::indexBlockY (int id, int *block) {
    return (id % (block[Y] * block[X])) / block[X];
}

int DirihleTask::indexBlockZ (int id, int *block) {
    return id / block[Y] / block[X];
}

void DirihleTask::iteration1 (int ax, const std::vector<int> &b, std::vector<double> &buff, std::vector<double> &data) {
    int *dim = dimension;
    int d[] = {0, 0, 0};
    d[ax] = 1;
    int qq[2];
    qq[ax % 2] = (ax + COUNT_OF_AXES - 1) % COUNT_OF_AXES;
    qq[(ax + 1) % 2] = (ax + (COUNT_OF_AXES - 1) * 2) % COUNT_OF_AXES;
    int dir = ax * 2;
    if (b[ax] < block[ax] - 1) {
        for (uint64_t k = 0; k < dim[qq[0]]; ++k) {
            for (uint64_t j = 0; j < dim[qq[1]]; ++j) {
                int coeff[3];
                uint64_t tmp[] = {j, k};
                for (uint64_t l = 0; l < ax; ++l) {
                    coeff[l] = tmp[l];
                }
                coeff[ax] = dim[ax] - 1;
                for (uint64_t l = ax + 1; l < COUNT_OF_AXES; ++l) {
                    coeff[l] = tmp[l - 1];
                }
                buff[k * dim[qq[1]] + j] = data[index(coeff[0], coeff[1], coeff[2], dim)];
            }
        }
        MPI_Send(&buff[0], dim[qq[0]] * dim[qq[1]], MPI_DOUBLE, indexBlock(b[X] + d[X], b[Y] + d[Y], b[Z] + d[Z], block), id, MPI_COMM_WORLD);
    }

    if (b[ax] > 0) {
        MPI_Recv(&buff[0], dim[qq[0]] * dim[qq[1]], MPI_DOUBLE, indexBlock(b[X] - d[X], b[Y] - d[Y], b[Z] - d[Z], block), indexBlock(b[X] - d[X], b[Y] - d[Y], b[Z] - d[Z], block), MPI_COMM_WORLD, &status);
        for (uint64_t k = 0; k < dim[qq[0]]; ++k) {
            for (uint64_t j = 0; j < dim[qq[1]]; ++j) {
                int coeff[3];
                uint64_t tmp[] = {j, k};
                for (uint64_t l = 0; l < ax; ++l) {
                    coeff[l] = tmp[l];
                }
                coeff[ax] = -1;
                for (uint64_t l = ax + 1; l < COUNT_OF_AXES; ++l) {
                    coeff[l] = tmp[l - 1];
                }
                data[index(coeff[0], coeff[1], coeff[2], dim)] = buff[k * dim[qq[1]] + j];
            }
        }
    } else {
        for (uint64_t k = 0; k < dim[qq[0]]; ++k) {
            for (uint64_t j = 0; j < dim[qq[1]]; ++j) {
                int coeff[3];
                uint64_t tmp[] = {j, k};
                for (uint64_t l = 0; l < ax; ++l) {
                    coeff[l] = tmp[l];
                }
                coeff[ax] = -1;
                for (uint64_t l = ax + 1; l < COUNT_OF_AXES; ++l) {
                    coeff[l] = tmp[l - 1];
                }
                data[index(coeff[0], coeff[1], coeff[2], dim)] = u[dir];
            }
        }
    }
}

void DirihleTask::iteration2 (int ax, const std::vector<int> &b, std::vector<double> &buff, std::vector<double> &data) {
    int *dim = dimension;
    int d[] = {0, 0, 0};
    d[ax] = 1;
    int qq[2];
    qq[ax % 2] = (ax + COUNT_OF_AXES - 1) % COUNT_OF_AXES;
    qq[(ax + 1) % 2] = (ax + (COUNT_OF_AXES - 1) * 2) % COUNT_OF_AXES;
    int dir = 1 + ax * 2;
    if (b[ax] > 0) {
        for (uint64_t k = 0; k < dim[qq[0]]; ++k) {
            for (uint64_t j = 0; j < dim[qq[1]]; ++j) {
                int coeff[3];
                uint64_t tmp[] = {j, k};
                for (uint64_t l = 0; l < ax; ++l) {
                    coeff[l] = tmp[l];
                }
                coeff[ax] = 0;
                for (uint64_t l = ax + 1; l < COUNT_OF_AXES; ++l) {
                    coeff[l] = tmp[l - 1];
                }
                buff[k * dim[qq[1]] + j] = data[index(coeff[0], coeff[1], coeff[2], dim)];
            }
        }
        MPI_Send(&buff[0], dim[qq[0]] * dim[qq[1]], MPI_DOUBLE, indexBlock(b[X] - d[X], b[Y] - d[Y], b[Z] - d[Z], block), id, MPI_COMM_WORLD);
    }

    if (b[ax] < block[ax] - 1) {
        MPI_Recv(&buff[0], dim[qq[0]] * dim[qq[1]], MPI_DOUBLE, indexBlock(b[X] + d[X], b[Y] + d[Y], b[Z] + d[Z], block), indexBlock(b[X] + d[X], b[Y] + d[Y], b[Z] + d[Z], block), MPI_COMM_WORLD, &status);
        for (uint64_t k = 0; k < dim[qq[0]]; ++k) {
            for (uint64_t j = 0; j < dim[qq[1]]; ++j) {
                int coeff[3];
                uint64_t tmp[] = {j, k};
                for (uint64_t l = 0; l < ax; ++l) {
                    coeff[l] = tmp[l];
                }
                coeff[ax] = dim[ax];
                for (uint64_t l = ax + 1; l < COUNT_OF_AXES; ++l) {
                    coeff[l] = tmp[l - 1];
                }
                data[index(coeff[0], coeff[1], coeff[2], dim)] = buff[k * dim[qq[1]] + j];
            }
        }
    } else {
        for (uint64_t k = 0; k < dim[qq[0]]; ++k) {
            for (uint64_t j = 0; j < dim[qq[1]]; ++j) {
                int coeff[3];
                uint64_t tmp[] = {j, k};
                for (uint64_t l = 0; l < ax; ++l) {
                    coeff[l] = tmp[l];
                }
                coeff[ax] = dim[ax];
                for (uint64_t l = ax + 1; l < COUNT_OF_AXES; ++l) {
                    coeff[l] = tmp[l - 1];
                }
                data[index(coeff[0], coeff[1], coeff[2], dim)] = u[dir];
            }
        }
    }
}

DirihleTask::DirihleTask (int argc, char *argv[]) {
    l.resize(3);
    u.resize(6);
    int numproc;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numproc);
    MPI_Comm_rank(MPI_COMM_WORLD, &id);
}

DirihleTask::~DirihleTask () {
    MPI_Finalize();
}

void DirihleTask::inputTaskInfo () {
    if (id == 0) {
        std::cin >> block[X] >> block[Y] >> block[Z];
        std::cin >> dimension[X] >> dimension[Y] >> dimension[Z];
        std::cin >> output;
        std::cin >> eps;
        std::cin >> l[X] >> l[Y] >> l[Z];
        std::cin >> u[DOWN] >> u[UP];
        std::cin >> u[LEFT] >> u[RIGHT];
        std::cin >> u[FRONT] >> u[BACK];
        std::cin >> u0;
    }

    MPI_Bcast(dimension, 3, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(block, 3, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&eps, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&l[0], l.size(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&u[0], u.size(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&u0, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
}

void DirihleTask::YakobiSolve () {
    double total_diff;
    std::vector<int> b(COUNT_OF_AXES);
    int *dim = dimension;

    inputTaskInfo();
    data.resize((dim[X] + 2) * (dim[Y] + 2) * (dim[Z] + 2));

    std::vector<double> buff, next;
    buff.resize(std::max(dim[X], std::max(dim[Y], dim[Z])) * std::max(dim[X], std::max(dim[Y], dim[Z])) + 2);
    next.resize((dim[X] + 2) * (dim[Y] + 2) * (dim[Z] + 2));

    for (uint64_t i = 0; i < dim[X]; ++i) {
        for (uint64_t j = 0; j < dim[Y]; ++j) {
            for (uint64_t k = 0; k < dim[Z]; ++k) {
                data[index(i, j, k, dim)] = u0;
            }
        }
    }

    b[X] = indexBlockX(id, block);
    b[Y] = indexBlockY(id, block);
    b[Z] = indexBlockZ(id, block);

    double hx = l[X] / (dim[X] * block[X]);
    double hy = l[Y] / (dim[Y] * block[Y]);
    double hz = l[Z] / (dim[Z] * block[Z]);

    total_diff = eps + 1;
    while (total_diff > eps) {
        for (int ax = X; ax < COUNT_OF_AXES; ++ax) {
            iteration1(ax, b, buff, data);
        }

        for (int ax = X; ax < COUNT_OF_AXES; ++ax) {
            iteration2(ax, b, buff, data);
        }

        MPI_Barrier(MPI_COMM_WORLD);
        total_diff = 0.0;
        for (uint64_t i = 0; i < dim[X]; ++i) {
			for (uint64_t j = 0; j < dim[Y]; ++j) {
				for (uint64_t k = 0; k < dim[Z]; ++k) {
					next[index(i, j, k, dim)] = 0.5 * ((data[index(i + 1, j, k, dim)] + data[index(i - 1, j, k, dim)]) / (hx * hx) +
						(data[index(i, j + 1, k, dim)] + data[index(i, j - 1, k, dim)]) / (hy * hy) +
						(data[index(i, j, k + 1, dim)] + data[index(i, j, k - 1, dim)]) / (hz * hz)) /
						(1.0 / (hx * hx) + 1.0 / (hy * hy) + 1.0 / (hz * hz));
					total_diff = std::max(total_diff, fabs(next[index(i, j, k, dim)] - data[index(i, j, k, dim)]));
				}
			}
		}
        std::swap(next, data);
        MPI_Allreduce(MPI_IN_PLACE, &total_diff, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    }

    if (id != 0) {
        for (uint64_t k = 0; k < dim[Z]; ++k) {
            for (uint64_t j = 0; j < dim[Y]; ++j) {
                for (uint64_t i = 0; i < dim[X]; ++i) {
                    buff[i] = data[index(i, j, k, dim)];
                }
                MPI_Send(&buff[0], dim[X], MPI_DOUBLE, 0, id, MPI_COMM_WORLD);
            }
        }
    } else {
        std::ofstream outFile(output, std::ios::out);
        outFile << std::scientific << std::setprecision(6);
        for (b[Z] = 0; b[Z] < block[Z]; ++b[Z]) {
            for (uint64_t k = 0; k < dim[Z]; ++k) {
                for (b[Y] = 0; b[Y] < block[Y]; ++b[Y]) {
                    for (uint64_t j = 0; j < dim[Y]; ++j) {
                        for (b[X] = 0; b[X] < block[X]; ++b[X]) {
                            if (indexBlock(b[X], b[Y], b[Z], block) == 0) {
                                for (uint64_t i = 0; i < dim[X]; ++i) {
                                    buff[i] = data[index(i, j, k, dim)];
                                }
                            } else {
                                MPI_Recv(&buff[0], dim[X], MPI_DOUBLE, indexBlock(b[X], b[Y], b[Z], block), indexBlock(b[X], b[Y], b[Z], block), MPI_COMM_WORLD, &status);
                            }

                            for (uint64_t i = 0; i < dim[X]; ++i) {
                                outFile << buff[i] << ' ';
                            }
                        }
                    }
                }
            }
        }
        outFile.close();
    }
}