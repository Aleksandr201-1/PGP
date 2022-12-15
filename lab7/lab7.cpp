#include <iostream>
#include "DirihleTask.hpp"

int main(int argc, char *argv[]){
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(nullptr);

    DirihleTask task(argc, argv);
    task.YakobiSolve();

    return 0;
}