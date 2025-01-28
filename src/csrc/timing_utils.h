#ifndef TIMING_UTILS_H
#define TIMING_UTILS_H

#include <chrono>
#include <string>
#include <iostream>

#define TIMED_CPU_FUNCTION() timers::FunctionTimer timer(__FUNCTION__)

namespace timers {
class FunctionTimer {
public:
    FunctionTimer(std::string function_name) :
    function_name_(function_name), start_(std::chrono::high_resolution_clock::now()) {}

    ~FunctionTimer() {
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start_).count();
        std::cout << "CPU function " << function_name_ << " finished in " << duration << " ms" <<std::endl;
    }

private:
std::string function_name_;
std::chrono::time_point<std::chrono::high_resolution_clock> start_;

};
}

#endif //TIMING_UTILS_H