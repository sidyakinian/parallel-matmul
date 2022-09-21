#include <functional>
#include <chrono>
#include <iostream>

#ifndef _PROFILER_H_
#define _PROFILER_H_

// Class for timing functions

template <class> struct ExeTime;

template <typename... Args>
struct ExeTime<void(Args ...)> {
public:
    ExeTime(std::function<void(Args...)> func): f_(func) { } 
    void operator()(Args ... args) {
        std::chrono::time_point<std::chrono::system_clock> start, end;
        std::chrono::duration<double> elasped;

        start = std::chrono::system_clock::now();
        f_(args...);    
        end = std::chrono::system_clock::now();
        elasped = end - start;
        std::cout << elasped.count() << " seconds" << std::endl;  
    }
private:
    std::function<void(Args ...)> f_; 
};

template <typename... Args>
ExeTime<void(Args ...)> make_decorator(void (*f)(Args ...)) {
    return ExeTime<void(Args...)>(std::function<void(Args...)>(f));    
}

#endif