#include "profiler.hpp"
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <time.h>
#include <sys/time.h>

#define DIM 1000

int matrix_a[DIM][DIM];
int matrix_b[DIM][DIM];
int matrix_c[DIM][DIM];

void init() {
    srand(time(NULL));
    for(int i = 0; i < DIM; i++) {
        for(int j = 0; j < DIM; j++) {
            matrix_a[i][j] = rand() % 10;
            matrix_b[i][j] = rand() % 10;
            matrix_c[i][j] = 0;
        }
    }
}

void multiply() {
    for(int i = 0; i < DIM; i++) {
        for(int j = 0; j < DIM; j++) {
            for(int k = 0; k < DIM; k++) {
                matrix_c[i][j] += matrix_a[i][k] * matrix_b[k][j];
            }
        }
    }
}

void print_matrix(int* matrix) {
    printf("matrix:\n");
    for(int i = 0; i < DIM; i++) {
        for(int j = 0; j < DIM; j++) {
            if (j == DIM - 1) {
                printf("%d", matrix[i * DIM + j]);
            } else {
                printf("%d,", matrix[i * DIM + j]);
            }
        }
        printf("\n");
    }
}

int main(void) {    
    auto timed_init = profile_time(init);
    timed_init()

    auto timed_multiply = profile_time(multiply);
    timed_multiply();

    return 0;
}