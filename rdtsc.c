// rdtsc.c
// lib gen with gcc -O2 -Wall -fPIC -shared -o librdtsc.so rdtsc.c
#include <stdint.h>

uint64_t read_cycles() {
    unsigned int hi, lo;
    __asm__ volatile (
        "lfence\n"             // Serialize
        "rdtsc\n"              // Read time-stamp counter
        : "=a"(lo), "=d"(hi)   // Output
    );
    return ((uint64_t)hi << 32) | lo;
}
